"""Run registered development jobs with input-addressed checkpoints and bounded concurrency."""
import argparse
import ast
from importlib import metadata
from importlib.util import resolve_name
from concurrent.futures import ThreadPoolExecutor
import fcntl
import glob
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

from research_grasping.phase3.experiment import digest, fingerprint, load, save

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'research/phase4/automation/workflow.json'


def inventory(patterns):
    hashes = {}
    for pattern in patterns:
        paths = sorted(Path(p) for p in glob.glob(pattern,recursive=True) if Path(p).is_file())
        if not paths:
            raise FileNotFoundError(pattern)
        for path in paths:
            hashes[str(path)] = digest(path)
    return hashes


def implementation_inputs(paths):
    """Follow repository Python imports, including local imports within functions."""
    found={}
    todo=[Path(name).resolve() for name in paths if str(name).endswith('.py')]
    while todo:
        path=todo.pop()
        if path in found or not path.is_file() or not path.is_relative_to(ROOT):
            continue
        found[path]=digest(path)
        package='.'.join(path.relative_to(ROOT).parts[:-1])
        imports=[]
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node,ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                module=resolve_name('.'*node.level+(node.module or ''),package)
                imports.append(module)
                imports.extend(module+'.'+alias.name for alias in node.names)
        for module in imports:
            if module.split('.')[0] in ('research_grasping','axisforge_bo'):
                base=ROOT.joinpath(*module.split('.'))
                candidate=base.with_suffix('.py')
                todo.append(candidate if candidate.is_file() else base/'__init__.py')
    return {str(k):v for k,v in found.items()}


def runtime_versions():
    versions={'python':sys.version}
    for library in ('numpy','scipy','scikit-learn','torch','torchvision','open_clip_torch',
                    'mujoco','transformers','gpytorch','botorch','trimesh','pillow','matplotlib'):
        try: versions[library]=metadata.version(library)
        except metadata.PackageNotFoundError: versions[library]=None
    return versions


def execute(name, spec, values, state, timeout):
    values=dict(values)
    for key,upstream in spec.get('upstream_hashes',{}).items():
        filename=upstream['path'].format_map(values)
        upstream_state=state/(upstream['stage']+'.json')
        if not upstream_state.exists():
            return dict(status='BLOCKED',reason='Missing upstream receipt')
        reference=load(upstream_state)
        expected=reference.get('outputs',{}).get(filename)
        if (reference.get('status')!='COMPLETED' or not expected or not Path(filename).is_file()
                or digest(filename)!=expected):
            return dict(status='BLOCKED',reason='Upstream receipt does not match its artifact')
        values[key]=expected
    def formatted(key):
        return [s.format_map(values) for s in spec[key]]
    state.mkdir(parents=True,exist_ok=True)
    receipt = state/(name+'.json')
    with (state/(name+'.lock')).open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        started=time.monotonic()
        try:
            inputs=inventory(formatted('inputs'))
        except FileNotFoundError as exc:
            return dict(status='BLOCKED',missing=str(exc))
        command=formatted('args')
        signature=fingerprint(dict(command=command,inputs=inputs,implementation=implementation_inputs(inputs),
                                   versions=runtime_versions(),threads=values['threads'],
                                   outputs=formatted('outputs'),workflow=digest(__file__)))
        if receipt.exists():
            old=load(receipt)
            if old['signature']!=signature:
                return dict(status='BLOCKED',reason='Inputs or implementation changed. Select a fresh work and state directory.')
            if old['status']=='COMPLETED':
                try: current=inventory(formatted('outputs'))
                except FileNotFoundError: current={}
                if current==old['outputs']:
                    return dict(status='REUSED',elapsed_s=old['elapsed_s'],signature=signature)
                return dict(status='BLOCKED',reason='Completed outputs changed. Preserve evidence and select a fresh work directory.')
        log=state/(name+'.log')
        entry=dict(status='RUNNING',signature=signature,args=command,log=str(log))
        save(receipt,entry)
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
                 OPENBLAS_NUM_THREADS=str(values['threads']),OMP_NUM_THREADS=str(values['threads']),
                 MKL_NUM_THREADS=str(values['threads']))
        try:
            with log.open('ab') as stream:
                subprocess.run(command,cwd=ROOT,env=env,check=True,timeout=timeout,stdout=stream,stderr=subprocess.STDOUT)
            entry.update(status='COMPLETED',outputs=inventory(formatted('outputs')))
        except (OSError, subprocess.SubprocessError) as exc:
            entry.update(status='FAILED',reason=str(exc))
        entry['elapsed_s']=time.monotonic()-started
        save(receipt,entry)
        return entry


def schedule(jobs, targets, runner, workers):
    if workers<1: raise ValueError('positive concurrency required')
    def dependencies(name, ancestors):
        if name in ancestors: raise ValueError('dependency cycle')
        if name not in jobs: raise ValueError(f'unknown job {name}')
        result={name}
        for parent in jobs[name].get('requires',[]):
            result.update(dependencies(parent,ancestors|{name}))
        return result
    pending=set().union(*(dependencies(t,set()) for t in targets))
    results={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending:
            ready=sorted(n for n in pending if set(jobs[n].get('requires',[]))<=results.keys())
            futures={}
            for n in ready:
                failed=[p for p in jobs[n].get('requires',[]) if results[p]['status'] not in ('COMPLETED','REUSED')]
                if failed: results[n]=dict(status='BLOCKED',requires=failed)
                else: futures[n]=pool.submit(runner,n,jobs[n])
            for n,task in futures.items(): results[n]=task.result()
            pending.difference_update(ready)
    return results



def foundation_jobs(plan, registry, device):
    """Register only model variants with independently supplied exact official pins."""
    jobs=plan['jobs']
    analysis=['{work}/model-analysis.json']
    deps=['visual_figure']
    for kind,pins in registry.items():
        if kind not in ('dinov2','dinov3'):
            raise ValueError('expected dinov2 or dinov3')
        for key,size in [('revision',40),('weights',64),('config',64),('processor',64)]:
            if not re.fullmatch('[a-f0-9]{'+str(size)+'}',str(pins.get(key,''))):
                raise ValueError('invalid independently verified model pin: '+key)
        directory='{work}/official-'+kind
        source='{root}/tools/research/phase4/'
        job='{root}/research/phase4/embedding_experiment/development-images.zip'
        destination='{work}/'+kind
        weight_files=[directory+'/'+n for n in ('model.safetensors','config.json','preprocessor_config.json')]
        download='download_'+kind
        extract='extract_'+kind
        scientific='analyze_'+kind
        def register(name,tool,arguments,inputs,outputs,previous):
            script=source+tool+'.py'
            jobs[name]=dict(args=['{py}',script]+arguments,inputs=[script]+inputs,outputs=outputs,requires=previous)
        register(download,'download_models',['--model',kind,'--revision',pins['revision'],'--out',directory],
                 [],weight_files,[])
        register(extract,'run_experiments',['--model',kind,'--checkpoint',directory,'--job',job,'--work',destination+'-work',
                 '--bundle',destination+'.zip','--device',device,'--revision',pins['revision'],'--weights-sha256',pins['weights'],
                 '--config-sha256',pins['config'],'--processor-sha256',pins['processor']], [job]+weight_files,
                 [destination+'.zip'],[download])
        output=destination+'-analysis.json'
        register(scientific,'import_embedding_bundle',['--job',job,'--bundle',destination+'.zip','--destination',
                 destination+'-import','--analysis-out',output,'--expected-revision',pins['revision'],
                 '--expected-checkpoint-sha256',pins['weights'],'--expected-processor-sha256',pins['processor']],
                 [job,destination+'.zip'],[output],[extract])
        analysis.append(output);deps.append(scientific)
    module='research_grasping.phase4.model_comparison'
    jobs['model_table']=dict(args=['{py}','-m',module,'--inputs']+analysis+['--out','{work}/foundation-comparison.json'],
        inputs=['{root}/research_grasping/phase4/model_comparison.py']+analysis,
        outputs=['{work}/foundation-comparison.json'],requires=deps)
    plan['modes']['foundation_model_comparison']=['model_table']
    manifest=load(ROOT/'research/phase3/dataset/manifest.lock.json')
    if 'development_assets' in jobs:
        outputs=jobs['development_assets']['outputs']
        for row in manifest['objects']:
            if row['split'] in ('train','validation'):
                orig=Path(row['original_mesh'])
                paths=[orig,orig.parent/'model.xml',Path(row['visual_mesh'])]
                if row['texture_availability']:paths.append(orig.parent/'texture.png')
                outputs.extend('{data}/'+str(p) for p in paths)



def main(argv=None):
    plan=load(PLAN)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=sorted(plan['modes']))
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/workspace/axisforge-phase3-data'))
    p.add_argument('--binary',type=Path,default=ROOT/'target/release/axisforge')
    p.add_argument('--selector',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--timeout',type=int,default=14400)
    p.add_argument('--only',help='one node of the registered graph and its dependencies')
    p.add_argument('--model-registry',type=Path,help='optional verified official DINO pins (revision, weights, config, processor)')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args(argv)
    if a.mode=='confirmatory':
        p.exit(2,'BLOCKED: M10 lacks an independently approved immutable protocol.\n')
    if min(a.threads,a.jobs,a.timeout)<1: p.error('resource limits must be positive')
    foundation_jobs(plan,load(a.model_registry) if a.model_registry else {},a.device)
    if a.only and a.only not in plan['jobs']:p.error('unknown workflow job')
    work=a.work.resolve(); data=a.data.resolve()
    values=dict(root=str(ROOT),py=sys.executable,work=str(work),data=str(data),binary=str(a.binary.resolve()),
                selector=str((a.selector or work/'build/release/axisforge').resolve()),threads=a.threads,
                checkpoint=str((a.checkpoint or data/'weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt').resolve()))
    work.mkdir(parents=True,exist_ok=True)
    targets=[a.only] if a.only else plan['modes'][a.mode]
    results=schedule(plan['jobs'],targets,lambda n,s:execute(n,s,values,work/'workflow-state',a.timeout),a.jobs)
    print(json.dumps(results,indent=2))
    return int(any(r['status'] not in ('COMPLETED','REUSED') for r in results.values()))


if __name__=='__main__':
    raise SystemExit(main())
