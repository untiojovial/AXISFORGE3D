"""Exploratory F1/F2 BO on verified phase-III development catalogues and oracle."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import fcntl
import time
import numpy as np
from research_grasping.phase3.experiment import DevelopmentData, digest, fingerprint, fit_history, load, run_policy, save
from research_grasping.phase3.pilot_cli import observer
from research_grasping.phase3.transfer_policy import FrozenScale
from research_grasping.phase4.feasibility_policy import FeasibilityTask
from research_grasping.phase4.feasibility_policy_v2b import PrevalenceFeasibilityPolicy as FeasibilityPolicy


def make_history(archive, history):
    tasks=[]
    for (catalog,log), task in zip(archive,history):
        rows=[r for r in log['rows'] if r['status'] in ('ok','physical_failure')]
        tasks.append(FeasibilityTask(task.object_id,task.available_at,task.context,task.features,
                                    np.array([r['status']=='ok' for r in rows],float),task.cost))
    return tasks


def run_fa(data,catalog,archive,*,method,seed,budget,target_time,observe,checkpoint,selected):
    data.require_catalogue(catalog,'validation')
    history,scaler=fit_history(data,archive,'native-f12')
    tasks=make_history(archive,history)
    initial=np.random.default_rng(seed).choice(catalog['ids'],2,replace=False).tolist()
    policy=FeasibilityPolicy(catalog['ids'],scaler.transform(catalog['features']),catalog['context'],tasks,
        manifest_splits=data.splits,manifest_digest=fingerprint(data.splits),target_id=catalog['object_id'],
        target_time=target_time,scaler=scaler,budget=budget,initial_ids=initial,mode=method,
        length_scale=selected[0],prior_strength=selected[1])
    definition=dict(catalogue_digest=fingerprint(catalog),manifest_sha256=data.manifest_digest,
        archive_digest=scaler.training_digest, initial=initial,budget=budget,seed=seed,method='v2b_'+method,
        fidelity='native-f12',object_id=catalog['object_id'],target_time=target_time)
    checkpoint=Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.with_suffix('.lock').open('a') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        state=load(checkpoint) if checkpoint.exists() else dict(definition=definition,rows=[],pending=None)
        if state['definition']!=definition:raise ValueError('run definition changed')
        for row in state['rows']:
            if policy.ask()!=row['candidate_id'] or policy.pending[1]!=row['prediction']:
                raise ValueError('replay diverged')
            policy.tell(row['candidate_id'],status=row['status'],utility=row['utility'],cost=row['cost'])
        while len(state['rows'])<budget:
            cid=policy.ask()
            if state['pending'] not in (None,cid):raise ValueError('reservation diverged')
            state['pending']=cid;save(checkpoint,state)
            result=observe(cid)
            policy.tell(cid,status=result['status'],utility=result['utility'],cost=result['cost'])
            state['rows'].append(dict(**policy.records[-1],details=result['details']))
            state['pending']=None;save(checkpoint,state)
        return state


def grouped_diagnostic(archive,history,scaler):
    tasks=make_history(archive,history);results=[]
    for held in tasks:
        others=[s for s in tasks if s.object_id!=held.object_id]
        manifest={s.object_id:'train' for s in tasks}
        training_x=np.concatenate([s.features for s in others])
        scale=training_x.std(axis=0);scale[scale<1e-8]=1.
        fold_scaler=FrozenScale(training_x.mean(axis=0),scale,fingerprint([s.object_id for s in others]))
        p=FeasibilityPolicy(['dummy'],np.zeros((1,len(scaler.center))),held.context,others,
            manifest_splits=manifest,manifest_digest=fingerprint(manifest),target_id=held.object_id,
            target_time=max(s.available_at for s in tasks)+1,scaler=fold_scaler,budget=1,initial_ids=['dummy'],mode='adaptive')
        pred=p.predict(fold_scaler.transform(held.features))[0];truth=held.feasible
        results.append({'object_id':held.object_id,'n':len(truth),'positives':int(truth.sum()),
            'brier':float(np.mean((pred-truth)**2)),'constant_half_brier':float(np.mean((.5-truth)**2)),
            'mean_prediction':float(pred.mean()),'empirical_rate':float(truth.mean())})
    return {'definition':'leave-one-train-object-out prior-only diagnostics; scaler fitted on other train objects',
            'warning':'not nested calibration and no hyperparameter tuning; fixed rule evaluated out-of-group',
            'objects':results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for f in ('data','protocol','catalogues','binary','archive','cache','out','nested','history-lock'):
        p.add_argument('--'+f,type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,29]);p.add_argument('--budget',type=int,default=10)
    p.add_argument('--archive-prefix',type=int,default=None)
    p.add_argument('--nested-sha256',required=True,help='expected digest of frozen nested result')
    p.add_argument('--methods',nargs='+',default=[])
    p.add_argument('--fa-modes',nargs='+',default=['target','adaptive'])
    a=p.parse_args()
    if digest(a.nested) != a.nested_sha256:
        raise ValueError('nested result digest mismatch')
    selected=load(a.nested)['selected_all_train'];data=DevelopmentData(a.protocol/'manifest.prepilot.json',load(a.protocol/'lock.json'),a.data)
    catalogs=[data.catalogue(path,a.binary) for path in sorted(a.catalogues.glob('*.json')) if path.name!='pca.json']
    train=[c for c in catalogs if c['split']=='train'];targets=[c for c in catalogs if c['split']=='validation']
    archive=[]; history_hashes={}
    directory=a.archive/'history' if (a.archive/'history').exists() else a.archive
    for c in train:
        path=directory/(c['object_id']+'.json'); log=load(path);history_hashes[c['object_id']]=digest(path)
        if a.archive_prefix is not None:
            if a.archive_prefix < 1 or len(log['rows']) < a.archive_prefix:
                raise ValueError('incomplete requested historical prefix')
            log=dict(log, selected=log['selected'][:a.archive_prefix], rows=log['rows'][:a.archive_prefix])
        archive.append((c,log))
    nested_lock=load(a.nested.with_suffix('.lock.json'))
    if (digest(a.history_lock)!=nested_lock['history_lock_sha256'] or a.archive_prefix!=nested_lock['prefix']
            or history_hashes!=load(a.history_lock)
            or nested_lock['catalogues']!={c['object_id']:fingerprint(c) for c in train}
            or digest(a.nested.with_suffix('.lock.json'))!=load(a.nested)['lock_sha256']):
        raise ValueError('nested selection does not match fixed history/catalogues')
    history,scaler=fit_history(data,archive,'native-f12')
    a.out.mkdir(parents=True,exist_ok=True)
    definition=dict(catalogues=[fingerprint(c) for c in catalogs],archive=scaler.training_digest,
                    seeds=a.seeds,budget=a.budget,manifest=data.manifest_digest,binary=digest(a.binary),
                    archive_prefix=a.archive_prefix, history_hashes=history_hashes, methods=a.methods,fa_modes=a.fa_modes,
                    nested_sha256=digest(a.nested), selected=selected,
                    code_hashes={str(path):digest(path) for path in (Path(__file__),Path('research_grasping/phase4/feasibility_policy.py'),Path('research_grasping/phase4/feasibility_policy_v2b.py'),
                        Path('research_grasping/phase3/transfer_policy.py'),Path('research_grasping/phase3/experiment.py'))})
    lock=a.out/'lock.json'
    if lock.exists():
        frozen=load(lock)
        if frozen['definition']!=definition:raise ValueError('experiment definition changed')
    else:
        frozen=dict(definition=definition,target_time=time.time());save(lock,frozen)
    with ProcessPoolExecutor(max_workers=1) as executor:
        for catalog in targets:
            observe=observer(executor,a.cache,a.binary,catalog,'native-f12')
            for seed in a.seeds:
                for method in a.methods:
                    checkpoint=a.out/'runs'/catalog['object_id']/f'{seed}-{method}.json'
                    run_policy(data,catalog,archive,method=method,seed=seed,budget=a.budget,initial_count=2,
                               fidelity='native-f12',target_time=frozen['target_time'],observe=observe,checkpoint=checkpoint)
                for mode in a.fa_modes:
                    checkpoint=a.out/'runs'/catalog['object_id']/f'{seed}-v2b_{mode}.json'
                    run_fa(data,catalog,archive,method=mode,seed=seed,budget=a.budget,
                           target_time=frozen['target_time'],observe=observe,checkpoint=checkpoint,selected=selected)

if __name__=='__main__':main()
