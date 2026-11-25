from pathlib import Path
import sys

import pytest
from research_grasping.phase4.workflow import execute, main, schedule


def test_dag_failure_isolated_and_cycle_blocked():
    jobs={'ok':{},'bad':{},'child':{'requires':['bad']},'leaf':{'requires':['ok']}}
    calls=[]
    def run(name,_):
        calls.append(name)
        return {'status':'FAILED' if name=='bad' else 'COMPLETED'}
    result=schedule(jobs,['child','leaf'],run,2)
    assert result['child']==dict(status='BLOCKED',requires=['bad']) and 'child' not in calls
    assert result['leaf']['status']=='COMPLETED'
    with pytest.raises(ValueError,match='cycle'):schedule({'a':{'requires':['a']}},['a'],run,1)
    with pytest.raises(ValueError,match='unknown'):schedule({},['a'],run,1)
    with pytest.raises(ValueError):schedule({},[],run,0)


def test_checkpoint_input_output_failure_and_retry(tmp_path):
    source=tmp_path/'in';dest=tmp_path/'out';script=tmp_path/'run.py'
    script.write_text("import sys\nfrom pathlib import Path\na,b=map(Path,sys.argv[1:])\nb.write_bytes(a.read_bytes())\n")
    spec=dict(inputs=[str(source),str(script)],outputs=[str(dest)],args=[sys.executable,str(script),str(source),str(dest)])
    def run(name='one',job=spec):
        return execute(name,job,{'threads':1},tmp_path/'state',3)['status']
    assert run()=='BLOCKED'
    source.write_bytes(b'first')
    assert run()=='COMPLETED' and run()=='REUSED'
    dest.write_bytes(b'corrupt')
    assert run()=='BLOCKED'
    dest.unlink()
    assert run()=='BLOCKED'
    source.write_bytes(b'changed')
    assert run()=='BLOCKED'
    assert run('failed',dict(spec,args=[sys.executable,'-c','raise SystemExit(2)']))=='FAILED'


def test_confirmatory_guard_precedes_job_and_data_access(tmp_path):
    with pytest.raises(SystemExit) as error:main(['confirmatory','--work',str(tmp_path/'fresh'),'--only','physical'])
    assert error.value.code==2 and not (tmp_path/'fresh').exists()


def test_receipt_seals_nested_result(tmp_path):
    from research_grasping.phase3.experiment import digest,save
    nested=tmp_path/'nested.json';nested.write_text('{"hyperparameter":2}')
    dest=tmp_path/'new';state=tmp_path/'state'
    spec=dict(inputs=[str(nested)],outputs=[str(dest)],args=[sys.executable,'-c',
        'import sys,pathlib;pathlib.Path(sys.argv[1]).write_text(sys.argv[2])',str(dest),'{nested_sha}'],
        upstream_hashes={'nested_sha':{'stage':'nested','path':str(nested)}})
    assert execute('next',spec,{'threads':1},state,3)['status']=='BLOCKED'
    save(state/'nested.json',dict(status='COMPLETED',outputs={str(nested):digest(nested)}))
    assert execute('next',spec,{'threads':1},state,3)['status']=='COMPLETED'
    assert dest.read_text()==digest(nested)
    nested.write_text('{}')
    assert execute('next',spec,{'threads':1},state,3)['status']=='BLOCKED'


def test_foundation_registry_separates_models_and_requires_external_pins():
    from research_grasping.phase4.workflow import foundation_jobs
    plan={'jobs':{},'modes':{}}
    pins={'revision':'c'*40,'weights':'a'*64,'config':'b'*64,'processor':'e'*64}
    foundation_jobs(plan,{'dinov2':pins,'dinov3':pins},'cpu')
    assert set(plan['jobs']['model_table']['requires'])=={'visual_figure','analyze_dinov2','analyze_dinov3'}
    assert plan['jobs']['extract_dinov2']['requires']==['download_dinov2']
    assert 'c'*40 in plan['jobs']['analyze_dinov3']['args']
    with pytest.raises(ValueError,match='pin'): foundation_jobs(plan,{'dinov2':dict(pins,weights='invalid')},'cpu')
    with pytest.raises(ValueError,match='expected'): foundation_jobs(plan,{'fake':pins},'cpu')
