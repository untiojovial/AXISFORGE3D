from copy import deepcopy
import subprocess
import pytest

import research_grasping.phase3.experiment as ex
from research_grasping.phase3 import pilot_cli as cli


@pytest.fixture
def setup(tmp_path, monkeypatch):
    raw = []
    for name, split in [('a','train'),('b','validation'),('t','test')]:
        mesh = tmp_path / name
        mesh.write_text(name)
        raw.append(dict(object_id=name, split=split, group=name, visual_mesh=name, processed_sha256=ex.digest(mesh)))
    manifest = tmp_path/'manifest'
    ex.save(manifest, dict(objects=raw))
    monkeypatch.setattr(ex, 'PREPILOT_MANIFEST', ex.digest(manifest))
    lock = dict(phase='development',manifest_sha256=ex.digest(manifest),train_size=1,validation_size=1,test_size=1)
    data = ex.DevelopmentData(manifest, lock, tmp_path)
    binary = tmp_path/'binary'
    binary.write_text('test-binary')
    catalogs = []
    for r in raw:
        name=r['object_id']
        row=dict(object_id=name,split=r['split'],mesh_file=str(tmp_path/name),project=str(tmp_path/name),
                 body=name,ids=['0','1','2','3'],context=[1.],features=[[0.],[1.],[2.],[3.]],
                 candidates={str(i):dict(position=[0,0,0],contacts=[]) for i in range(4)},
                 points=[[0,0,0]],frame='world',
                 metadata=dict(manifest_sha256=ex.digest(manifest),processed_sha256=r['processed_sha256'],
                    binary_sha256=ex.digest(binary),project_sha256=ex.digest(tmp_path/name)))
        path=tmp_path/(name+'.json')
        ex.save(path,row)
        if r['split'] != 'test':
            catalogs.append(data.catalogue(path,binary))
    a,b=catalogs
    log=dict(object_id='a', selected=['0','2'], fidelity='f', catalogue_digest=ex.fingerprint(a),
             completed_at=1.,rows=[dict(candidate_id=i,status='ok',utility=v,cost=.1) for i,v in [('0',.4),('2',.6)]])
    return data, a, b, log, lock, binary


def test_entry_and_replay(setup,tmp_path):
    data,a,b,log,*_=setup
    seen=[]
    def oracle(candidate):
        seen.append(candidate)
        if len(seen)==2:
            raise OSError('lost worker')
        return dict(status='ok',utility=.5,cost=.1,details={})
    def run():
        return ex.run_policy(data,b,[(a,log)],method='adaptive',seed=7,budget=3,initial_count=1,
                   fidelity='f',target_time=2.,observe=oracle,checkpoint=tmp_path/'run')
    with pytest.raises(OSError):
        run()
    assert ex.load(tmp_path/'run')['pending']==seen[1]
    r=run()
    assert len(r['rows'])==3 and seen[1]==seen[2] and len(set(seen))==3
    assert run()==r and len(seen)==4
    assert r['rows'][0]['prediction'] is not None
    assert ex.viewer_report([b],[r])['schema']=='axisforge.research.v1'


def test_sealed_and_tamper(setup,tmp_path):
    data,a,b,log,lock,binary=setup
    with pytest.raises(ValueError,match='development access denied'):
        data.catalogue(tmp_path/'t.json',binary)
    for key, value in [('split','train'),('features',[[0.]]*4)]:
        fake=deepcopy(b);fake[key]=value
        with pytest.raises(ValueError):
            data.require_catalogue(fake,'train')
    ex.save(tmp_path/'fake',dict(objects=[]))
    with pytest.raises(ValueError):
        ex.DevelopmentData(tmp_path/'fake',dict(lock,manifest_sha256=ex.digest(tmp_path/'fake')),tmp_path)
    with pytest.raises(ValueError):
        ex.fit_history(data,[(b,log)], 'f')
    for bad in [dict(log,selected=['0','0']),dict(log,fidelity='f3'),dict(log,catalogue_digest='forged')]:
        with pytest.raises(ValueError):
            ex.fit_history(data,[(a,bad)],'f')
    with pytest.raises(ValueError):
        ex.run_policy(data,b,[(a,dict(log,completed_at=3.))],method='adaptive',seed=7,budget=3,
            initial_count=1,fidelity='f',target_time=2.,observe=None,checkpoint=tmp_path/'unused')
    (tmp_path/'b').write_text('modified')
    with pytest.raises(ValueError,match='provenance'):
        data.catalogue(tmp_path/'b.json',binary)


def test_named_observation_infra_and_cache(tmp_path,monkeypatch):
    calls=[]
    def fake(binary,project,body,i):
        calls.append(i)
        if i==3:
            raise subprocess.TimeoutExpired('native',5)
        return dict(records=[dict(quality=dict(measure='wrench_space',score=.4,quality=.2,forceClosure=True),
            plan=None,rejection='unreachable',qualityElapsedS=.01,attemptElapsedS=.02)])
    monkeypatch.setattr(cli,'evaluate_candidate',fake)
    def get(i):
        return cli.cached_observation(tmp_path,'bin','proj','body','sig',i,'native-f12')
    assert get('2')['utility']==0. and get('2')['details']['cached']
    assert calls==[2]
    infra=get('3')
    assert infra['status']=='infrastructure_failure' and infra['utility'] is None and len(infra['details']['errors'])==3
    assert calls==[2,3,3,3]
