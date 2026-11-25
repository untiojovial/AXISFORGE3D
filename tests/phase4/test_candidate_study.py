import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from research_grasping.phase4 import catalogue_study as study
from research_grasping.phase4 import placement_study as placement
from research_grasping.phase4.candidate_results import summarize

D=Path('/workspace/axisforge-phase4-data')
DATA=Path('/workspace/axisforge-phase3-data')


def test_sampling_and_contact_frame():
    proto={'train_measurements_per_object':2,'uniform_seed':4,'validation_measurements_per_object':3}
    assert study.plans([3,1,2],'train','stratified',proto)==[3,1]
    assert len(set(study.plans([3,1,2],'validation','uniform',proto)))==3
    for ids,split,method in [([1,1],'train','uniform'),([1,2],'test','uniform'),([1,2],'train','x')]:
        with pytest.raises(ValueError):study.plans(ids,split,method,proto)
    def record(points):
        return {'candidate':{'contacts':[dict(point=dict(zip('xyz',p)),normal=dict(x=0,y=0,z=1)) for p in points]}}
    a=record([(0,0,1),(1,0,1)])
    assert placement.contacts_match(a,record([(2,0,1),(1,0,1)]),np.array([1,0,0]))
    assert not placement.contacts_match(a,record([(1,0,1)]),np.zeros(3))
    assert not placement.contacts_match(a,record([(0,0,1),(1.1,0,1)]),np.zeros(3))
    assert placement.contacts_match(record([]),record([]),np.zeros(3))


@pytest.mark.skipif(not (D/'selection.json').is_file(),reason='external measured native catalogues required')
def test_resume_authentic_catalogues_without_running_oracle(tmp_path,monkeypatch):
    binary=D/'bin/axisforge-phase3';selector=D/'build/release/axisforge'
    def fail(*args,**kwargs):pytest.fail('resume reran a paid evaluation')
    monkeypatch.setattr(study,'cached_observation',fail)
    lock=study.prepare(DATA,binary,selector,D)
    assert len(lock['objects'])==22
    study.measure(DATA,binary,D,2)
    totals=summarize(D,tmp_path)
    assert totals['unique_object_candidates']==1841 and totals['fresh_distinct']==1800
    with gzip.open(tmp_path/'native_observations.json.gz','rt') as f: rows=json.load(f)
    assert len(rows['history/stratified'])==16 and len(rows['validation_panel/uniform'])==6
    name=next(iter(lock['objects']))
    c=study.load(D/'catalogues/diverse128'/(name+'.json'))
    proj=study.load(c['project'])
    assert placement.supported(proj,c,np.zeros(3)) and not placement.supported(proj,c,np.array([20,0,0]))
    from copy import deepcopy
    other=deepcopy(proj)
    table=next(b for b in other['project']['environment']['bodies'] if b['name']=='table')
    table['geometries'][0]['shape']['kind']='sphere'
    with pytest.raises(ValueError,match='box'):placement.supported(other,c,np.zeros(3))


@pytest.mark.skipif(not (D/'selection.json').is_file(),reason='external inputs needed for rebuilding')
def test_rebuild_and_recover_selection_on_native_fixture(tmp_path,monkeypatch):
    # One development object through both real executables, retaining the sealed
    # manifest validation in DevelopmentData before narrowing this test fixture.
    original_class=study.DevelopmentData
    def one_object(*args,**kwargs):
        data=original_class(*args,**kwargs)
        chosen=next(k for k in sorted(data.splits) if data.splits[k]=='train')
        from types import SimpleNamespace
        return SimpleNamespace(splits={chosen:'train'},manifest_digest=data.manifest_digest,catalogue=data.catalogue)
    monkeypatch.setattr(study,'DevelopmentData',one_object)
    oracle=D/'bin/axisforge-phase3';selector=D/'build/release/axisforge'
    lock=study.prepare(DATA,oracle,selector,tmp_path)
    name,entry=next(iter(lock['objects'].items()))
    reference=study.load(D/'selection.json')['objects'][name]
    assert entry['sampling']==reference['sampling'] and entry['catalogues']==reference['catalogues']
    study.measure(DATA,oracle,tmp_path,1)
    for scheme in ('stratified','uniform'):
        result=study.load(tmp_path/'history'/scheme/(name+'.json'))
        prior=study.load(D/'history'/scheme/(name+'.json'))
        assert [(r['candidate_id'],r['status'],r['utility']) for r in result['rows']]==[
            (r['candidate_id'],r['status'],r['utility']) for r in prior['rows']]
    path=tmp_path/'history/stratified'/(name+'.json')
    invalid=study.load(path);invalid['selected'].reverse();study.save(path,invalid)
    with pytest.raises(ValueError,match='results differ'): study.measure(DATA,oracle,tmp_path,1)
    bad=study.load(tmp_path/'selection.json');bad['protocol_sha256']='different'
    study.save(tmp_path/'selection.json',bad)
    with pytest.raises(ValueError,match='another configuration'): study.prepare(DATA,oracle,selector,tmp_path)


@pytest.mark.skipif(not (D/'selection.json').is_file(),reason='observational locks needed')
def test_training_archive_requires_full_roster_and_authentic_sampling(tmp_path):
    from research_grasping.phase4.history_lock import seal
    from research_grasping.phase3.experiment import load,save
    import shutil
    shutil.copyfile(D/'selection.json',tmp_path/'selection.json')
    with pytest.raises(ValueError,match='full training'):seal(tmp_path)
    shutil.copytree(D/'history/stratified',tmp_path/'history/stratified',dirs_exist_ok=True)
    sealed=seal(tmp_path)
    assert sealed==load(study.ROOT/'research/phase4/candidates/history_stratified.lock.json')
    path=next((tmp_path/'history/stratified').glob('*.json'))
    original=load(path);bad=dict(original,selected=list(reversed(original['selected'])))
    save(path,bad)
    with pytest.raises(ValueError,match='selection differ'):seal(tmp_path)
    save(path,original)
    sealed[path.stem]='unexpected';save(tmp_path/'history_stratified.lock.json',sealed)
    with pytest.raises(ValueError,match='lock differs'):seal(tmp_path)


@pytest.mark.skipif(not (D/'selection.json').is_file(),reason='real pinned native selection required')
def test_no_paid_queries_on_mismatched_protocol_roster_or_native_ids(tmp_path,monkeypatch):
    from research_grasping.phase3.experiment import load,save
    from copy import deepcopy
    source=load(D/'selection.json')
    path=tmp_path/'selection.json'
    binary=D/'bin/axisforge-phase3'
    monkeypatch.setattr(study,'cached_observation',lambda *args:pytest.fail('queried after integrity error'))
    (tmp_path/'catalogues').symlink_to(D/'catalogues',target_is_directory=True)
    for field,value,error in [('protocol_sha256','different','protocol or oracle'),('objects',{},'roster')]:
        save(path,dict(source,**{field:value}))
        with pytest.raises(ValueError,match=error):study.measure(DATA,binary,tmp_path,1)
    changed=deepcopy(source)
    for entry in changed['objects'].values():
        for key in ('uniform','stratified'):entry['sampling'][key]=['not_a_native_id']
    save(path,changed)
    with pytest.raises(ValueError,match='differs from lock'):study.measure(DATA,binary,tmp_path,1)
    with pytest.raises(ValueError,match='catalogue or selection'):study.prepare(DATA,binary,D/'build/release/axisforge',tmp_path)
