"""Tests for the frozen prospective selection; no validation outcomes opened."""
import json
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[2]


def test_frozen_protocol_and_selection():
    path=ROOT/'research/phase4/candidates/reach_selection.json'
    protocol=ROOT/'research/phase4/candidates/reach_protocol.json'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(path)=='6833566d2e0ef828a1ad731a2e7aea2922ca23060cd98dfc445df55735b7cdc2'
    assert sha(protocol)=='c445bc275676b5df7257ffd3294a2f6979b1a8b3eef0eecb52cbeb4eae4081df'
    d=json.loads(path.read_text())
    assert d['training_n']==800 and d['pregrasp_train_n']==594 and d['ok_positive']==17
    assert len(d['objects'])==6
    for row in d['objects'].values():
        top,rand=row['top'],row['random']
        assert len(top)==len(rand)==12 and len(set(top+rand))==24
        assert not (set(top+rand)&set(row['excluded_previously_queried']))
        assert not (set(top+rand)&set(row['excluded_too_wide']))
        ordered=sorted(row['eligible_scores'],key=lambda x:(-row['eligible_scores'][x],int(x)))
        assert ordered[:12]==top


import shutil
import numpy as np
import pytest


@pytest.fixture
def isolated_reach(tmp_path, monkeypatch):
    """Rebuild from sealed train files and ID-only snapshot, never existing labels."""
    import research_grasping.phase4.reach_study as reach
    live=Path('/workspace/axisforge-phase4-data')
    if not (live/'history/stratified').is_dir() or not reach.DATA.is_dir():
        pytest.skip('external Phase IV development assets unavailable')
    frozen=json.loads((ROOT/'research/phase4/candidates/reach_selection.json').read_text())
    work=tmp_path/'work'; work.mkdir()
    shutil.copytree(live/'history'/'stratified',work/'history'/'stratified')
    shutil.copytree(live/'catalogues'/'diverse128',work/'catalogues'/'diverse128')
    cache=tmp_path/'cache'; cache.mkdir()
    for row in frozen['objects'].values():
        target=cache/row['signature']/'native-f12'; target.mkdir(parents=True,exist_ok=True)
        for cid in row['cache_existing_ids']:
            # Filenames only: labels intentionally do not exist in the fixture.
            (target/(cid+'.json')).write_text('DO NOT READ VALIDATION LABELS')
    monkeypatch.setattr(reach,'WORK',work)
    monkeypatch.setattr(reach,'CACHE',cache)
    monkeypatch.setattr(reach,'SELECTION',tmp_path/'selection.json')
    original_load=reach.load
    def only_authorized_load(path):
        p=Path(path)
        if 'validation_panel' in p.parts or cache in p.parents:
            pytest.fail('selection attempted to read a validation outcome')
        return original_load(path)
    monkeypatch.setattr(reach,'load',only_authorized_load)
    return reach,work,frozen


def test_rebuild_from_training_and_prior_ids_without_validation_labels(isolated_reach):
    reach,_,frozen=isolated_reach
    reach.train_and_select()
    actual=json.loads(reach.SELECTION.read_text())
    assert actual['model_sha256']==frozen['model_sha256']
    for name, row in actual['objects'].items():
        assert row['top']==frozen['objects'][name]['top']
        assert row['random']==frozen['objects'][name]['random']
        assert row['excluded_previously_queried']==frozen['objects'][name]['excluded_previously_queried']
        assert row['eligible_scores']==pytest.approx(frozen['objects'][name]['eligible_scores'],abs=1e-14)


def test_reject_training_corruption_before_selection(isolated_reach):
    reach,work,_=isolated_reach
    target=next((work/'history'/'stratified').glob('*.json'))
    target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError,match='training data changed'): reach.train_and_select()
    assert not reach.SELECTION.exists()


def test_changed_train_labels_change_scores_and_duplicate_id_rejected(isolated_reach,monkeypatch):
    reach,work,frozen=isolated_reach
    histories=list((work/'history'/'stratified').glob('*.json'))
    target=None
    for p in histories:
        doc=json.loads(p.read_text())
        if any(r['status']=='ok' for r in doc['rows']): target=p; break
    assert target is not None
    original_digest=reach.digest(target)
    row=next(r for r in doc['rows'] if r['status']=='ok')
    row['status']='physical_failure';row['details']['rejection']={'reason':'pregrasp_unreachable'}
    target.write_text(json.dumps(doc))
    real_digest=reach.digest
    # Simulate a separately resealed train dataset; the other test verifies unsealed corruption fails.
    monkeypatch.setattr(reach,'digest',lambda path: original_digest if Path(path)==target else real_digest(path))
    reach.train_and_select(); changed=json.loads(reach.SELECTION.read_text())
    assert changed['ok_positive']==16
    assert changed['model_sha256']!=frozen['model_sha256']
    assert any(abs(score-frozen['objects'][name]['eligible_scores'][cid])>1e-10
               for name,obj in changed['objects'].items() for cid,score in obj['eligible_scores'].items())
    reach.SELECTION.unlink()
    doc['rows'][1]['candidate_id']=doc['rows'][0]['candidate_id']
    target.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='training order differs'): reach.train_and_select()
    assert not reach.SELECTION.exists()


def test_results_match_144_raw_observations():
    import gzip
    from collections import Counter, defaultdict
    root=ROOT/'research/phase4/candidates'
    with gzip.open(root/'reach_observations.json.gz','rt') as f: records=json.load(f)
    summary=json.loads((root/'reach_results.json').read_text()); selection=json.loads((root/'reach_selection.json').read_text())
    counts=defaultdict(Counter); seen=set()
    for record in records:
        key=(record['object_id'],record['arm']); assert key not in seen; seen.add(key)
        assert [r['candidate_id'] for r in record['rows']]==selection['objects'][key[0]][key[1]]
        for row in record['rows']:
            reason=(row['details'].get('rejection') or {}).get('reason',row['status'])
            counts[key[1]][reason]+=1
    assert len(seen)==12 and sum(sum(c.values()) for c in counts.values())==144
    for arm,counter in counts.items():
        assert dict(counter)==summary['arms'][arm]['causes']
        assert counter['ok']==0


@pytest.mark.parametrize('mutation',['corrupt','duplicate'])
def test_measure_rejects_changed_selection_before_oracle(isolated_reach,monkeypatch,mutation):
    reach,_,frozen=isolated_reach
    selection=json.loads(json.dumps(frozen))
    row=next(iter(selection['objects'].values()))
    if mutation=='duplicate': row['random'][0]=row['top'][0]
    else: row['top'][0]='999999'
    reach.SELECTION.write_text(json.dumps(selection))
    if mutation=='duplicate':
        # Exercise the structural check even if an attacker also substitutes a digest.
        monkeypatch.setattr(reach,'SELECTION_HASH',reach.digest(reach.SELECTION))
    calls=[]
    monkeypatch.setattr(reach,'cached_observation',lambda *a,**kw: calls.append(a))
    with pytest.raises(ValueError,match='selection seal changed|invalid arms'):
        reach.measure()
    assert not calls


def test_measure_resume_after_call_before_record_and_detect_corruption(isolated_reach,monkeypatch,tmp_path):
    reach,work,frozen=isolated_reach
    # The sealed configuration is copied into the temporary run, never edited.
    reach.SELECTION.write_bytes((ROOT/'research/phase4/candidates/reach_selection.json').read_bytes())
    monkeypatch.setattr(reach,'STAMP',tmp_path/'prequery.json')
    calls=[]; interrupted=[False]
    def fake_observation(cache,binary,project,body,signature,candidate,fidelity):
        path=Path(cache)/signature/fidelity/(str(int(candidate))+'.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            value=json.loads(path.read_text()); value['details']['cached']=True
            return value
        value={'status':'physical_failure','utility':0.,'cost':12.34,
               'details':{'rejection':{'reason':'pregrasp_unreachable'},'cached':False}}
        path.write_text(json.dumps(value)); calls.append((body,candidate))
        if len(calls)==3 and not interrupted[0]:
            interrupted[0]=True
            raise RuntimeError('simulated interruption after cache commit')
        return value
    monkeypatch.setattr(reach,'cached_observation',fake_observation)
    with pytest.raises(RuntimeError,match='simulated interruption'): reach.measure()
    assert len(calls)==3 and reach.STAMP.exists()
    partial=list((work/'reach_results').glob('*.json'))
    assert len(partial)==1 and len(json.loads(partial[0].read_text())['rows'])==2
    reach.measure()
    assert len(calls)==144  # No duplicate physical query; third is recovered from cache.
    all_rows=[r for p in (work/'reach_results').glob('*.json') for r in json.loads(p.read_text())['rows']]
    assert len(all_rows)==144
    recovered=[r for r in all_rows if r['cost_provenance']=='recovered_cache_unknown']
    assert len(recovered)==1 and recovered[0]['cost']==12.34 and recovered[0]['details']['cached']
    before=len(calls); reach.measure(); assert len(calls)==before
    # A partial or completed result modified after its checksum must not resume.
    data=json.loads(partial[0].read_text()); data['rows'][0]['cost']=99
    partial[0].write_text(json.dumps(data))
    with pytest.raises(ValueError,match='checksum'): reach.measure()
    assert len(calls)==before


def test_measure_writer_lock_serializes_concurrent_calls(tmp_path,monkeypatch):
    import threading
    import research_grasping.phase4.reach_study as reach
    monkeypatch.setattr(reach,'WORK',tmp_path)
    entered=threading.Event(); finished=threading.Event()
    def fake_measure(): entered.set()
    monkeypatch.setattr(reach,'_measure_locked',fake_measure)
    def invoke(): reach.measure(); finished.set()
    with reach.writer_lock(tmp_path/'reach_results.lock'):
        worker=threading.Thread(target=invoke); worker.start()
        assert not entered.wait(.2)
    assert finished.wait(3)
    worker.join()


def test_resume_rejects_reordered_rows_even_with_recomputed_checksums(tmp_path):
    import research_grasping.phase4.reach_study as reach
    entry={'top':['7','8'],'catalogue_digest':'catalog','signature':'signature'}
    path=tmp_path/'arm.json'
    reach._save_arm(path,'object','top',entry,'selection',[{'candidate_id':'7','cost':1}])
    data=json.loads(path.read_text()); data['rows'][0]['candidate_id']='8'
    data['row_sha256']=[reach.fingerprint(r) for r in data['rows']]
    data['arm_sha256']=reach.fingerprint({k:v for k,v in data.items() if k!='arm_sha256'})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='candidate IDs mismatch'):
        reach._verified_arm(path,'object','top',entry,'selection')


@pytest.mark.parametrize('field,kind,error',[
    ('object_id','other','provenance'),('arm','random','provenance'),
    ('catalogue_digest','other','catalogue'),('signature','other','catalogue'),
    ('selected_ids',['7','9'],'row checksum or selection'),('row_sha256',['0'*64],'row checksum or selection'),
])
def test_arm_provenance_rejects_resealed_inconsistency(tmp_path,field,kind,error):
    import research_grasping.phase4.reach_study as reach
    entry={'top':['7','8'],'catalogue_digest':'catalog','signature':'signature'}
    path=tmp_path/'arm.json';reach._save_arm(path,'object','top',entry,'selection',[{'candidate_id':'7','cost':1}])
    data=json.loads(path.read_text());data[field]=kind
    data['arm_sha256']=reach.fingerprint({k:v for k,v in data.items() if k!='arm_sha256'})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match=error):reach._verified_arm(path,'object','top',entry,'selection')


def test_legacy_report_and_record_corruption_rejected(tmp_path,monkeypatch):
    import research_grasping.phase4.reach_study as reach
    report=tmp_path/'legacy.json';arm=tmp_path/'legacy-arm.json'
    record={'object_id':'object','arm':'top','selection_sha256':'s','rows':[{'candidate_id':'7'}]}
    arm.write_text(json.dumps(record));report.write_text(json.dumps({'selection_sha256':'s','measurement_file_sha256':{arm.name:reach.digest(arm)}}))
    monkeypatch.setattr(reach,'LEGACY_REPORT',report);monkeypatch.setattr(reach,'LEGACY_REPORT_HASH',reach.digest(report))
    entry={'top':['7']}
    assert reach._verified_arm(arm,'object','top',entry,'s')==record
    arm.write_text(json.dumps(record)+' ')
    with pytest.raises(ValueError,match='legacy arm checksum'):reach._verified_arm(arm,'object','top',entry,'s')
    report.write_text(report.read_text()+' ')
    with pytest.raises(ValueError,match='legacy report seal'):reach._verified_arm(arm,'object','top',entry,'s')


def test_reject_nonfinite_prequery_features():
    import research_grasping.phase4.reach_study as reach
    cat={'ids':['0'],'features':[[0,0,0,1,0,0,0,1,0,float('nan')]],'candidates':{'0':{'position':[0,0,0]}},'metadata':{'source_bounds_m':[[0,0,0],[1,1,1]]}}
    with pytest.raises(ValueError,match='invalid prequery features'):reach.features(cat,'0')


def test_infrastructure_failure_not_persisted_and_stamp_corruption_blocked(isolated_reach,monkeypatch,tmp_path):
    reach,work,_=isolated_reach
    reach.SELECTION.write_bytes((ROOT/'research/phase4/candidates/reach_selection.json').read_bytes())
    monkeypatch.setattr(reach,'STAMP',tmp_path/'prequery.json')
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1);return {'status':'infrastructure_failure','details':{},'cost':42.0}
    monkeypatch.setattr(reach,'cached_observation',fail)
    with pytest.raises(RuntimeError,match='infrastructure failure'):reach.measure()
    assert len(calls)==1 and not list((work/'reach_results').glob('*.json'))
    contents=json.loads(reach.STAMP.read_text());contents['selection_sha256']='0'*64
    reach.STAMP.write_text(json.dumps(contents))
    with pytest.raises(ValueError,match='stamp mismatch'):reach.measure()
    assert len(calls)==1
