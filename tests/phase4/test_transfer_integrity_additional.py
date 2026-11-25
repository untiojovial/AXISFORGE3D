"""Integrity regressions for real development artifacts, with isolated outputs."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

from research_grasping.phase3 import experiment
from research_grasping.phase3.transfer_policy import FrozenScale, HistoricalTask, _MultidimensionalRGPE
from research_grasping.phase4 import transfer_diagnostics, transfer_statistics

ROOT=Path('/workspace/axisforge-phase3-data')
DATA4=Path('/workspace/axisforge-phase4-data')
PROTOCOL=Path('research/phase3/pilot')
BINARY=DATA4/'bin/axisforge-phase3'


@pytest.fixture(scope='module')
def assets():
    cats=DATA4/'catalogues/diverse128'; hist=DATA4/'history/stratified'
    if not all(p.exists() for p in (cats,hist,BINARY,ROOT)):
        pytest.skip('development artifacts not available')
    guard=experiment.DevelopmentData(PROTOCOL/'manifest.prepilot.json',experiment.load(PROTOCOL/'lock.json'),ROOT)
    train=[];validation=None
    for path in sorted(cats.glob('*.json')):
        raw=experiment.load(path)
        if raw['split']=='train' and len(train)<2:
            c=guard.catalogue(path,BINARY);log=experiment.load(hist/(c['object_id']+'.json'))
            train.append((c,log))
        elif raw['split']=='validation' and validation is None:
            validation=guard.catalogue(path,BINARY)
        if len(train)==2 and validation:break
    return guard,train,validation


@pytest.mark.parametrize('mutation,message',[
    ('object','own training catalogue'), ('digest','different labels'), ('fidelity','different labels'),
    ('duplicate','inconsistent historical'), ('candidate','inconsistent historical'),
    ('status','inconsistent historical'), ('negative_cost','inconsistent historical'),
    ('no_measurements','no measurements')])
def test_history_rejects_incompatible_provenance_and_status(assets,mutation,message):
    guard,train,_=assets;c,old=train[0];log=copy.deepcopy(old)
    if mutation=='object':log['object_id']='other'
    elif mutation=='digest':log['catalogue_digest']='corrupt'
    elif mutation=='fidelity':log['fidelity']='wrong'
    elif mutation=='duplicate':log['selected'][1]=log['selected'][0]
    elif mutation=='candidate':log['rows'][0]['candidate_id']='nonexistent'
    elif mutation=='status':log['rows'][0]['status']='unrecognized'
    elif mutation=='negative_cost':log['rows'][0]['cost']=-1
    elif mutation=='no_measurements':
        for row in log['rows']:row.update(status='infrastructure_failure',utility=None)
    with pytest.raises(ValueError,match=message):experiment.fit_history(guard,[(c,log)],'native-f12')


def test_empty_history_and_unverified_catalogue(assets):
    guard,train,_=assets
    with pytest.raises(ValueError,match='frozen train archive'):
        experiment.fit_history(guard,[],'native-f12')
    altered=copy.deepcopy(train[0][0]);altered['context'][0]+=1
    with pytest.raises(ValueError,match='unverified catalogue'):
        experiment.fit_history(guard,[(altered,train[0][1])],'native-f12')


def test_scale_and_archive_validation():
    scaler=FrozenScale(np.array([0.]),np.array([0.]),'digest')
    with pytest.raises(ValueError,match='scaling'):scaler.transform([[1.]])
    scaler=FrozenScale(np.array([0.]),np.array([1.]),'digest')
    task=HistoricalTask('s',1.,np.array([0.]),np.array([[1.]]),np.array([float('nan')]),1.)
    from research_grasping.phase3.transfer_policy import approved_history
    with pytest.raises(ValueError,match='historical observations'):
        approved_history([task],{'s':'train','t':'validation'},target_id='t',target_time=2,scaler=scaler)


def test_rgpe_rank_error_rolls_back_state(monkeypatch):
    model=_MultidimensionalRGPE([],dimension=1,noise=.1,length_scale=1.,seed=3,samples=16)
    model.observe([.2],.4)
    original_x=model.x.copy();original_y=model.y.copy();original_weights=model.weights.copy();original_target=model.target
    def fail():raise ArithmeticError('injected ranking failure')
    monkeypatch.setattr(model,'rank_weights',fail)
    with pytest.raises(ArithmeticError,match='injected'):model.observe([.7],.5)
    assert np.array_equal(model.x,original_x) and np.array_equal(model.y,original_y)
    assert np.array_equal(model.weights,original_weights) and model.target is original_target


def test_statistics_cli_figures_and_unpaired_rejection(tmp_path,monkeypatch):
    v1=DATA4/'run-transfer-50';v2=DATA4/'run-transfer-v2b'
    if not (v1/'runs').exists() or not (v2/'runs').exists():pytest.skip('completed pilot missing')
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'mpl'))
    monkeypatch.setenv('XDG_CACHE_HOME',str(tmp_path/'cache'))
    out=tmp_path/'result'
    monkeypatch.setattr(sys,'argv',['transfer_statistics','--v1',str(v1),'--v2b',str(v2),'--out',str(out)])
    transfer_statistics.main()
    result=experiment.load(out/'paired_bootstrap.json')
    assert result['comparisons']['v2b_adaptive_minus_v2b_target']['n_objects']==6
    assert (out/'paired_by_object.svg').stat().st_size>0 and (out/'paired_by_object.png').stat().st_size>0
    assert result['input_sha256']
    with pytest.raises(ValueError,match='unpaired'):
        transfer_statistics.compare({('obj',11):.4},{('obj',29):.4})


def test_statistics_rejects_incomplete_or_duplicate_run(tmp_path):
    d=tmp_path/'runs'/'obj';d.mkdir(parents=True)
    run={'definition':{'method':'x','budget':1,'object_id':'obj','seed':11},'rows':[],'pending':'candidate'}
    (d/'a.json').write_text(json.dumps(run))
    with pytest.raises(ValueError,match='incomplete'):transfer_statistics.load_method(tmp_path,'x')
    run['pending']=None;run['rows']=[{'status':'ok','utility':.5}]
    (d/'a.json').write_text(json.dumps(run));(d/'b.json').write_text(json.dumps(run))
    with pytest.raises(ValueError,match='duplicate run'):transfer_statistics.load_method(tmp_path,'x')


def test_run_policy_rejects_corrupted_checkpoint_and_bad_context(assets,tmp_path):
    guard,archive,catalog=assets
    cutoff=max(log['completed_at'] for _,log in archive)+1
    kwargs=dict(method='target_local',seed=7,budget=3,initial_count=2,fidelity='native-f12',
                target_time=cutoff,checkpoint=tmp_path/'run.json')
    observe=lambda _: {'status':'physical_failure','utility':0.,'cost':1.,'details':{}}
    state=experiment.run_policy(guard,catalog,archive,observe=observe,**kwargs)
    assert len(state['rows'])==3
    with pytest.raises(ValueError,match='unknown context mode'):
        experiment.run_policy(guard,catalog,archive,observe=observe,context_mode='bad',**kwargs)
    state['rows'][0]['prediction']['mean']+=1
    kwargs['checkpoint'].write_text(json.dumps(state))
    with pytest.raises(ValueError,match='posterior diverged'):
        experiment.run_policy(guard,catalog,archive,observe=observe,**kwargs)


def test_statistics_cli_rejects_changed_initialization(tmp_path,monkeypatch):
    first=tmp_path/'first';second=tmp_path/'second'
    def write(root,method,initial):
        path=root/'runs'/'obj'/f'11-{method}.json';path.parent.mkdir(parents=True,exist_ok=True)
        definition={'method':method,'budget':1,'object_id':'obj','seed':11,
            'catalogue_digest':'c','manifest_sha256':'m','archive_digest':'a',
            'initial':[initial],'fidelity':'native-f12'}
        path.write_text(json.dumps({'definition':definition,'rows':[{'status':'ok','utility':.2}],'pending':None}))
    write(second,'v2b_adaptive','a');write(second,'v2b_target','b')
    monkeypatch.setattr(sys,'argv',['transfer_statistics','--v1',str(first),'--v2b',str(second),'--out',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='different protocol or initialization'):
        transfer_statistics.main()
    assert not (tmp_path/'out'/'paired_bootstrap.json').exists()


def test_diagnostic_cli_replays_real_observations(tmp_path,monkeypatch):
    cats=ROOT/'functional-catalogs';run=ROOT/'run-functional';binary=Path('target/release/axisforge')
    if not all(p.exists() for p in (cats,run,binary)):pytest.skip('functional panel unavailable')
    output=tmp_path/'gate.json'
    monkeypatch.setattr(sys,'argv',['transfer_diagnostics','--data',str(ROOT),'--protocol',str(PROTOCOL),
        '--catalogues',str(cats),'--binary',str(binary),'--run',str(run),'--out',str(output)])
    transfer_diagnostics.main()
    result=experiment.load(output)
    assert result['runs']==18 and result['compared_decisions']==144
    assert len(result['runs_detail'])==18


def test_catalogue_rejects_wrong_split_path_and_duplicate_ids(assets,tmp_path):
    guard,train,_=assets
    original=train[0][0]
    path=tmp_path/'candidate.json'
    altered=copy.deepcopy(original);altered['split']='validation';path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='development access denied'):
        guard.catalogue(path,BINARY)
    altered=copy.deepcopy(original);altered['mesh_file']=str(tmp_path/'outside.stl');path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='unexpected mesh path'):
        guard.catalogue(path,BINARY)
    altered=copy.deepcopy(original);altered['ids'][1]=altered['ids'][0];path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='invalid pre-query candidate catalogue'):
        guard.catalogue(path,BINARY)


def test_grouped_feasibility_uses_only_other_objects(assets):
    from research_grasping.phase4.transfer_pilot import grouped_diagnostic
    guard,archive,_=assets
    history,scaler=experiment.fit_history(guard,archive,'native-f12')
    result=grouped_diagnostic(archive,history,scaler)
    assert len(result['objects'])==2
    assert {r['object_id'] for r in result['objects']}=={c['object_id'] for c,_ in archive}
    assert all(0<=r['brier']<=1 and r['constant_half_brier']==pytest.approx(.25) for r in result['objects'])


def test_gate_replay_rejects_history_mismatch(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(transfer_diagnostics,'fit_history',lambda *args: ([],SimpleNamespace(training_digest='expected')))
    catalog={'object_id':'validation'}
    run={'definition':{'fidelity':'native-f12','catalogue_digest':experiment.fingerprint(catalog),'archive_digest':'other'}}
    with pytest.raises(ValueError,match='mismatched catalog or history'):
        transfer_diagnostics.compare(None,catalog,[],run)


def test_policy_rejects_bad_cost_and_preserves_reservation():
    from research_grasping.phase3.transfer_policy import CataloguePolicy
    manifest={'t':'validation'}
    policy=CataloguePolicy(['a','b'],[[0.],[1.]],[0.],[],manifest_splits=manifest,
        manifest_digest=experiment.fingerprint(manifest),target_id='t',scaler=FrozenScale([0.],[1.],'train'),
        target_time=2,method='random',budget=2,seed=5,initial_ids=[])
    cid=policy.ask()
    with pytest.raises(ValueError,match='invalid status or cost'):
        policy.tell(cid,status='ok',cost=-1,utility=.5)
    with pytest.raises(ValueError,match='candidate is not reserved'):
        policy.tell('wrong',status='ok',cost=1,utility=.5)
    policy.tell(cid,status='ok',cost=1,utility=.5)
    assert policy.records[0]['candidate_id']==cid


@pytest.mark.parametrize('mode,method',[('geometry','target_local'),('visual','visual_fixed')])
def test_run_policy_context_modes_use_locked_features(assets,tmp_path,mode,method):
    guard,archive,catalog=assets
    cutoff=max(log['completed_at'] for _,log in archive)+1
    result=experiment.run_policy(guard,catalog,archive,method=method,seed=17,budget=2,initial_count=2,
        fidelity='native-f12',target_time=cutoff,context_mode=mode,checkpoint=tmp_path/(mode+'.json'),
        observe=lambda _: {'status':'physical_failure','utility':0.,'cost':.2,'details':{}})
    assert result['definition']['context_mode']==mode and len(result['rows'])==2


def test_replay_rejects_changed_candidate_reservation(assets,tmp_path):
    guard,archive,catalog=assets
    cutoff=max(log['completed_at'] for _,log in archive)+1
    checkpoint=tmp_path/'interrupted.json'
    kwargs=dict(method='target_local',seed=19,budget=3,initial_count=2,fidelity='native-f12',
        target_time=cutoff,checkpoint=checkpoint)
    with pytest.raises(RuntimeError,match='interrupted'):
        experiment.run_policy(guard,catalog,archive,observe=lambda _: (_ for _ in ()).throw(RuntimeError('interrupted')),**kwargs)
    state=experiment.load(checkpoint);state['pending']='tampered-id';checkpoint.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='reserved candidate changed'):
        experiment.run_policy(guard,catalog,archive,observe=lambda _: pytest.fail('oracle must not run'),**kwargs)


def test_visual_policy_requires_valid_provenance():
    from research_grasping.phase3.transfer_policy import CataloguePolicy
    manifest={'t':'validation'}
    kwargs=dict(ids=['a'],features=[[0.]],context=[0.],history=[],manifest_splits=manifest,
        manifest_digest=experiment.fingerprint(manifest),target_id='t',scaler=FrozenScale([0.],[1.],'train'),
        target_time=2,method='dinov2_contextual',budget=1,seed=5,initial_ids=['a'])
    with pytest.raises(ValueError,match='verified pretrained visual provenance'):
        CataloguePolicy(**kwargs,visual_provenance={'model':'dinov3','checkpoint_sha256':'f'*64,'preprocessing':'known'})
    assert CataloguePolicy(**kwargs,visual_provenance={'model':'dinov2','checkpoint_sha256':'f'*64,'preprocessing':'known'}).ask()=='a'
