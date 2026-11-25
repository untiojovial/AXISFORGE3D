"""Integration regressions using locally available, locked development artifacts.

All new checkpoints are in pytest's temporary directory. No test object or new
physical evaluation is opened. These tests skip when the local dataset is absent.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from research_grasping.phase3.experiment import DevelopmentData, load
from research_grasping.phase4 import transfer_diagnostics, transfer_nested_cli, transfer_pilot, transfer_v2b_pilot


REPO = Path(__file__).resolve().parents[2]
DATA = Path('/workspace/axisforge-phase3-data')
PHASE4 = Path('/workspace/axisforge-phase4-data')
PROTOCOL = REPO / 'research/phase3/pilot'
BINARY = PHASE4 / 'bin/axisforge-phase3'
HISTORY_LOCK = REPO / 'research/phase4/candidates/history_stratified.lock.json'


@pytest.fixture(scope='module')
def real_development():
    catalog_dir = PHASE4 / 'catalogues/diverse128'
    archive_dir = PHASE4 / 'history/stratified'
    if not all(p.exists() for p in (catalog_dir, archive_dir, BINARY, HISTORY_LOCK, DATA)):
        pytest.skip('locked phase IV development artifacts are not locally available')
    guard = DevelopmentData(PROTOCOL/'manifest.prepilot.json', load(PROTOCOL/'lock.json'), DATA)
    catalogs = []
    for path in sorted(catalog_dir.glob('*.json')):
        obj = load(path)['object_id']
        if guard.splits.get(obj) in ('train', 'validation'):
            catalogs.append(guard.catalogue(path, BINARY))
    train = [c for c in catalogs if c['split'] == 'train']
    val = [c for c in catalogs if c['split'] == 'validation']
    if len(train) != 16 or not val:
        pytest.skip('development roster incomplete')
    archive=[]
    for c in train:
        row = load(archive_dir / (c['object_id']+'.json'))
        # Training labels were already measured; use a small fixed prefix.
        row = dict(row, selected=row['selected'][:3], rows=row['rows'][:3])
        archive.append((c,row))
    cutoff = max(row['completed_at'] for _, row in archive)+1
    return guard, val[0], archive, cutoff


@pytest.mark.parametrize('module,extra', [(transfer_pilot, {}),
                                        (transfer_v2b_pilot, {'selected': [2., 2.]})])
def test_run_fa_checkpoint_resume_and_changed_definition(real_development, tmp_path, module, extra):
    guard, catalog, archive, cutoff = real_development
    checkpoint = tmp_path/'run.json'
    calls=[]
    def interrupted(candidate):
        calls.append(candidate)
        if len(calls) == 2:
            raise RuntimeError('simulated oracle interruption')
        return {'status':'physical_failure','utility':0.,'cost':.2,'details':{'fixture':True}}
    kwargs=dict(method='target', seed=11, budget=3, target_time=cutoff,
                checkpoint=checkpoint, **extra)
    with pytest.raises(RuntimeError, match='interruption'):
        module.run_fa(guard,catalog,archive,observe=interrupted,**kwargs)
    partial=load(checkpoint)
    assert len(partial['rows']) == 1 and partial['pending'] == calls[-1]
    resumed=[]
    def complete(candidate):
        resumed.append(candidate)
        return {'status':'physical_failure','utility':0.,'cost':.2,'details':{'fixture':True}}
    final=module.run_fa(guard,catalog,archive,observe=complete,**kwargs)
    assert len(final['rows']) == 3 and final['pending'] is None
    assert resumed[0] == calls[-1] and calls[0] not in resumed
    assert len({r['candidate_id'] for r in final['rows']}) == 3
    assert module.run_fa(guard,catalog,archive,observe=lambda _: pytest.fail('unexpected oracle call'),**kwargs) == final
    with pytest.raises(ValueError, match='definition changed'):
        module.run_fa(guard,catalog,archive,observe=complete,**dict(kwargs,budget=4))


def test_diagnostic_replay_rejects_tampered_prediction():
    catalog_dir = DATA/'functional-catalogs'
    run_dir = DATA/'run-functional'
    binary = REPO/'target/release/axisforge'
    if not all(p.exists() for p in (catalog_dir, run_dir, binary)):
        pytest.skip('phase III functional run is not locally available')
    guard=DevelopmentData(PROTOCOL/'manifest.prepilot.json',load(PROTOCOL/'lock.json'),DATA)
    catalogs=[guard.catalogue(p,binary) for p in sorted(catalog_dir.glob('*.json')) if p.name!='pca.json']
    archive=[(c,load(run_dir/'history'/(c['object_id']+'.json'))) for c in catalogs if c['split']=='train']
    val=next(c for c in catalogs if c['split']=='validation')
    path=next((run_dir/'runs'/val['object_id']).glob('*-adaptive.json'))
    run=load(path)
    result=transfer_diagnostics.compare(guard,val,archive,run)
    assert len(result['rows']) == len(run['rows'])
    changed=copy.deepcopy(run);changed['rows'][0]['prediction']['mean'] += 1.
    with pytest.raises(ValueError, match='prediction mismatch'):
        transfer_diagnostics.compare(guard,val,archive,changed)


def test_nested_cli_uses_only_train_and_rejects_changed_lock(real_development, tmp_path, monkeypatch):
    guard, _, _, _ = real_development
    output=tmp_path/'nested.json'
    argv=['transfer_nested_cli','--data',str(DATA),'--protocol',str(PROTOCOL),
          '--catalogues',str(PHASE4/'catalogues/diverse128'),'--binary',str(BINARY),
          '--archive',str(PHASE4/'history/stratified'),'--history-lock',str(HISTORY_LOCK),
          '--prefix','3','--out',str(output)]
    seen=[]
    def fake_nested(tasks):
        seen.extend(t.object_id for t in tasks)
        return {'selected_all_train':[1.,2.]}
    monkeypatch.setattr(transfer_nested_cli,'nested',fake_nested)
    monkeypatch.setattr(sys,'argv',argv)
    transfer_nested_cli.main()
    assert len(seen)==16 and all(guard.splits[name]=='train' for name in seen)
    lock=output.with_suffix('.lock.json')
    mutated=load(lock);mutated['prefix']=99;lock.write_text(json.dumps(mutated))
    with pytest.raises(ValueError,match='definition changed'):
        transfer_nested_cli.main()


@pytest.mark.parametrize('module,is_v2b', [(transfer_pilot,False),(transfer_v2b_pilot,True)])
def test_cli_freezes_inputs_and_visits_validation_only(real_development, tmp_path, monkeypatch, module, is_v2b):
    guard, _, _, _ = real_development
    nested=REPO/'research/phase4/transfer/v2b_nested_final.json'
    if is_v2b and not nested.exists():
        pytest.skip('nested development result not locally available')
    class DummyExecutor:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    calls=[]
    def fake_fa(data, catalog, archive, **kwargs):
        calls.append((catalog['object_id'], kwargs['seed'], kwargs['method']))
    monkeypatch.setattr(module,'ProcessPoolExecutor',DummyExecutor)
    monkeypatch.setattr(module,'observer',lambda *args: lambda _: pytest.fail('oracle called'))
    monkeypatch.setattr(module,'run_fa',fake_fa)
    monkeypatch.setattr(module,'run_policy',lambda *args,**kwargs: None)
    if not is_v2b:
        monkeypatch.setattr(module,'grouped_diagnostic',lambda *args: {})
    out=tmp_path/'run'
    argv=['transfer_pilot','--data',str(DATA),'--protocol',str(PROTOCOL),
          '--catalogues',str(PHASE4/'catalogues/diverse128'),'--binary',str(BINARY),
          '--archive',str(PHASE4/'history/stratified'),'--archive-prefix',str(50 if is_v2b else 3),
          '--cache',str(tmp_path/'cache'),'--out',str(out),'--seeds','11','--budget','3',
          '--fa-modes','target']
    if is_v2b: argv += ['--nested',str(nested),'--nested-sha256',hashlib.sha256(nested.read_bytes()).hexdigest(),'--history-lock',str(HISTORY_LOCK)]
    monkeypatch.setattr(sys,'argv',argv)
    module.main()
    assert len(calls)==6 and all(guard.splits[name]=='validation' for name,_,_ in calls)
    assert (out/'lock.json').exists()
    monkeypatch.setattr(sys,'argv',argv+['--budget','4'])
    with pytest.raises(ValueError,match='definition changed'):
        module.main()


def test_v2b_checkpoint_rejects_changed_selected_parameters(real_development, tmp_path):
    guard, catalog, archive, cutoff = real_development
    checkpoint=tmp_path/'selected.json'
    kwargs=dict(method='adaptive',seed=11,budget=3,target_time=cutoff,checkpoint=checkpoint)
    observer=lambda _: {'status':'physical_failure','utility':0.,'cost':.2,'details':{}}
    transfer_v2b_pilot.run_fa(guard,catalog,archive,observe=observer,selected=[2.,2.],**kwargs)
    with pytest.raises(ValueError,match='replay diverged'):
        transfer_v2b_pilot.run_fa(guard,catalog,archive,observe=observer,selected=[.5,8.],**kwargs)


def test_v2b_cli_rejects_tampered_nested_result(real_development, tmp_path, monkeypatch):
    nested=REPO/'research/phase4/transfer/v2b_nested_final.json'
    if not nested.exists(): pytest.skip('nested result not locally available')
    original=nested.read_bytes(); expected=hashlib.sha256(original).hexdigest()
    altered=json.loads(original);altered['selected_all_train']=[.5,8.]
    fake=tmp_path/'nested.json';fake.write_text(json.dumps(altered))
    argv=['transfer_v2b_pilot','--data',str(DATA),'--protocol',str(PROTOCOL),
          '--catalogues',str(PHASE4/'catalogues/diverse128'),'--binary',str(BINARY),
          '--archive',str(PHASE4/'history/stratified'),'--cache',str(tmp_path/'cache'),
          '--out',str(tmp_path/'out'),'--nested',str(fake),'--nested-sha256',expected,
          '--history-lock',str(HISTORY_LOCK)]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='nested result digest mismatch'):
        transfer_v2b_pilot.main()
    assert not (tmp_path/'out').exists()
