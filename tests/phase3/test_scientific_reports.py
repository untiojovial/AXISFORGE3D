"""Numerical and provenance checks for phase-III scientific reports."""
import copy
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from research_grasping.phase3.contact_similarity import distances, evaluate
from research_grasping.phase3.representation_gp import design
from research_grasping.phase3.summarize_pilot import summarize_logs, summarize_run
from research_grasping.phase3.supplement import export

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/workspace/axisforge-phase3-data')


def example_run(seed=1):
    return {'definition': {'object_id':'target', 'method':'random', 'seed':seed, 'budget':2},
            'pending':None,'rows':[{'candidate_id':'a','status':'physical_failure','cost':2.,'utility':0.,
                                   'details':{'epsilon':.4,'cached':False}},
                                  {'candidate_id':'b','status':'ok','cost':3.,'utility':.8,
                                   'details':{'epsilon':.2,'cached':True}}]}


def test_report_calculations_and_invalid_runs():
    run=example_run(); summary=summarize_run(run)
    assert summary['curve']==[0.,.8] and summary['auc']==pytest.approx(.4)
    assert summary['epsilon_all']==[.4,.2] and summary['epsilon_feasible']==[.2]
    assert summary['nominal_online_s']==5 and summary['fresh_online_s']==2
    duplicate=copy.deepcopy(run);duplicate['rows'][1]['candidate_id']='a'
    with pytest.raises(ValueError,match='duplicate'):summarize_run(duplicate)
    invalid=copy.deepcopy(run);invalid['rows'][0]['cost']=-1
    with pytest.raises(ValueError,match='invalid'):summarize_run(invalid)
    mixed=example_run(2);mixed['definition']['budget']=1;mixed['rows']=mixed['rows'][:1]
    with pytest.raises(ValueError,match='mixed budgets'):summarize_logs([run,mixed],[])
    failure=copy.deepcopy(run);failure['rows'][0].update(status='infrastructure_failure',utility=None)
    assert summarize_run(failure)['auc'] is None


def test_supplement_roundtrip_and_refuses_incomplete(tmp_path):
    source=tmp_path/'source';(source/'runs'/'target').mkdir(parents=True);(source/'history').mkdir()
    run=example_run(); (source/'runs'/'target'/'1.json').write_text(json.dumps(run))
    (source/'history'/'h.json').write_text(json.dumps({'rows':[]}))
    for name in ('selection-lock','archive-lock'):(source/(name+'.json')).write_text('{}')
    out=tmp_path/'out';export(source,out)
    recovered=json.loads(gzip.decompress((out/'observations.json.gz').read_bytes()))
    assert recovered['runs'][0]==run
    with (out/'paired.csv').open() as f: row=next(csv.DictReader(f))
    assert row['object']=='target' and row['method']=='random'
    assert float(row['mean_auc'])==pytest.approx(.4) and int(row['f1_rejected'])==1
    run['pending']='b';(source/'runs'/'target'/'1.json').write_text(json.dumps(run))
    with pytest.raises(ValueError,match='incomplete'):export(source,tmp_path/'bad')


def descriptor(g,v):
    return {'geometry':[g], 'global_embedding':np.asarray(v,float),'local':np.asarray(v,float)}


def label(epsilon,status):return {'details':{'epsilon':epsilon},'status':status}


def test_similarity_ties_constant_and_metrics_manual():
    source=[descriptor(1,[1,0]),descriptor(1,[1,0]),descriptor(1,[1,0]),descriptor(1,[1,0])]
    target=[descriptor(1,[1,0])]
    result=evaluate(source,target,[label(0,'physical_failure'),label(1,'ok'),label(2,'ok'),label(3,'ok')],[label(2,'ok')])
    assert result['geometry']['epsilon_predictions']==[1.5]  # all four tied, not arbitrary three
    assert result['geometry']['epsilon_mae']==pytest.approx(.5)
    assert result['geometry']['f1_brier']==pytest.approx(.0625)
    assert result['geometry']['distance_vs_epsilon_delta'] is None
    assert result['train_constant']['epsilon_predictions']==[1.5]
    assert distances(source,target)['equal_fusion'].shape==(1,4)


def test_representation_fit_does_not_use_heldout_values():
    rng=np.random.default_rng(17)
    rows=[{'geometry':rng.normal(size=3),'global_embedding':rng.normal(size=7),'local':rng.normal(size=8)} for _ in range(12)]
    x, _ = design(rows[:10], rows[10:])
    altered=copy.deepcopy(rows[10:])
    for r in altered:
        for key in r:r[key]=np.asarray(r[key])*10000+999
    x2, _ = design(rows[:10], altered)
    np.testing.assert_allclose(x,x2,atol=1e-12)
    assert np.isfinite(x).all()


@pytest.mark.skipif(not (DATA/'run-development'/'history').exists() or not (ROOT/'target/release/axisforge').exists(),reason='phase III development artifacts unavailable')
def test_summarize_cli_real_panel_matches_locks(tmp_path):
    run=DATA/'run-development';out=tmp_path/'report'
    env=dict(os.environ,MPLCONFIGDIR=str(tmp_path/'mpl'),OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,'-m','research_grasping.phase3.summarize_pilot',str(run),'--out',str(out)],
                   cwd=ROOT,env=env,check=True,capture_output=True,text=True,timeout=120)
    report=json.loads((out/'summary.json').read_text()); manifest=json.loads((ROOT/'research/phase3/pilot/manifest.prepilot.json').read_text())
    validation={o['object_id'] for o in manifest['objects'] if o['split']=='validation'}
    assert {g['object_id'] for g in report['groups']}==validation
    assert report['objects']==len(validation) and report['incomplete_runs_excluded']==0
    runs=[json.loads(p.read_text()) for p in sorted((run/'runs').glob('*/*.json'))]
    histories=list((run/'history').glob('*.json'))
    assert report['completed_runs']==len(runs)
    assert report['historical']['objects']==len(histories)
    group=report['groups'][0]
    matching=[summarize_run(x) for x in runs if x['definition']['object_id']==group['object_id'] and x['definition']['method']==group['method']]
    assert group['auc']['mean']==pytest.approx(np.mean([r['auc'] for r in matching]))
    assert (out/'summary.csv').exists() and (out/'best_so_far.svg').exists()


@pytest.mark.skipif(not (DATA/'vision-development-v3').exists() or not (DATA/'run-development'/'history').exists() or not (ROOT/'target/release/axisforge').exists(),reason='phase III development artifacts unavailable')
@pytest.mark.parametrize('module',['contact_similarity','representation_gp'])
def test_real_representation_cli_uses_declared_panel(tmp_path,module):
    pytest.importorskip('torch')
    pytest.importorskip('sklearn')
    out=tmp_path/(module+'.json')
    common=['--data',str(DATA),'--catalogues',str(DATA/'pilot-expanded/catalogs'),
            '--protocol',str(ROOT/'research/phase3/pilot'),'--binary',str(ROOT/'target/release/axisforge'),
            '--vision',str(DATA/'vision-development-v3'),'--run',str(DATA/'run-development'),'--out',str(out)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(tmp_path/'mpl'))
    subprocess.run([sys.executable,'-m','research_grasping.phase3.'+module,*common],
                   cwd=ROOT,env=env,check=True,capture_output=True,text=True,timeout=180)
    report=json.loads(out.read_text());manifest=json.loads((ROOT/'research/phase3/pilot/manifest.prepilot.json').read_text())
    from research_grasping.phase3.experiment import digest
    train={o['object_id'] for o in manifest['objects'] if o['split']=='train'}
    val={o['object_id'] for o in manifest['objects'] if o['split']=='validation'}
    assert set(report['provenance']['inputs'])==train|val
    assert report['provenance']['manifest_sha256']==digest(ROOT/'research/phase3/pilot/manifest.prepilot.json')
    for oid,entry in report['provenance']['inputs'].items():
        assert entry['split']==('train' if oid in train else 'validation')
        assert entry['catalogue_sha256']==digest(DATA/'pilot-expanded/catalogs'/(oid+'.json'))
    if module=='contact_similarity':
        assert {o['object_id'] for o in report['objects']}==val
        for obj in report['objects']:
            run=json.loads((DATA/'run-development/runs'/obj['object_id']/'11-random.json').read_text())
            assert obj['ids']==[row['candidate_id'] for row in run['rows']]
            for result in obj['methods'].values():
                assert result['epsilon_mae']>=0 and 0<=result['f1_brier']<=1
                assert len(result['epsilon_predictions'])==len(obj['ids'])
    else:
        assert report['train']==160 and report['target']==60
        assert {'train_constant','rbf','matern52','dkl_11','dkl_29','dkl_47'}<=set(report['models'])
        for model in report['models'].values():
            assert set(model['by_object'])==val
            assert all(0<=v['coverage90']<=1 and v['mae']>=0 for v in model['by_object'].values())
