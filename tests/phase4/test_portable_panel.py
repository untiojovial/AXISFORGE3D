import gzip
import json
from pathlib import Path

import pytest
from research_grasping.phase4 import portable_panel, embedding_report
from research_grasping.phase4.history_lock import seal
from research_grasping.phase3.experiment import load

WORK=Path('/workspace/axisforge-phase4-data')
DATA=Path('/workspace/axisforge-phase3-data')
ROOT=Path(__file__).resolve().parents[2]
PANEL=ROOT/'research/phase4/embedding_experiment'


def test_panel_uses_inductive_predictors_and_matched_random_baselines():
    summary=embedding_report.summarize(PANEL/'openclip_analysis.json')
    assert set(summary['results'])=={'geometry','global','local','global_local','random_control'}
    assert len({r['auc']['random'] for r in summary['results'].values()})==1
    assert len(summary['results']['local']['adaptive_minus_geometry_by_object'])==6


@pytest.mark.skipif(not (WORK/'bin/axisforge-phase3').exists(),reason='requires original native development data')
def test_archive_reconstruction_excludes_target_outcomes_from_features(tmp_path):
    portable_panel.prepare(DATA,WORK/'bin/axisforge-phase3',tmp_path)
    assert load(tmp_path/'lock.json')==load(PANEL/'lock.json')
    with gzip.open(tmp_path/'candidates.json.gz','rt') as fp:features=json.load(fp)
    assert all(set(x)=={'split','ids','features','dimensions','source_shape','visible'} for x in features.values())
    with gzip.open(tmp_path/'observations.json.gz','rt') as fp:observations=json.load(fp)
    assert all(set(observations[n])==set(x['ids']) for n,x in features.items())
    assert sum(v['split']=='train' for v in features.values())==16
    assert sum(v['split']=='validation' for v in features.values())==6
    assert seal(WORK)==load(ROOT/'research/phase4/candidates/history_stratified.lock.json')


def test_cross_model_table_declares_missing_models(tmp_path):
    import json
    from research_grasping.phase4.model_comparison import compare
    path=PANEL/'openclip_analysis.json'
    assert compare([path])['comparison_status']=='SINGLE_MODEL_ONLY'
    with pytest.raises(ValueError,match='duplicate'): compare([path,path])
    changed=load(path);changed['job_sha256']='changed'
    other=tmp_path/'report.json';other.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='different input'): compare([path,other])
