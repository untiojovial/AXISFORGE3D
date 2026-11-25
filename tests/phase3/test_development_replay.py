"""Optional offline integration with the pinned public data supplied by the caller."""
import os
from pathlib import Path

import numpy as np
import pytest

from research_grasping.phase3 import pilot_cli
from research_grasping.phase3.experiment import DevelopmentData,digest,load
from research_grasping.phase3.gso_contact_probe import source_parts,scene,simulate
from research_grasping.phase3.native_catalog import evaluate_candidate
from research_grasping.phase3.representation_gp import design


def test_feature_compression_uses_train_only():
    def row(n):
        return dict(geometry=[n,n],global_embedding=[1.,n],local=[n,0.,1.])
    train=[row(n) for n in range(8)]
    a,b=design(train,[row(9)])
    same,others=design(train,[row(9),row(150)])
    assert np.array_equal(a,same) and np.allclose(b,others[:1])
    assert a.shape==(8,4)


@pytest.fixture
def supplied():
    path=os.getenv('AXISFORGE_PHASE3_DATA')
    if not path:
        pytest.skip('set AXISFORGE_PHASE3_DATA to the prepared offline development bundle')
    base=Path(path)
    protocol=Path(__file__).resolve().parents[2]/'research/phase3/pilot'
    binary=Path(os.environ.get('AXISFORGE_BIN',str(protocol.parents[2]/'target/release/axisforge')))
    return base,protocol,binary


def test_repeat_real_cli_without_revealing_hidden_outcomes(supplied,tmp_path):
    base,protocol,binary=supplied
    names=['2_of_Jenga_Classic_Game','Great_Dinos_Triceratops_Toy']
    cats=[str(base/'pilot-expanded'/'catalogs'/(name+'.json')) for name in names]
    out=tmp_path/'pilot'
    args=['--catalogues',*cats,'--binary',str(binary),'--data',str(base),'--protocol',str(protocol),
          '--out',str(out),'--budget','3','--methods','random','target_local','adaptive','--seeds','11']
    assert pilot_cli.main(args)==0
    before=digest(out/'viewer.json')
    assert pilot_cli.main(args)==0 and before==digest(out/'viewer.json')
    runs=load(out/'viewer.json')['runs']
    assert len(runs)==3
    assert len({tuple(row['candidate_id'] for row in r['rows'][:2]) for r in runs})==1
    assert all(row['details']['f3_executions']==0 for r in runs for row in r['rows'])


def test_native_pose_lift_and_support_reference(supplied):
    pytest.importorskip('mujoco')
    source=os.getenv('AXISFORGE_PHASE3_GSO')
    if not source:pytest.skip('set AXISFORGE_PHASE3_GSO to pinned provider checkout')
    base,protocol,binary=supplied
    data=DevelopmentData(protocol/'manifest.prepilot.json',load(protocol/'lock.json'),base)
    name='Great_Dinos_Triceratops_Toy'
    c=data.catalogue(base/'pilot-expanded'/'catalogs'/(name+'.json'),binary)
    assets,hulls=source_parts(Path(source),name);assert len(hulls)>1
    meta=c['metadata']
    for i,expected in [(0,False),(24,True)]:
        record=evaluate_candidate(binary,c['project'],c['body'],i)['records'][0]
        assert record['plan'] is not None
        model,sim,opening=scene(load(c['project']),record['graspPose'],assets,
                 np.asarray(meta['source_origin_m']),np.asarray(meta['base_placement_world_m']))
        assert np.isclose(model.body_mass[model.body('object').id],.1)
        observed=simulate(model,sim,opening)
        assert observed['held'] is expected
