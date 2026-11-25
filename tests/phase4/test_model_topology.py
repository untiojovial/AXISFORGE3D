"""Require a complete robot tree and independent held-out objects before scoring."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_grasping.phase4.full_robot_kinematics import build_kinematic_model
from research_grasping.phase4.transfer_nested import choose,fold_predict,nested

PROJECT=Path(__file__).resolve().parents[2]/'examples/pick_and_place_cell.axisforge'


@pytest.mark.parametrize('change,reason',[
    ('unknown_joint','unsupported joint kind'),('duplicate_link','link names must be unique'),
    ('orphan_link','disconnected or unsupported')])
def test_reject_unrepresentable_model_before_fk_comparison(change,reason):
    project=json.loads(PROJECT.read_text())
    robot=project['project']['robot']
    if change=='unknown_joint':robot['joints'][0]['kind']='unknown'
    elif change=='duplicate_link':robot['links'][1]['name']=robot['links'][0]['name']
    else:robot['links'].append(dict(robot['links'][0],id='unconnected_link',name='unconnected link'))
    with pytest.raises(ValueError,match=reason):build_kinematic_model(project)


def test_nested_transfer_rejects_target_leakage_and_duplicate_source_ids():
    source=SimpleNamespace(object_id='source')
    target=SimpleNamespace(object_id='target')
    with pytest.raises(ValueError,match='held-out'):fold_predict([source,target],target,1.,2.)
    with pytest.raises(ValueError,match='duplicate'):fold_predict([source,source],target,1.,2.)


def test_insufficient_independent_sources_do_not_yield_a_calibration_score():
    source=SimpleNamespace(object_id='source')
    with pytest.raises(ValueError,match='two training objects'):choose([source])
    with pytest.raises(ValueError,match='three objects'):nested([source])
