"""Whole-window and cross-timestep integration smoke on a synthetic fixture."""
import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('mujoco')
from research_grasping.phase3.gso_contact_probe import scene
from research_grasping.phase4.physics_simulation import simulate


def test_no_external_grasp_stays_on_table_for_all_dt():
    project = json.loads((Path(__file__).resolve().parents[2] / 'examples/pick_and_place_cell.axisforge').read_text())
    pose = {'translation': {'x': .72, 'y': -.1, 'z': .52}, 'rotation': {'x':0.,'y':0.,'z':0.,'w':1.}}
    assets = {'fixture.obj': b'v 0 0 0\nv .011 0 0\nv 0 .023 0\nv 0 0 .037\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'}
    for dt in (.0005, .001, .002):
        model, data, opening = scene(project, pose, assets, np.zeros(3), np.array([.42,-.1,.42]))
        model.opt.timestep = dt
        result = simulate(model, data, opening, close_jaws=False)
        assert not result['sustained']
        assert result['table_hold_fraction'] > .99
        assert result['max_penetration_all_m'] >= result['max_object_penetration_all_m'] >= 0
        assert result['settle']['speed_m_s'] >= 0


def test_locked_physical_sample_is_train_only_and_unique():
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads((root/'research/phase4/robotics/physical_protocol.json').read_text())
    manifest = json.loads((root/'research/phase3/pilot/manifest.prepilot.json').read_text())
    splits = {obj['object_id']:obj['split'] for obj in manifest['objects']}
    objects = protocol['objects']
    assert len(objects) == 14
    assert len({obj['object_id'] for obj in objects}) == 14
    assert all(splits[obj['object_id']] == 'train' for obj in objects)
    assert sum(len(obj['candidate_ids']) for obj in objects) == 84
    assert all(len(set(obj['candidate_ids'])) == 6 for obj in objects)


def test_nonfinite_initial_state_never_becomes_physical_success():
    project = json.loads((Path(__file__).resolve().parents[2] / 'examples/pick_and_place_cell.axisforge').read_text())
    pose = {'translation': {'x': .72, 'y': -.1, 'z': .52}, 'rotation': {'x':0.,'y':0.,'z':0.,'w':1.}}
    assets = {'fixture.obj': b'v 0 0 0\nv .011 0 0\nv 0 .023 0\nv 0 0 .037\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'}
    model, data, opening = scene(project, pose, assets, np.zeros(3), np.array([.42,-.1,.42]))
    data.qvel[0] = np.nan
    result = simulate(model, data, opening)
    assert result['numerical_valid'] is False
    assert result['sustained'] is None and result['eligible_sustained'] is None
    assert not result['numerical_issue']['finite']['qvel']


def test_existing_mujoco_warning_invalidates_before_any_step():
    project = json.loads((Path(__file__).resolve().parents[2] / 'examples/pick_and_place_cell.axisforge').read_text())
    pose = {'translation': {'x': .72, 'y': -.1, 'z': .52}, 'rotation': {'x':0.,'y':0.,'z':0.,'w':1.}}
    assets = {'fixture.obj': b'v 0 0 0\nv .011 0 0\nv 0 .023 0\nv 0 0 .037\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'}
    model, data, opening = scene(project, pose, assets, np.zeros(3), np.array([.42,-.1,.42]))
    data.warning[0].number = 1
    result = simulate(model, data, opening)
    assert not result['numerical_valid'] and result['numerical_issue']['step'] == -1
    assert result['sustained'] is None


def test_runtime_warning_or_reset_invalidates_result():
    project = json.loads((Path(__file__).resolve().parents[2] / 'examples/pick_and_place_cell.axisforge').read_text())
    pose = {'translation': {'x': .72, 'y': -.1, 'z': .52}, 'rotation': {'x':0.,'y':0.,'z':0.,'w':1.}}
    assets = {'fixture.obj': b'v 0 0 0\nv .011 0 0\nv 0 .023 0\nv 0 0 .037\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'}
    model, data, opening = scene(project, pose, assets, np.zeros(3), np.array([.42,-.1,.42]))
    data.qvel[0] = 1e30
    result = simulate(model, data, opening)
    assert not result['numerical_valid'] and result['sustained'] is None
    assert any(result['numerical_issue']['warning_counts']) or result['numerical_issue']['time_reset']


def test_full_physical_artifact_conserves_sample_and_invalid_warning_states():
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads((root/'research/phase4/robotics/physical_protocol.json').read_text())
    result = json.loads((root/'research/phase4/robotics/physical_pilot.json').read_text())
    expected = {(obj['object_id'],cid) for obj in protocol['objects'] for cid in obj['candidate_ids']}
    rows = result['rows']
    observed = [(row['object_id'],row['candidate_id']) for row in rows]
    assert len(observed) == len(expected) and set(observed) == expected
    for row in rows:
        if not row.get('f3'):
            assert row['f3_status'] in ('blocked_proxy_local_disagreement', 'infrastructure_error')
            continue
        variants = [value for key,value in row['f3'].items() if key != 'numerics'] + row['f3']['numerics']
        invalid = [variant for variant in variants if not variant['metrics']['numerical_valid']]
        assert row['numerical_failures'] == len(invalid)
        for variant in invalid:
            assert variant['metrics']['sustained'] is None
            assert 'numerical_issue' in variant['metrics']
