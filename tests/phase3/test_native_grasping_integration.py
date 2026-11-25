"""Small synthetic integration fixtures for native hands and phase-III guards."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
import pytest

from research_grasping.phase3 import native_dev, native_multieffector
from research_grasping.phase3.native_catalog import load_catalog, evaluate_candidate

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / 'target/release/axisforge'
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason='release AxisForge CLI unavailable')


def cube_stl():
    vertices = [(-.02,-.02,0),(.02,-.02,0),(.02,.02,0),(-.02,.02,0),
                (-.02,-.02,.04),(.02,-.02,.04),(.02,.02,.04),(-.02,.02,.04)]
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    return b'fixture'.ljust(80,b'\0') + struct.pack('<I', len(faces)) + b''.join(
        struct.pack('<12fH', 0,0,0, *(n for i in face for n in vertices[i]), 0) for face in faces)


def fixture_manifest(root, split='train', valid_hash=True):
    mesh = root / 'fixture.stl'
    mesh.write_bytes(cube_stl())
    sha = hashlib.sha256(mesh.read_bytes()).hexdigest() if valid_hash else '0'*64
    obj = {'object_id':'fixture', 'split':split, 'visual_mesh':'fixture.stl',
           'processed_sha256':sha, 'physical_scale':'fixture SI'}
    (root/'manifest.json').write_text(json.dumps({'objects':[obj]}))
    return obj


@pytest.mark.parametrize('effector', ['three-finger','hand'])
def test_real_demo_hand_catalog_and_selected_identity(tmp_path, effector):
    project = tmp_path / f'{effector}.axisforge'
    subprocess.run([BINARY,'demo','--gripper',effector,'--out',project],check=True,capture_output=True)
    before = load_catalog(BINARY,project,'cube',timeout_s=30)['records']
    assert len(before) >= 3
    candidate = before[0]
    assert candidate['candidate']['hand']['fingersInContact'] >= 2
    assert len(candidate['candidate']['contacts']) >= 2
    evaluated = evaluate_candidate(BINARY,project,'cube',0,timeout_s=30)['records'][0]
    assert evaluated['candidate'] == candidate['candidate']
    assert evaluated['graspPose'] == candidate['graspPose']
    assert (evaluated['plan'] is None) != (evaluated['rejection'] is None)
    assert evaluated['quality'] is not None


def test_multieffector_guards_reject_test_split_and_wrong_mesh(tmp_path, monkeypatch):
    monkeypatch.setattr(native_multieffector,'OBJECTS',('fixture',))
    fixture_manifest(tmp_path,split='test')
    with pytest.raises(ValueError,match='train split'):
        native_multieffector.run(BINARY,tmp_path.resolve(),30)
    fixture_manifest(tmp_path,valid_hash=False)
    with pytest.raises(ValueError,match='provenance'):
        native_multieffector.run(BINARY,tmp_path.resolve(),30)


def test_native_dev_refuses_held_out_before_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(native_dev,'OBJECT_IDS',('fixture',))
    fixture_manifest(tmp_path,split='test')
    monkeypatch.setattr(sys,'argv',['native_dev','--data-root',str(tmp_path),'--binary',str(BINARY),
                                    '--output',str(tmp_path/'out.json')])
    with pytest.raises(ValueError,match='only the development train split'):
        native_dev.main()
    assert not (tmp_path/'out.json').exists()


def test_multieffector_empty_catalog_and_infrastructure_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(native_multieffector,'OBJECTS',('fixture',))
    fixture_manifest(tmp_path)
    monkeypatch.setattr(native_multieffector,'load_catalog',lambda *a: {
        'records':[], 'project_sha256':'fixture', 'elapsed_s':0.})
    result = native_multieffector.run(BINARY,tmp_path.resolve(),30)
    assert len(result['runs']) == 2
    assert all(row['status']=='empty_catalog' and row['evaluations']==[] for row in result['runs'])


def test_native_dev_real_synthetic_object_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(native_dev,'OBJECT_IDS',('fixture',))
    fixture_manifest(tmp_path)
    output = tmp_path/'out.json'
    monkeypatch.setattr(sys,'argv',['native_dev','--data-root',str(tmp_path),'--binary',str(BINARY),
                                    '--output',str(output),'--timeout-s','30'])
    native_dev.main()
    result = json.loads(output.read_text())
    assert result['physical_status'] == 'F3 not evaluated'
    obj = result['objects'][0]
    assert obj['object_id'] == 'fixture' and obj['split'] == 'train'
    assert obj['selected_ids'] == [0, obj['candidate_count']//2, obj['candidate_count']-1]
    assert [r['id'] for r in obj['evaluations']] == obj['selected_ids']
    assert all(r['quality'] is not None and ((r['plan'] is None) != (r['rejection'] is None))
               for r in obj['evaluations'])


def test_multieffector_real_synthetic_object_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(native_multieffector,'OBJECTS',('fixture',))
    fixture_manifest(tmp_path)
    result = native_multieffector.run(BINARY,tmp_path.resolve(),30)
    assert len(result['runs']) == 2 and not result['f3_executed']
    for row in result['runs']:
        assert row['status'] == 'complete'
        assert len(row['evaluations']) == 3
        assert all(item['status'] == 'ok' for item in row['evaluations'])
        for item in row['evaluations']:
            candidate=item['record']['candidate']
            assert candidate['hand']['fingersInContact'] >= 2
            assert len(candidate['contacts']) >= 2


def test_multieffector_retains_failed_candidate_as_infrastructure(tmp_path, monkeypatch):
    monkeypatch.setattr(native_multieffector,'OBJECTS',('fixture',))
    fixture_manifest(tmp_path)
    monkeypatch.setattr(native_multieffector,'load_catalog',lambda *a: {
        'records':[{'candidate':{'label':'fixture'}}], 'project_sha256':'fixture', 'elapsed_s':0.})
    def broken(*args):
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(native_multieffector,'evaluate_candidate',broken)
    result=native_multieffector.run(BINARY,tmp_path.resolve(),30)
    for row in result['runs']:
        assert row['selected_ids'] == [0]
        assert row['evaluations'][0]['status'] == 'infrastructure_error'
        assert 'TimeoutExpired' in row['evaluations'][0]['error']
        assert 'record' not in row['evaluations'][0]


def test_native_dev_rejects_corrupted_mesh(tmp_path, monkeypatch):
    monkeypatch.setattr(native_dev,'OBJECT_IDS',('fixture',))
    fixture_manifest(tmp_path,valid_hash=False)
    monkeypatch.setattr(sys,'argv',['native_dev','--data-root',str(tmp_path),'--binary',str(BINARY),
                                    '--output',str(tmp_path/'out.json')])
    with pytest.raises(ValueError,match='mesh provenance mismatch'):
        native_dev.main()


def test_native_dev_refuses_catalog_with_fewer_than_three_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(native_dev,'OBJECT_IDS',('fixture',))
    fixture_manifest(tmp_path)
    monkeypatch.setattr(native_dev,'load_catalog',lambda *a:{'records':[{},{}]})
    monkeypatch.setattr(sys,'argv',['native_dev','--data-root',str(tmp_path),'--binary',str(BINARY),
                                    '--output',str(tmp_path/'out.json')])
    with pytest.raises(ValueError,match='fewer than three candidates'):
        native_dev.main()
    assert not (tmp_path/'out.json').exists()


def test_multieffector_candidate_drift_is_recorded_not_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(native_multieffector,'OBJECTS',('fixture',))
    fixture_manifest(tmp_path)
    monkeypatch.setattr(native_multieffector,'load_catalog',lambda *a:{
        'records':[{'candidate':{'label':'expected'}}], 'project_sha256':'fixture', 'elapsed_s':0.})
    monkeypatch.setattr(native_multieffector,'evaluate_candidate',lambda *a:{
        'records':[{'candidate':{'label':'different'}}], 'elapsed_s':0.})
    result=native_multieffector.run(BINARY,tmp_path.resolve(),30)
    for row in result['runs']:
        attempt=row['evaluations'][0]
        assert attempt['status']=='infrastructure_error'
        assert 'candidate drift' in attempt['error'] and 'record' not in attempt
