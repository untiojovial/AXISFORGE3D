import os
from pathlib import Path
import pytest
from research_grasping.phase3.native_catalog import load_catalog, evaluate_candidate, evaluate_batch

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / 'target/release/axisforge'
PROJECT = ROOT / 'examples/pick_and_place_cell.axisforge'
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason='release CLI not built')


def test_catalog_unranked_and_selected_evaluation():
    catalog = load_catalog(BINARY, PROJECT, 'cube')
    records = catalog['records']
    assert [r['id'] for r in records] == list(range(len(records)))
    assert all(r['quality'] is None and r['plan'] is None and r['rejection'] is None for r in records)
    evaluated = evaluate_candidate(BINARY, PROJECT, 'cube', 0)['records'][0]
    assert evaluated['candidate'] == records[0]['candidate']
    assert evaluated['graspPose'] == records[0]['graspPose']
    assert evaluated['quality'] is not None
    assert (evaluated['plan'] is None) != (evaluated['rejection'] is None)


def test_failures_are_not_scores():
    result = evaluate_batch(BINARY, PROJECT, 'cube', [999999])[0]
    assert result['status'] == 'infrastructure_error'
    assert 'result' not in result
