"""Integrity of the fixed multi-effector development smoke evidence."""
import json
from pathlib import Path


def test_multieffector_results_preserve_all_prespecified_outcomes():
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / 'research/phase3/robot/native_multieffector_results.json').read_text())
    assert not data['f3_executed']
    assert len(data['runs']) == 6
    assert {r['effector'] for r in data['runs']} == {'three-finger', 'hand'}
    assert len({r['object_id'] for r in data['runs']}) == 3
    for run in data['runs']:
        if run['status'] == 'empty_catalog':
            assert run['candidate_count'] == 0 and not run['evaluations']
        elif run['status'] == 'complete':
            n = run['candidate_count']
            assert run['selected_ids'] == list(dict.fromkeys((0, n // 2, n - 1)))
            assert [r['id'] for r in run['evaluations']] == run['selected_ids']
            for item in run['evaluations']:
                if item['status'] == 'ok':
                    record = item['record']
                    assert record['id'] == item['id']
                    assert record['quality'] is not None
                    assert (record['plan'] is None) != (record['rejection'] is None)
        else:
            assert run['status'] == 'infrastructure_error' and run['error']
