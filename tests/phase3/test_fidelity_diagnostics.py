import pytest
from research_grasping.phase3.fidelity_diagnostics import summarize_pairs


def test_counts_thresholds_and_missing():
    rows = [{'object_id':'a','status':'ok','f2':.9,'f3':0},
            {'object_id':'a','status':'ok','f2':.1,'f3':1},
            {'object_id':'a','status':'ok','f2':.5,'f3':1},
            {'object_id':'b','status':'infrastructure_failure'}]
    result = summarize_pairs(rows, f2_threshold=.5, f3_threshold=.5)
    assert result['confusion'] == {'tp':1,'fp':1,'tn':0,'fn':1}
    assert result['false_positive_rate'] == 1
    assert result['false_negative_rate'] == .5
    assert result['status_counts']['infrastructure_failure'] == 1
    assert result['b15_status'] == 'PENDING'


def test_reject_missing_values_for_ok():
    with pytest.raises(ValueError):
        summarize_pairs([{'object_id':'a','status':'ok','f2':None,'f3':1}], f2_threshold=.5, f3_threshold=.5)


def test_physical_failure_recorded_even_when_unpaired():
    rows = [{'object_id':'a','status':'physical_failure','f2':.8,'f3':0.},
            {'object_id':'b','status':'physical_failure','f2':.2,'f3':None}]
    result = summarize_pairs(rows, f2_threshold=.5, f3_threshold=.5)
    assert result['status_counts']['physical_failure'] == 2
    assert result['physical_failure_unpaired'] == 1
    assert result['confusion']['fp'] == 1
