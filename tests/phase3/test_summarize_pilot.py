import pytest
from research_grasping.phase3.summarize_pilot import summarize_run, summarize_logs


def run():
    return {'definition': {'object_id':'a', 'method':'random', 'seed':1, 'budget':3}, 'pending':None,
            'rows':[{'candidate_id':'1','status':'physical_failure','cost':2,'utility':0.,'details':{'epsilon':.7,'cached':False}},
                    {'candidate_id':'2','status':'ok','cost':3,'utility':.8,'details':{'epsilon':.3,'cached':True}},
                    {'candidate_id':'3','status':'infrastructure_failure','cost':1,'utility':None,'details':{'cached':False}}]}


def test_status_costs_and_no_imputation():
    result = summarize_run(run())
    assert result['curve'] == [0, .8, .8]
    assert result['auc'] == pytest.approx(1.6/3)
    assert result['nominal_online_s'] == 6 and result['fresh_online_s'] == 3
    assert result['cache_hits'] == 1
    assert result['epsilon_all'] == [.7,.3] and result['epsilon_feasible'] == [.3]
    groups = summarize_logs([run()], [])['groups']
    assert groups[0]['f1_rejection_pct'] == 50
    assert groups[0]['infrastructure_error_pct'] == pytest.approx(100/3)
    assert groups[0]['epsilon_unconditional']['mean'] == .5
    assert groups[0]['auc']['low'] is None


def test_pending_or_incomplete_excluded():
    r = run(); r['pending'] = '4'
    assert summarize_run(r) is None
    assert summarize_logs([r], [])['incomplete_runs_excluded'] == 1
