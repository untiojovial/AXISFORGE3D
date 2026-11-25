import pytest
from research_grasping.phase4.transfer_statistics import compare, trajectory_auc


def test_hierarchical_paired_bootstrap_and_symmetry():
    a={('x',1):.2,('x',2):.4,('y',1):.3,('y',2):.5}
    b={key:value-.1 for key,value in a.items()}
    result=compare(a,b,draws=200,seed=1)
    assert result['mean_delta']==pytest.approx(.1)
    assert result['bootstrap_ci95_percentile']==pytest.approx([.1,.1])
    with pytest.raises(ValueError): compare(a,{('x',1):.1})


def test_missing_evaluations_not_silently_imputed():
    assert trajectory_auc([{'status':'ok','utility':.2},{'status':'physical_failure','utility':0}])==pytest.approx(.2)
    with pytest.raises(ValueError): trajectory_auc([{'status':'infrastructure_failure','utility':None}])
