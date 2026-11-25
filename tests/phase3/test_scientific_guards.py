"""Errors and controlled baselines for phase-III statistical helpers."""
from types import SimpleNamespace
import numpy as np
import pytest

from research_grasping.phase3.fidelity_diagnostics import summarize_pairs
from research_grasping.phase3.policy_baselines import PooledGP, _ConditionalKernel
from research_grasping.phase3.summarize_pilot import summarize_run


def test_fidelity_preserves_unpaired_and_validates_inputs():
    kwargs={'f2_threshold':.5,'f3_threshold':.5}
    with pytest.raises(ValueError,match='finite'):summarize_pairs([],f2_threshold=float('nan'),f3_threshold=.5)
    with pytest.raises(ValueError,match='invalid paired'):summarize_pairs([{'object_id':'','status':'ok'}],**kwargs)
    with pytest.raises(ValueError,match='successful pairs'):summarize_pairs([{'object_id':'a','status':'ok','f2':.3}],**kwargs)
    with pytest.raises(ValueError,match='finite'):summarize_pairs([{'object_id':'a','status':'ok','f2':float('inf'),'f3':.1}],**kwargs)
    result=summarize_pairs([{'object_id':'a','status':'physical_failure','f2':None,'f3':None},
                            {'object_id':'b','status':'infrastructure_failure'}],**kwargs)
    assert result['physical_failure_unpaired']==1 and result['n_pairs']==0
    assert result['pooled_spearman_descriptive'] is None and result['false_positive_rate'] is None


def test_gp_guards_and_no_target_ei():
    with pytest.raises(ValueError,match='correlation'):_ConditionalKernel(1.,1.)
    with pytest.raises(ValueError,match='mode'):PooledGP([],np.array([1.]),dimension=1,noise=.01,length_scale=1.,mode='bad')
    with pytest.raises(ValueError,match='context'):PooledGP([],np.array([float('nan')]),dimension=1,noise=.01,length_scale=1.,mode='icm')
    bad=SimpleNamespace(x=np.ones((2,2)),context=np.array([1.]),y=np.array([0.,1.]))
    with pytest.raises(ValueError,match='incompatible'):PooledGP([bad],np.array([1.]),dimension=1,noise=.01,length_scale=1.,mode='icm')
    gp=PooledGP([],np.array([1.]),dimension=1,noise=.01,length_scale=1.,mode='icm')
    with pytest.raises(ValueError,match='target observations'):gp.acquisition(np.array([[.1]]))
    with pytest.raises(ValueError,match='features'):gp.predict(np.array([[float('nan')]]))
    with pytest.raises(ValueError,match='observation'):gp.observe([.1],float('nan'))
    gp.observe([.1],.2)
    assert np.isfinite(gp.acquisition(np.array([[.2]]))).all()


def test_summary_rejects_nonfinite_measured_values():
    row={'candidate_id':'a','status':'ok','cost':1.,'utility':None,'details':{}}
    run={'definition':{'object_id':'x','method':'a','seed':1,'budget':1},'pending':None,'rows':[row]}
    with pytest.raises(ValueError,match='missing measured utility'):summarize_run(run)
    row['utility']=.2;row['details']['epsilon']=float('nan')
    with pytest.raises(ValueError,match='invalid epsilon'):summarize_run(run)
