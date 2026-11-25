import numpy as np
from research_grasping.phase3.policy_baselines import PooledGP, _ConditionalKernel
from research_grasping.contextual_transfer import Source


def test_icm_psd_and_pooled_causality():
    x = np.array([[.1,.2,0], [.8,.3,1], [.2,.7,2]])
    kernel = _ConditionalKernel(1., .5)
    assert np.linalg.eigvalsh(kernel(x,x)).min() > -1e-10
    source = Source('s', np.array([.3]), np.array([[0.,0.],[1.,1.]]), np.array([0.,1.]), 0.)
    a = PooledGP([source], [.2], dimension=2, noise=.1, length_scale=1., mode='icm')
    b = PooledGP([source], [.2], dimension=2, noise=.1, length_scale=1., mode='icm')
    a.observe([.1,.2], .4); b.observe([.1,.2], .4)
    before = a.predict([[.3,.4]])
    assert all(np.allclose(x,y) for x,y in zip(before,b.predict([[.3,.4]])))
    b.observe([.8,.3], .9)
    assert all(np.allclose(x,y) for x,y in zip(before,a.predict([[.3,.4]])))


def test_contextual_model_depends_on_context():
    source = Source('s', np.array([0.]), np.array([[0.,0.]]), np.array([1.]), 0.)
    a = PooledGP([source], [0.], dimension=2, noise=.01, length_scale=1., mode='contextual')
    b = PooledGP([source], [3.], dimension=2, noise=.01, length_scale=1., mode='contextual')
    assert a.predict([[0.,0.]])[0][0] > b.predict([[0.,0.]])[0][0]
