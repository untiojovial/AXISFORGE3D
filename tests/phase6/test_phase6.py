"""Phase VI: fusion orderings, the gate, and the dynamic conditions."""
import numpy as np
import pytest

from research_grasping.phase6 import fusion


def cat(n=6, makeable=None):
    rng = np.random.default_rng(0)
    return {'label': np.array([f'c{i}' for i in range(n)]), 'preliminary': rng.random(n),
            'makeable': np.ones(n, bool) if makeable is None else np.asarray(makeable, bool),
            'center_offset': np.linspace(0, .06, n), 'feasible': np.ones(n, bool)}


def test_pareto_fronts():
    p = np.array([1., .5, .2, .9, .1])
    q = np.array([.1, .5, .9, .05, .0])
    f = fusion.pareto_fronts(p, q)
    assert list(f) == [0, 0, 0, 1, 2]


def test_scores_reduce_to_their_inputs():
    p, q = np.array([.2, .9, .5]), np.array([.8, .1, .5])
    assert np.argmax(fusion.score('M3', p, q, alpha=1.)) == 1
    assert np.argmax(fusion.score('M3', p, q, alpha=0.)) == 0
    assert np.argmax(fusion.score('M4', p, q, alpha=1.)) == 1
    # lexicographic: only p >= theta enter first, ordered by q
    s = fusion.score('M5', p, q, theta=.4)
    assert list(np.argsort(-s)) == [2, 1, 0]
    # filter with fallback: q >= tau first, ordered by p
    s = fusion.score('M6', p, q, tau=.4)
    assert list(np.argsort(-s)) == [2, 0, 1]


def test_gate_stays_first_with_a_huge_weight():
    c = cat(6, makeable=[0, 1, 0, 1, 0, 1])
    s = np.array([1., 0., .9, .1, .8, .2])  # the fused score prefers the non-makeable ones
    o = fusion.order(c, s)
    assert all(c['makeable'][o[:3]]) and not any(c['makeable'][o[3:]])
    assert list(o[:3]) == [5, 3, 1]


def test_quality_terms():
    c = cat()
    assert fusion.quality(c, 'cen')[0] == pytest.approx(1.) and fusion.quality(c, 'cen')[-1] == pytest.approx(0.)
    assert fusion.quality(c, 'pre').max() == pytest.approx(1.)


def test_phase6_conditions_use_the_intended_masses():
    from tools_import import sweep_conditions
    conditions = dict(sweep_conditions())
    assert conditions['mass x2'] == {'mass': .2} and conditions['mass x0.5'] == {'mass': .05}


def test_phase5_mass_labels_were_absolute_kilograms():
    """Documents the Phase V erratum: its 'mass x2' simulated 2 kg."""
    import re
    from research_grasping.phase5 import dynamics as v1
    box = np.array([[x, y, z] for x in (-.02, .02) for y in (-.02, .02) for z in (0, .04)])
    cand = dict(center=dict(x=0, y=0, z=.02), approach=dict(x=0, y=0, z=-1), closing=dict(x=0, y=1, z=0), width=.04)
    g = dict(fingerWidth=.025, fingerDepth=.045, maxGripForce=90)
    xml, _ = v1.scene([box], np.eye(3), np.zeros(3), 0., cand, g, **dict(v1.PERTURBATIONS)['mass x2'])
    assert re.search(r'<inertial pos="0 0 0" mass="([^"]+)"', xml).group(1) == '2.0'


def test_v2_moves_only_the_centre_of_mass():
    from research_grasping.phase6 import physics
    box = np.array([[x, y, z] for x in (-.02, .02) for y in (-.02, .02) for z in (0, .04)])
    cand = dict(center=dict(x=0, y=0, z=.02), approach=dict(x=0, y=0, z=-1), closing=dict(x=0, y=1, z=0), width=.04)
    g = dict(fingerWidth=.025, fingerDepth=.045, maxGripForce=90)
    a, _ = physics.scene([box], np.eye(3), np.zeros(3), 0., cand, g)
    b, _ = physics.scene([box], np.eye(3), np.zeros(3), 0., cand, g, com=[.01, 0, .02])
    import re
    pos = lambda xml: [float(v) for v in re.search(r'<inertial pos="([^"]+)" mass="0.1"', xml).group(1).split()]
    assert np.allclose(pos(a), 0) and np.allclose(pos(b), [.01, 0, 0])
    assert re.sub(r'<inertial pos="[^"]+" mass="0.1"', '', a) == re.sub(r'<inertial pos="[^"]+" mass="0.1"', '', b)


def test_planner_com_is_recovered():
    from research_grasping.phase6 import physics
    rng = np.random.default_rng(1)
    com = np.array([.1, -.2, .3])
    P = com + rng.normal(0, .03, (40, 3))
    K = rng.normal(size=(40, 3)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    v = com - P
    co = np.linalg.norm(v - (v*K).sum(1, keepdims=True)*K, axis=1)
    got, residual = physics.planner_com({'center': P, 'closing': K, 'center_offset': co})
    assert np.allclose(got, com, atol=1e-7) and residual < 1e-8
