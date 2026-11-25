import numpy as np
import pytest

from research_grasping.phase5 import baselines as bl
from research_grasping.phase5 import catalogue, stats


def cat(n=6, feasible=(3,), makeable=None, preliminary=None):
    rng = np.random.default_rng(0)
    a = np.tile([0., 0., -1.], (n, 1)); k = np.tile([1., 0., 0.], (n, 1))
    return dict(label=np.array([f'c{i}' for i in range(n)]),
                preliminary=np.array(preliminary if preliminary is not None else np.linspace(.9, .4, n)),
                makeable=np.array(makeable if makeable is not None else [True]*n),
                feasible=np.isin(np.arange(n), feasible), reason=np.array(['x']*n),
                width=np.linspace(.01, .06, n), center_offset=np.zeros(n),
                center=rng.normal(size=(n, 3))*.01+[.4, 0, .45], approach=a, closing=k,
                normalA=k.copy(), normalB=-k, grasp_pose=np.zeros((n, 7)))


def test_order_follows_the_planner_rules():
    c = cat(makeable=[False, True, True, True, True, True])
    # makeable first, then by preliminary score; candidate 0 is examined last
    assert list(catalogue.order(c)) == [1, 2, 3, 4, 5, 0]
    assert catalogue.examined(c) == (3, True)
    # a prior on the feasible candidate brings it first
    prior = {'c3': 1.}
    assert catalogue.examined(c, prior)[0] == 1
    # equal scores keep generator order (stable, like Rust's sort_by)
    flat = cat(preliminary=[.5]*6, feasible=(5,))
    assert list(catalogue.order(flat)) == list(range(6))
    assert catalogue.examined(cat(feasible=()), None) == (6, False)


def test_heuristics_are_scores_in_unit_range():
    c = cat()
    for h in bl.HEURISTICS:
        s = bl.heuristic(h, c, [.38, -.02, .42], [.42, .02, .48])
        assert s.shape == (6,) and s.min() >= 0 and s.max() <= 1
    assert bl.heuristic('topdown', c, [0, 0, 0], [1, 1, 1]).min() == 1
    assert bl.heuristic('narrow', c, [0, 0, 0], [1, 1, 1])[0] == 1
    with pytest.raises(ValueError):
        bl.heuristic('magic', c, [0, 0, 0], [1, 1, 1])


def test_d2_sees_size():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)*.05
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    small, same, large = bl.d2_descriptor(v, f), bl.d2_descriptor(v, f, seed=1), bl.d2_descriptor(v*3, f)
    assert np.isclose(np.linalg.norm(small), 1)
    assert small@same > .99 and small@large < .5


def rows(values_a, values_b, a='m', b='ref'):
    out = []
    for i, (x, y) in enumerate(zip(values_a, values_b)):
        for m, e in ((a, x), (b, y)):
            out.append(dict(part=f'p{i}', method=m, feasible=1, examined1=e, found1=True, examined3=e, found3=False))
    return out


def test_contrast_ratio_and_equivalence():
    same = stats.contrast(rows([10]*40, [10]*40), 'm', 'ref')
    assert same['ratio'] == 1 and same['equal'] == 40 and same['equivalent']
    halved = stats.contrast(rows([5]*20+[6]*20, [10]*20+[12]*20), 'm', 'ref')
    assert halved['ratio'] == pytest.approx(.5) and not halved['equivalent'] and halved['better'] == 40
    assert halved['wilcoxon_p'] < .001


def test_holm_is_step_down():
    r = stats.holm([dict(wilcoxon_p=.01), dict(wilcoxon_p=.04), dict(wilcoxon_p=.03)])
    assert [x['holm_p'] for x in r] == pytest.approx([.03, .06, .06])


def test_budget_counts_parts_without_a_grasp_as_failures():
    rs = [dict(part='a', method='m', found1=True, examined1=1), dict(part='b', method='m', found1=True, examined1=8),
          dict(part='c', method='m', found1=False, examined1=900)]
    b = stats.budget(rs, 'm', budgets=(1, 10))
    assert b['budgets']['1']['rate'] == pytest.approx(1/3) and b['budgets']['10']['rate'] == pytest.approx(2/3)


def test_vit_random_pools_its_initialisations():
    rs = [dict(part='a', method=f'visual:vit-random-s{s}', feasible=1, examined1=e) for s, e in enumerate([2, 4, 6])]
    assert stats.per_part(rs, 'visual:vit-random') == {'a': 4.}
