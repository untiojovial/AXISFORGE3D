import numpy as np
import pytest
from axisforge_bo.benchmarks import PROBLEMS, multiobjective
from axisforge_bo.decisions import (
    entropy_search,
    diverse_solutions,
    constrained_improvement,
)


@pytest.mark.parametrize(
    "name,x",
    [
        ("branin", [(np.pi + 5) / 15, 2.275 / 15]),
        ("hartmann3", [0.114614, 0.555649, 0.852547]),
        ("hartmann6", [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]),
        ("sphere", [0.5] * 3),
        ("levy", [0.55] * 3),
        ("rosenbrock", [0.75] * 3),
        ("rastrigin", [0.5] * 3),
        ("ackley", [0.5] * 3),
    ],
)
def test_optima(name, x):
    p = PROBLEMS[name]
    assert p([x])[0] == pytest.approx(p.optimum, abs=2e-6)
    with pytest.raises(ValueError):
        p(x)
    with pytest.raises(ValueError):
        p(np.full((1, p.dimension), -1))


def test_pareto_examples():
    np.testing.assert_allclose(multiobjective("zdt1", [[0.25, 0, 0]]), [[0.25, 0.5]])
    np.testing.assert_allclose(
        multiobjective("dtlz2", [[0.5, 0.5, 0.5]]), [[2**-0.5, 2**-0.5]]
    )
    with pytest.raises(ValueError):
        multiobjective("missing", [[0, 0.5]])


def test_mutual_information():
    scores = entropy_search([0, 0], np.eye(2), [[0, 1], [0, 0]], [1, 1], seed=13)
    assert scores[0] == pytest.approx(0, abs=1e-12)
    assert 0.13 < scores[1] < 0.26
    with pytest.raises(ValueError):
        entropy_search([0, 0], np.eye(2), [[2], [0]], [1])


def test_diverse_and_feasible():
    assert diverse_solutions([[0], [0.1], [1]], [1.0, 2.0, 3.0], 0.5, 3).tolist() == [
        0,
        2,
    ]
    np.testing.assert_array_equal(constrained_improvement([1, 2], [0, 0.5]), [0, 1])


def test_unscented_integrates_correlated_predictions():
    from axisforge_bo.robust import unscented_improvement

    class LinearPosterior:
        def predict_joint(self, x):
            return x[:, 0] ** 2, 1 + x @ x.T

    mu, var, ei = unscented_improvement(LinearPosterior(), [0.2], [[0.04]], 0.5)
    assert mu == pytest.approx(0.08)
    assert var == pytest.approx(1.04)
    assert ei > 0
    with pytest.raises(ValueError):
        unscented_improvement(LinearPosterior(), [0], [[-1]], 0)


def test_lipschitz_feasible_set():
    from axisforge_bo.robust import certified_candidates

    np.testing.assert_array_equal(
        certified_candidates([[0], [0.4], [1]], [[0]], [[-0.5]], [1]),
        [True, True, False],
    )
    assert not certified_candidates([[0]], [[0]], [[1]], [1])[0]
