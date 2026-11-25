"""Invalid shape, units and posterior checks for the research API."""

import numpy as np
import pytest

from axisforge_bo.decisions import (
    upper_cvar,
    diverse_solutions,
    probability_feasible,
    constrained_improvement,
    entropy_search,
    pareto_mask,
    hypervolume_2d,
    discrete_knowledge_gradient,
)
from axisforge_bo.robust import certified_candidates, unscented_improvement


@pytest.mark.parametrize(
    "function,args",
    [
        (upper_cvar, ([[1]], 1)),
        (upper_cvar, ([[]], 0)),
        (upper_cvar, ([[float("nan")]], 0.5)),
        (diverse_solutions, ([[0]], [0], 0, 1)),
        (diverse_solutions, ([[0]], [0], 0.1, 0)),
        (probability_feasible, ([[0]], [[0, 1]])),
        (constrained_improvement, ([-1], [1])),
        (pareto_mask, ([[]],)),
        (hypervolume_2d, ([[0, 1]], [2])),
        (discrete_knowledge_gradient, ([], [[0]], [1], [1])),
        (discrete_knowledge_gradient, ([1], [[1]], [1], [1], 130)),
        (entropy_search, ([0], [[1, 2]], [[1]], [1])),
        (entropy_search, ([0], [[1]], [[1]], [0])),
        (entropy_search, ([0], [[1]], [[1]], [1])),
        (entropy_search, ([0, 0], [[1, 1], [0, 1]], [[0], [0]], [1])),
        (certified_candidates, ([[0]], [[0]], [[1]], [-1])),
        (unscented_improvement, (None, [0, 0], [[1, 0], [1, 1]], 0)),
        (unscented_improvement, (None, [0, 0], [[1]], 0)),
    ],
)
def test_rejects_inconsistent_inputs(function, args):
    options = (
        {"paths": 1}
        if function is entropy_search
        and len(args[0]) == 1
        and args[3] == [1]
        and args[1] == [[1]]
        else {}
    )
    with pytest.raises(ValueError):
        function(*args, **options)


def test_pareto_duplicates_and_empty_front():
    assert pareto_mask([[1, 2], [1, 2], [2, 3], [3, 1]]).tolist() == [
        True,
        True,
        False,
        True,
    ]
    assert hypervolume_2d([[2, 2]], [1, 1]) == 0
    assert upper_cvar([[1]], 0.99)[0] == 1


def test_gp_invalid_covariance_propagates():
    class Broken:
        def predict_joint(self, x):
            return np.zeros(len(x)), -np.eye(len(x))

    with pytest.raises(ArithmeticError):
        unscented_improvement(Broken(), [0], [[1]], 0)
