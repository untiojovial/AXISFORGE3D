"""Independent auditor's mathematical and adversarial checks."""

import math
import numpy as np
import pytest
from scipy.stats import norm
from axisforge_bo.decisions import (
    probability_feasible,
    upper_cvar,
    hypervolume_2d,
    discrete_knowledge_gradient,
)


def test_feasibility_independent_and_deterministic_limits():
    got = probability_feasible(
        [[0.0, -1.0, 1.0], [-1.0, -1.0, -1.0]], [[1.0, 0.0, 4.0], [0.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(got, [0.5 * norm.cdf(-0.5), 1.0], rtol=1e-14)
    with pytest.raises(ValueError):
        probability_feasible([[1.0]], [[-1.0]])


def test_cvar_fractional_tail():
    assert upper_cvar([[0.0, 2.0, 5.0, 10.0]], 0.625)[0] == pytest.approx(
        (10.0 + 0.5 * 5.0) / 1.5
    )
    assert upper_cvar([[0.0, 2.0, 5.0, 10.0]], 0)[0] == pytest.approx(4.25)


def test_knowledge_gradient_gaussian_analytic():
    # Two equal-mean alternatives updated with posterior slopes +1 and -1.
    # Before: min(0,0)=0. After: min(Z,-Z)=-|Z|. Cost doubles denominator.
    result = discrete_knowledge_gradient(
        [0.0, 0.0], [[1.0], [-1.0]], [1.0], [2.0], quadrature_order=128
    )
    assert result[0] == pytest.approx(math.sqrt(2 / math.pi) / 2, rel=0.004)
    assert discrete_knowledge_gradient([0.0, 1.0], [[0.0], [0.0]], [0.0], [1.0])[0] == 0
    with pytest.raises(ValueError):
        discrete_knowledge_gradient([0.0], [[1.0]], [0.0], [1.0])
    with pytest.raises(ValueError):
        discrete_knowledge_gradient([0.0], [[0.0]], [1.0], [0.0])


def test_hypervolume_against_botorch():
    torch = pytest.importorskip("torch")
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    values = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    ref = np.array([5.0, 5.0])
    expected = Hypervolume(torch.tensor(-ref)).compute(torch.tensor(-values))
    assert hypervolume_2d(values, ref) == pytest.approx(expected, abs=1e-12)
    assert hypervolume_2d(values, ref) == pytest.approx(11.0)


def test_botorch_scalar_logei_matches_independent_formula():
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    from axisforge_bo.botorch_backend import TorchExperiment

    model = TorchExperiment(
        [[0.0], [0.5], [1.0]],
        [[0.0], [1.0], [0.0]],
        variances=[[0.01], [0.01], [0.01]],
        fit=False,
    )
    point = torch.tensor([[0.3]], dtype=torch.double)
    posterior = model.model.posterior(point)
    mu = posterior.mean.item()
    sigma = posterior.variance.sqrt().item()
    z = (mu - 1.0) / sigma
    expected = math.log((mu - 1.0) * norm.cdf(z) + sigma * norm.pdf(z))
    actual = model.scores([[[0.3]]], method="log_ei").item()
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-10)
    with pytest.raises(ValueError):
        model.scores([[[0.3]]], method="log_ei", pending=[[0.5]])
    with pytest.raises(ValueError):
        model.scores([[[1.1]]], method="log_ei")


def test_finite_extremes_fail_instead_of_silent_nonfinite():
    from axisforge_bo.acquisitions import expected_improvement, lower_confidence_bound
    from axisforge_bo.models import StationaryKernel

    with pytest.raises(ArithmeticError):
        expected_improvement([-1e308], [1.0], 1e308)
    with pytest.raises(ArithmeticError):
        lower_confidence_bound([-1e308], [1e308], 1e308)
    with np.errstate(all="ignore"), pytest.raises(ArithmeticError):
        StationaryKernel(kind="matern52")([[-1e308]], [[1e308]])


def test_multifidelity_backend_respects_unit_cube_and_positive_noise():
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    from axisforge_bo.botorch_backend import TorchExperiment

    x = [[0.1, 0.3], [0.4, 0.7], [0.8, 1.0]]
    y = [[0.2], [0.6], [0.8]]
    m = TorchExperiment(x, y, variances=[[0.01]] * 3, fidelity=1, fit=False)
    posterior = m.model.posterior(torch.tensor([[0.4, 1.0]], dtype=torch.double))
    assert torch.isfinite(posterior.mean).all() and (posterior.variance >= 0).all()
    with pytest.raises(ValueError):
        TorchExperiment(x, y, variances=[[0.0]] * 3, fidelity=1, fit=False)
    with pytest.raises(ValueError):
        TorchExperiment(x, y, fidelity=2, fit=False)
    with pytest.raises(ValueError):
        m.scores([[[0.4, 1.1]]])


def test_local_optimizer_restart_keeps_global_history():
    from axisforge_bo.local_optimizer import LocalBayesianOptimizer
    from axisforge_bo.space import Box

    opt = LocalBayesianOptimizer(Box([[0.0, 1.0]]), seed=12, pool_size=4)
    first = opt.ask()
    opt.tell(first, -10.0)
    opt.region.restart_required = True
    second = opt.ask()
    opt.tell(second, 1.0)
    assert len(opt.observations) == 2
    point, value = opt.best()
    np.testing.assert_array_equal(point, first)
    assert value == -10.0
