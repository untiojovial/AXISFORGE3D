import numpy as np
import pytest

pytest.importorskip("torch")
from axisforge_bo.nonlinear_meta import mlp_parameter_count, nonlinear_task_gradient


def test_nonlinear_implicit_matches_refit_for_multiple_weights():
    rng = np.random.default_rng(21)
    theta = rng.normal(0, 0.3, mlp_parameter_count(2, 2, 1))
    sx = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -0.3]])
    sy = np.array([-0.8, 0.4, 0.7])
    qx = np.array([[0.5, 0.1], [-0.4, 0.6]])
    qy = np.array([0.35, -0.15])
    value, grad, u = nonlinear_task_gradient(
        theta, sx, sy, qx, qy, hidden_dim=2, feature_dim=1
    )
    assert np.isfinite(value) and np.isfinite(u).all()
    for i in (0, 2, 4, 6):
        step = np.zeros_like(theta)
        step[i] = 2e-4
        plus = nonlinear_task_gradient(
            theta + step, sx, sy, qx, qy, hidden_dim=2, feature_dim=1
        )[0]
        minus = nonlinear_task_gradient(
            theta - step, sx, sy, qx, qy, hidden_dim=2, feature_dim=1
        )[0]
        assert grad[i] == pytest.approx((plus - minus) / 4e-4, rel=4e-3, abs=2e-5)


def test_nonlinear_rejects_invalid_or_bad_condition():
    t = np.ones(mlp_parameter_count(1, 2, 1)) * 0.1
    args = (
        t,
        np.array([[-1.0], [0.0], [1.0]]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([[0.5]]),
        np.array([0.5]),
    )
    with pytest.raises(ValueError):
        nonlinear_task_gradient(*args, hidden_dim=2, feature_dim=1, prior_precision=-1)
    with pytest.raises(ValueError):
        nonlinear_task_gradient(t[:-1], *args[1:], hidden_dim=2, feature_dim=1)
    with pytest.raises(ArithmeticError, match="curvature"):
        nonlinear_task_gradient(
            *args, hidden_dim=2, feature_dim=1, max_condition=1.00001
        )


def test_nonlinear_training_step_and_separate_heldout_task():
    from axisforge_bo.nonlinear_meta import nonlinear_meta_step

    rng = np.random.default_rng(24)
    t = rng.normal(0, 0.4, mlp_parameter_count(1, 2, 1))
    train = (
        np.array([[-1.0], [0.0], [1.0]]),
        np.array([-0.8, 0.0, 0.9]),
        np.array([[-0.4], [0.5]]),
        np.array([-0.3, 0.45]),
    )
    heldout = (
        np.array([[-0.9], [0.1], [0.8]]),
        np.array([-0.7, 0.1, 0.7]),
        np.array([[-0.6], [0.3]]),
        np.array([-0.5, 0.25]),
    )
    new, old_loss = nonlinear_meta_step(
        t, [train], hidden_dim=2, feature_dim=1, learning_rate=0.01
    )
    assert (
        nonlinear_task_gradient(new, *train, hidden_dim=2, feature_dim=1)[0] < old_loss
    )
    assert np.isfinite(
        nonlinear_task_gradient(new, *heldout, hidden_dim=2, feature_dim=1)[0]
    )
    with pytest.raises(ValueError):
        nonlinear_meta_step(t, [], hidden_dim=2, feature_dim=1)
