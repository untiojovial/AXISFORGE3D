import numpy as np
import pytest

pytest.importorskip("torch")
from axisforge_bo.meta_learning import task_meta_gradient


def test_task_adaptation_implicit_gradient_matches_refit():
    theta = np.array([[0.7]])
    sx, sy = np.array([[-1.0], [0.0], [1.0]]), np.array([-0.8, 0.0, 0.9])
    qx, qy = np.array([[-0.4], [0.6]]), np.array([-0.3, 0.5])
    value, grad, u = task_meta_gradient(theta, sx, sy, qx, qy)
    eps = 1e-4
    plus = task_meta_gradient(theta + eps, sx, sy, qx, qy)[0]
    minus = task_meta_gradient(theta - eps, sx, sy, qx, qy)[0]
    assert np.isfinite(value) and np.isfinite(u).all()
    assert grad[0, 0] == pytest.approx((plus - minus) / (2 * eps), rel=2e-3, abs=1e-4)


def test_meta_step_improves_query_loss_locally():
    from axisforge_bo.meta_learning import meta_step

    theta = np.array([[0.7]])
    task = (
        np.array([[-1.0], [0.0], [1.0]]),
        np.array([-0.8, 0.0, 0.9]),
        np.array([[-0.4], [0.6]]),
        np.array([-0.3, 0.5]),
    )
    updated, before = meta_step(theta, [task], learning_rate=0.01)
    after = task_meta_gradient(updated, *task)[0]
    assert after < before
    np.testing.assert_array_equal(theta, [[0.7]])


def test_multidimensional_meta_gradient_refits_and_bad_inputs():
    theta = np.array([[0.6], [0.3]])
    sx = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -0.2]])
    sy = np.array([-0.6, 0.3, 0.4])
    qx = np.array([[0.4, 0.5], [-0.3, 0.7]])
    qy = np.array([0.4, -0.1])
    _, grad, _ = task_meta_gradient(theta, sx, sy, qx, qy)
    for i in range(2):
        step = np.zeros_like(theta)
        step[i, 0] = 1e-4
        fd = (
            task_meta_gradient(theta + step, sx, sy, qx, qy)[0]
            - task_meta_gradient(theta - step, sx, sy, qx, qy)[0]
        ) / (2e-4)
        assert grad[i, 0] == pytest.approx(fd, rel=3e-3, abs=1e-4)
    with pytest.raises(ValueError):
        task_meta_gradient(theta, sx, sy, qx, [])
    with pytest.raises(ValueError):
        task_meta_gradient(theta, sx, sy, qx, qy, regularization=0)
    from axisforge_bo.meta_learning import meta_step

    with pytest.raises(ValueError):
        meta_step(theta, [])
    with pytest.raises(ValueError):
        meta_step(theta, [], learning_rate=-1)
