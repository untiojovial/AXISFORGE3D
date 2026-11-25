import numpy as np
import pytest

from research_grasping.contextual_transfer import AdaptiveTransfer, Source


def make():
    source = Source(
        "s", np.array([0.0]), np.array([[0.2], [0.8]]), np.array([1.0, -1.0]), 0
    )
    return AdaptiveTransfer(
        [0.0], [source], target_time=1, dimension=1, noise_variance=0.1
    )


def test_mixture_and_psd():
    m = make()
    x = np.array([[0.1], [0.5], [0.9]])
    mu, var = m.predict(x)
    joint, cov = m.predict_joint(x)
    means, vs = m.components(x)
    w = m.weights[:, None]
    assert np.allclose(mu, (w * means).sum(0))
    assert np.allclose(var, (w * (vs + (means - mu) ** 2)).sum(0))
    assert np.allclose(joint, mu) and np.allclose(np.diag(cov), var)
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_ei_matches_monte_carlo():
    m = make()
    m.observe([0.3], 0.2)
    x = np.array([[0.6]])
    mu, v = m.components(x)
    w = m.weights
    rng = np.random.default_rng(91)
    indices = rng.choice(len(w), 300000, p=w)
    samples = rng.normal(mu[indices, 0], np.sqrt(v[indices, 0]))
    assert abs(m.acquisition(x)[0] - np.maximum(samples - 0.2, 0).mean()) < 0.004


def test_prequential_and_no_sources():
    m = make()
    m.observe([0.4], 0.5)
    snapshot = (m.weights.copy(), m.predict(np.array([[0.7]]))[0].copy())
    other = make()
    other.observe([0.4], 0.5)
    assert np.allclose(other.weights, snapshot[0])
    other.observe([0.6], -0.8)
    assert np.allclose(m.weights, snapshot[0]) and np.allclose(
        m.predict(np.array([[0.7]]))[0], snapshot[1]
    )
    empty = AdaptiveTransfer([0.0], [], target_time=1, dimension=1)
    assert np.array_equal(empty.weights, [1.0])
    with pytest.raises(ValueError):
        empty.acquisition([[0.5]])


def test_future_source_rejected():
    src = Source("future", np.array([0.0]), np.array([[0.2]]), np.array([1.0]), 1)
    with pytest.raises(ValueError):
        AdaptiveTransfer([0.0], [src], target_time=1, dimension=1)


def test_failed_fit_does_not_mutate_state(monkeypatch):
    m = make()
    m.observe([0.3], 0.2)
    old_weights = m.weights.copy()
    old_y = m.y.copy()
    old_prediction = m.predict([[0.4]])

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("injected failure")

    monkeypatch.setattr("axisforge_bo.models.np.linalg.cholesky", fail)
    with pytest.raises(np.linalg.LinAlgError):
        m.observe([0.7], 0.5)
    assert np.array_equal(m.y, old_y)
    assert np.array_equal(m.weights, old_weights)
    assert all(np.array_equal(a, b) for a, b in zip(m.predict([[0.4]]), old_prediction))
