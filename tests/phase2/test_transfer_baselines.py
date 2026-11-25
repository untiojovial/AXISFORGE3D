import numpy as np

from axisforge_bo.models import ExactGP
from research_grasping.contextual_transfer import Source
from research_grasping.contextual_transfer.baselines import RGPE, FixedICM, ranking_loss


def source():
    return Source(
        "s", np.array([0.0]), np.array([[0.1], [0.7]]), np.array([1.0, -0.3]), 0
    )


def test_rank_loss_and_loo():
    y = np.array([0.1, 0.5, -0.1])
    draws = np.array([[1.0, 3.0, 0.0]])
    assert ranking_loss(draws, y)[0] == 0
    model = RGPE([source()], noise=0.05, length_scale=0.2, seed=10, samples=64)
    for x, value in zip([0.1, 0.4, 0.8], y):
        model.observe([x], value)
    assert np.isclose(model.weights.sum(), 1)
    assert np.array_equal(model.weights, model.rank_weights())
    for i in range(3):
        mask = np.arange(3) != i
        loo = ExactGP(model.kernel, model.noise).fit(model.x[mask], model.y[mask])
        assert np.isfinite(loo.predict(model.x)[0]).all()


def test_sum_covariance_and_fixed_icm_psd():
    model = RGPE([source()], noise=0.05, length_scale=0.2, seed=10, samples=64)
    model.observe([0.2], 0.1)
    model.observe([0.5], 0.4)
    x = np.array([[0.2], [0.6], [0.9]])
    mu, cov = model.predict_joint(x)
    parts = [m.predict_joint(x) for m in model.source_models + [model.target]]
    assert np.allclose(mu, sum(w * p[0] for w, p in zip(model.weights, parts)))
    assert np.allclose(cov, sum(w * w * p[1] for w, p in zip(model.weights, parts)))
    assert np.linalg.eigvalsh(cov).min() > -1e-10
    icm = FixedICM([source()], noise=0.05, length_scale=0.2, correlation=0.5)
    matrix = icm.kernel(
        np.array([[0.2, 0], [0.2, 1], [0.7, 1]]),
        np.array([[0.2, 0], [0.2, 1], [0.7, 1]]),
    )
    assert np.linalg.eigvalsh(matrix).min() > -1e-10


def test_rgpe_prefix_does_not_depend_on_future():
    first = RGPE([source()], noise=0.05, length_scale=0.2, seed=10, samples=64)
    second = RGPE([source()], noise=0.05, length_scale=0.2, seed=10, samples=64)
    for model in (first, second):
        model.observe([0.2], 0.1)
        model.observe([0.5], 0.4)
    before = first.weights.copy()
    assert np.array_equal(before, second.weights)
    second.observe([0.9], -0.8)
    assert np.array_equal(first.weights, before)
