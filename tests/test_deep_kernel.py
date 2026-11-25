import numpy as np
import pytest

torch = pytest.importorskip("torch")
from axisforge_bo.deep_kernel import DeepKernelGP


def test_dkl_train_posterior_and_gradients():
    x = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([-1.0, 0.0, 1.0])
    model = DeepKernelGP(1, seed=7)
    start = float(model.loss(x, y).detach())
    model.fit(x, y, steps=40)
    end = float(model.loss(x, y).detach())
    assert end < start
    mean, variance = model.predict(x)
    assert mean.shape == variance.shape == (3,)
    assert np.isfinite(mean).all() and (variance >= 0).all()
    loss = model.loss(x, y)
    grad = torch.autograd.grad(loss, model.log_amplitude)[0].item()
    with torch.no_grad():
        model.log_amplitude += 1e-5
    plus = model.loss(x, y).item()
    with torch.no_grad():
        model.log_amplitude -= 2e-5
    minus = model.loss(x, y).item()
    assert grad == pytest.approx((plus - minus) / 2e-5, rel=1e-5, abs=1e-6)


def test_dkl_rejects_bad_shapes():
    m = DeepKernelGP(1)
    with pytest.raises(ValueError):
        m.fit([[1]], [float("nan")])
    with pytest.raises(ValueError):
        m.predict([[1, 2]])


def test_dkl_matches_independent_numpy_posterior():
    x = np.array([[-1.0], [0.5], [1.0]])
    y = np.array([1.0, -0.5, 0.25])
    q = np.array([[-0.25], [0.75]])
    m = DeepKernelGP(1, seed=9).fit(x, y, steps=0)
    with torch.no_grad():
        z = m.features(torch.tensor(x, dtype=torch.double)).numpy()
        zq = m.features(torch.tensor(q, dtype=torch.double)).numpy()
        amp = m.log_amplitude.exp().item()
        length = m.log_lengthscale.exp().item()
        noise = m.log_noise.exp().item()

    def k(a, b):
        return amp**2 * np.exp(
            -((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) / (2 * length**2)
        )

    K = k(z, z) + (noise + m.jitter) * np.eye(3)
    cross = k(z, zq)
    mean, variance = m.predict(q)
    np.testing.assert_allclose(
        mean, cross.T @ np.linalg.solve(K, y), rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        variance,
        np.diag(k(zq, zq)) - np.sum(cross * np.linalg.solve(K, cross), axis=0),
        rtol=1e-10,
        atol=1e-10,
    )


def test_dkl_refuses_nonfinite_parameters():
    m = DeepKernelGP(1)
    with torch.no_grad():
        m.log_noise.fill_(float("nan"))
    with pytest.raises(ArithmeticError):
        m.predict([[0.0]])


def test_dkl_prior_invalid_configs_and_fit():
    with pytest.raises(ValueError):
        DeepKernelGP(0)
    with pytest.raises(ValueError):
        DeepKernelGP(1, jitter=0)
    model = DeepKernelGP(1)
    mean, var = model.predict([[0.0]])
    assert mean[0] == 0 and var[0] > 0
    with pytest.raises(ValueError):
        model.fit([[0.0]], [1.0], steps=-1)
    with pytest.raises(ValueError):
        model.fit([], [])
    model.float()
    with pytest.raises(ValueError):
        model.predict([[0.0]])
