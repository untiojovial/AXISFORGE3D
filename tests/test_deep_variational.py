import numpy as np
import pytest

pytest.importorskip("gpytorch")
from axisforge_bo.deep_gp import DeepVariationalGP


def test_stochastic_layers_preserve_noise_and_rng():
    import torch

    model = DeepVariationalGP(steps=8, samples=8, seed=91)
    with pytest.raises(RuntimeError):
        model.predict([[0]])
    with pytest.raises(ValueError):
        model.fit([[0]], [1])
    x = np.linspace(0, 1, 8)[:, None]
    state = torch.random.get_rng_state().clone()
    model.fit(x, np.cos(x[:, 0] * 2))
    mu, latent = model.predict(x)
    obs_mu, obs = model.predict(x, observation=True)
    np.testing.assert_allclose(mu, obs_mu)
    assert len(mu) == len(x) and np.all(latent > 0) and np.all(obs > latent)
    np.testing.assert_allclose(
        obs - latent, model.model.likelihood.noise.item() * model.scale.item() ** 2
    )
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.model.features.output_dims == 2
    with pytest.raises(ValueError):
        model.predict([[0, 1]])


@pytest.mark.parametrize("opts", [{"inducing": 0}, {"samples": 0}, {"seed": -1}])
def test_invalid_budgets(opts):
    with pytest.raises(ValueError):
        DeepVariationalGP(**opts)
