import numpy as np
import pytest
from axisforge_bo.models import StationaryKernel
from axisforge_bo.heteroscedastic import ReplicatedNoiseGP


def test_replicate_means_and_variances():
    # Exact, reproducible within-group variances with unequal replicate counts.
    X = np.array([[0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
    y = np.array([-1.0, 1.0, -2.0, -2.0, 2.0, 2.0])
    m = ReplicatedNoiseGP(StationaryKernel(), StationaryKernel()).fit(X, y)
    np.testing.assert_array_equal(m.group_counts, [2, 4])
    np.testing.assert_allclose(m.group_means, [0, 0])
    np.testing.assert_allclose(m.group_variances, [2, 16 / 3])
    np.testing.assert_allclose(m.mean_model.noise_variance, [1, 4 / 3])
    mean, latent, observed = m.predict([[0.0], [1.0]])
    np.testing.assert_allclose(mean, [0, 0])
    assert np.all(latent >= 0) and np.all(observed > latent)


def test_floor_invalid_and_transactional_fit():
    m = ReplicatedNoiseGP(StationaryKernel(), StationaryKernel(), noise_floor=0.1)
    with pytest.raises(ValueError):
        m.predict([[0.0]])
    with pytest.raises(ValueError):
        m.fit([[0.0], [1.0]], [1.0, 2.0])
    with pytest.raises(ValueError):
        m.fit([[0.0], [0.0]], [1.0, float("nan")])
    m.fit([[0.0], [0.0]], [2.0, 2.0])
    np.testing.assert_allclose(m.group_variances, [0.1])
    before = m.predict([[0.0]])[0]
    with pytest.raises(ValueError):
        m.fit([[0.0], [1.0]], [1.0, 2.0])
    np.testing.assert_array_equal(m.predict([[0.0]])[0], before)
    with pytest.raises(ValueError):
        ReplicatedNoiseGP(StationaryKernel(), StationaryKernel(), 0)


def test_synthetic_distinct_noise_levels():
    rng = np.random.default_rng(84)
    n = 200
    X = np.repeat([[0.0], [2.0]], n, axis=0)
    y = np.concatenate([rng.normal(0, 0.3, n), rng.normal(0, 1.5, n)])
    model = ReplicatedNoiseGP(
        StationaryKernel(), StationaryKernel(length_scale=0.4)
    ).fit(X, y)
    _, latent, observed = model.predict([[0.0], [2.0]])
    noise = observed - latent
    assert noise[1] > 5 * noise[0]
    np.testing.assert_allclose(model.group_variances, [0.09, 2.25], rtol=0.35)
