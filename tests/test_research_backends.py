import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
from axisforge_bo.botorch_backend import TorchExperiment
from axisforge_bo.surrogates import ResearchGP

X = np.array([[0.0], [0.33], [0.65], [0.98]])
Y = np.sin(X * 4)
BATCHES = np.array([[[0.2], [0.5]], [[0.12], [0.48]]])


@pytest.mark.parametrize("name", ["qlog_ei", "qlog_nei", "qlog_ehvi", "qlog_nehvi"])
def test_joint_acquisitions_with_pending(name):
    multi = name.endswith("hvi")
    y = np.hstack((Y, 1 - Y)) if multi else Y
    model = TorchExperiment(X, y, variances=np.full_like(y, 1e-4), fit=False)
    options = dict(
        method=name,
        reference=[-1, -1] if multi else None,
        pending=[[0.85]],
        samples=16,
        seed=21,
    )
    scores = model.scores(BATCHES, **options)
    assert scores.shape == (2,)
    assert torch.allclose(scores, model.scores(BATCHES, **options))


@pytest.mark.parametrize("name", ["mes", "pes"])
def test_information_gain(name):
    model = TorchExperiment(X, Y, fit=False)
    a = model.information_scores(
        [[0.1], [0.7]], [[0.01], [0.42], [0.83]], method=name, samples=4
    )
    assert a.shape == (2,)
    assert torch.isfinite(a).all()


@pytest.mark.parametrize("kind", ["variational", "multitask"])
def test_learned_gp(kind):
    gp = ResearchGP(kind, steps=10, inducing=3)
    with pytest.raises(RuntimeError):
        gp.predict(X)
    x, y = (
        (X, Y.ravel())
        if kind == "variational"
        else (
            np.column_stack((np.tile(X.ravel(), 2), np.repeat([0, 1], 4))),
            np.r_[Y.ravel(), 2 + Y.ravel()],
        )
    )
    gp.fit(x, y)
    mu, var = gp.predict(x)
    assert mu.shape == var.shape == y.shape and np.all(var > 0)
    if kind == "multitask":
        assert mu[4:].mean() > mu[:4].mean() + 1
        with pytest.raises(ValueError):
            gp.predict([[0.5, 3]])
    assert np.isfinite(mu).all()


def test_saas_mixture():
    pytest.importorskip("numpyro")
    from axisforge_bo.surrogates import SaasGP

    # Tiny chain exercises the inference path only, not posterior convergence.
    model = SaasGP(warmup=8, draws=8, thinning=2, seed=9).fit(X, Y.ravel())
    mean, var = model.predict(X)
    posterior = model.model.posterior(torch.as_tensor(X, dtype=torch.double))
    assert np.isfinite(mean).all() and np.all(var >= 0)
    np.testing.assert_allclose(
        var,
        (posterior.variance.mean(0) + posterior.mean.var(0, unbiased=False))
        .detach()
        .numpy()
        .ravel(),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "unknown"},
        {"samples": 0},
        {"seed": -1},
        {"reference": [1]},
        {"method": "qlog_ehvi", "reference": [1, 2]},
        {"pending": [[0.1, 0.2]]},
        {"method": "qlog_ei", "pending": [[1.1]]},
    ],
)
def test_scoring_rejects_invalid_requests(kwargs):
    model = TorchExperiment(X, Y, fit=False)
    with pytest.raises(ValueError):
        model.scores(BATCHES, **kwargs)


def test_batch_shapes_and_references():
    model = TorchExperiment(X, np.hstack((Y, Y)), fit=False)
    for opts in [{}, {"reference": [1]}]:
        with pytest.raises(ValueError):
            model.scores(BATCHES, method="qlog_ehvi", **opts)
    model = TorchExperiment(X, Y, fit=False)
    for invalid in [[], [[0.1]], [[[np.nan]]]]:
        with pytest.raises(ValueError):
            model.scores(invalid)
    for kwargs in [dict(method="bad"), dict(samples=1)]:
        with pytest.raises(ValueError):
            model.information_scores(X, X, **kwargs)
    with pytest.raises(ValueError):
        model.information_scores(X, X + 1)


def test_backend_fits_and_copies_input():
    x = X.copy()
    model = TorchExperiment(x, Y, variances=np.full_like(Y, 0.05), fit=True)
    x[:] = 0
    np.testing.assert_allclose(model.x, X)
    assert torch.isfinite(model.scores(BATCHES[:, :1])).all()


def test_research_gp_validation():
    for opts in [dict(kind="absent"), dict(inducing=0)]:
        with pytest.raises(ValueError):
            ResearchGP(**opts)
    gp = ResearchGP("multitask", steps=2)
    with pytest.raises(ValueError):
        gp.fit(X, [1, 2])
    with pytest.raises(ValueError):
        gp.fit(X, Y.ravel())
    with pytest.raises(ValueError):
        gp.fit(np.column_stack((X, X)), Y.ravel())
    gp = ResearchGP("variational", steps=2).fit(X, np.ones(len(X)))
    with pytest.raises(ValueError):
        gp.predict([[1, 2]])
    assert np.isfinite(gp.predict([[0.3]])[0]).all()


def test_saas_validation():
    from axisforge_bo.surrogates import SaasGP

    with pytest.raises(ValueError):
        SaasGP(draws=1, thinning=2)
    with pytest.raises(RuntimeError):
        SaasGP().predict(X)
    with pytest.raises(ValueError):
        SaasGP().fit(X, X)
