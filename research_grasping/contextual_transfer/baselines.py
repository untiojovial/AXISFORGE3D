"""Diagnostic RGPE-inspired ensemble and fixed-covariance ICM baselines."""

import numpy as np
from scipy.special import ndtr

from axisforge_bo.models import ExactGP, StationaryKernel


def _ei(mu, variance, best):
    sd = np.sqrt(np.maximum(variance, 0))
    delta = mu - best
    z = np.divide(delta, sd, out=np.zeros_like(delta), where=sd > 0)
    return np.where(
        sd > 0,
        delta * ndtr(z) + sd * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
        np.maximum(delta, 0),
    )


def _samples(model, x, n, rng):
    mean, cov = model.predict_joint(x)
    values, vectors = np.linalg.eigh((cov + cov.T) / 2)
    if values.min(initial=0) < -1e-8:
        raise ArithmeticError("non-PSD posterior")
    factor = vectors * np.sqrt(np.maximum(values, 0))
    return mean + rng.standard_normal((n, len(x))) @ factor.T


def ranking_loss(samples, y):
    """Ordered pair loss; shape S,n for sources or S,n,n for LOO."""
    target = y[:, None] < y[None, :]
    if samples.ndim == 2:
        predicted = samples[:, :, None] < samples[:, None, :]
    else:
        predicted = np.diagonal(samples, axis1=1, axis2=2)[:, :, None] < samples
    return np.logical_xor(predicted, target).sum(axis=(1, 2))


class RGPE:
    """Tutorial-inspired rank weights, independently summed GPs, EI acquisition."""

    def __init__(self, sources, *, noise, length_scale, seed, samples=256):
        self.kernel = StationaryKernel(length_scale=length_scale)
        self.noise, self.seed, self.samples = noise, seed, samples
        self.source_models = [
            ExactGP(self.kernel, noise).fit(s.x, s.y) for s in sources
        ]
        self.target = ExactGP(self.kernel, noise)
        self.x = np.empty((0, 1))
        self.y = np.empty(0)
        self.weights = np.r_[np.zeros(len(sources)), 1.0]

    def observe(self, x, y):
        new_x = np.vstack((self.x, np.asarray(x, float).reshape(1, 1)))
        new_y = np.append(self.y, float(y))
        target = ExactGP(self.kernel, self.noise).fit(new_x, new_y)
        self.x, self.y, self.target = new_x, new_y, target
        self.weights = self.rank_weights()

    def rank_weights(self):
        n = len(self.y)
        if n < 2:
            return np.r_[np.zeros(len(self.source_models)), 1.0]
        rng = np.random.default_rng(self.seed + n)
        losses = [
            ranking_loss(_samples(m, self.x, self.samples, rng), self.y)
            for m in self.source_models
        ]
        loo_samples = []
        for i in range(n):
            mask = np.arange(n) != i
            model = ExactGP(self.kernel, self.noise).fit(self.x[mask], self.y[mask])
            loo_samples.append(_samples(model, self.x, self.samples, rng))
        losses.append(ranking_loss(np.stack(loo_samples, axis=1), self.y))
        winners = np.argmin(np.stack(losses), axis=0)
        return np.bincount(winners, minlength=len(losses)) / self.samples

    def predict_joint(self, x):
        models = self.source_models + [self.target]
        predictions = [m.predict_joint(x) for m in models]
        mean = sum(w * p[0] for w, p in zip(self.weights, predictions))
        covariance = sum(w * w * p[1] for w, p in zip(self.weights, predictions))
        return mean, covariance

    def acquisition(self, x):
        mean, covariance = self.predict_joint(x)
        return _ei(mean, np.diag(covariance), max(self.y))


class _ICMKernel:
    def __init__(self, length_scale, correlation):
        if not 0 <= correlation < 1:
            raise ValueError("correlation must be in [0,1)")
        self.base = StationaryKernel(length_scale=length_scale)
        self.correlation = correlation

    def __call__(self, x, y):
        task_cov = self.correlation + (1 - self.correlation) * (
            x[:, -1, None] == y[None, :, -1]
        )
        return self.base(x[:, :-1], y[:, :-1]) * task_cov

    def diagonal(self, x):
        return self.base.diagonal(x[:, :-1])


class FixedICM:
    """ICM GP with fixed equicorrelation, not a learned multitask baseline."""

    def __init__(self, sources, *, noise, length_scale, correlation):
        self.kernel = _ICMKernel(length_scale, correlation)
        self.noise = noise
        self.task = len(sources)
        self.x = np.vstack(
            [
                np.column_stack((s.x, np.full(len(s.y), i)))
                for i, s in enumerate(sources)
            ]
        )
        self.y = np.concatenate([s.y for s in sources])
        self.target_y = []
        self.model = ExactGP(self.kernel, noise).fit(self.x, self.y)

    def observe(self, x, y):
        new_x = np.vstack((self.x, [[float(x[0]), self.task]]))
        new_y = np.append(self.y, y)
        model = ExactGP(self.kernel, self.noise).fit(new_x, new_y)
        self.x, self.y, self.model = new_x, new_y, model
        self.target_y.append(float(y))

    def acquisition(self, x):
        x = np.asarray(x, float)
        mean, variance = self.model.predict(
            np.column_stack((x, np.full(len(x), self.task)))
        )
        return _ei(mean, variance, max(self.target_y))
