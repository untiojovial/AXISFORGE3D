"""Prequential mixture of fixed-hyperparameter AxisForgeBO Gaussian processes."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, ndtr

from axisforge_bo.models import ExactGP, StationaryKernel


@dataclass(frozen=True)
class Source:
    """Source in common decision/utility units; available_at must predate target."""

    task_id: str
    context: np.ndarray
    x: np.ndarray
    y: np.ndarray
    available_at: float


class AdaptiveTransfer:
    """Latent mixture, exact marginal EI, and prequential observation update.

    Input scaling, task permissions, common units, and fixed hyperparameters are
    supplied by the caller. Identical observation noise is assumed across tasks.
    Models are independent conditional experts, not a posterior GP mixture fit.
    """

    def __init__(
        self,
        context,
        sources=(),
        *,
        target_time,
        dimension,
        length_scale=0.2,
        amplitude=1.0,
        noise_variance=0.01,
        k=3,
        tau=1.0,
        eta=1.0,
        floor=0.1,
        use_evidence=True,
        contextual=True,
    ):
        c = np.asarray(context, dtype=float)
        if c.ndim != 1 or not len(c) or not np.isfinite(c).all():
            raise ValueError("invalid context")
        if (
            type(dimension) is not int
            or dimension < 1
            or type(k) is not int
            or k < 0
            or not np.isfinite([target_time, tau, eta, floor, noise_variance]).all()
            or tau <= 0
            or eta <= 0
            or not 0 <= floor < 1
            or noise_variance <= 0
        ):
            raise ValueError("invalid configuration")
        self.dimension, self.noise = dimension, float(noise_variance)
        self.eta, self.floor, self.use_evidence = eta, floor, use_evidence
        self.kernel = StationaryKernel(length_scale=length_scale, amplitude=amplitude)
        self.target = ExactGP(self.kernel, noise_variance=noise_variance)
        self.x = np.empty((0, dimension))
        self.y = np.empty(0)
        selected = []
        ids = set()
        for src in sources:
            sc, sx, sy = (
                np.asarray(src.context, float),
                np.asarray(src.x, float),
                np.asarray(src.y, float),
            )
            if (
                not src.task_id
                or src.task_id in ids
                or not np.isfinite(src.available_at)
                or src.available_at >= target_time
                or sc.shape != c.shape
                or sx.ndim != 2
                or sx.shape[1] != dimension
                or sy.shape != (len(sx),)
                or not len(sx)
                or not all(np.isfinite(a).all() for a in (sc, sx, sy))
            ):
                raise ValueError(
                    "invalid, duplicate, unavailable or incompatible source"
                )
            ids.add(src.task_id)
            selected.append(
                (float(np.linalg.norm(sc - c)), src.task_id, sx.copy(), sy.copy())
            )
        selected.sort(key=lambda v: (v[0], v[1]))
        selected = selected[:k]
        self.source_ids = tuple(s[1] for s in selected)
        self.models = [self.target] + [
            ExactGP(self.kernel, noise_variance=noise_variance).fit(s[2], s[3])
            for s in selected
        ]
        self.log_prior = np.array(
            [0.0] + [(-s[0] / tau if contextual else 0.0) for s in selected]
        )
        self.log_evidence = np.zeros(len(self.models))

    def _points(self, x):
        a = np.asarray(x, float)
        if a.ndim != 2 or a.shape[1] != self.dimension or not np.isfinite(a).all():
            raise ValueError("expected finite candidate matrix")
        return a

    @property
    def weights(self):
        score = self.log_prior + (
            self.eta * self.log_evidence if self.use_evidence else 0.0
        )
        q = np.exp(score - logsumexp(score))
        w = (1 - self.floor) * q
        w[0] += self.floor
        return w

    def components(self, x):
        x = self._points(x)
        means, variances = zip(*(m.predict(x) for m in self.models))
        return np.array(means), np.array(variances)

    def predict(self, x):
        means, variances = self.components(x)
        w = self.weights[:, None]
        mean = np.sum(w * means, axis=0)
        variance = np.sum(w * (variances + (means - mean) ** 2), axis=0)
        return mean, variance

    def predict_joint(self, x):
        x = self._points(x)
        predictions = [m.predict_joint(x) for m in self.models]
        means = np.array([p[0] for p in predictions])
        w = self.weights
        mean = w @ means
        cov = sum(
            weight * (p[1] + np.outer(mu - mean, mu - mean))
            for weight, p, mu in zip(w, predictions, means)
        )
        return mean, (cov + cov.T) / 2

    def acquisition(self, x):
        if not len(self.y):
            raise ValueError("EI requires an observed incumbent")
        means, var = self.components(x)
        delta = means - np.max(self.y)
        sd = np.sqrt(var)
        z = np.divide(delta, sd, out=np.zeros_like(delta), where=sd > 0)
        ei = np.where(
            sd > 0,
            delta * ndtr(z) + sd * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
            np.maximum(delta, 0),
        )
        return self.weights @ ei

    def observe(self, x, y):
        x = self._points(np.asarray(x, float).reshape(1, -1))
        y = float(y)
        if not np.isfinite(y):
            raise ValueError("invalid observation")
        means, variances = self.components(x)
        v = variances[:, 0] + self.noise
        ll = -0.5 * (np.log(2 * np.pi * v) + (y - means[:, 0]) ** 2 / v)
        if not np.isfinite(ll).all():
            raise ArithmeticError("nonfinite predictive log likelihood")
        new_x = np.vstack((self.x, x))
        new_y = np.append(self.y, y)
        new_target = ExactGP(self.kernel, noise_variance=self.noise).fit(new_x, new_y)
        self.target = new_target
        self.models[0] = new_target
        self.x, self.y = new_x, new_y
        self.log_evidence += ll
