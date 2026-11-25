"""Exact fixed-hyperparameter Gaussian process regression in float64."""

import numpy as np
from scipy.linalg import cho_solve, solve_triangular


class StationaryKernel:
    def __init__(self, length_scale=1.0, amplitude=1.0, kind="rbf"):
        self.length_scale = np.asarray(length_scale, dtype=float)
        self.amplitude = float(amplitude)
        self.kind = kind
        if (
            self.length_scale.ndim > 1
            or self.length_scale.size == 0
            or not np.isfinite(self.length_scale).all()
            or np.any(self.length_scale <= 0)
            or not np.isfinite(self.amplitude)
            or self.amplitude <= 0
            or kind
            not in {"rbf", "matern12", "matern32", "matern52", "rational_quadratic"}
        ):
            raise ValueError("invalid kernel parameters")

    def diagonal(self, x):
        x = np.asarray(x, dtype=float)
        if (
            x.ndim != 2
            or not np.isfinite(x).all()
            or (self.length_scale.ndim == 1 and self.length_scale.size != x.shape[1])
        ):
            raise ValueError("invalid kernel input")
        value = self.amplitude * self.amplitude
        if not np.isfinite(value):
            raise ArithmeticError("nonfinite kernel diagonal")
        return np.full(len(x), value)

    def to_config(self):
        return {
            "length_scale": self.length_scale.tolist(),
            "amplitude": self.amplitude,
            "kind": self.kind,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __call__(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (
            x.ndim != 2
            or y.ndim != 2
            or x.shape[1] != y.shape[1]
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or (self.length_scale.ndim == 1 and self.length_scale.size != x.shape[1])
        ):
            raise ValueError("kernel inputs have incompatible shapes or values")
        r2 = np.sum(((x[:, None, :] - y[None, :, :]) / self.length_scale) ** 2, axis=-1)
        r = np.sqrt(r2)
        if self.kind == "rbf":
            k = np.exp(-r2 / 2)
        elif self.kind == "matern12":
            k = np.exp(-r)
        elif self.kind == "matern32":
            z = np.sqrt(3) * r
            k = (1 + z) * np.exp(-z)
        elif self.kind == "matern52":
            z = np.sqrt(5) * r
            k = (1 + z + z * z / 3) * np.exp(-z)
        else:
            k = 1 / (1 + r2 / 2)  # alpha=1
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.amplitude**2 * k
        if not np.isfinite(result).all():
            raise ArithmeticError("nonfinite kernel result")
        return result


class ExactGP:
    def __init__(self, kernel, noise_variance=1e-6, jitter=1e-10):
        self.kernel = kernel
        self.noise_variance = np.asarray(noise_variance, dtype=float)
        self.jitter = float(jitter)
        if (
            self.noise_variance.ndim > 1
            or not np.isfinite(self.noise_variance).all()
            or np.any(self.noise_variance < 0)
            or not np.isfinite(self.jitter)
            or self.jitter < 0
        ):
            raise ValueError("noise variance and jitter must be finite and nonnegative")
        self.X = self.y = self.L = self.alpha = None

    def to_config(self):
        return {
            "kernel": self.kernel.to_config(),
            "noise_variance": self.noise_variance.tolist(),
            "jitter": self.jitter,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            StationaryKernel.from_config(config["kernel"]),
            config["noise_variance"],
            config["jitter"],
        )

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if (
            X.ndim != 2
            or X.shape[1] == 0
            or y.shape != (X.shape[0],)
            or not np.isfinite(X).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("invalid training data")
        if self.noise_variance.ndim and self.noise_variance.shape != y.shape:
            raise ValueError("noise variance must be scalar or per observation")
        K = self.kernel(X, X)
        if not len(y):
            self.X, self.y, self.L, self.alpha = (
                X.copy(),
                y.copy(),
                np.empty((0, 0)),
                np.empty(0),
            )
            return self
        K = K + np.diag(np.broadcast_to(self.noise_variance, y.shape) + self.jitter)
        if not np.isfinite(K).all():
            raise ArithmeticError("nonfinite covariance matrix")
        L = np.linalg.cholesky(K)
        alpha = cho_solve((L, True), y)
        if not np.isfinite(L).all() or not np.isfinite(alpha).all():
            raise ArithmeticError("nonfinite GP fit")
        self.X, self.y, self.L, self.alpha = X.copy(), y.copy(), L, alpha
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("invalid prediction data")
        variance = (
            self.kernel.diagonal(X)
            if hasattr(self.kernel, "diagonal")
            else np.diag(self.kernel(X, X)).copy()
        )
        prior_scale = np.max(np.abs(variance), initial=0)
        mean = np.zeros(len(X))
        if self.X is not None:
            if X.shape[1] != self.X.shape[1]:
                raise ValueError("wrong dimension")
            if len(self.y):
                K = self.kernel(self.X, X)
                mean = K.T @ self.alpha
                v = solve_triangular(self.L, K, lower=True)
                variance -= np.sum(v * v, axis=0)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ArithmeticError("nonfinite posterior")
        tolerance = 1e-10 * max(1.0, float(prior_scale))
        if np.any(variance < -tolerance):
            raise ArithmeticError("materially negative posterior variance")
        return mean, np.maximum(variance, 0.0)

    def predict_joint(self, X):
        X = np.asarray(X, dtype=float)
        mean, _ = self.predict(X)
        cov = self.kernel(X, X)
        if self.X is not None and len(self.y):
            v = solve_triangular(self.L, self.kernel(self.X, X), lower=True)
            cov = cov - v.T @ v
        cov = (cov + cov.T) / 2
        if not np.isfinite(cov).all():
            raise ArithmeticError("nonfinite covariance")
        return mean, cov

    def log_marginal_likelihood(self):
        if self.X is None:
            raise ValueError("fit first")
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(
                -0.5 * self.y @ self.alpha
                - np.log(np.diag(self.L)).sum()
                - 0.5 * len(self.y) * np.log(2 * np.pi)
            )
        if not np.isfinite(result):
            raise ArithmeticError("nonfinite log marginal likelihood")
        return result


class LinearKernel:
    def __init__(self, variance=1.0, offset=0.0):
        self.variance, self.offset = float(variance), float(offset)
        if not np.isfinite([self.variance, self.offset]).all() or self.variance <= 0:
            raise ValueError("invalid linear parameters")

    def diagonal(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("invalid inputs")
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.variance * np.sum((x - self.offset) ** 2, axis=1)
        if not np.isfinite(result).all():
            raise ArithmeticError("nonfinite kernel diagonal")
        return result

    def __call__(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if (
            x.ndim != 2
            or y.ndim != 2
            or x.shape[1] != y.shape[1]
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("invalid inputs")
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.variance * ((x - self.offset) @ (y - self.offset).T)
        if not np.isfinite(result).all():
            raise ArithmeticError("nonfinite kernel")
        return result


class PeriodicKernel:
    def __init__(self, length_scale=1.0, period=1.0, amplitude=1.0):
        self.length_scale, self.period, self.amplitude = (
            float(length_scale),
            float(period),
            float(amplitude),
        )
        if (
            not np.isfinite([self.length_scale, self.period, self.amplitude]).all()
            or min(self.length_scale, self.period, self.amplitude) <= 0
        ):
            raise ValueError("invalid periodic parameters")

    def diagonal(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("invalid inputs")
        value = self.amplitude * self.amplitude
        if not np.isfinite(value):
            raise ArithmeticError("nonfinite kernel diagonal")
        return np.full(len(x), value)

    def __call__(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if (
            x.ndim != 2
            or y.ndim != 2
            or x.shape[1] != y.shape[1]
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("invalid inputs")
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.amplitude**2 * np.exp(
                -2
                * np.sum(
                    np.sin(np.pi * (x[:, None, :] - y[None, :, :]) / self.period) ** 2,
                    axis=-1,
                )
                / self.length_scale**2
            )
        if not np.isfinite(result).all():
            raise ArithmeticError("nonfinite kernel")
        return result


class CompositeKernel:
    def __init__(self, left, right, operation):
        if operation not in ("sum", "product"):
            raise ValueError("invalid operation")
        self.left, self.right, self.operation = left, right, operation

    def diagonal(self, x):
        a = (
            self.left.diagonal(x)
            if hasattr(self.left, "diagonal")
            else np.diag(self.left(x, x))
        )
        b = (
            self.right.diagonal(x)
            if hasattr(self.right, "diagonal")
            else np.diag(self.right(x, x))
        )
        with np.errstate(over="ignore", invalid="ignore"):
            result = a + b if self.operation == "sum" else a * b
        if not np.isfinite(result).all():
            raise ArithmeticError("nonfinite kernel diagonal")
        return result

    def __call__(self, x, y):
        with np.errstate(over="ignore", invalid="ignore"):
            a, b = self.left(x, y), self.right(x, y)
            result = a + b if self.operation == "sum" else a * b
        if not np.isfinite(result).all():
            raise ArithmeticError("nonfinite kernel")
        return result
