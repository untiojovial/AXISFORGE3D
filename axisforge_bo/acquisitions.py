"""Analytic acquisitions for a fixed incumbent and latent Gaussian posterior."""

import numpy as np
from scipy.special import ndtr


def _inputs(mean, variance):
    m, v = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)
    )
    if not np.isfinite(m).all() or not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError("invalid posterior")
    return m, np.sqrt(v)


def expected_improvement(mean, variance, best, xi=0.0):
    m, sigma = _inputs(mean, variance)
    if not np.isfinite(best) or not np.isfinite(xi) or xi < 0:
        raise ValueError("invalid incumbent or offset")
    with np.errstate(over="ignore", invalid="ignore"):
        d = best - m - xi
    if not np.isfinite(d).all():
        raise ArithmeticError("nonfinite improvement")
    z = np.divide(d, sigma, out=np.zeros_like(d), where=sigma > 0)
    ei = d * ndtr(z) + sigma * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    result = np.maximum(np.where(sigma > 0, ei, np.maximum(d, 0)), 0)
    if not np.isfinite(result).all():
        raise ArithmeticError("nonfinite expected improvement")
    return result


def probability_improvement(mean, variance, best, xi=0.0):
    m, sigma = _inputs(mean, variance)
    if not np.isfinite(best) or not np.isfinite(xi) or xi < 0:
        raise ValueError("invalid incumbent or offset")
    with np.errstate(over="ignore", invalid="ignore"):
        d = best - m - xi
    if not np.isfinite(d).all():
        raise ArithmeticError("nonfinite improvement")
    z = np.divide(d, sigma, out=np.zeros_like(d), where=sigma > 0)
    return np.where(sigma > 0, ndtr(z), (d > 0).astype(float))


def lower_confidence_bound(mean, variance, beta=2.0):
    m, sigma = _inputs(mean, variance)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        result = m - np.sqrt(beta) * sigma
    if not np.isfinite(result).all():
        raise ArithmeticError("nonfinite confidence bound")
    return result


def log_expected_improvement(mean, variance, best, xi=0.0):
    """Log EI for positive variance, using stable negative-tail expressions."""
    from scipy.special import log_ndtr

    m, sigma = _inputs(mean, variance)
    if not np.isfinite(best) or not np.isfinite(xi) or xi < 0 or np.any(sigma <= 0):
        raise ValueError("LogEI requires positive variance and finite incumbent")
    with np.errstate(over="ignore", invalid="ignore"):
        d = best - m - xi
        z = d / sigma
    if not np.isfinite(z).all():
        raise ArithmeticError("nonfinite scaled improvement")
    out = np.empty_like(z)
    positive = z >= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[positive] = np.logaddexp(
            -(z[positive] ** 2) / 2 - 0.5 * np.log(2 * np.pi),
            np.log(z[positive]) + log_ndtr(z[positive]),
        )
        mid = (z < 0) & (z >= -1e4)
        zz = z[mid]
        logphi = -(zz**2) / 2 - 0.5 * np.log(2 * np.pi)
        ratio = np.log(-zz) + log_ndtr(zz) - logphi
        out[mid] = logphi + np.log(-np.expm1(ratio))
        tail = z < -1e4
        zz = z[tail]
        out[tail] = -(zz**2) / 2 - 0.5 * np.log(2 * np.pi) - 2 * np.log(-zz)
        result = out + np.log(sigma)
    if not np.isfinite(result).all():
        raise ArithmeticError("LogEI outside numeric range")
    return result


def _lcb_score(mean, variance, best):
    """Adapt LCB to the shared three-argument acquisition protocol."""
    return lower_confidence_bound(mean, variance)


# Public registry maps names to (callable, direction). Callables take posterior
# mean, latent variance, and a canonical-minimization incumbent.
ACQUISITIONS = {
    "ei": (expected_improvement, "max"),
    "pi": (probability_improvement, "max"),
    "lcb": (_lcb_score, "min"),
    "log_ei": (log_expected_improvement, "max"),
}


def register_acquisition(name, function, direction="max"):
    if (
        not isinstance(name, str)
        or not name
        or name in ACQUISITIONS
        or not callable(function)
        or direction not in {"min", "max"}
    ):
        raise ValueError("invalid or existing acquisition")
    ACQUISITIONS[name] = (function, direction)


def thompson_sample(mean, covariance, rng):
    """Draw one correlated latent function sample over a finite candidate set.

    The caller minimizes the draw. Reject covariance that is materially indefinite.
    """
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if (
        mean.ndim != 1
        or covariance.shape != (len(mean), len(mean))
        or not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("invalid joint posterior")
    scale = max(1.0, float(np.max(np.abs(covariance), initial=0)))
    tol = 1e-10 * scale
    if not np.allclose(covariance, covariance.T, atol=tol, rtol=0):
        raise ValueError("covariance must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
    if np.any(eigenvalues < -tol):
        raise ValueError("covariance is not positive semidefinite")
    with np.errstate(over="ignore", invalid="ignore"):
        draw = mean + eigenvectors @ (
            np.sqrt(np.maximum(eigenvalues, 0)) * rng.standard_normal(len(mean))
        )
    if not np.isfinite(draw).all():
        raise ArithmeticError("nonfinite Thompson sample")
    return draw
