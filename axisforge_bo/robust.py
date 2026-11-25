"""Finite-domain uncertainty integration and confidence-based design restrictions."""

import numpy as np
from scipy.spatial.distance import cdist

from .acquisitions import expected_improvement
from .decisions import _finite


def unscented_improvement(
    model, center, perturbation_covariance, incumbent, *, kappa=1.0
):
    """EI of the input-averaged latent response under positive-weight sigma points.

    Center and covariance are in model feature units. The caller must ensure
    every sigma point lies in the evaluator domain. Incumbent is the best known
    input-averaged loss. This approximates the objective integral, not the loss
    at one randomly perturbed input. Averages can be correlated, hence w'Kw.
    """
    center, cov = _finite(center, 1), _finite(perturbation_covariance, 2)
    d = len(center)
    if d == 0 or cov.shape != (d, d) or not np.isfinite(kappa) or kappa < 0:
        raise ValueError("invalid covariance shape or sigma-point spread")
    tol = (
        np.finfo(float).eps
        * max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
        * d
        * 32
    )
    if np.max(np.abs(cov - cov.T)) > tol:
        raise ValueError("perturbation covariance must be symmetric")
    eig, vec = np.linalg.eigh((cov + cov.T) * 0.5)
    if eig.min() < -tol:
        raise ValueError("perturbation covariance must be positive semidefinite")
    offsets = (vec * np.sqrt((d + kappa) * np.maximum(eig, 0))).T
    points = np.vstack((center, center + offsets, center - offsets))
    weights = np.full(2 * d + 1, 0.5 / (d + kappa))
    weights[0] = kappa / (d + kappa)
    mu, K = model.predict_joint(points)
    mean = float(weights @ mu)
    var = float(weights @ K @ weights)
    numerical_tol = (
        64
        * np.finfo(float).eps
        * max(float(np.max(np.abs(K))), np.finfo(float).tiny)
        * len(weights)
    )
    if var < -numerical_tol:
        raise ArithmeticError("invalid covariance of input-averaged prediction")
    variance = max(var, 0.0)
    return mean, variance, float(expected_improvement(mean, variance, incumbent))


def certified_candidates(points, known_points, upper_bounds, lipschitz):
    """Feasible by Lipschitz extension from supplied simultaneous upper bounds.

    For each constraint g <= 0, min_i(U_i + L ||x-x_i||) bounds g(x)
    only if the supplied global Lipschitz constants and upper bounds hold.
    Unknown constants or pointwise credible bands cannot certify real safety.
    """
    x, known, upper = (
        _finite(points, 2),
        _finite(known_points, 2),
        _finite(upper_bounds, 2),
    )
    L = _finite(lipschitz, 1)
    if (
        not len(known)
        or not len(L)
        or x.shape[1] != known.shape[1]
        or upper.shape != (len(known), len(L))
        or (L < 0).any()
    ):
        raise ValueError(
            "known bounds and nonnegative constants required for every constraint"
        )
    distance = cdist(x, known)
    with np.errstate(over="raise", invalid="raise"):
        bounds = (upper[None, :, :] + distance[:, :, None] * L).min(axis=1)
    return (bounds <= 0).all(axis=1)
