"""Decision utilities for minimization, with explicit cost and risk conventions."""

from collections.abc import Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from scipy.special import ndtr


def _finite(value: ArrayLike, ndim: int) -> NDArray:
    data = np.asarray(value, dtype=np.float64)
    if data.ndim != ndim or not np.isfinite(data).all():
        raise ValueError(f"expected finite array with rank {ndim}")
    return data


def probability_feasible(means: ArrayLike, variances: ArrayLike) -> NDArray:
    """Joint P(all constraints <= 0) for independent Gaussian constraint models."""
    mean, var = _finite(means, 2), _finite(variances, 2)
    if mean.shape != var.shape or (var < 0).any():
        raise ValueError(
            "constraint moments must match and variances must be nonnegative"
        )
    prob = np.asarray(mean <= 0, dtype=float)
    stochastic = var > 0
    with np.errstate(over="ignore"):
        prob[stochastic] = ndtr(-mean[stochastic] / np.sqrt(var[stochastic]))
    return prob.prod(axis=1)


def constrained_improvement(improvement: ArrayLike, feasibility: ArrayLike) -> NDArray:
    """Expected feasible improvement under posterior independence from feasibility."""
    ei, prob = _finite(improvement, 1), _finite(feasibility, 1)
    if ei.shape != prob.shape or (ei < 0).any() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("invalid improvement or feasibility probability")
    return ei * prob


def pareto_mask(objectives: ArrayLike) -> NDArray:
    """Non-dominated rows for componentwise minimization; preserve equal solutions."""
    y = _finite(objectives, 2)
    if y.shape[1] < 1:
        raise ValueError("an objective is required")
    return np.array(
        [not np.any(np.all(y <= row, axis=1) & np.any(y < row, axis=1)) for row in y],
        dtype=bool,
    )


def hypervolume_2d(objectives: ArrayLike, reference: Sequence[float]) -> float:
    """Area of the union of rectangles [y, reference] for two losses."""
    y, ref = _finite(objectives, 2), _finite(reference, 1)
    if y.shape[1] != 2 or ref.shape != (2,):
        raise ValueError("two objectives required")
    front = y[np.all(y < ref, axis=1)]
    floor = ref[1]
    area = 0.0
    with np.errstate(over="raise", invalid="raise"):
        for x, height in front[np.argsort(front[:, 0])]:
            if height < floor:
                area += (ref[0] - x) * (floor - height)
                floor = height
    return float(area)


def upper_cvar(losses: ArrayLike, alpha: float = 0.9) -> NDArray:
    """Empirical upper-tail expected loss, interpolating the quantile atom.

    Rows are decisions and columns are equally probable perturbations.
    alpha=0 gives the mean. A non-integer tail contains a fractional sample.
    """
    y = _finite(losses, 2)
    if y.shape[1] == 0 or not 0 <= alpha < 1:
        raise ValueError("nonempty samples and 0 <= alpha < 1 required")
    mass = y.shape[1] * (1 - alpha)
    weights = np.clip(mass - np.arange(y.shape[1]), 0, 1) / mass
    with np.errstate(over="raise", invalid="raise"):
        return np.sort(y, axis=1)[:, ::-1] @ weights


def diverse_solutions(
    points: ArrayLike, losses: ArrayLike, radius: float, limit: int
) -> NDArray:
    """Greedy increasing-loss archive under Euclidean distance in supplied units."""
    x, y = _finite(points, 2), _finite(losses, 1)
    if len(x) != len(y) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("points, losses, and positive radius required")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    selected: list[int] = []
    for i in np.argsort(y, kind="stable"):
        if not selected or np.all(cdist(x[[i]], x[selected]) >= radius):
            selected.append(int(i))
            if len(selected) == limit:
                break
    return np.asarray(selected, dtype=int)


def discrete_knowledge_gradient(
    target_mean: ArrayLike,
    cross_covariance: ArrayLike,
    observation_variance: ArrayLike,
    costs: ArrayLike,
    quadrature_order: int = 32,
) -> NDArray:
    """One-step value of information per cost for evaluations at any fidelity.

    Rows of cross_covariance are finite full-fidelity recommendations, columns
    are possible observations. All moments must belong to one joint posterior.
    Observation variances include measurement noise. Conditional target means
    become mu_i + cov(i,j)/sd_j * Z. Integrate the best updated recommendation
    over a standard normal Z using Gauss-Hermite quadrature.
    """
    mu, cov = _finite(target_mean, 1), _finite(cross_covariance, 2)
    var, cost = _finite(observation_variance, 1), _finite(costs, 1)
    if mu.size == 0 or cov.shape != (mu.size, var.size) or cost.shape != var.shape:
        raise ValueError("incompatible target and observation shapes")
    if (var < 0).any() or (cost <= 0).any() or np.any(cov[:, var == 0] != 0):
        raise ValueError(
            "invalid cost, variance, or nonzero covariance with a constant"
        )
    if type(quadrature_order) is not int or not 2 <= quadrature_order <= 128:
        raise ValueError("quadrature order must be an integer between 2 and 128")
    nodes, weights = hermgauss(quadrature_order)
    with np.errstate(over="raise", invalid="raise"):
        slope = np.divide(
            cov, np.sqrt(var), where=var[None, :] > 0, out=np.zeros_like(cov)
        )
        # Subtract current best before integrating to avoid a large common offset.
        updated = (mu - mu.min())[:, None, None] + slope[:, :, None] * (
            np.sqrt(2) * nodes
        )
        gain = -np.min(updated, axis=0) @ (weights / np.sqrt(np.pi))
        return np.maximum(gain, 0.0) / cost


def entropy_search(
    target_mean,
    target_covariance,
    cross_covariance,
    observation_variance,
    *,
    seed=0,
    paths=1024,
    fantasies=16,
):
    """Discrete entropy search by Gaussian conditioning and sampled minimizers.

    Returns estimated mutual information in nats between each possible scalar
    observation and the minimizing index on a fixed finite recommendation set.
    Finite Monte Carlo estimates can be negative. Work scales cubically in
    the number of target locations per prospective observation.
    """
    mu = _finite(target_mean, 1)
    cov = _finite(target_covariance, 2)
    cross = _finite(cross_covariance, 2)
    var = _finite(observation_variance, 1)
    if (
        mu.size == 0
        or cov.shape != (mu.size, mu.size)
        or cross.shape != (mu.size, var.size)
    ):
        raise ValueError("incompatible moment dimensions")
    if (var < 0).any() or np.any(cross[:, var == 0] != 0):
        raise ValueError("inconsistent observation variance")
    if (
        type(paths) is not int
        or paths < 2
        or type(fantasies) is not int
        or not 2 <= fantasies <= 128
    ):
        raise ValueError("invalid integration budget")

    def factor(matrix):
        scale = max(np.max(np.abs(matrix)), np.finfo(float).tiny)
        tol = 32 * np.finfo(float).eps * len(matrix) * scale
        if np.max(np.abs(matrix - matrix.T)) > tol:
            raise ValueError("asymmetric covariance")
        eigenvalues, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
        if eigenvalues.min() < -tol:
            raise ValueError("inconsistent joint covariance")
        # Symmetric principal root provides common-random-number coupling.
        return (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T

    normal = np.random.default_rng(seed).normal(size=(paths, len(mu)))

    def entropy(draws):
        probability = np.bincount(draws.argmin(axis=1), minlength=len(mu)) / paths
        nonzero = probability[probability > 0]
        return -float(nonzero @ np.log(nonzero))

    initial = entropy(mu + normal @ factor(cov))
    nodes, weights = hermgauss(fantasies)
    scores = np.zeros(len(var))
    for j in np.flatnonzero(var > 0):
        slope = cross[:, j] / np.sqrt(var[j])
        conditional = mu + normal @ factor(cov - np.outer(slope, slope))
        entropies = [entropy(conditional + np.sqrt(2) * node * slope) for node in nodes]
        scores[j] = initial - weights @ entropies / np.sqrt(np.pi)
    return scores
