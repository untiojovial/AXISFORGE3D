"""Nonlinear task-adaptive deep kernel research prototype, CPU/float64.

Support-fitted GP parameters and an exact local implicit derivative of the
joint query negative log density. This is not the molecular ADKF experiment.
"""

import math

import numpy as np
import torch
from scipy.optimize import minimize

from .implicit import implicit_hypergradient


def mlp_parameter_count(input_dim, hidden_dim, feature_dim):
    if any(
        isinstance(d, bool) or not isinstance(d, int) or d <= 0
        for d in (input_dim, hidden_dim, feature_dim)
    ):
        raise ValueError("dimensions must be positive integers")
    return input_dim * hidden_dim + hidden_dim + hidden_dim * feature_dim + feature_dim


def nonlinear_task_gradient(
    theta,
    support_x,
    support_y,
    query_x,
    query_y,
    *,
    hidden_dim,
    feature_dim,
    prior_precision=1.0,
    stationarity_tol=1e-6,
    max_condition=1e8,
):
    """Adapt GP on support; return query NLL, MLP hypergradient and fitted GP logs.

    Flat MLP parameter order: W1 (input, hidden), b1, W2 (hidden, feature), b2.
    GP log parameters are lengthscale, signal std and noise variance. The
    optional quadratic penalty is on raw GP logs (zero disables the penalty).
    The caller must ensure disjoint, provenance-safe support/query tasks.
    """
    theta, sx, sy, qx, qy = [
        np.asarray(a, dtype=float)
        for a in (theta, support_x, support_y, query_x, query_y)
    ]
    if (
        sx.ndim != 2
        or not sx.shape[1]
        or qx.ndim != 2
        or qx.shape[1] != sx.shape[1]
        or not len(sx)
        or not len(qx)
        or sy.shape != (len(sx),)
        or qy.shape != (len(qx),)
    ):
        raise ValueError("invalid task shapes")
    d = sx.shape[1]
    n = mlp_parameter_count(d, hidden_dim, feature_dim)
    if theta.shape != (n,) or not all(
        np.isfinite(a).all() for a in (theta, sx, sy, qx, qy)
    ):
        raise ValueError("invalid weights or data")
    if (
        not math.isfinite(prior_precision)
        or prior_precision < 0
        or not math.isfinite(stationarity_tol)
        or stationarity_tol <= 0
        or not math.isfinite(max_condition)
        or max_condition <= 1
    ):
        raise ValueError("invalid prior or tolerances")
    T = torch.tensor(theta, dtype=torch.double)
    S, Y, Q, Z = [torch.tensor(a, dtype=torch.double) for a in (sx, sy, qx, qy)]

    def features(t, x):
        i = d * hidden_dim
        w1 = t[:i].reshape(d, hidden_dim)
        b1 = t[i : i + hidden_dim]
        i += hidden_dim
        w2 = t[i : i + hidden_dim * feature_dim].reshape(hidden_dim, feature_dim)
        i += hidden_dim * feature_dim
        return torch.tanh(x @ w1 + b1) @ w2 + t[i:]

    def kernel(t, u, x, y):
        a, b = features(t, x), features(t, y)
        r2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return torch.exp(2 * u[1] - 0.5 * r2 * torch.exp(-2 * u[0]))

    def support(t, u):
        return kernel(t, u, S, S) + (torch.exp(u[2]) + 1e-8) * torch.eye(
            len(S), dtype=torch.double
        )

    def inner(t, u):
        L = torch.linalg.cholesky(support(t, u))
        return (
            0.5 * Y.dot(torch.cholesky_solve(Y[:, None], L)[:, 0])
            + L.diagonal().log().sum()
            + 0.5 * len(S) * math.log(2 * math.pi)
            + 0.5 * prior_precision * u.dot(u)
        )

    def outer(t, u):
        L = torch.linalg.cholesky(support(t, u))
        cross = kernel(t, u, S, Q)
        mu = cross.T @ torch.cholesky_solve(Y[:, None], L)[:, 0]
        v = torch.linalg.solve_triangular(L, cross, upper=False)
        cov = (
            kernel(t, u, Q, Q)
            + (torch.exp(u[2]) + 1e-8) * torch.eye(len(Q), dtype=torch.double)
            - v.T @ v
        )
        Lq = torch.linalg.cholesky(cov)
        residual = Z - mu
        return (
            0.5 * residual.dot(torch.cholesky_solve(residual[:, None], Lq)[:, 0])
            + Lq.diagonal().log().sum()
            + 0.5 * len(Q) * math.log(2 * math.pi)
        )

    def objective(u):
        z = torch.tensor(u, dtype=torch.double, requires_grad=True)
        value = inner(T, z)
        grad = torch.autograd.grad(value, z)[0]
        return value.item(), grad.detach().numpy()

    result = minimize(
        objective,
        np.zeros(3),
        jac=True,
        method="BFGS",
        options={"gtol": stationarity_tol / 10, "maxiter": 500},
    )
    if (
        not np.isfinite(result.x).all()
        or np.linalg.norm(objective(result.x)[1]) > stationarity_tol
    ):
        raise ArithmeticError("inner GP fit did not reach stationarity")
    U = torch.tensor(result.x, dtype=torch.double)
    H = torch.autograd.functional.hessian(lambda u: inner(T, u), U)
    eig = torch.linalg.eigvalsh(H)
    if (
        not torch.isfinite(eig).all()
        or eig[0].item() <= 0
        or eig[-1].item() / eig[0].item() > max_condition
    ):
        raise ArithmeticError("inner curvature is non-positive or ill-conditioned")
    grad = implicit_hypergradient(inner, outer, T, U, stationarity_tol=stationarity_tol)
    return float(outer(T, U).item()), grad.numpy(), result.x.copy()


def nonlinear_meta_step(
    theta, tasks, *, hidden_dim, feature_dim, learning_rate=0.01, **kwargs
):
    """Average query hypergradients over training tasks and return new MLP weights.

    Construct disjoint support/query episodes and reserve separate meta-test tasks
    outside this function. No updates are committed if any task fit fails.
    """
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("at least one task is required")
    results = [
        nonlinear_task_gradient(
            theta, *task, hidden_dim=hidden_dim, feature_dim=feature_dim, **kwargs
        )
        for task in tasks
    ]
    updated = np.asarray(theta, dtype=float) - learning_rate * np.mean(
        [r[1] for r in results], axis=0
    )
    if not np.isfinite(updated).all():
        raise ArithmeticError("nonfinite update")
    return updated, float(np.mean([r[0] for r in results]))
