"""Small task-adaptive deep kernel GP with implicit outer derivative.

CPU float64; linear feature map, regularized GP hyperparameter adaptation.
Research prototype: this is an ADKF-style instantiation, not a reproduction.
"""

import math

import numpy as np
import torch
from scipy.optimize import minimize

from .implicit import implicit_hypergradient


def task_meta_gradient(
    theta, support_x, support_y, query_x, query_y, *, regularization=1.0, tolerance=1e-5
):
    """Fit per-task log GP parameters on support; differentiate joint query NLL.

    theta is an input_dim by feature_dim matrix. A quadratic regularizer on
    GP log parameters makes the inner objective different from unregularized
    ADKF-IFT; it is explicit and its stationary optimum is checked. The GP
    kernel has an RBF with amplitude and noise variance both learned.
    """
    arrays = [
        np.asarray(a, dtype=float)
        for a in (theta, support_x, support_y, query_x, query_y)
    ]
    theta, sx, sy, qx, qy = arrays
    if (
        theta.ndim != 2
        or 0 in theta.shape
        or sx.ndim != 2
        or qx.ndim != 2
        or sx.shape[1] != theta.shape[0]
        or qx.shape[1] != theta.shape[0]
        or sy.shape != (len(sx),)
        or qy.shape != (len(qx),)
        or len(sx) == 0
        or len(qx) == 0
        or not all(np.isfinite(a).all() for a in arrays)
    ):
        raise ValueError("invalid task shapes or values")
    if (
        not math.isfinite(regularization)
        or regularization <= 0
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("regularization and tolerance must be positive")
    T = torch.tensor(theta.reshape(-1), dtype=torch.double)
    S, Y, Q, Z = [torch.tensor(a, dtype=torch.double) for a in (sx, sy, qx, qy)]
    shape = theta.shape

    def kernel(t, u, a, b):
        fa, fb = a @ t.reshape(shape), b @ t.reshape(shape)
        dist = ((fa[:, None, :] - fb[None, :, :]) ** 2).sum(-1)
        return torch.exp(2 * u[1]) * torch.exp(-0.5 * dist * torch.exp(-2 * u[0]))

    def support(t, u):
        return kernel(t, u, S, S) + (torch.exp(u[2]) + 1e-8) * torch.eye(
            len(S), dtype=torch.double
        )

    def inner(t, u):
        L = torch.linalg.cholesky(support(t, u))
        alpha = torch.cholesky_solve(Y[:, None], L)[:, 0]
        return (
            0.5 * Y.dot(alpha)
            + torch.log(L.diagonal()).sum()
            + 0.5 * len(S) * math.log(2 * math.pi)
            + 0.5 * regularization * u.dot(u)
        )

    def outer(t, u):
        L = torch.linalg.cholesky(support(t, u))
        cross = kernel(t, u, S, Q)
        mu = cross.T @ torch.cholesky_solve(Y[:, None], L)[:, 0]
        V = torch.linalg.solve_triangular(L, cross, upper=False)
        cov = (
            kernel(t, u, Q, Q)
            + (torch.exp(u[2]) + 1e-8) * torch.eye(len(Q), dtype=torch.double)
            - V.T @ V
        )
        Lq = torch.linalg.cholesky(cov)
        residual = Z - mu
        return (
            0.5 * residual.dot(torch.cholesky_solve(residual[:, None], Lq)[:, 0])
            + torch.log(Lq.diagonal()).sum()
            + 0.5 * len(Q) * math.log(2 * math.pi)
        )

    def objective(u):
        v = torch.tensor(u, dtype=torch.double, requires_grad=True)
        loss = inner(T, v)
        grad = torch.autograd.grad(loss, v)[0]
        return loss.item(), grad.detach().numpy()

    result = minimize(
        objective,
        np.zeros(3),
        jac=True,
        method="BFGS",
        options={"gtol": tolerance / 10, "maxiter": 500},
    )
    u = torch.tensor(result.x, dtype=torch.double)
    if (
        not np.isfinite(result.x).all()
        or np.linalg.norm(objective(result.x)[1]) > tolerance
    ):
        raise ArithmeticError(
            f"inner optimizer did not reach stationarity: {result.message}"
        )
    grad = implicit_hypergradient(inner, outer, T, u, stationarity_tol=tolerance)
    return float(outer(T, u).item()), grad.numpy().reshape(shape), result.x.copy()


def meta_step(theta, tasks, *, learning_rate=0.01, regularization=1.0, tolerance=1e-5):
    """Return one averaged held-out-query gradient update across training tasks.

    Each task is (support_x, support_y, query_x, query_y); task construction,
    disjointness and time-safe provenance remain the caller's responsibility.
    No in-place update occurs if any task fails.
    """
    theta = np.asarray(theta, dtype=float)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("at least one task required")
    results = [
        task_meta_gradient(
            theta, *task, regularization=regularization, tolerance=tolerance
        )
        for task in tasks
    ]
    gradient = np.mean([r[1] for r in results], axis=0)
    updated = theta - learning_rate * gradient
    if not np.isfinite(updated).all():
        raise ArithmeticError("nonfinite update")
    return updated, float(np.mean([r[0] for r in results]))
