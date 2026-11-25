"""Implicit derivative for a small stationary unconstrained inner problem.

The caller supplies differentiable scalar losses and an already optimized inner
parameter. This helper does not optimize tasks or implement ADKF-IFT.
"""

import math

import torch


def implicit_hypergradient(
    inner_loss, outer_loss, outer, inner, *, stationarity_tol=1e-7, residual_tol=1e-8
):
    if not all(math.isfinite(t) and t > 0 for t in (stationarity_tol, residual_tol)):
        raise ValueError("tolerances must be positive and finite")
    if (
        outer.ndim != 1
        or inner.ndim != 1
        or not outer.numel()
        or not inner.numel()
        or outer.dtype != torch.double
        or inner.dtype != torch.double
    ):
        raise ValueError("parameters must be nonempty float64 vectors")
    if not torch.isfinite(outer).all() or not torch.isfinite(inner).all():
        raise ValueError("parameters must be finite")
    theta = outer.detach().clone().requires_grad_(True)
    u = inner.detach().clone().requires_grad_(True)
    loss = inner_loss(theta, u)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("inner loss must be finite scalar")
    grad_inner = torch.autograd.grad(loss, u, create_graph=True)[0]
    if (
        not torch.isfinite(grad_inner).all()
        or torch.linalg.vector_norm(grad_inner).item() > stationarity_tol
    ):
        raise ValueError("inner point is not stationary within tolerance")
    H = torch.autograd.functional.hessian(lambda v: inner_loss(theta, v), u)
    J = torch.autograd.functional.jacobian(
        lambda t: torch.autograd.grad(inner_loss(t, u), u, create_graph=True)[0], theta
    )
    output = outer_loss(theta, u)
    if output.ndim != 0 or not torch.isfinite(output):
        raise ValueError("outer loss must be finite scalar")
    dtheta, du = torch.autograd.grad(output, (theta, u))
    if not all(torch.isfinite(v).all() for v in (H, J, dtheta, du)):
        raise ArithmeticError("nonfinite derivatives")
    try:
        v = torch.linalg.solve(H.T, du)
    except RuntimeError as exc:
        raise ArithmeticError("singular inner Hessian") from exc
    residual = torch.linalg.vector_norm(H.T @ v - du)
    if not torch.isfinite(v).all() or residual > residual_tol * (
        1 + torch.linalg.vector_norm(du)
    ):
        raise ArithmeticError("inner Hessian solve failed residual check")
    gradient = dtheta - v @ J
    if not torch.isfinite(gradient).all():
        raise ArithmeticError("nonfinite hypergradient")
    return gradient.detach()
