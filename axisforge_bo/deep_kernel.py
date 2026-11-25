"""Optional PyTorch DKL regression, trained by exact GP marginal likelihood.

This is single-task DKL, not ADKF-IFT. Parameters are point estimates and
predictive variances are conditional on them. Requires PyTorch. CPU float64 only.
"""

import math

import torch
from torch import nn


class DeepKernelGP(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, feature_dim=2, *, seed=0, jitter=1e-8):
        super().__init__()
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0
            for n in (input_dim, hidden_dim, feature_dim)
        ):
            raise ValueError("dimensions must be positive integers")
        if not math.isfinite(jitter) or jitter <= 0:
            raise ValueError("jitter must be positive and finite")
        # fork_rng preserves the caller's global random state.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.features = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, feature_dim),
            ).double()
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0, dtype=torch.double))
        self.log_amplitude = nn.Parameter(torch.tensor(0.0, dtype=torch.double))
        self.log_noise = nn.Parameter(torch.tensor(-3.0, dtype=torch.double))
        self.input_dim, self.jitter = input_dim, float(jitter)
        self._X = self._y = None

    def _tensor(self, x):
        if (
            next(self.parameters()).device.type != "cpu"
            or next(self.parameters()).dtype != torch.double
        ):
            raise ValueError("DeepKernelGP supports CPU float64 only")
        if any(not torch.isfinite(p).all() for p in self.parameters()):
            raise ArithmeticError("nonfinite model parameters")
        x = torch.as_tensor(x, dtype=torch.double, device="cpu")
        if x.ndim != 2 or x.shape[1] != self.input_dim or not torch.isfinite(x).all():
            raise ValueError("invalid input matrix")
        return x

    def kernel(self, x, y):
        a, b = self.features(x), self.features(y)
        r2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(dim=-1)
        return self.log_amplitude.mul(2).exp() * torch.exp(
            -0.5 * r2 / self.log_lengthscale.mul(2).exp()
        )

    def loss(self, X, y):
        X = self._tensor(X)
        y = torch.as_tensor(y, dtype=torch.double)
        if not len(X) or y.shape != (len(X),) or not torch.isfinite(y).all():
            raise ValueError("invalid training targets")
        K = self.kernel(X, X) + (self.log_noise.exp() + self.jitter) * torch.eye(
            len(X), dtype=X.dtype, device=X.device
        )
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y[:, None], L)[:, 0]
        return (
            0.5 * y.dot(alpha)
            + torch.log(torch.diagonal(L)).sum()
            + 0.5 * len(y) * math.log(2 * math.pi)
        )

    def fit(self, X, y, *, steps=100, learning_rate=0.01):
        X = self._tensor(X).detach().clone()
        y = torch.as_tensor(y, dtype=torch.double).detach().clone()
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 0
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("invalid training configuration")
        # An unsuccessful fit invalidates the previously fitted posterior.
        self._X = self._y = None
        # Validate even for steps=0.
        loss = self.loss(X, y)
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for _ in range(steps):
            opt.zero_grad()
            loss = self.loss(X, y)
            if not torch.isfinite(loss):
                raise ArithmeticError("nonfinite marginal likelihood")
            loss.backward()
            if any(
                p.grad is None or not torch.isfinite(p.grad).all()
                for p in self.parameters()
                if p.requires_grad
            ):
                raise ArithmeticError("nonfinite or missing training gradient")
            opt.step()
            if any(not torch.isfinite(p).all() for p in self.parameters()):
                raise ArithmeticError("nonfinite trained parameters")
        if not torch.isfinite(self.loss(X, y)):
            raise ArithmeticError("nonfinite fitted model")
        self._X, self._y = X, y
        return self

    def predict(self, X):
        X = self._tensor(X)
        with torch.no_grad():
            Kss = self.kernel(X, X)
            var = torch.diagonal(Kss).clone()
            mu = torch.zeros(len(X), dtype=torch.double)
            if self._X is not None:
                K = self.kernel(self._X, self._X) + (
                    self.log_noise.exp() + self.jitter
                ) * torch.eye(len(self._X), dtype=torch.double)
                L = torch.linalg.cholesky(K)
                cross = self.kernel(self._X, X)
                mu = cross.T @ torch.cholesky_solve(self._y[:, None], L)[:, 0]
                v = torch.linalg.solve_triangular(L, cross, upper=False)
                var -= (v * v).sum(dim=0)
            tol = 1e-10 * max(
                1.0, float(torch.diagonal(Kss).abs().max()) if len(X) else 0.0
            )
            if (var < -tol).any():
                raise ArithmeticError("materially negative variance")
            if not torch.isfinite(mu).all() or not torch.isfinite(var).all():
                raise ArithmeticError("nonfinite posterior")
            return mu.numpy(), var.clamp_min(0).numpy()
