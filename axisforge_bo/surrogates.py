"""Optional trained GPyTorch surrogates with scalar NumPy fit/predict interfaces."""

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.approximate_gp import SingleTaskVariationalGP
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO

from .botorch_backend import _matrix


class ResearchGP:
    """Sparse variational RBF GP or intrinsic coregionalization exact GP.

    The task column is an integer identifier for observed tasks and is retained
    at prediction time. The variational model uses a deterministic subset of
    training inputs for its initial inducing points. Output scaling is fitted
    on training observations only. fit replaces the previous model on success.
    """

    def __init__(
        self,
        kind="variational",
        *,
        inducing=16,
        steps=60,
        task_column=-1,
        device="cpu",
        seed=0,
    ):
        if (
            kind not in {"variational", "multitask"}
            or type(steps) is not int
            or steps < 1
        ):
            raise ValueError("invalid model or optimization budget")
        if type(inducing) is not int or inducing < 1 or type(task_column) is not int:
            raise ValueError("invalid inducing budget or task column")
        self.kind, self.inducing, self.steps = kind, inducing, steps
        self.task_column, self.device, self.seed = (
            task_column,
            torch.device(device),
            seed,
        )
        self.model = None

    def fit(self, X, y):
        x = _matrix(X, self.device)
        target = torch.as_tensor(y, dtype=x.dtype, device=self.device)
        if target.shape != (len(x),) or not torch.isfinite(target).all() or len(x) < 2:
            raise ValueError("need at least two finite target observations")
        center = target.mean()
        scale = target.std()
        if scale == 0:
            scale = torch.ones_like(scale)
        standardized = (target - center) / scale
        devices = [] if x.device.type == "cpu" else [x.device.index]
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.seed)
            if devices:
                with torch.cuda.device(x.device):
                    torch.cuda.manual_seed(self.seed)
            if self.kind == "multitask":
                if not -x.shape[1] <= self.task_column < x.shape[1] or x.shape[1] < 2:
                    raise ValueError("feature and task columns required")
                ids = x[:, self.task_column]
                if (
                    (ids < 0).any()
                    or not torch.equal(ids, ids.floor())
                    or ids.unique().numel() < 2
                ):
                    raise ValueError("need multiple tasks with nonnegative integer IDs")
                model = MultiTaskGP(
                    x,
                    standardized[:, None],
                    task_feature=self.task_column,
                    outcome_transform=None,
                )
                fit_gpytorch_mll(
                    ExactMarginalLogLikelihood(model.likelihood, model),
                    max_attempts=1,
                    optimizer_kwargs={"options": {"maxiter": self.steps}},
                )
            else:
                indices = torch.linspace(
                    0, len(x) - 1, min(self.inducing, len(x)), device=x.device
                ).long()
                model = SingleTaskVariationalGP(
                    x, standardized[:, None], inducing_points=x[indices].clone()
                )
                criterion = VariationalELBO(
                    model.likelihood, model.model, num_data=len(x)
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
                model.train()
                for _ in range(self.steps):
                    optimizer.zero_grad()
                    loss = -criterion(model.model(x), standardized)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("variational objective diverged")
                    loss.backward()
                    optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("non-finite fitted GP")
        self.model = model.eval()
        self.center, self.scale, self.dimension = center, scale, x.shape[1]
        self.task_ids = (
            set(x[:, self.task_column].tolist()) if self.kind == "multitask" else set()
        )
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("fit before predicting")
        x = _matrix(X, self.device)
        if x.shape[1] != self.dimension:
            raise ValueError("unexpected dimension")
        if (
            self.kind == "multitask"
            and not set(x[:, self.task_column].tolist()) <= self.task_ids
        ):
            raise ValueError("unseen or noninteger task")
        with torch.no_grad():
            posterior = self.model.posterior(x)
            mu = posterior.mean.flatten() * self.scale + self.center
            variance = posterior.variance.flatten() * self.scale.square()
            if not torch.isfinite(mu).all() or not torch.isfinite(variance).all():
                raise FloatingPointError("invalid GP prediction")
            return mu.cpu().numpy(), variance.cpu().numpy()


class SaasGP:
    """Fully Bayesian sparse axis-aligned subspace GP using BoTorch and NumPyro.

    Inputs are unit-cube features; outputs retain their physical units. Predictive
    variance integrates both conditional GP and hyperparameter uncertainty.
    Sampling diagnostics and mixing must be evaluated for each scientific fit.
    """

    def __init__(self, *, warmup=512, draws=256, thinning=16, seed=0):
        if (
            any(type(n) is not int or n < 1 for n in (warmup, draws, thinning))
            or draws < thinning
        ):
            raise ValueError(
                "positive MCMC budgets and at least one retained draw required"
            )
        self.warmup, self.draws, self.thinning, self.seed = (
            warmup,
            draws,
            thinning,
            seed,
        )
        self.model = None

    def fit(self, X, y):
        from botorch.fit import fit_fully_bayesian_model_nuts
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

        x = _matrix(X, "cpu")
        target = torch.as_tensor(y, dtype=x.dtype)
        if (
            target.shape != (len(x),)
            or not torch.isfinite(target).all()
            or ((x < 0) | (x > 1)).any()
            or len(x) < 2
        ):
            raise ValueError("need matching finite observations on the unit cube")
        from botorch.models.transforms.outcome import Standardize

        model = SaasFullyBayesianSingleTaskGP(
            x, target[:, None], outcome_transform=Standardize(m=1)
        )
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=self.warmup,
            num_samples=self.draws,
            thinning=self.thinning,
            disable_progbar=True,
            seed=self.seed,
        )
        self.model = model.eval()
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("fit before predicting")
        x = _matrix(X, "cpu")
        if (
            x.shape[1] != self.model.train_inputs[0].shape[-1]
            or ((x < 0) | (x > 1)).any()
        ):
            raise ValueError("incompatible prediction input")
        with torch.no_grad():
            posterior = self.model.posterior(x)
            mean, var = (
                posterior.mixture_mean.flatten(),
                posterior.mixture_variance.flatten(),
            )
            if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
                raise FloatingPointError("nonfinite MCMC mixture posterior")
            return mean.numpy(), var.numpy()
