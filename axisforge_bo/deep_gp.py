"""Optional two-layer variational deep Gaussian process for scalar regression."""

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.settings import num_likelihood_samples
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from .botorch_backend import _matrix


class _Layer(DeepGPLayer):
    def __init__(self, inducing, inputs, outputs=None):
        batch = torch.Size([] if outputs is None else [outputs])
        distribution = CholeskyVariationalDistribution(
            inducing.size(-2), batch_shape=batch
        )
        strategy = VariationalStrategy(
            self, inducing.clone(), distribution, learn_inducing_locations=True
        )
        super().__init__(strategy, inputs, outputs)
        self.mean = ConstantMean(batch_shape=batch)
        self.kernel = ScaleKernel(
            RBFKernel(ard_num_dims=inputs, batch_shape=batch), batch_shape=batch
        )

    def forward(self, x):
        return MultivariateNormal(self.mean(x), self.kernel(x))


class _Model(DeepGP):
    def __init__(self, initial, hidden):
        super().__init__()
        self.features = _Layer(initial.repeat(hidden, 1, 1), initial.size(-1), hidden)
        self.output = _Layer(
            torch.randn(len(initial), hidden, dtype=initial.dtype), hidden
        )
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        return self.output(self.features(x))


class DeepVariationalGP:
    """Two stochastic GP layers, Gaussian likelihood and sample-based variational ELBO.

    CPU/f64 full-batch optimization. Predict returns mixture moments integrating
    hidden path samples. The output variance is latent by default. Randomness
    during training and inference is private and seeded for repeatable queries.
    This approximation does not give the posterior of a single exact GP.
    """

    def __init__(self, *, hidden=2, inducing=12, steps=50, samples=16, seed=0):
        if any(type(n) is not int or n < 1 for n in (hidden, inducing, steps, samples)):
            raise ValueError("positive integer layer and training budgets required")
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("nonnegative 31-bit seed required")
        self.hidden, self.inducing, self.steps, self.samples, self.seed = (
            hidden,
            inducing,
            steps,
            samples,
            seed,
        )
        self.model = None

    def fit(self, X, y):
        x = _matrix(X, "cpu")
        target = torch.as_tensor(y, dtype=x.dtype)
        if target.shape != (len(x),) or len(x) < 2 or not torch.isfinite(target).all():
            raise ValueError("two or more finite scalar observations required")
        center = target.mean()
        scale = target.std()
        if scale == 0:
            scale = torch.ones_like(scale)
        standardized = (target - center) / scale
        with torch.random.fork_rng(devices=[]), num_likelihood_samples(self.samples):
            torch.random.default_generator.manual_seed(self.seed)
            indices = torch.linspace(0, len(x) - 1, min(len(x), self.inducing)).long()
            model = _Model(x[indices], self.hidden).double()
            criterion = DeepApproximateMLL(
                VariationalELBO(model.likelihood, model, num_data=len(x))
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = -criterion(model(x), standardized)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite deep GP objective")
                loss.backward()
                optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("non-finite parameters")
        self.model = model.eval()
        self.dimension, self.center, self.scale = x.shape[-1], center, scale
        return self

    def predict(self, X, *, observation=False):
        if self.model is None:
            raise RuntimeError("fit before prediction")
        x = _matrix(X, "cpu")
        if x.shape[1] != self.dimension:
            raise ValueError("dimension mismatch")
        with (
            torch.no_grad(),
            torch.random.fork_rng(devices=[]),
            num_likelihood_samples(self.samples),
        ):
            torch.random.default_generator.manual_seed(self.seed)
            mixture = self.model(x)
            if observation:
                mixture = self.model.likelihood(mixture)
            mean = mixture.mean.mean(dim=0) * self.scale + self.center
            var = (
                mixture.variance.mean(dim=0) + mixture.mean.var(dim=0, unbiased=False)
            ) * self.scale.square()
            if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
                raise FloatingPointError("non-finite mixture")
            return mean.numpy(), var.numpy()
