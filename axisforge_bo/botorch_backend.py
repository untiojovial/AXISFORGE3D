"""Optional CPU/GPU finite-set Bayesian design with BoTorch and GPyTorch.

All outcomes passed here are utilities to maximize. Negate minimization outputs
and their reference coordinates before fitting; noise variances keep their sign.
Candidate batches have shape (alternatives, q, dimensions), allowing genuine
joint scoring and asynchronous pending points without changing their ordering.
"""

import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from gpytorch.mlls import ExactMarginalLogLikelihood


def _matrix(value, device) -> torch.Tensor:
    matrix = torch.as_tensor(value, dtype=torch.double, device=device)
    if matrix.ndim != 2 or 0 in matrix.shape or not torch.isfinite(matrix).all():
        raise ValueError("expected a nonempty finite matrix")
    return matrix


class TorchExperiment:
    """Exact GP with standardized outputs in double precision; raw-unit posteriors."""

    def __init__(
        self, x, utilities, *, variances=None, fidelity=None, device="cpu", fit=True
    ):
        self.device = torch.device(device)
        self.x = _matrix(x, self.device).detach().clone()
        self.y = _matrix(utilities, self.device).detach().clone()
        if self.x.shape[0] != self.y.shape[0] or ((self.x < 0) | (self.x > 1)).any():
            raise ValueError("need matching observations and unit-cube inputs")
        noise = (
            None
            if variances is None
            else _matrix(variances, self.device).detach().clone()
        )
        if noise is not None and (noise.shape != self.y.shape or (noise <= 0).any()):
            raise ValueError("observation variances must match outputs and be positive")
        if fidelity is None:
            self.model = SingleTaskGP(self.x, self.y, train_Yvar=noise)
        else:
            if type(fidelity) is not int or not 0 <= fidelity < self.x.shape[1]:
                raise ValueError("invalid fidelity column")
            self.model = SingleTaskMultiFidelityGP(
                self.x, self.y, train_Yvar=noise, data_fidelities=[fidelity]
            )
        if fit:
            # One deterministic initial fit; surface optimization failures.
            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(self.model.likelihood, self.model),
                max_attempts=1,
            )
        self.model.eval()

    def scores(
        self,
        batches,
        *,
        method="log_ei",
        pending=None,
        reference=None,
        samples=128,
        seed=0,
    ):
        """Acquisition value for each provided alternative. Higher is preferable.

        log_ei: analytic, scalar, q=1, fixed observed incumbent.
        qlog_ei / qlog_nei: QMC joint improvement with fixed/noisy incumbent.
        qlog_ehvi / qlog_nehvi: joint log hypervolume improvement against a
        finite reference point in maximization units. Pending points are shared
        across alternatives and accounted for by the Monte Carlo acquisition.
        """
        x = torch.as_tensor(batches, dtype=torch.double, device=self.device)
        if x.ndim != 3 or 0 in x.shape or x.shape[-1] != self.x.shape[-1]:
            raise ValueError("expected nonempty alternatives x q x decision dimensions")
        px = None if pending is None else _matrix(pending, self.device)
        if px is not None and px.shape[-1] != x.shape[-1]:
            raise ValueError("pending dimension mismatch")
        for points in (x,) if px is None else (x, px):
            if not torch.isfinite(points).all() or ((points < 0) | (points > 1)).any():
                raise ValueError("candidates must be finite in the unit cube")
        if (
            type(samples) is not int
            or samples < 1
            or type(seed) is not int
            or not 0 <= seed < 2**31
        ):
            raise ValueError(
                "positive sample count and nonnegative 31-bit seed required"
            )
        multi = self.y.shape[-1] > 1
        if multi != (method in {"qlog_ehvi", "qlog_nehvi"}):
            raise ValueError("acquisition and number of objectives do not match")
        if not multi and reference is not None:
            raise ValueError("reference is only meaningful for hypervolume")
        sampler = SobolQMCNormalSampler(torch.Size([samples]), seed=seed)
        common = dict(model=self.model, sampler=sampler, X_pending=px)
        if multi:
            if reference is None:
                raise ValueError("reference required")
            ref = torch.as_tensor(reference, dtype=self.y.dtype, device=self.device)
            if ref.shape != self.y.shape[1:] or not torch.isfinite(ref).all():
                raise ValueError("invalid reference utilities")
            if method == "qlog_ehvi":
                partition = FastNondominatedPartitioning(ref_point=ref, Y=self.y)
                acq = qLogExpectedHypervolumeImprovement(
                    **common, ref_point=ref, partitioning=partition
                )
            else:
                acq = qLogNoisyExpectedHypervolumeImprovement(
                    **common, ref_point=ref, X_baseline=self.x, prune_baseline=False
                )
        elif method == "log_ei":
            if x.shape[1] != 1 or px is not None:
                raise ValueError("analytic EI requires q=1 and no pending observations")
            acq = LogExpectedImprovement(self.model, best_f=self.y.max())
        elif method == "qlog_ei":
            acq = qLogExpectedImprovement(**common, best_f=self.y.max())
        elif method == "qlog_nei":
            acq = qLogNoisyExpectedImprovement(
                **common, X_baseline=self.x, prune_baseline=False
            )
        else:
            raise ValueError(f"unknown acquisition {method!r}")
        with torch.no_grad():
            result = acq(x)
        if not torch.isfinite(result).all():
            raise FloatingPointError("non-finite acquisition value")
        return result

    def information_scores(
        self, points, representers, *, method="mes", samples=32, seed=0
    ):
        """Scalar unit-cube MES/PES, with a finite representer set for the maximum.

        PES conditions on sampled maximizers of correlated posterior paths over
        the supplied set. MES delegates to the backend's maximum-value entropy
        estimator. Both approximations depend on the domain discretization.
        """
        from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
        from botorch.acquisition.predictive_entropy_search import (
            qPredictiveEntropySearch,
        )

        query = _matrix(points, self.device)
        support = _matrix(representers, self.device)
        if self.y.shape[1] != 1 or method not in {"mes", "pes"}:
            raise ValueError("MES or PES requires a scalar utility")
        if (
            type(samples) is not int
            or samples < 2
            or type(seed) is not int
            or not 0 <= seed < 2**31
        ):
            raise ValueError("at least two samples and a 31-bit seed are required")
        for matrix in (query, support):
            if (
                matrix.shape[1] != self.x.shape[1]
                or ((matrix < 0) | (matrix > 1)).any()
            ):
                raise ValueError("wrong input dimension or bounds")
        devices = [] if self.device.type == "cpu" else [self.x.device.index]
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            if method == "mes":
                acquisition = qMaxValueEntropy(
                    self.model,
                    candidate_set=support,
                    num_mv_samples=samples,
                    num_y_samples=samples,
                )
            else:
                draws = SobolQMCNormalSampler(torch.Size([samples]), seed=seed)(
                    self.model.posterior(support)
                )
                maxima = support[draws[..., 0].argmax(dim=-1)]
                acquisition = qPredictiveEntropySearch(
                    self.model, optimal_inputs=maxima
                )
            scores = acquisition(query.unsqueeze(-2))
            if not torch.isfinite(scores).all():
                raise FloatingPointError("non-finite information gain")
            return scores
