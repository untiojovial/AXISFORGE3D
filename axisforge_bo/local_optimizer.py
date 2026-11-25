"""Sequential single-region Bayesian optimization with finite-pool Thompson sampling.

This is an explicit local-search variant, not a faithful TuRBO implementation.
"""

import numpy as np

from .acquisitions import thompson_sample
from .models import ExactGP, StationaryKernel
from .space import Box
from .trust_region import TrustRegion


class LocalBayesianOptimizer:
    def __init__(
        self, space, seed=0, pool_size=128, noise_variance=1e-6, initial_length=0.8
    ):
        if (
            not isinstance(space, Box)
            or not isinstance(pool_size, int)
            or isinstance(pool_size, bool)
            or pool_size < 2
        ):
            raise ValueError("expected a continuous box and pool size >= 2")
        self.space = space
        self.rng = np.random.default_rng(seed)
        self.pool_size = pool_size
        self.initial_length = initial_length
        self.region = TrustRegion(space.dimension, length=initial_length)
        self.model = ExactGP(
            StationaryKernel(length_scale=np.ones(space.dimension), kind="matern52"),
            noise_variance=noise_variance,
        )
        self._pending = None
        self._observations = []
        self._region_observations = []

    def ask(self):
        if self._pending is not None:
            raise RuntimeError("outstanding evaluation")
        if self.region.restart_required:
            self.region.restart(self.initial_length)
            self._region_observations = []
        if not self._region_observations:
            point = self.space.sample(self.rng, 1)[0]
        else:
            bounds = self.space.bounds
            width = bounds[:, 1] - bounds[:, 0]
            records = self._region_observations
            center = min(records, key=lambda item: item[1])[0]
            X = np.array([(p - bounds[:, 0]) / width for p, y in records])
            y = np.array([y for p, y in records])
            self.model.fit(X, y)
            pool = self.region.sample(
                self.space,
                center,
                self.pool_size,
                self.rng,
                self.model.kernel.length_scale,
            )
            # Perturb a sparse set of coordinates for higher-dimensional inputs.
            probability = min(20 / self.space.dimension, 1.0)
            mask = self.rng.random((self.pool_size, self.space.dimension)) < probability
            empty = np.flatnonzero(~mask.any(axis=1))
            if len(empty):
                mask[
                    empty, self.rng.integers(self.space.dimension, size=len(empty))
                ] = True
            pool = np.where(mask, pool, center)
            candidates = (pool - bounds[:, 0]) / width
            mean, cov = self.model.predict_joint(candidates)
            sample = thompson_sample(mean, cov, self.rng)
            point = pool[int(np.argmin(sample))]
        self._pending = point.copy()
        return point.copy()

    def tell(self, point, value):
        if self._pending is None:
            raise RuntimeError("no outstanding evaluation")
        point = self.space.validate(point)
        if not np.array_equal(point, self._pending):
            raise ValueError("point does not match reservation")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("objective value must be finite")
        self.region.update([value])
        entry = (point.copy(), value)
        self._observations.append(entry)
        self._region_observations.append(entry)
        self._pending = None

    @property
    def observations(self):
        return [(p.copy(), y) for p, y in self._observations]

    def best(self):
        if not self._observations:
            raise RuntimeError("no observations")
        p, y = min(self._observations, key=lambda item: item[1])
        return p.copy(), y
