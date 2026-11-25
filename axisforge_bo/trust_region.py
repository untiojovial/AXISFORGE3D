"""Single-region bounded local candidate search state, not a TuRBO reproduction."""

from dataclasses import dataclass

import numpy as np

from .space import Box


@dataclass
class TrustRegion:
    """Adaptive local box in normalized coordinates for minimization.

    Updates compare batches of completed objective observations. A restart is a
    request to the caller, who must choose and record a new center; it never
    hides additional objective evaluations.
    """

    dimension: int
    length: float = 0.8
    min_length: float = 0.5**7
    max_length: float = 1.6
    success_tolerance: int = 3
    failure_tolerance: int = 4
    relative_improvement: float = 1e-3
    best_value: float = float("inf")
    successes: int = 0
    failures: int = 0
    restart_required: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.dimension, int)
            or self.dimension < 1
            or not np.isfinite(
                [
                    self.length,
                    self.min_length,
                    self.max_length,
                    self.relative_improvement,
                ]
            ).all()
            or not 0 < self.min_length <= self.length <= self.max_length
            or self.success_tolerance < 1
            or self.failure_tolerance < 1
            or self.relative_improvement < 0
        ):
            raise ValueError("invalid trust region configuration")

    def update(self, values):
        if self.restart_required:
            raise ValueError("restart must be acknowledged before update")
        y = np.asarray(values, dtype=float)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
            raise ValueError("batch must contain finite values")
        best = float(np.min(y))
        improved = not np.isfinite(
            self.best_value
        ) or best < self.best_value - self.relative_improvement * abs(self.best_value)
        self.best_value = min(best, self.best_value)
        if improved:
            self.successes += 1
            self.failures = 0
            if self.successes >= self.success_tolerance:
                self.length = min(2 * self.length, self.max_length)
                self.successes = 0
        else:
            self.failures += 1
            self.successes = 0
            if self.failures >= self.failure_tolerance:
                self.length /= 2
                self.failures = 0
                if self.length < self.min_length:
                    self.restart_required = True

    def restart(self, length=0.8):
        if not np.isfinite(length) or not self.min_length <= length <= self.max_length:
            raise ValueError("invalid restart length")
        self.length = float(length)
        self.best_value = float("inf")
        self.successes = self.failures = 0
        self.restart_required = False

    def bounds(self, space, center, length_scales=None):
        if not isinstance(space, Box) or space.dimension != self.dimension:
            raise ValueError("incompatible box")
        center = space.validate(center)
        if self.restart_required:
            raise ValueError("restart required")
        widths = space.bounds[:, 1] - space.bounds[:, 0]
        unit = (center - space.bounds[:, 0]) / widths
        weights = (
            np.ones(self.dimension)
            if length_scales is None
            else np.asarray(length_scales, dtype=float)
        )
        if (
            weights.shape != (self.dimension,)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError("invalid length scales")
        # Unit geometric mean preserves the untruncated volume of the region.
        log_weights = np.log(weights)
        scaled = np.exp(log_weights - np.mean(log_weights))
        if not np.isfinite(scaled).all():
            raise ArithmeticError("length-scale ratio overflows")
        half = self.length * scaled / 2
        lower = np.maximum(0, unit - half)
        upper = np.minimum(1, unit + half)
        return np.column_stack(
            (space.bounds[:, 0] + lower * widths, space.bounds[:, 0] + upper * widths)
        )

    def sample(self, space, center, n, rng, length_scales=None):
        if not isinstance(n, int) or n < 0:
            raise ValueError("invalid sample count")
        bounds = self.bounds(space, center, length_scales)
        return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, self.dimension))
