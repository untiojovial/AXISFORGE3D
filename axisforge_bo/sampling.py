"""Seeded designs over continuous boxes; Sobol size is not silently rounded."""

import numpy as np
from scipy.stats import qmc

from .space import Box


def design(space, n, method="random", seed=None):
    if not isinstance(space, Box) or not isinstance(n, int) or n < 0:
        raise ValueError("expected box and nonnegative count")
    if method == "random":
        return space.sample(np.random.default_rng(seed), n)
    if method == "sobol":
        if n and n & (n - 1):
            raise ValueError("Sobol count must be a power of two")
        sampler = qmc.Sobol(d=space.dimension, scramble=True, seed=seed)
        unit = (
            sampler.random_base2(int(np.log2(n)))
            if n
            else np.empty((0, space.dimension))
        )
    elif method == "latin_hypercube":
        unit = qmc.LatinHypercube(d=space.dimension, seed=seed).random(n)
    else:
        raise ValueError("unknown design method")
    return qmc.scale(unit, space.bounds[:, 0], space.bounds[:, 1]) if n else unit
