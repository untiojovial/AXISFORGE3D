"""Development-only equal-evaluation Sphere benchmark; run from repository root."""

import argparse
import json
import time
import numpy as np
from axisforge_bo.space import Box
from axisforge_bo.local_optimizer import LocalBayesianOptimizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--confirmatory", action="store_true")
args = parser.parse_args()
seeds = range(301, 331) if args.confirmatory else range(5)
space = Box([(-5.0, 5.0)] * 10)
runs = []
for seed in seeds:
    start = time.perf_counter()
    optimizer = LocalBayesianOptimizer(space, seed=seed, pool_size=32)
    for _ in range(25):
        x = optimizer.ask()
        optimizer.tell(x, float(x @ x))
    elapsed = time.perf_counter() - start
    rng = np.random.default_rng(seed)
    x = space.sample(rng, 25)
    random_best = float(np.min(np.sum(x * x, axis=1)))
    runs.append(
        {
            "seed": seed,
            "local_bo_best": optimizer.best()[1],
            "random_best": random_best,
            "local_bo_seconds": elapsed,
        }
    )
print(
    json.dumps(
        {
            "problem": "10D Sphere sum(x_i^2), bounds [-5,5], minimum 0",
            "budget_per_method": 25,
            "local_pool_size": 32,
            "seeds": list(seeds),
            "results": runs,
            "status": "confirmatory seed cohort"
            if args.confirmatory
            else "development benchmark; no inferential or superiority claim",
        },
        indent=2,
    )
)
