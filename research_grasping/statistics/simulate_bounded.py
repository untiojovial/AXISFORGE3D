"""Predefined bounded skewed synthetic object differences, not robotic data."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [float(center - half), float(center + half)]


def run(reps=10000, seed=20261126):
    rng = np.random.default_rng(seed)
    rows = []
    for n in (50, 100, 200, 1000, 5000):
        for mu in (0.0, 0.05, 0.10, 0.15):
            r1 = np.sqrt(2 * np.log(20) / n)
            r2 = np.sqrt(2 * np.log(40) / n)
            rej_h = rej_t = cov_h = cov_t = 0
            for start in range(0, reps, 128):
                count = min(128, reps - start)
                x = rng.beta(1, 5, (count, n)) - 1 / 6 + mu
                means = x.mean(axis=1)
                se = x.std(axis=1, ddof=1) / np.sqrt(n)
                rej_h += int(np.sum(means > r1))
                rej_t += int(np.sum(means > t.ppf(0.95, n - 1) * se))
                cov_h += int(np.sum((means - r2 <= mu) & (mu <= means + r2)))
                cov_t += int(
                    np.sum(
                        (means - t.ppf(0.975, n - 1) * se <= mu)
                        & (mu <= means + t.ppf(0.975, n - 1) * se)
                    )
                )
            rows.append(
                {
                    "n": n,
                    "mu": mu,
                    "hoeffding_one_sided_radius": float(r1),
                    "hoeffding_two_sided_radius": float(r2),
                    "nominal_evaluations": n * 1000,
                    "hoeffding_rejection": rej_h / reps,
                    "hoeffding_rejection_wilson": wilson(rej_h, reps),
                    "t_rejection": rej_t / reps,
                    "t_rejection_wilson": wilson(rej_t, reps),
                    "hoeffding_coverage": cov_h / reps,
                    "t_coverage": cov_t / reps,
                    "hoeffding_coverage_wilson": wilson(cov_h, reps),
                    "t_coverage_wilson": wilson(cov_t, reps),
                }
            )
    return {
        "schema_version": 1,
        "design": "Beta(1,5) - 1/6 + mu, independent objects",
        "repetitions": reps,
        "seed": seed,
        "alpha": 0.05,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repetitions", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repetitions < 1:
        p.error("repetitions must be positive")
    args.output.write_text(
        json.dumps(run(args.repetitions, args.seed), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
