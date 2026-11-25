"""Synthetic study design sensitivity; no robot observations or held-out data."""

import argparse
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def run(reps, seed):
    rng = np.random.default_rng(seed)
    rows = []
    # Each row is a distinct Monte Carlo experiment, generated in bounded chunks.
    scenarios = list(
        itertools.product(
            ("normal", "skew", "mixture"),
            (20, 50, 100, 200),
            (5, 10, 30),
            (0.1, 0.2),
            (0.0, 0.05, 0.1),
            (0.0, 0.05, 0.2),
            (1,),
        )
    )
    scenarios += list(
        itertools.product(
            ("normal", "skew", "mixture"),
            (50,),
            (10,),
            (0.2,),
            (0.0, 0.05, 0.1),
            (0.0,),
            (5, 20),
        )
    )
    for shape, n, s, sigma, mu, missing, histories in scenarios:
        reject = covered = usable = zero_pairs = 0
        estimates = []
        for start in range(0, reps, 64):
            r = min(64, reps - start)
            size = (r, histories, n)
            if shape == "normal":
                u = rng.normal(size=size)
            elif shape == "skew":
                # Standardized lognormal(0,1), exactly centered and unit variance.
                u = (rng.lognormal(size=size) - np.exp(0.5)) / np.sqrt(
                    (np.exp(1) - 1) * np.exp(1)
                )
            else:
                u = (
                    rng.normal(size=size)
                    * np.where(rng.random(size=size) < 0.1, 3, 1)
                    / np.sqrt(1.8)
                )
            counts = rng.binomial(s, 1 - missing, size=size)
            zero_pairs += int((counts == 0).sum())
            eps = rng.normal(size=size) * np.sqrt(counts)
            d = (
                mu
                + sigma * u
                + 0.2 * np.divide(eps, counts, out=np.zeros_like(eps), where=counts > 0)
            )
            d[counts == 0] = np.nan
            # With H>1, the histories are the independent units and contain a shared effect.
            if histories > 1:
                d += rng.normal(0, 0.1, size=(r, histories, 1))
                units = np.nanmean(d, axis=2)
            else:
                units = d[:, 0, :]
            count = np.sum(np.isfinite(units), axis=1)
            # Drop trials with <2 independent units, and report them explicitly.
            valid = count >= 2
            if not np.any(valid):
                continue
            unit = units[valid]
            count = count[valid]
            mean = np.nanmean(unit, axis=1)
            sd = np.nanstd(unit, axis=1, ddof=1)
            se = sd / np.sqrt(count)
            crit = t.ppf(0.95, count - 1)
            crit_ci = t.ppf(0.975, count - 1)
            reject += int(np.sum((mean > 0) & (mean > crit * se)))
            covered += int(
                np.sum((mean - crit_ci * se <= mu) & (mu <= mean + crit_ci * se))
            )
            usable += len(mean)
            estimates.extend(mean.tolist())
        rows.append(
            {
                "shape": shape,
                "objects_per_history": n,
                "seeds": s,
                "sigma_object": sigma,
                "sigma_seed": 0.2,
                "mu": mu,
                "missing_pair_probability": missing,
                "histories": histories,
                "history_sd": 0.1 if histories > 1 else 0,
                "attempted": reps,
                "usable": usable,
                "zero_pair_objects": zero_pairs,
                "rejection": reject / usable if usable else None,
                "rejection_wilson": wilson(reject, usable) if usable else None,
                "coverage": covered / usable if usable else None,
                "coverage_wilson": wilson(covered, usable) if usable else None,
                "mean_estimate": float(np.mean(estimates)) if estimates else None,
                "nominal_evaluations": 2 * histories * n * s * 50,
            }
        )
    return {
        "schema_version": 1,
        "config": {
            "repetitions": reps,
            "seed": seed,
            "alpha": 0.05,
            "budget": 50,
            "history": "fixed conditional when H=1; random independent otherwise",
        },
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
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.repetitions < 1:
        p.error("repetitions must be positive")
    a.output.write_text(
        json.dumps(run(a.repetitions, a.seed), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
