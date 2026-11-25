"""Run the frozen protocol in docs/axisforge-bo/BENCHMARK_PROTOCOL.md."""

import argparse
import csv
import gzip
import io
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from axisforge_bo.benchmarks import PROBLEMS
from axisforge_bo.acquisitions import expected_improvement
from axisforge_bo.models import ExactGP, StationaryKernel

METHODS = ("random", "sobol", "native", "sklearn")


def posterior(method, x, y, test):
    scale = np.std(y)
    if scale == 0:
        scale = 1.0
    center = y.mean()
    target = (y - center) / scale
    if method == "native":
        model = ExactGP(StationaryKernel(0.25), noise_variance=1e-6, jitter=0)
        mean, var = model.fit(x, target).predict(test)
        std = np.sqrt(var)
    else:
        model = GaussianProcessRegressor(kernel=RBF(0.25), alpha=1e-6, optimizer=None)
        mean, std = model.fit(x, target).predict(test, return_std=True)
    return mean * scale + center, std * scale


def design(problem, seed, method):
    rng_init, rng_pool, rng_random, rng_sobol, rng_test = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)
    ]
    d = problem.dimension
    initial = rng_init.random((4, d))
    truth_points = rng_test.random((32, d))
    sobol = qmc.Sobol(
        d, scramble=True, seed=int(rng_sobol.integers(2**31))
    ).random_base2(4)
    x, y = initial, problem(initial)
    tracemalloc.start()
    start = time.perf_counter()
    for step in range(12):
        pool = rng_pool.random((128, d))
        if method == "random":
            new = rng_random.random(d)
        elif method == "sobol":
            new = sobol[step]
        else:
            mean, std = posterior(method, x, y, pool)
            if method == "native":
                scores = expected_improvement(mean, std**2, y.min())
            else:
                delta = y.min() - mean
                z = np.divide(delta, std, out=np.zeros_like(std), where=std > 0)
                scores = np.where(
                    std > 0,
                    delta * ndtr(z) + std * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi),
                    np.maximum(delta, 0),
                )
            new = pool[scores.argmax()]
        x = np.vstack((x, new))
        y = np.append(y, problem(new[None, :]))
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    regret = y - problem.optimum
    best = np.minimum.accumulate(regret)
    hit = np.flatnonzero(best <= 0.1 * best[3])
    row = dict(
        problem=problem.name,
        seed=seed,
        method=method,
        regret=best[-1],
        cumulative_regret=regret.sum(),
        first_hit=int(hit[0] + 1) if len(hit) else None,
        seconds=seconds,
        python_peak_bytes=peak,
        rmse=None,
        nll=None,
        coverage95=None,
        points=x.tolist(),
        values=y.tolist(),
        best=best.tolist(),
    )
    if method in {"native", "sklearn"}:
        mu, std = posterior(method, x, y, truth_points)
        error = problem(truth_points) - mu
        row.update(
            rmse=float(np.sqrt(np.mean(error**2))),
            nll=float(
                np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * (error / std) ** 2)
            ),
            coverage95=float(np.mean(np.abs(error) <= 1.95996398454 * std)),
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("at least one seed required")
    args.output.mkdir(parents=True, exist_ok=True)
    stats = []
    with (
        gzip.GzipFile(
            filename=str(args.output / "scalar.jsonl.gz"), mode="wb", mtime=0
        ) as raw,
        io.TextIOWrapper(raw, encoding="utf-8") as records,
    ):
        for problem in PROBLEMS.values():
            runs = {m: [] for m in METHODS}
            for seed in range(1001, 1001 + args.seeds):
                for m in METHODS:
                    row = design(problem, seed, m)
                    records.write(
                        json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                    )
                    runs[m].append(row["regret"])
            for baseline in ["random", "sobol", "sklearn"]:
                diff = np.array(runs["native"]) - runs[baseline]
                boot = (
                    np.random.default_rng(218)
                    .choice(diff, size=(2000, len(diff)))
                    .mean(axis=1)
                )
                stats.append(
                    (
                        problem.name,
                        baseline,
                        np.mean(runs["native"]),
                        np.mean(runs[baseline]),
                        diff.mean(),
                        *np.quantile(boot, [0.025, 0.975]),
                    )
                )
            print(problem.name, flush=True)
    with (args.output / "paired.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "problem",
                "comparator",
                "mean_native_regret",
                "mean_comparator_regret",
                "paired_mean_difference",
                "bootstrap_low",
                "bootstrap_high",
            ]
        )
        writer.writerows(stats)


if __name__ == "__main__":
    main()
