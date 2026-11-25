"""Frozen paired development benchmark, run from repository root.

Synthetic 1D deterministic minimization; 30 seeds; 4 historical observations,
8 current-task evaluations per method. Historical labels never include target.
"""

import csv
import json
import time
from pathlib import Path
import numpy as np
from axisforge_bo.context import ContextMemory
from axisforge_bo.models import ExactGP, StationaryKernel
from axisforge_bo.optimizer import BayesianOptimizer
from axisforge_bo.space import Box

SEEDS = tuple(range(30))
METHODS = ("cold", "context", "context_permuted")
SCENARIOS = ("related", "adversarial")
BUDGET, N_HISTORY, POOL = 8, 4, 128
OUT = Path("experiments/contextual_transfer_results")


def run(seed, scenario, method):
    rng = np.random.default_rng(seed + 10000)
    history_x = rng.uniform(0.0, 1.0, N_HISTORY)
    target_opt = 0.7
    source_opt = 0.6 if scenario == "related" else 0.1
    memory = ContextMemory(1, 1)
    for i, x in enumerate(history_x):
        memory.append([x], [0.1 * (i + 1)], (x - source_opt) ** 2, i, "source")
    history = (
        memory.warm_start(10, allowed_groups={"source"}, target_group="target")
        if method != "cold"
        else ()
    )
    opt = BayesianOptimizer(
        Box([[0.0, 1.0]]),
        ExactGP(StationaryKernel(length_scale=0.3), noise_variance=1e-5),
        seed=seed,
        context_dimension=1,
        n_initial=2,
        pool_size=POOL,
        max_evaluations=BUDGET + len(history),
    )
    start = time.perf_counter()
    for i, r in enumerate(history):
        permuted = history[(i + 1) % len(history)].context
        opt.add_observation(
            r.decision,
            r.value,
            context=permuted if method == "context_permuted" else r.context,
        )
    best = float("inf")
    trajectory = []
    for _ in range(BUDGET):
        candidate = opt.ask(context=[0.0])
        y = (candidate.x[0] - target_opt) ** 2
        opt.tell(candidate, y)
        best = min(best, y)
        trajectory.append(best)
    return {
        "seed": seed,
        "scenario": scenario,
        "method": method,
        "simple_regret": best,
        "seconds": time.perf_counter() - start,
        "trajectory": trajectory,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        run(seed, scenario, method)
        for scenario in SCENARIOS
        for seed in SEEDS
        for method in METHODS
    ]
    with (OUT / "raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=(
                "seed",
                "scenario",
                "method",
                "simple_regret",
                "seconds",
                "trajectory",
            ),
        )
        w.writeheader()
        for row in rows:
            w.writerow({**row, "trajectory": json.dumps(row["trajectory"])})
    bootstrap_rng = np.random.default_rng(90210)
    summary = {}
    for scenario in SCENARIOS:
        by_method = {
            m: np.array(
                [
                    r["simple_regret"]
                    for r in rows
                    if r["scenario"] == scenario and r["method"] == m
                ]
            )
            for m in METHODS
        }
        summary[scenario] = {}
        for method in METHODS:
            times = [
                r["seconds"]
                for r in rows
                if r["scenario"] == scenario and r["method"] == method
            ]
            delta = by_method[method] - by_method["cold"]
            samples = delta[
                bootstrap_rng.integers(0, len(delta), size=(10000, len(delta)))
            ].mean(axis=1)
            summary[scenario][method] = {
                "mean_regret": float(by_method[method].mean()),
                "median_regret": float(np.median(by_method[method])),
                "mean_seconds": float(np.mean(times)),
                "paired_mean_regret_delta_vs_cold": float(delta.mean()),
                "bootstrap_95_percentile_ci": np.quantile(
                    samples, [0.025, 0.975]
                ).tolist(),
            }
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "budget": BUDGET,
                "history": N_HISTORY,
                "pool": POOL,
                "bootstrap_seed": 90210,
                "bootstrap_resamples": 10000,
                "results": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
