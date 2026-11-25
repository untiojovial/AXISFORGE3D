"""Reproduce advanced diagnostic experiments on mathematical objectives only."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc
import torch

from axisforge_bo.acquisitions import expected_improvement
from axisforge_bo.benchmarks import multiobjective
from axisforge_bo.botorch_backend import TorchExperiment
from axisforge_bo.decisions import (
    constrained_improvement,
    discrete_knowledge_gradient,
    diverse_solutions,
    hypervolume_2d,
    probability_feasible,
    upper_cvar,
)
from axisforge_bo.models import ExactGP, StationaryKernel


def multi(seed, name, method):
    init, search, sobol = np.random.SeedSequence(seed).spawn(3)
    x = np.random.default_rng(init).random((4, 3))
    stream = np.random.default_rng(search)
    seq = qmc.Sobol(3, scramble=True, seed=np.random.default_rng(sobol)).random_base2(3)
    ref = np.array([1.1, 11] if name == "zdt1" else [3.0, 3.0])
    y = multiobjective(name, x)
    trace = [hypervolume_2d(y, ref)]
    for iteration in range(3):
        batches = stream.random((16, 2, 3))
        if method == "qlog_ehvi":
            model = TorchExperiment(x, -y, variances=np.full_like(y, 1e-5), fit=False)
            scores = model.scores(
                batches, method=method, reference=-ref, samples=32, seed=seed
            )
            pair = batches[int(scores.argmax())]
        elif method == "random":
            pair = batches[0]
        else:
            pair = seq[2 * iteration : 2 * (iteration + 1)]
        x = np.vstack((x, pair))
        y = np.vstack((y, multiobjective(name, pair)))
        trace.append(hypervolume_2d(y, ref))
    return dict(
        points=x.tolist(),
        objectives=y.tolist(),
        hypervolume=trace,
        reference=ref.tolist(),
    )


def constrained(seed, method):
    """Noisy latent quadratic with g(x)=x-.55; optimum on feasible boundary."""
    init, proposals, obs = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    x = init.random((4, 1))

    def observe(points):
        return (points[:, 0] - 0.8) ** 2 + obs.normal(scale=0.01, size=len(points))

    y = observe(x)
    for _ in range(8):
        pool = proposals.random((128, 1))
        if method == "random":
            j = 0
        else:
            obj = ExactGP(StationaryKernel(0.23), noise_variance=0.01**2).fit(x, y)
            con = ExactGP(StationaryKernel(0.23), noise_variance=1e-8).fit(
                x, x[:, 0] - 0.55
            )
            cm, cv = con.predict(pool)
            prob = probability_feasible(cm[:, None], cv[:, None])
            feasible = x[:, 0] <= 0.55
            if feasible.any():
                mu, var = obj.predict(pool)
                rank = constrained_improvement(
                    expected_improvement(mu, var, y[feasible].min()), prob
                )
            else:
                rank = prob
            j = int(rank.argmax())
        x = np.vstack((x, pool[j]))
        y = np.append(y, observe(pool[j : j + 1]))
    feasible = x[:, 0] <= 0.55
    latent = (x[:, 0] - 0.8) ** 2
    return dict(
        points=x[:, 0].tolist(),
        observations=y.tolist(),
        violation_rate=float((~feasible).mean()),
        best_feasible_regret=float(latent[feasible].min() - 0.25**2)
        if feasible.any()
        else None,
    )


def fidelity(seed, method):
    """Discrete finite-target KG at equal cost. Cheap evaluations have smooth bias."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 21)
    points = np.array([(x, f) for f in (0.0, 1.0) for x in grid])
    target = points[len(grid) :]
    phase = rng.uniform(0.15, 0.6)

    def losses(x):
        return np.sin(2 * np.pi * (x[:, 0] - phase)) + (1 - x[:, 1]) * 0.35 * np.cos(
            3 * np.pi * x[:, 0]
        )

    costs = np.where(points[:, 1] == 0, 1, 3)
    ids = [4, len(grid) + 14]
    spent = int(costs[ids].sum())
    while spent < 16:
        model = ExactGP(StationaryKernel([0.20, 1.0]), noise_variance=1e-6).fit(
            points[ids], losses(points[ids])
        )
        permitted = np.array(
            [i for i, c in enumerate(costs) if c <= 16 - spent and i not in ids]
        )
        if method == "kg_per_cost":
            mu, cov = model.predict_joint(points)
            scores = discrete_knowledge_gradient(
                mu[len(grid) :],
                cov[len(grid) :, permitted],
                np.diag(cov)[permitted] + 1e-6,
                costs[permitted],
            )
            j = permitted[scores.argmax()]
        else:
            j = rng.choice(permitted)
        ids.append(int(j))
        spent += int(costs[j])
    model = ExactGP(StationaryKernel([0.2, 1.0]), noise_variance=1e-6).fit(
        points[ids], losses(points[ids])
    )
    m, _ = model.predict(target)
    truth = losses(target)
    return dict(
        ids=ids,
        cost=spent,
        recommendation=float(grid[m.argmin()]),
        target_simple_regret=float(truth[m.argmin()] - truth.min()),
    )


def robust(seed):
    """Finite-grid robust selection vs nominal selection under common disturbances."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1, 1, 121)

    def loss(z):
        return (
            (z * z - 0.5) ** 2 + 0.08 * z + 0.75 * np.exp(-(((z + 0.55) / 0.06) ** 2))
        )

    train = np.clip(grid[:, None] + rng.normal(0, 0.12, (1, 128)), -1, 1)
    risk = upper_cvar(loss(train), 0.9)
    test = np.clip(grid[:, None] + rng.normal(0, 0.12, (1, 4096)), -1, 1)
    test_risk = upper_cvar(loss(test), 0.9)
    archive = diverse_solutions(grid[:, None], risk, radius=0.25, limit=3)
    return dict(
        nominal_x=grid[loss(grid).argmin()],
        robust_x=grid[risk.argmin()],
        nominal_test_cvar=test_risk[loss(grid).argmin()],
        robust_test_cvar=test_risk[risk.argmin()],
        diverse_x=grid[archive].tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--confirmatory", action="store_true", help="use 30 reserved seeds 301–330"
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    with args.output.open("w") as log:
        for seed in range(301, 331) if args.confirmatory else range(201, 206):
            tasks = [
                (name, m, lambda n=name, s=m: multi(seed, n, s))
                for name in ("zdt1", "dtlz2")
                for m in ("random", "sobol", "qlog_ehvi")
            ]
            tasks += [
                ("noisy_constraint", m, lambda s=m: constrained(seed, s))
                for m in ("random", "feasible_ei")
            ]
            tasks += [
                ("two_fidelities", m, lambda s=m: fidelity(seed, s))
                for m in ("random", "kg_per_cost")
            ]
            tasks += [("input_risk", "empirical_cvar", lambda: robust(seed))]
            for name, method, run in tasks:
                start = time.perf_counter()
                row = run()
                row.update(
                    task=name,
                    method=method,
                    seed=seed,
                    seconds=time.perf_counter() - start,
                )
                log.write(json.dumps(row, allow_nan=False) + "\n")
            print(seed, flush=True)


if __name__ == "__main__":
    main()
