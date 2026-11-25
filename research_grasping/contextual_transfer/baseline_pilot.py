"""Execute the predeclared diagnostic synthetic pilot, not a grasping benchmark."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import qmc

from .baselines import RGPE, FixedICM
from .model import AdaptiveTransfer, Source


def run(config):
    records = []
    budget = config["budget"]
    initial = config["initial"]
    n = config["candidate_count"]
    for scenario in config["scenarios"]:
        for seed in range(config["seed_start"], config["seed_start"] + config["seeds"]):
            rng = np.random.default_rng(seed)
            phase = rng.uniform(-0.25, 0.25)
            sources = []
            for i in range(config["history_tasks"]):
                xs = rng.uniform(0, 1, (config["history_points"], 1))
                sign = 1 if scenario == "related" else -1
                ys = sign * np.sin(
                    2 * np.pi * xs[:, 0] + phase + 0.08 * i
                ) + rng.normal(0, np.sqrt(config["noise_variance"]), len(xs))
                sources.append(Source(str(i), np.array([(i + 1) / 10]), xs, ys, i))
            sobol = qmc.Sobol(1, scramble=True, seed=seed).random_base2(4)
            init = sobol[:initial]
            candidates = np.random.default_rng(seed + 10000).uniform(
                0, 1, (budget - initial, n, 1)
            )
            noises = np.random.default_rng(seed + 20000).normal(
                0, np.sqrt(config["noise_variance"]), budget
            )
            for method in config["methods"]:
                start = time.perf_counter()
                if method == "rgpe":
                    model = RGPE(
                        sources,
                        noise=config["noise_variance"],
                        length_scale=config["length_scale"],
                        seed=seed + 40000,
                        samples=config["rgpe_samples"],
                    )
                elif method == "icm_fixed":
                    model = FixedICM(
                        sources,
                        noise=config["noise_variance"],
                        length_scale=config["length_scale"],
                        correlation=config["icm_correlation"],
                    )
                else:
                    model = AdaptiveTransfer(
                        [0.0],
                        [] if method in ("target", "random", "sobol") else sources,
                        target_time=100,
                        dimension=1,
                        length_scale=config["length_scale"],
                        noise_variance=config["noise_variance"],
                        eta=config["eta"],
                        floor=config["floor"],
                        tau=config["tau"],
                        use_evidence=method == "adaptive",
                        contextual=True,
                    )
                trajectory = []
                for t in range(budget):
                    if t < initial:
                        x = init[t]
                    elif method == "random":
                        x = candidates[t - initial, 0]
                    elif method == "sobol":
                        x = sobol[t]
                    else:
                        x = candidates[
                            t - initial,
                            np.argmax(model.acquisition(candidates[t - initial])),
                        ]
                    true = float(np.sin(2 * np.pi * x[0] + phase))
                    y = true + noises[t]
                    model.observe(x, y)
                    trajectory.append(
                        {
                            "x": float(x[0]),
                            "y": float(y),
                            "true": true,
                            "weights": model.weights.tolist()
                            if hasattr(model, "weights")
                            else None,
                        }
                    )
                true_best = np.maximum.accumulate([p["true"] for p in trajectory])
                records.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "phase": phase,
                        "source_x": [s.x.tolist() for s in sources],
                        "source_y": [s.y.tolist() for s in sources],
                        "trajectory": trajectory,
                        "auc": float(np.mean(true_best)),
                        "seconds": time.perf_counter() - start,
                    }
                )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transfer/synthetic_baselines.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = Path(args.config).read_bytes()
    config = json.loads(raw)
    records = run(config)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = "unavailable"
    result = {
        "config": config,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "git_head": commit,
        "note": "Working tree may be dirty; elapsed time is non-deterministic.",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "device": "CPU",
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in [
                "research_grasping/contextual_transfer/model.py",
                "research_grasping/contextual_transfer/baselines.py",
                "research_grasping/contextual_transfer/baseline_pilot.py",
            ]
        },
        "records": records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
