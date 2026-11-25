"""Frozen supplemental synthetic diagnostic requested by the independent reviewer."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np


def saas():
    from numpyro.diagnostics import summary
    from axisforge_bo.surrogates import SaasGP

    rng = np.random.default_rng(521)
    x = rng.random((16, 6))
    y = np.sin(5 * x[:, 0]) + x[:, 1] + rng.normal(0, 0.04, len(x))
    chains = []
    for seed in range(40, 44):
        model = SaasGP(warmup=256, draws=256, thinning=1, seed=seed).fit(x, y).model
        cov = model.covar_module
        lengths = cov.base_kernel.lengthscale.detach().reshape(256, 6).numpy()
        chain = {
            "mean": model.mean_module.constant.detach().reshape(256).tolist(),
            "noise": model.likelihood.noise.detach().reshape(256).tolist(),
            "outputscale": cov.outputscale.detach().reshape(256).tolist(),
        }
        chain.update({f"lengthscale_{i}": lengths[:, i].tolist() for i in range(6)})
        chains.append(chain)
        print(f"finished chain {seed}", flush=True, file=sys.stderr)
    draws = {k: np.asarray([chain[k] for chain in chains]) for k in chains[0]}
    metrics = {
        k: {m: float(s[m]) for m in ("r_hat", "n_eff", "mean", "std")}
        for k, s in summary(draws, group_by_chain=True).items()
    }
    return {
        "seeds": list(range(40, 44)),
        "inputs": x.tolist(),
        "observations": y.tolist(),
        "warmup": 256,
        "draws_per_chain": 256,
        "metrics": metrics,
        "draws": draws_to_json(draws),
    }


def draws_to_json(draws):
    return {k: v.tolist() for k, v in draws.items()}


def profile(n):
    from axisforge_bo.surrogates import ResearchGP

    rng = np.random.default_rng(521)
    x = rng.random((n, 6))
    q = rng.random((128, 6))

    def signal(points):
        return np.sin(5 * points[:, 0]) + points[:, 1]

    y = signal(x) + rng.normal(0, 0.04, n)
    start = time.perf_counter()
    model = ResearchGP(steps=25, inducing=16).fit(x, y)
    trained = time.perf_counter()
    mean, var = model.predict(q)
    if not np.isfinite(var).all() or (var < 0).any():
        raise ArithmeticError("invalid posterior")
    return {
        "n": n,
        "fit_seconds": trained - start,
        "predict_seconds": time.perf_counter() - trained,
        "rmse": float(np.sqrt(np.mean((signal(q) - mean) ** 2))),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["saas", "variational", "profile-case"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "profile-case":
        print(json.dumps(profile(args.n)))
        return
    if args.output is None:
        parser.error("--output required")
    if args.mode == "saas":
        result = saas()
    else:
        result = []
        for n in (32, 128, 512):
            for replicate in range(3):
                out = subprocess.check_output(
                    [sys.executable, "-I", __file__, "profile-case", "--n", str(n)],
                    env={
                        **os.environ,
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                    },
                )
                result.append(dict(json.loads(out), replicate=replicate))
    args.output.write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
