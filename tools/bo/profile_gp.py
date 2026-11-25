"""Measure wall time and peak process RSS in isolated Linux CPU subprocesses."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def measurement(n, m):
    import numpy as np
    from axisforge_bo import ExactGP, StationaryKernel

    rng = np.random.default_rng(37)
    x, query = rng.random((n, 6)), rng.random((m, 6))
    y = np.sin(3 * x[:, 0]) + x[:, 1] ** 2
    gp = ExactGP(StationaryKernel(0.3))
    start = time.perf_counter()
    gp.fit(x, y)
    trained = time.perf_counter()
    means, variances = gp.predict(query)
    if not np.isfinite(means).all() or not (variances >= 0).all():
        raise ArithmeticError("nonfinite profile result")
    return dict(
        observations=n,
        queries=m,
        fit_ms=1000 * (trained - start),
        predict_ms=1000 * (time.perf_counter() - trained),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(measurement(*args.case)))
    else:
        if args.output is None:
            parser.error("--output is required")
        runs = []
        for n in (32, 128, 512):
            for m in (128, 1024):
                for replicate in range(3):
                    out = subprocess.check_output(
                        [sys.executable, "-I", __file__, "--case", str(n), str(m)],
                        env={
                            **os.environ,
                            "OPENBLAS_NUM_THREADS": "1",
                            "OMP_NUM_THREADS": "1",
                        },
                    )
                    runs.append(dict(json.loads(out), replicate=replicate))
        args.output.write_text(json.dumps(runs, indent=2) + "\n")
