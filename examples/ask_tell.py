"""Seeded minimization with failure handling and a resumable reservation."""

from pathlib import Path
from tempfile import TemporaryDirectory

from axisforge_bo import BayesianOptimizer, Box, ExactGP, StationaryKernel


def main():
    optimizer = BayesianOptimizer(
        Box([(-2, 2)]),
        ExactGP(StationaryKernel(length_scale=0.2)),
        seed=12,
        max_evaluations=12,
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "experiment.json"
        candidate = optimizer.ask()
        optimizer.save_checkpoint(path)
        optimizer = BayesianOptimizer.load_checkpoint(path)
        optimizer.tell(candidate, (candidate.x[0] - 0.4) ** 2)
        for _ in range(11):
            candidate = optimizer.ask()
            optimizer.tell(candidate, (candidate.x[0] - 0.4) ** 2)
    print(optimizer.best())


if __name__ == "__main__":
    main()
