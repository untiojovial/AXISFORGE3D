"""Analytic benchmark functions on the unit cube; all objectives are minimized."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Problem:
    name: str
    dimension: int
    optimum: float
    function: Callable[[NDArray], NDArray]

    def __call__(self, points: ArrayLike) -> NDArray:
        x = np.asarray(points, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("expected rows of points with the problem dimension")
        if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError("points must belong to the unit cube")
        return self.function(x)


def _branin(x):
    a, b = 15 * x[:, 0] - 5, 15 * x[:, 1]
    return (
        (b - 5.1 * a * a / (4 * np.pi**2) + 5 * a / np.pi - 6) ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(a)
        + 10
    )


def _hartmann(x):
    if x.shape[1] == 3:
        a = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
        p = [
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ]
    else:
        a = [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
        p = [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    distances = np.sum(np.array(a) * (x[:, None, :] - np.array(p) * 1e-4) ** 2, axis=-1)
    return -np.exp(-distances) @ [1, 1.2, 3, 3.2]


def _ackley(x):
    z = 65.536 * (x - 0.5)
    return (
        -20 * np.exp(-0.2 * np.sqrt((z * z).mean(axis=1)))
        - np.exp(np.cos(2 * np.pi * z).mean(axis=1))
        + 20
        + np.e
    )


def _levy(x):
    w = 1 + (20 * x - 11) / 4
    return (
        np.sin(np.pi * w[:, 0]) ** 2
        + np.sum(
            (w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1
        )
        + (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    )


def _rosenbrock(x):
    z = 4 * x - 2
    return (100 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (1 - z[:, :-1]) ** 2).sum(axis=1)


def _rastrigin(x):
    z = 10.24 * (x - 0.5)
    return np.sum(z * z + 10 * (1 - np.cos(2 * np.pi * z)), axis=1)


PROBLEMS = {
    p.name: p
    for p in [
        Problem("branin", 2, 0.39788735772973816, _branin),
        Problem("hartmann3", 3, -3.86277978733266, _hartmann),
        Problem("hartmann6", 6, -3.32236801142, _hartmann),
        Problem("sphere", 3, 0, lambda x: np.sum((10 * x - 5) ** 2, axis=1)),
        Problem("ackley", 3, 0, _ackley),
        Problem("rosenbrock", 3, 0, _rosenbrock),
        Problem("rastrigin", 3, 0, _rastrigin),
        Problem("levy", 3, 0, _levy),
    ]
}


def multiobjective(name: str, points: ArrayLike) -> NDArray:
    """ZDT1 or two-objective DTLZ2, variable decision dimension >= 2."""
    x = np.asarray(points, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] < 2
        or not np.isfinite(x).all()
        or ((x < 0) | (x > 1)).any()
    ):
        raise ValueError("expected unit-cube inputs with at least two columns")
    if name == "zdt1":
        g = 1 + 9 * x[:, 1:].mean(axis=1)
        return np.column_stack((x[:, 0], g * (1 - np.sqrt(x[:, 0] / g))))
    if name == "dtlz2":
        r = 1 + np.sum((x[:, 1:] - 0.5) ** 2, axis=1)
        angle = np.pi * x[:, 0] / 2
        return r[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    raise ValueError("unknown multiobjective problem")
