"""Validated continuous and finite mixed search spaces."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Real:
    low: float
    high: float

    def __post_init__(self):
        if not np.isfinite([self.low, self.high]).all() or self.low >= self.high:
            raise ValueError("real bounds must be finite and strictly ordered")


@dataclass(frozen=True)
class Integer:
    low: int
    high: int

    def __post_init__(self):
        if (
            isinstance(self.low, bool)
            or isinstance(self.high, bool)
            or not isinstance(self.low, (int, np.integer))
            or not isinstance(self.high, (int, np.integer))
            or self.low > self.high
        ):
            raise ValueError("integer bounds must be ordered integers")


@dataclass(frozen=True)
class Categorical:
    choices: tuple

    def __post_init__(self):
        object.__setattr__(self, "choices", tuple(self.choices))
        if not self.choices or len(set(self.choices)) != len(self.choices):
            raise ValueError("choices must be nonempty and unique")


class MixedSpace:
    """Typed search space; categorical entries remain labels, not metric embeddings."""

    def __init__(self, variables):
        self.variables = tuple(variables)
        if not self.variables or not all(
            isinstance(v, (Real, Integer, Categorical)) for v in self.variables
        ):
            raise ValueError("space requires typed variables")
        self.dimension = len(self.variables)

    def validate(self, x):
        if len(x) != self.dimension:
            raise ValueError("wrong dimension")
        result = []
        for value, variable in zip(x, self.variables):
            if isinstance(variable, Categorical):
                if value not in variable.choices:
                    raise ValueError("unknown category")
            elif isinstance(variable, Integer):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or not variable.low <= value <= variable.high
                ):
                    raise ValueError("invalid integer")
            else:
                if not np.isfinite(value) or not variable.low <= value <= variable.high:
                    raise ValueError("invalid real")
            result.append(value)
        return tuple(result)

    def sample(self, rng, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("invalid count")
        return [
            tuple(
                rng.uniform(v.low, v.high)
                if isinstance(v, Real)
                else int(rng.integers(v.low, v.high, endpoint=True))
                if isinstance(v, Integer)
                else v.choices[int(rng.integers(len(v.choices)))]
                for v in self.variables
            )
            for _ in range(n)
        ]

    @property
    def encoded_dimension(self):
        return sum(
            len(v.choices) if isinstance(v, Categorical) else 1 for v in self.variables
        )

    def encode(self, x):
        x = self.validate(x)
        out = []
        for value, v in zip(x, self.variables):
            if isinstance(v, Categorical):
                out.extend(float(value == a) for a in v.choices)
            elif v.high == v.low:
                out.append(0.0)
            else:
                out.append((value - v.low) / (v.high - v.low))
        return np.asarray(out, dtype=float)

    def decode(self, encoded):
        z = np.asarray(encoded, dtype=float)
        if (
            z.shape != (self.encoded_dimension,)
            or not np.isfinite(z).all()
            or np.any(z < 0)
            or np.any(z > 1)
        ):
            raise ValueError("invalid encoding")
        out = []
        i = 0
        for v in self.variables:
            if isinstance(v, Categorical):
                block = z[i : i + len(v.choices)]
                if not np.isclose(block.sum(), 1) or not np.all(
                    (block == 0) | (block == 1)
                ):
                    raise ValueError("category must be one-hot")
                out.append(v.choices[int(np.argmax(block))])
                i += len(v.choices)
            else:
                value = v.low + z[i] * (v.high - v.low)
                if isinstance(v, Integer):
                    value = int(round(value))
                    if not np.isclose(
                        z[i],
                        0 if v.high == v.low else (value - v.low) / (v.high - v.low),
                    ):
                        raise ValueError("integer encoding is off-grid")
                out.append(value)
                i += 1
        return self.validate(out)


class Box:
    def __init__(self, bounds):
        a = np.asarray(bounds, dtype=float)
        if (
            a.ndim != 2
            or a.shape[1] != 2
            or not len(a)
            or not np.isfinite(a).all()
            or np.any(a[:, 0] >= a[:, 1])
        ):
            raise ValueError(
                "bounds must have shape (d,2), finite and strictly ordered"
            )
        with np.errstate(over="ignore"):
            widths = a[:, 1] - a[:, 0]
        if not np.isfinite(widths).all():
            raise ValueError("bound width overflows")
        self.bounds = a.copy()
        self.dimension = len(a)

    def validate(self, x):
        a = np.asarray(x, dtype=float)
        if (
            a.shape != (self.dimension,)
            or not np.isfinite(a).all()
            or np.any(a < self.bounds[:, 0])
            or np.any(a > self.bounds[:, 1])
        ):
            raise ValueError("point outside space or wrong dimension")
        return a.copy()

    def sample(self, rng, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("invalid count")
        return rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(n, self.dimension)
        )
