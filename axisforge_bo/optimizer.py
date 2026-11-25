"""Fixed-context Bayesian optimization with explicit asynchronous reservations."""

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import RLock

import numpy as np

from .acquisitions import ACQUISITIONS
from .acquisitions import register_acquisition as register_acquisition
from .models import ExactGP, StationaryKernel
from .space import Box, Categorical, Integer, MixedSpace, Real


def synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _validate_metadata(metadata):
    if metadata is None:
        return {}
    if not isinstance(metadata, dict) or set(metadata) - {
        "units",
        "cost_seconds",
        "provenance",
    }:
        raise ValueError("invalid evaluation metadata")
    for key in ("units", "provenance"):
        if (
            key in metadata
            and metadata[key] is not None
            and not isinstance(metadata[key], str)
        ):
            raise ValueError("invalid evaluation metadata text")
    cost = metadata.get("cost_seconds")
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not np.isfinite(cost)
        or cost < 0
    ):
        raise ValueError("invalid evaluation cost")
    return deepcopy(metadata)


@dataclass(frozen=True)
class Candidate:
    id: int
    x: tuple
    context: tuple


class BayesianOptimizer:
    """Pool-search BO with constant-liar asynchronous reservations.

    Context coordinates are fixed, compared exactly for the incumbent, and not
    optimized. Caller owns context scaling and decision-time provenance. A new
    context without observations starts randomly; GP uses all observed contexts.
    """

    def __init__(
        self,
        space,
        surrogate,
        seed=0,
        context_dimension=0,
        n_initial=3,
        pool_size=512,
        maximize=False,
        max_evaluations=None,
        acquisition="ei",
        sampler="random",
        stop=None,
    ):
        if not isinstance(space, (Box, MixedSpace)):
            raise TypeError("unsupported search space")
        if isinstance(surrogate, ExactGP) and surrogate.noise_variance.ndim != 0:
            raise ValueError("optimizer requires scalar observation noise")
        if (
            not isinstance(context_dimension, int)
            or isinstance(context_dimension, bool)
            or context_dimension < 0
        ):
            raise ValueError("invalid context dimension")
        if (
            not isinstance(n_initial, int)
            or isinstance(n_initial, bool)
            or n_initial < 1
        ):
            raise ValueError("invalid initial count")
        if (
            not isinstance(pool_size, int)
            or isinstance(pool_size, bool)
            or pool_size < 1
        ):
            raise ValueError("invalid pool size")
        if not isinstance(maximize, bool):
            raise ValueError("maximize must be bool")
        if max_evaluations is not None and (
            not isinstance(max_evaluations, int)
            or isinstance(max_evaluations, bool)
            or max_evaluations < 1
        ):
            raise ValueError("invalid evaluation budget")
        if acquisition not in ACQUISITIONS:
            raise ValueError("unknown acquisition")
        if sampler not in {"random", "sobol", "lhs"}:
            raise ValueError("unknown sampler")
        if stop is not None and not callable(stop):
            raise ValueError("stop must be callable")
        if isinstance(space, MixedSpace) and sampler != "random":
            raise ValueError("mixed spaces require random sampling")
        self._lock = RLock()
        self.max_evaluations, self.acquisition, self.sampler, self.stop = (
            max_evaluations,
            acquisition,
            sampler,
            stop,
        )
        self.space, self.surrogate = space, surrogate
        self.context_dimension, self.n_initial, self.pool_size, self.maximize = (
            context_dimension,
            n_initial,
            pool_size,
            maximize,
        )
        self.rng = np.random.default_rng(seed)
        self._records = []
        self._pending = {}
        self.next_id = 0

    def _context(self, context):
        c = np.asarray(() if context is None else context, dtype=float)
        if c.shape != (self.context_dimension,) or not np.isfinite(c).all():
            raise ValueError("invalid context")
        return tuple(c.tolist())

    def _encode(self, x):
        if isinstance(self.space, MixedSpace):
            return self.space.encode(x).tolist()
        a = np.asarray(x, dtype=float)
        return (
            (a - self.space.bounds[:, 0])
            / (self.space.bounds[:, 1] - self.space.bounds[:, 0])
        ).tolist()

    def _fit(self, include_pending=False):
        done = [r for r in self._records if r["status"] == "success"]
        if done:
            X = np.array(
                [self._encode(r["x"]) + r["context"] for r in done], dtype=float
            )
            y = np.array([r["value"] for r in done], dtype=float)
            y = -y if self.maximize else y
            if include_pending:
                pending_x, pending_y = [], []
                for candidate in self._pending.values():
                    same = [r for r in done if tuple(r["context"]) == candidate.context]
                    if same:
                        lie = max(
                            (-r["value"] if self.maximize else r["value"]) for r in same
                        )
                        pending_x.append(
                            self._encode(candidate.x) + list(candidate.context)
                        )
                        pending_y.append(lie)
                if pending_x:
                    X = np.vstack((X, np.asarray(pending_x, dtype=float)))
                    y = np.concatenate((y, pending_y))
            self.surrogate.fit(X, y)
        return done

    @property
    @synchronized
    def pending(self):
        """Single reservation for legacy callers, or a tuple when multiple exist."""
        values = tuple(self._pending.values())
        return None if not values else values[0] if len(values) == 1 else values

    @property
    @synchronized
    def pending_candidates(self):
        return tuple(self._pending.values())

    @property
    @synchronized
    def records(self):
        """A detached snapshot of the evaluation history."""
        return deepcopy(self._records)

    @synchronized
    def ask(self, context=None) -> Candidate:
        state = deepcopy(self.rng.bit_generator.state)
        try:
            return self._ask(context)
        except Exception:
            self.rng.bit_generator.state = state
            raise

    @synchronized
    def ask_batch(self, count: int, context=None) -> tuple[Candidate, ...]:
        """Reserve a batch atomically, rolling back on failure."""
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("batch count must be a positive integer")
        state = deepcopy(self.rng.bit_generator.state)
        pending = self._pending.copy()
        next_id = self.next_id
        try:
            return tuple(self._ask(context) for _ in range(count))
        except Exception:
            self.rng.bit_generator.state = state
            self._pending = pending
            self.next_id = next_id
            raise

    def _ask(self, context=None):
        c = self._context(context)
        if (
            self.max_evaluations is not None
            and len(self._records) + len(self._pending) >= self.max_evaluations
        ):
            raise RuntimeError("evaluation budget exhausted")
        if self.stop is not None and self.stop(self.records):
            raise RuntimeError("stopping callback requested stop")
        done = self._fit(include_pending=True)
        same = [r for r in done if tuple(r["context"]) == c]
        if self.sampler == "random":
            pool = self.space.sample(self.rng, self.pool_size)
        else:
            from scipy.stats import qmc

            seed = int(self.rng.integers(0, 2**32))
            engine = (
                qmc.Sobol(self.space.dimension, scramble=True, seed=seed)
                if self.sampler == "sobol"
                else qmc.LatinHypercube(self.space.dimension, seed=seed)
            )
            unit = (
                engine.random_base2(int(np.ceil(np.log2(self.pool_size))))[
                    : self.pool_size
                ]
                if self.sampler == "sobol"
                else engine.random(self.pool_size)
            )
            pool = qmc.scale(unit, self.space.bounds[:, 0], self.space.bounds[:, 1])
        used = {tuple(r["x"]) for r in self._records if tuple(r["context"]) == c} | {
            p.x for p in self._pending.values() if p.context == c
        }
        pool = [tuple(x) for x in pool if tuple(x) not in used]
        if not len(pool):
            raise RuntimeError("no untried points sampled")
        if len(same) < self.n_initial:
            x = pool[0]
        else:
            X = np.array([self._encode(x) + list(c) for x in pool], dtype=float)
            mu, var = self.surrogate.predict(X)
            incumbent = min(
                (-r["value"] if self.maximize else r["value"]) for r in same
            )
            function, direction = ACQUISITIONS[self.acquisition]
            scores = np.asarray(function(mu, var, incumbent), dtype=float)
            if scores.shape != mu.shape:
                raise ValueError("acquisition shape mismatch")
            if direction == "min":
                scores = -scores
            if not np.isfinite(scores).all():
                raise ArithmeticError("non-finite acquisition")
            x = pool[int(np.argmax(scores))]
        candidate = Candidate(self.next_id, tuple(x), c)
        self.next_id += 1
        self._pending[candidate.id] = candidate
        return candidate

    @synchronized
    def tell(self, candidate: Candidate, value: float, *, metadata=None) -> None:
        if (
            not isinstance(candidate, Candidate)
            or self._pending.get(candidate.id) != candidate
        ):
            raise ValueError("candidate does not match pending reservation")
        if (
            isinstance(value, (bool, np.bool_))
            or np.ndim(value) != 0
            or not np.isfinite(value)
        ):
            raise ValueError("objective must be finite scalar")
        metadata = _validate_metadata(metadata)
        self._records.append(
            {
                **({"metadata": metadata} if metadata else {}),
                "id": candidate.id,
                "x": list(candidate.x),
                "context": list(candidate.context),
                "status": "success",
                "value": float(value),
            }
        )
        del self._pending[candidate.id]

    @synchronized
    def fail(
        self,
        candidate: Candidate,
        status: str = "error",
        detail: str = "",
        *,
        metadata=None,
    ) -> None:
        if (
            not isinstance(candidate, Candidate)
            or self._pending.get(candidate.id) != candidate
        ):
            raise ValueError("candidate does not match pending reservation")
        if status not in {
            "error",
            "timeout",
            "cancelled",
            "infeasible",
        } or not isinstance(detail, str):
            raise ValueError("invalid failure status or detail")
        metadata = _validate_metadata(metadata)
        self._records.append(
            {
                **({"metadata": metadata} if metadata else {}),
                "id": candidate.id,
                "x": list(candidate.x),
                "context": list(candidate.context),
                "status": status,
                "detail": detail,
            }
        )
        del self._pending[candidate.id]

    @synchronized
    def add_observation(self, x, value, context=None):
        """Import one successful warm-start observation; counts toward budget."""
        if self._pending:
            raise RuntimeError("cannot import while a reservation is pending")
        if (
            self.max_evaluations is not None
            and len(self._records) + len(self._pending) >= self.max_evaluations
        ):
            raise RuntimeError("evaluation budget exhausted")
        point = tuple(self.space.validate(x))
        c = self._context(context)
        if any(
            tuple(r["x"]) == point and tuple(r["context"]) == c for r in self._records
        ):
            raise ValueError("duplicate warm-start observation")
        if (
            isinstance(value, (bool, np.bool_))
            or np.ndim(value) != 0
            or not np.isfinite(value)
        ):
            raise ValueError("objective must be finite scalar")
        candidate = Candidate(self.next_id, point, c)
        self.next_id += 1
        self._pending[candidate.id] = candidate
        self.tell(candidate, value)

    @synchronized
    def predict(self, x, context=None):
        x = self.space.validate(x)
        c = self._context(context)
        self._fit()
        mean, var = self.surrogate.predict(np.array([self._encode(x) + list(c)]))
        return float(-mean[0] if self.maximize else mean[0]), float(var[0])

    @synchronized
    def best(self, context=None):
        c = self._context(context)
        done = [
            r
            for r in self._records
            if r["status"] == "success" and tuple(r["context"]) == c
        ]
        return (
            deepcopy((max if self.maximize else min)(done, key=lambda r: r["value"]))
            if done
            else None
        )

    def _space_config(self):
        if isinstance(self.space, Box):
            return {"type": "box", "bounds": self.space.bounds.tolist()}
        variables = []
        for v in self.space.variables:
            if isinstance(v, Categorical):
                if not all(isinstance(x, str) for x in v.choices):
                    raise TypeError("checkpoint supports string categories only")
                variables.append({"type": "categorical", "choices": list(v.choices)})
            else:
                variables.append(
                    {
                        "type": "real" if isinstance(v, Real) else "integer",
                        "low": v.low,
                        "high": v.high,
                    }
                )
        return {"type": "mixed", "variables": variables}

    @staticmethod
    def _space_from_config(config):
        if config["type"] == "box":
            return Box(config["bounds"])
        if config["type"] != "mixed":
            raise ValueError("unsupported space type")
        variables = []
        for item in config["variables"]:
            if item["type"] == "real":
                variables.append(Real(item["low"], item["high"]))
            elif item["type"] == "integer":
                variables.append(Integer(item["low"], item["high"]))
            elif item["type"] == "categorical":
                variables.append(Categorical(tuple(item["choices"])))
            else:
                raise ValueError("unsupported variable type")
        return MixedSpace(variables)

    @synchronized
    def save_checkpoint(self, path: str | Path) -> None:
        if not isinstance(self.surrogate, ExactGP) or not isinstance(
            self.surrogate.kernel, StationaryKernel
        ):
            raise TypeError("checkpoint supports ExactGP and StationaryKernel only")
        k = self.surrogate.kernel
        state = {
            "version": 1,
            "space": self._space_config(),
            "context_dimension": self.context_dimension,
            "n_initial": self.n_initial,
            "pool_size": self.pool_size,
            "maximize": self.maximize,
            "kernel": {
                "length_scale": k.length_scale.tolist(),
                "amplitude": k.amplitude,
                "kind": k.kind,
            },
            "noise_variance": self.surrogate.noise_variance.tolist(),
            "jitter": self.surrogate.jitter,
            "rng": self.rng.bit_generator.state,
            "next_id": self.next_id,
            "records": self._records,
            "pending": [
                {"id": p.id, "x": list(p.x), "context": list(p.context)}
                for p in self._pending.values()
            ],
        }
        state.update(
            {
                "max_evaluations": self.max_evaluations,
                "acquisition": self.acquisition,
                "sampler": self.sampler,
            }
        )
        if self.stop is not None:
            raise TypeError(
                "callbacks are not serializable; remove callback before checkpointing"
            )
        encoded = json.dumps(state, allow_nan=False, separators=(",", ":"))
        path = Path(path)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, delete=False
            ) as f:
                tmp = f.name
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            tmp = None
        finally:
            if tmp is not None:
                os.unlink(tmp)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "BayesianOptimizer":
        def bad_constant(x):
            raise ValueError("non-finite JSON value")

        with open(path, encoding="utf-8") as f:
            s = json.load(f, parse_constant=bad_constant)
        if not isinstance(s, dict) or s.get("version") != 1:
            raise ValueError("unsupported checkpoint version")
        obj = cls(
            cls._space_from_config(
                s.get("space", {"type": "box", "bounds": s.get("bounds")})
            ),
            ExactGP(StationaryKernel(**s["kernel"]), s["noise_variance"], s["jitter"]),
            context_dimension=s["context_dimension"],
            n_initial=s["n_initial"],
            pool_size=s["pool_size"],
            maximize=s["maximize"],
            max_evaluations=s.get("max_evaluations"),
            acquisition=s.get("acquisition", "ei"),
            sampler=s.get("sampler", "random"),
        )
        if not isinstance(s["records"], list):
            raise ValueError("invalid records")
        ids = set()
        for r in s["records"]:
            if (
                not isinstance(r, dict)
                or not isinstance(r.get("id"), int)
                or isinstance(r.get("id"), bool)
                or r["id"] < 0
                or r["id"] in ids
            ):
                raise ValueError("invalid record ID")
            ids.add(r["id"])
            x = obj.space.validate(r["x"])
            c = obj._context(r["context"])
            candidate = Candidate(r["id"], tuple(x), c)
            obj._pending[candidate.id] = candidate
            if r["status"] == "success":
                obj.tell(candidate, r["value"], metadata=r.get("metadata"))
            else:
                obj.fail(
                    candidate, r["status"], r["detail"], metadata=r.get("metadata")
                )
        pending = s["pending"]
        if pending is None:
            pending = []
        if isinstance(pending, dict):
            pending = [pending]
        if not isinstance(pending, list):
            raise ValueError("invalid pending reservations")
        for p in pending:
            if (
                not isinstance(p["id"], int)
                or isinstance(p["id"], bool)
                or p["id"] < 0
                or p["id"] in ids
            ):
                raise ValueError("invalid pending ID")
            candidate = Candidate(
                p["id"], tuple(obj.space.validate(p["x"])), obj._context(p["context"])
            )
            obj._pending[candidate.id] = candidate
            ids.add(p["id"])
        if (
            not isinstance(s["next_id"], int)
            or isinstance(s["next_id"], bool)
            or s["next_id"] < 0
            or (ids and s["next_id"] <= max(ids))
        ):
            raise ValueError("invalid next ID")
        points = [(tuple(r["x"]), tuple(r["context"])) for r in obj._records]
        points.extend((p.x, p.context) for p in obj._pending.values())
        if len(points) != len(set(points)):
            raise ValueError("duplicate checkpoint point")
        if obj.max_evaluations is not None and len(points) > obj.max_evaluations:
            raise ValueError("checkpoint exceeds evaluation budget")
        obj.next_id = s["next_id"]
        obj.rng.bit_generator.state = s["rng"]
        obj._fit()
        return obj
