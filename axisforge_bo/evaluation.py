"""External evaluation outcomes; objective failures never become observations."""

import math
from dataclasses import dataclass
from typing import Protocol

from .optimizer import Candidate


@dataclass(frozen=True)
class Evaluation:
    status: str
    value: float | None = None
    detail: str = ""
    units: str | None = None
    cost_seconds: float | None = None
    provenance: str | None = None

    def __post_init__(self):
        if self.status not in {
            "success",
            "error",
            "timeout",
            "cancelled",
            "infeasible",
        }:
            raise ValueError("invalid status")
        if not isinstance(self.detail, str):
            raise ValueError("detail must be a string")
        if self.units is not None and not isinstance(self.units, str):
            raise ValueError("units must be text")
        if self.provenance is not None and not isinstance(self.provenance, str):
            raise ValueError("provenance must be text")
        if self.cost_seconds is not None and (
            isinstance(self.cost_seconds, bool)
            or not isinstance(self.cost_seconds, (int, float))
            or not math.isfinite(self.cost_seconds)
            or self.cost_seconds < 0
        ):
            raise ValueError("invalid cost")
        if self.status == "success":
            if (
                isinstance(self.value, bool)
                or self.value is None
                or not math.isfinite(self.value)
            ):
                raise ValueError("success requires a finite value")
        elif self.value is not None:
            raise ValueError("failure must not carry an objective")


class Evaluator(Protocol):
    def evaluate(self, candidate: Candidate) -> Evaluation: ...


def record(optimizer, candidate: Candidate, result: Evaluation) -> None:
    if not isinstance(result, Evaluation):
        raise TypeError("expected Evaluation")
    metadata = {
        key: value
        for key, value in {
            "units": result.units,
            "cost_seconds": result.cost_seconds,
            "provenance": result.provenance,
        }.items()
        if value is not None
    }
    if result.status == "success":
        optimizer.tell(candidate, result.value, metadata=metadata)
    else:
        optimizer.fail(candidate, result.status, result.detail, metadata=metadata)


@dataclass(frozen=True)
class MockEvaluator:
    """Deterministic synthetic evaluator for contract tests, not a simulator."""

    value: float = 0.0
    status: str = "success"
    detail: str = ""

    def evaluate(self, candidate: Candidate) -> Evaluation:
        if not isinstance(candidate, Candidate):
            raise TypeError("expected Candidate")
        return Evaluation(
            self.status, self.value if self.status == "success" else None, self.detail
        )
