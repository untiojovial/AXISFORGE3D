"""Time-safe, explicit contextual observations for external evaluators."""

import math
import numbers
from dataclasses import dataclass


def _vector(value, name):
    try:
        vector = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonempty vector") from exc
    if not vector or not all(math.isfinite(x) for x in vector):
        raise ValueError(f"{name} must be a finite nonempty vector")
    return vector


def _time(value):
    # One explicitly defined clock domain: finite real seconds chosen by caller.
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
    ):
        raise ValueError("time must be finite real seconds in the same clock domain")
    return float(value)


@dataclass(frozen=True)
class ContextRecord:
    decision: tuple
    context: tuple
    value: float
    available_at: float
    group: str


class ContextMemory:
    """Append-only immutable copies; availability must precede the decision strictly.

    Time units and common clock domain are the caller's responsibility. Group exclusion
    is explicit; this filter cannot detect leakage already present in inputs or features.
    """

    def __init__(self, decision_dimension, context_dimension):
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0
            for n in (decision_dimension, context_dimension)
        ):
            raise ValueError("dimensions must be positive integers")
        self.decision_dimension = decision_dimension
        self.context_dimension = context_dimension
        self._records = []

    def append(self, decision, context, value, available_at, group):
        d, c = _vector(decision, "decision"), _vector(context, "context")
        if len(d) != self.decision_dimension or len(c) != self.context_dimension:
            raise ValueError("wrong dimension")
        if not isinstance(group, str) or not group:
            raise ValueError("group must be a nonempty string")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        record = ContextRecord(d, c, value, _time(available_at), group)
        self._records.append(record)
        return record

    def before(self, decision_at, *, excluded_groups=()):
        at = _time(decision_at)
        if isinstance(excluded_groups, str):
            raise ValueError("excluded_groups must be a collection of group IDs")
        excluded = frozenset(excluded_groups)
        if any(not isinstance(g, str) or not g for g in excluded):
            raise ValueError("invalid group ID")
        return tuple(
            r for r in self._records if r.available_at < at and r.group not in excluded
        )

    def warm_start(self, decision_at, *, allowed_groups, target_group):
        """Return eligible prior records from explicit allowed groups, excluding target.

        Data authenticity and whether features were computed without future data are
        caller obligations. Empty selection is valid and supports a cold-start fallback.
        """
        if isinstance(allowed_groups, str):
            raise ValueError("allowed_groups must be a collection")
        allowed = frozenset(allowed_groups)
        if (
            not isinstance(target_group, str)
            or not target_group
            or any(not isinstance(g, str) or not g for g in allowed)
        ):
            raise ValueError("invalid group IDs")
        if target_group in allowed:
            raise ValueError("target group cannot be allowed")
        return tuple(
            r
            for r in self.before(decision_at, excluded_groups={target_group})
            if r.group in allowed
        )

    def to_dict(self):
        """Return a JSON-compatible independent snapshot, not proof of provenance."""
        return {
            "version": 1,
            "decision_dimension": self.decision_dimension,
            "context_dimension": self.context_dimension,
            "records": [
                {
                    "decision": list(r.decision),
                    "context": list(r.context),
                    "value": r.value,
                    "available_at": r.available_at,
                    "group": r.group,
                }
                for r in self._records
            ],
        }

    @classmethod
    def from_dict(cls, data):
        if (
            not isinstance(data, dict)
            or set(data)
            != {"version", "decision_dimension", "context_dimension", "records"}
            or type(data.get("version")) is not int
            or data.get("version") != 1
            or not isinstance(data.get("records"), list)
        ):
            raise ValueError("invalid memory snapshot")
        obj = cls(data["decision_dimension"], data["context_dimension"])
        for record in data["records"]:
            if not isinstance(record, dict) or set(record) != {
                "decision",
                "context",
                "value",
                "available_at",
                "group",
            }:
                raise ValueError("invalid record")
            obj.append(**record)
        return obj
