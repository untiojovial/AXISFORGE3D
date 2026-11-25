"""Explicit past-task archive and conservative embedding-based retrieval.

This is an offline selection aid, not a guarantee against negative transfer.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def _time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must have a timezone")
    return parsed


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite JSON data") from exc


def _embedding(value):
    try:
        x = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid embedding") from exc
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("embedding must be a nonempty finite vector")
    return x


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    group: str
    completed_at: str
    context: dict
    embedding: tuple[float, ...]
    observations: tuple[dict, ...]
    versions: dict
    seeds: dict
    model_hyperparameters: dict

    def __post_init__(self):
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.group, str)
            or not self.group
        ):
            raise ValueError("task_id and group must be nonempty strings")
        completed = _time(self.completed_at)
        object.__setattr__(
            self, "embedding", tuple(_embedding(self.embedding).tolist())
        )
        for name in ("context", "versions", "seeds", "model_hyperparameters"):
            value = _json_copy(getattr(self, name))
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be an object")
            object.__setattr__(self, name, value)
        observations = _json_copy(self.observations)
        if not isinstance(observations, list) or any(
            not isinstance(x, dict) for x in observations
        ):
            raise ValueError("observations must be a list of objects")
        for observation in observations:
            if (
                "observed_at" in observation
                and _time(observation["observed_at"]) > completed
            ):
                raise ValueError("observation occurs after task completion")
            cost = observation.get("cost_seconds")
            if cost is not None and (
                isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0
            ):
                raise ValueError("observation cost must be nonnegative")
        object.__setattr__(self, "observations", tuple(observations))

    def to_dict(self):
        return _json_copy(
            {
                "task_id": self.task_id,
                "group": self.group,
                "completed_at": self.completed_at,
                "context": self.context,
                "embedding": self.embedding,
                "observations": self.observations,
                "versions": self.versions,
                "seeds": self.seeds,
                "model_hyperparameters": self.model_hyperparameters,
            }
        )


@dataclass(frozen=True)
class Selection:
    tasks: tuple[TaskRecord, ...]
    distances: tuple[float, ...]
    weights: tuple[float, ...]
    reason: str


class TaskArchive:
    def __init__(self, records=()):
        self._records = []
        for record in records:
            self.add(record)

    @property
    def records(self):
        return tuple(TaskRecord(**r.to_dict()) for r in self._records)

    def add(self, record):
        if not isinstance(record, TaskRecord):
            raise TypeError("expected TaskRecord")
        if any(r.task_id == record.task_id for r in self._records):
            raise ValueError("duplicate task id")
        self._records.append(TaskRecord(**record.to_dict()))

    def select(
        self,
        embedding,
        *,
        decision_at,
        allowed_groups,
        excluded_groups=(),
        max_distance,
        limit=5,
        paired_validation_losses=None,
    ):
        """Retrieve past eligible tasks; smaller loss means better validation.

        Paired losses (transfer, baseline) supplied by the caller are historical
        evidence, not inferred from target-task results. Mean transfer > baseline
        rejects transfer; no significance or safety guarantee is implied.
        """
        query, decision = _embedding(embedding), _time(decision_at)
        if isinstance(allowed_groups, (str, bytes)) or isinstance(
            excluded_groups, (str, bytes)
        ):
            raise ValueError("groups must be collections of names")
        allowed, excluded = set(allowed_groups), set(excluded_groups)
        if any(not isinstance(g, str) or not g for g in allowed | excluded):
            raise ValueError("invalid group")
        if (
            isinstance(max_distance, bool)
            or not isinstance(max_distance, (int, float))
            or not math.isfinite(max_distance)
            or max_distance < 0
        ):
            raise ValueError("invalid distance threshold")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("invalid selection limit")
        if paired_validation_losses is not None:
            losses = np.asarray(paired_validation_losses, dtype=float)
            if (
                losses.ndim != 2
                or losses.shape[1] != 2
                or not len(losses)
                or not np.isfinite(losses).all()
            ):
                raise ValueError("paired losses must be a finite nonempty (n,2) array")
            if float(np.mean(losses[:, 0] - losses[:, 1])) > 0:
                return Selection((), (), (), "negative_transfer_rejected")
        eligible = []
        for r in self._records:
            if (
                r.group not in allowed
                or r.group in excluded
                or _time(r.completed_at) >= decision
            ):
                continue
            if len(r.embedding) != len(query):
                continue
            distance = float(np.linalg.norm(np.asarray(r.embedding) - query))
            if math.isfinite(distance) and distance <= max_distance:
                eligible.append((distance, r.task_id, r))
        eligible.sort(key=lambda entry: (entry[0], entry[1]))
        eligible = eligible[:limit]
        if not eligible:
            return Selection((), (), (), "no_eligible_task_within_support")
        distances = np.asarray([x[0] for x in eligible])
        logits = -distances
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return Selection(
            tuple(TaskRecord(**r.to_dict()) for _, _, r in eligible),
            tuple(distances.tolist()),
            tuple(weights.tolist()),
            "selected",
        )

    def save(self, path):
        payload = json.dumps(
            {"version": 1, "tasks": [r.to_dict() for r in self._records]},
            allow_nan=False,
        )
        path = Path(path)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, delete=False
            ) as f:
                tmp = f.name
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            tmp = None
        finally:
            if tmp is not None:
                os.unlink(tmp)

    @classmethod
    def load(cls, path):
        def reject(value):
            raise ValueError("nonfinite JSON value")

        with open(path, encoding="utf-8") as f:
            data = json.load(f, parse_constant=reject)
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] != 1
            or not isinstance(data.get("tasks"), list)
        ):
            raise ValueError("invalid archive schema")
        try:
            return cls(TaskRecord(**r) for r in data["tasks"])
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid task record") from exc
