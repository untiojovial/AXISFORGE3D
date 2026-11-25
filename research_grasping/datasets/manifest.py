"""Conservative split and prequential history validation; no labels are read."""

import hashlib
import json
import math
import re

_SPLITS = {"train", "validation", "test"}
_UNKNOWN = {"", "unknown", "unverified", "pending", "none", "n/a"}


def _declared(value):
    return isinstance(value, str) and value.strip().lower() not in _UNKNOWN


def geometry_digest(geometry):
    """SHA-256 of canonical-json-v1: sorted keys, compact UTF-8, finite JSON."""
    data = json.dumps(
        geometry,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def validate_manifest(records):
    """Validate metadata and exact canonical geometry, returning id-indexed records.

    Family assignment and license evidence are declarations for human review;
    this function does not prove near-duplicate absence or legal rights.
    """
    by_id, families, digests = {}, {}, {}
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must be a nonempty list")
    for item in records:
        if not isinstance(item, dict):
            raise TypeError("invalid object record")
        for key in (
            "id",
            "family",
            "family_evidence",
            "origin",
            "license",
            "license_evidence",
            "units",
        ):
            if not _declared(item.get(key)):
                raise ValueError(f"missing or unverified {key}")
        if item["units"] != "m" or item.get("split") not in _SPLITS:
            raise ValueError("units must be metres and split must be declared")
        if item["id"] in by_id:
            raise ValueError("duplicate object id, including experimental variants")
        if (
            item.get("geometry_hash_format") != "canonical-json-v1"
            or not isinstance(item.get("geometry"), dict)
            or not item["geometry"]
        ):
            raise ValueError("geometry and canonical-json-v1 required")
        try:
            digest = geometry_digest(item["geometry"])
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid canonical geometry") from exc
        if (
            not re.fullmatch("[0-9a-f]{64}", item.get("sha256", ""))
            or item["sha256"] != digest
        ):
            raise ValueError("asset hash mismatch")
        split = item["split"]
        for registry, value in ((families, item["family"]), (digests, digest)):
            if value in registry and registry[value] != split:
                raise ValueError("family or asset crosses splits")
            registry[value] = split
        by_id[item["id"]] = item
    return by_id


def validate_history(records, target_id, cutoff, events, mode, allowed_source_splits):
    """Check complete earlier events against explicit frozen or online policy.

    Equality with cutoff is forbidden. Caller supplies only the prefix to be used;
    future records are rejected rather than silently filtered.
    """
    objects = validate_manifest(records)
    if (
        target_id not in objects
        or mode not in ("frozen", "online")
        or not isinstance(cutoff, (int, float))
        or isinstance(cutoff, bool)
        or not math.isfinite(cutoff)
    ):
        raise ValueError("invalid target, mode or cutoff")
    if (
        not isinstance(allowed_source_splits, (set, frozenset))
        or not allowed_source_splits <= _SPLITS
    ):
        raise ValueError("explicit source split policy required")
    if not isinstance(events, list):
        raise TypeError("events must be a list")
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("object_id") not in objects
            or event.get("complete") is not True
        ):
            raise ValueError("unknown or incomplete event")
        time = event.get("time")
        if (
            not isinstance(time, (int, float))
            or isinstance(time, bool)
            or not math.isfinite(time)
            or time >= cutoff
        ):
            raise ValueError("event must precede cutoff")
        source = event["object_id"]
        expected_kind = "observation" if source == target_id else "completed_task"
        if event.get("kind") != expected_kind:
            raise ValueError(
                "target observations and completed historical tasks must be distinguished"
            )
        if (
            source != target_id
            and objects[source]["split"] not in allowed_source_splits
        ):
            raise ValueError("source split not allowed by policy")
        if (
            mode == "frozen"
            and source != target_id
            and objects[source]["split"] == "test"
        ):
            raise ValueError("frozen test cannot use history from other test objects")
    return True
