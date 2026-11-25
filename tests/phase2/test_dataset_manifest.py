"""Synthetic fixtures only: no final dataset or test labels."""

import copy

import pytest

from research_grasping.datasets.manifest import (
    geometry_digest,
    validate_history,
    validate_manifest,
)


def obj(name, family, split, width):
    geom = {"type": "box", "size_m": [width, 0.1, 0.1]}
    return {
        "id": name,
        "family": family,
        "family_evidence": "manual-fixture",
        "origin": "synthetic-fixture",
        "license": "test-only",
        "license_evidence": "fixture-author",
        "units": "m",
        "split": split,
        "geometry": geom,
        "geometry_hash_format": "canonical-json-v1",
        "sha256": geometry_digest(geom),
    }


@pytest.fixture
def records():
    return [
        obj("train", "a", "train", 0.1),
        obj("test1", "b", "test", 0.2),
        obj("test2", "c", "test", 0.3),
    ]


def test_valid_and_no_labels_read(records):
    records[1]["label"] = object()
    assert set(validate_manifest(records)) == {"train", "test1", "test2"}
    assert geometry_digest({"b": 2, "a": 1}) == geometry_digest({"a": 1, "b": 2})


def test_cross_split_collisions_and_unknown(records):
    for field, value in [
        ("family", "a"),
        ("family", "unknown"),
        ("license", "unknown"),
        ("family_evidence", "pending"),
        ("units", "cm"),
    ]:
        data = copy.deepcopy(records)
        data[1][field] = value
        with pytest.raises(ValueError):
            validate_manifest(data)
    data = copy.deepcopy(records)
    data[1]["geometry"] = data[0]["geometry"]
    data[1]["sha256"] = data[0]["sha256"]
    with pytest.raises(ValueError):
        validate_manifest(data)
    data = copy.deepcopy(records)
    data[1]["geometry"]["size_m"][0] = 0.9
    with pytest.raises(ValueError):
        validate_manifest(data)
    with pytest.raises(ValueError):
        validate_manifest(records + [copy.deepcopy(records[0])])


def test_prefix_policy(records):
    event = lambda name, t: {
        "object_id": name,
        "time": t,
        "complete": True,
        "kind": "observation" if name == "test1" else "completed_task",
    }
    assert validate_history(
        records, "test1", 5, [event("train", 1), event("test1", 4)], "frozen", {"train"}
    )
    for e in [
        event("test1", 5),
        event("test1", 6),
        event("missing", 1),
        {"object_id": "train", "time": 1, "complete": False},
    ]:
        with pytest.raises(ValueError):
            validate_history(records, "test1", 5, [e], "frozen", {"train"})
    wrong = event("train", 1)
    wrong["kind"] = "observation"
    with pytest.raises(ValueError):
        validate_history(records, "test1", 5, [wrong], "frozen", {"train"})
    with pytest.raises(ValueError):
        validate_history(
            records, "test1", 5, [event("test2", 1)], "frozen", {"train", "test"}
        )
    assert validate_history(
        records, "test1", 5, [event("test2", 1)], "online", {"test"}
    )
    with pytest.raises(ValueError):
        validate_history(records, "test1", 5, [event("test2", 1)], "online", {"train"})
