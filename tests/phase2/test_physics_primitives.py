"""Identity, procedural provenance and interventions in the primitive pilot."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
from research_grasping.simulation import mujoco_pilot, primitive_pilot
from research_grasping.simulation.run_primitive_pilot import OBJECTS, digest


def test_control_cube_matches_original_exactly():
    assert primitive_pilot.evaluate() == mujoco_pilot.evaluate()


def test_geometry_selection_is_fixed_and_rejects_unknown():
    assert primitive_pilot.GEOMETRIES == {
        "cube": ("box", ".02 .02 .02"),
        "elongated_box": ("box", ".02 .03 .02"),
        "vertical_cylinder": ("cylinder", ".02 .02"),
        "ellipsoid": ("ellipsoid", ".02 .03 .02"),
    }
    with pytest.raises(ValueError):
        primitive_pilot.evaluate(object_id="mesh")


def test_manifest_and_control_interventions():
    path = (
        Path(__file__).resolve().parents[2] / "research/results/primitive_pilot.json.gz"
    )
    results = json.loads(gzip.decompress(path.read_bytes()))
    assert len(results["manifest"]) == 3
    assert len(results["runs"]) == 36
    assert results["identity_control_passed"]
    assert (
        results["primitive_module_sha256"]
        == hashlib.sha256(Path(primitive_pilot.__file__).read_bytes()).hexdigest()
    )
    for obj in results["manifest"]:
        assert obj["geometry_sha256"] == digest(obj["geometry"])
        assert obj["units"] == "m"
    assert {obj["id"] for obj in results["manifest"]} == {obj["id"] for obj in OBJECTS}
    for run in results["runs"]:
        result = run["result"]
        conditions = result["parameters"]
        if conditions["offset_x_m"] == 0.15:
            assert not result["success"]
    centered_box = primitive_pilot.evaluate(object_id="elongated_box")
    no_friction = primitive_pilot.evaluate(object_id="elongated_box", friction=0)
    heavy = primitive_pilot.evaluate(object_id="elongated_box", mass_kg=5)
    assert (
        centered_box["success"] and not no_friction["success"] and not heavy["success"]
    )
