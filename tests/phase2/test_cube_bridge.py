import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from research_grasping.experiments.cube import extract_scene, propose


def vector(x, y, z):
    return {"x": x, "y": y, "z": z}


def test_cube_plan_frame_and_restrictions():
    pose = {
        "translation": vector(0.42, -0.10, 0.44),
        "rotation": {**vector(0, 0, 0), "w": 1},
    }
    geom = {
        "origin": {**pose, "translation": vector(0, 0, 0)},
        "shape": {"kind": "box", "size": vector(0.04, 0.04, 0.04)},
    }
    cube = {
        "name": "cube",
        "pose": pose,
        "geometries": [geom],
        "id": "part",
        "physical": {"mass": 0.18, "frictionCoefficient": 0.6},
    }
    scene = {
        "version": 2,
        "format": "axisforge-project",
        "project": {"environment": {"bodies": [cube]}},
    }
    candidate = {
        "label": "test",
        "closing": vector(1, 0, 0),
        "approach": vector(0, 0, -1),
        "width": 0.04,
        "center": vector(0.427, -0.1, 0.445),
    }
    report = {
        "bodyId": "part",
        "plans": [
            {
                "candidate": candidate,
                "quality": {"measure": "wrench_space", "quality": 0.3},
            }
        ],
    }
    frame = extract_scene(scene, report)
    np.testing.assert_allclose(frame["center_object_m"], [0.007, 0, 0.005], atol=1e-12)
    np.testing.assert_allclose(
        np.add(frame["center_object_m"], frame["object_origin_m"]),
        frame["center_world_m"],
    )
    wrong = deepcopy(report)
    wrong["plans"][0]["candidate"]["closing"] = vector(0, 1, 0)
    with pytest.raises(ValueError):
        extract_scene(scene, wrong)
    pose["rotation"] = {"x": 0, "y": 0, "z": 1, "w": 0}
    with pytest.raises(ValueError):
        extract_scene(scene, report)


def test_paired_prefix_and_resumable_policy():
    config = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/pilot_study/cube.json"
        ).read_text()
    )
    prefix = []
    for _ in range(config["initial"]):
        candidates = [
            propose(method, 11, prefix, config, [])[0] for method in config["methods"]
        ]
        assert len(set(candidates)) == 1
        prefix.append({"u": candidates[0], "quality": 0.2})
    x, weights = propose("target", 11, prefix, config, [])
    assert propose("adaptive", 11, json.loads(json.dumps(prefix)), config, []) == (
        x,
        weights,
    )
    assert weights == [1.0] and x not in [r["u"] for r in prefix]
