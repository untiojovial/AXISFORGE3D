"""Couple one verified AxisForge3D plan to a reduced MuJoCo lateral-offset pilot."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from research_grasping.contextual_transfer import AdaptiveTransfer, Source
from research_grasping.simulation.mujoco_pilot import evaluate


def digest(data):
    return hashlib.sha256(data).hexdigest()


def xyz(vector):
    return np.array([vector[axis] for axis in "xyz"], dtype=float)


def extract_scene(scene, report):
    """Accept only the verified identity-aligned 40 mm box and top-down +X plan."""
    project = scene["project"]
    cubes = [b for b in project["environment"]["bodies"] if b["name"] == "cube"]
    if (
        len(cubes) != 1
        or scene["format"] != "axisforge-project"
        or scene["version"] != 2
    ):
        raise ValueError("unsupported AxisForge project or ambiguous cube")
    cube = cubes[0]
    (geometry,) = cube["geometries"]
    for pose in (cube["pose"], geometry["origin"]):
        rot = pose["rotation"]
        if not np.allclose([rot[k] for k in "xyzw"], [0, 0, 0, 1], rtol=0, atol=1e-10):
            raise ValueError("fixture requires the identity orientation")
    if (
        geometry["shape"]["kind"] != "box"
        or not np.allclose(xyz(geometry["shape"]["size"]), 0.04, rtol=0, atol=1e-9)
        or not np.allclose(xyz(geometry["origin"]["translation"]), 0, atol=1e-10)
        or cube["id"] != report["bodyId"]
    ):
        raise ValueError("shape or planner target does not match the dynamic fixture")
    for plan in report["plans"]:
        candidate = plan["candidate"]
        if (
            np.allclose(xyz(candidate["closing"]), [1, 0, 0], atol=1e-9)
            and np.allclose(xyz(candidate["approach"]), [0, 0, -1], atol=1e-9)
            and math.isclose(candidate["width"], 0.04, abs_tol=1e-9)
        ):
            break
    else:
        raise ValueError("no reachable top-down plan closing along positive world X")
    center = xyz(candidate["center"])
    origin = xyz(cube["pose"]["translation"])
    local = center - origin
    if not np.isfinite(local).all() or np.max(np.abs(local)) > 0.02:
        raise ValueError("invalid candidate center for this cube")
    return {
        "body_id": cube["id"],
        "plan_label": candidate["label"],
        "center_world_m": center.tolist(),
        "object_origin_m": origin.tolist(),
        "center_object_m": local.tolist(),
        "static_measure": plan["quality"]["measure"],
        "static_score": plan["quality"]["quality"],
        "mass_kg": cube["physical"]["mass"],
        "friction": cube["physical"]["frictionCoefficient"],
    }


def build_inputs(binary):
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory) / "cell.axisforge"
        subprocess.run(
            [binary, "demo", "--out", project], check=True, capture_output=True
        )
        report = subprocess.check_output([binary, "plan", project, "cube", "--json"])
        raw = project.read_bytes()
    fixture = extract_scene(json.loads(raw), json.loads(report))
    return {
        **fixture,
        "project_sha256": digest(raw),
        "plan_sha256": digest(report),
        "binary_sha256": digest(Path(binary).read_bytes()),
    }


def atomic_write(path, data):
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(blob)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def measurement(u, mass, friction, config, scene):
    offset_m = scene["center_object_m"][0] + (2 * u - 1) * config["offset_limit_m"]
    start = time.perf_counter()
    result = evaluate(offset_m, mass, friction, config["dt_s"], config["solver"])
    duration = time.perf_counter() - start
    quality = float(result["quality"])
    if not math.isfinite(quality) or not 0 <= quality <= 1:
        raise ArithmeticError("nonfinite or out-of-range physical response")
    return {
        "u": float(u),
        "offset_m": offset_m,
        "quality": quality,
        "success": bool(result["success"]),
        "reason": result["reason"],
        "min_hold_lift_m": result["lift_m"],
        "max_hold_drift_m": result["relative_drift_m"],
        "table_contact": result["table_contact"],
        "seconds": duration,
    }


def propose(method, seed, prefix, config, archive):
    t = len(prefix)
    budget = config["budget"]
    sobol = qmc.Sobol(1, scramble=True, seed=seed).random_base2(
        (budget - 1).bit_length()
    )[:, 0]
    available = archive if method == "adaptive" else ()
    model = AdaptiveTransfer(
        [1.0],
        available,
        dimension=1,
        target_time=2.0,
        length_scale=config["length_scale"],
        noise_variance=config["noise_variance"],
    )
    if method in ("adaptive", "target"):
        for row in prefix:
            model.observe([row["u"]], row["quality"])
    if t < config["initial"] or method == "sobol":
        u = sobol[t]
    elif method == "random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, t, 1]))
        u = rng.uniform()
    else:
        candidates = np.linspace(0.0, 1.0, config["candidate_grid"])
        seen = np.array([row["u"] for row in prefix])
        candidates = candidates[
            np.min(np.abs(candidates[:, None] - seen), axis=1) > 1e-10
        ]
        u = candidates[np.argmax(model.acquisition(candidates[:, None]))]
    return float(u), model.weights.tolist() if method in ("target", "adaptive") else []


def run(config_path, output, binary):
    config_file = Path(config_path)
    cfg = json.loads(config_file.read_bytes())
    if (
        cfg["methods"] != ["random", "sobol", "target", "adaptive"]
        or not 1 <= cfg["initial"] < cfg["budget"] <= cfg["candidate_grid"]
    ):
        raise ValueError("unsupported design")
    scene = build_inputs(binary)
    from axisforge_bo import models
    from research_grasping.contextual_transfer import model
    from research_grasping.simulation import mujoco_pilot

    files = [
        Path(__file__),
        Path(model.__file__),
        Path(mujoco_pilot.__file__),
        Path(models.__file__),
    ]
    versions = {
        pkg: metadata.version(pkg)
        for pkg in ["axisforge-bo", "mujoco", "numpy", "scipy"]
    }
    versions["python"] = platform.python_version()
    signature = {
        "scene": scene,
        "config_sha256": digest(config_file.read_bytes()),
        "source_sha256": {p.name: digest(p.read_bytes()) for p in files},
        "versions": versions,
    }
    if output.exists():
        data = json.loads(output.read_bytes())
        if data["signature"] != signature:
            raise ValueError("checkpoint inputs differ")
        if data["pending_errors"]:
            raise ValueError(
                "recorded simulator error invalidates this pilot; preserve the aborted run"
            )
    else:
        data = {
            "signature": signature,
            "config": cfg,
            "git": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "device": platform.platform(),
            "architecture": platform.machine(),
            "available_cpu": len(os.sched_getaffinity(0)),
            "weight_order": ["target"] + sorted(task["id"] for task in cfg["history"]),
            "historical": {},
            "runs": {},
            "pending_errors": [],
        }
        atomic_write(output, data)

    def append_measurement(
        rows, u, mass, friction, label, weights, decision_seconds=0.0
    ):
        try:
            row = measurement(u, mass, friction, cfg, scene)
        except Exception as exc:
            data["pending_errors"].append(
                {
                    "stream": label,
                    "index": len(rows),
                    "u": u,
                    "error": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            atomic_write(output, data)
            raise
        rows.append(
            {**row, "weights_before": weights, "decision_seconds": decision_seconds}
        )
        atomic_write(output, data)

    archive = []
    for task in cfg["history"]:
        rows = data["historical"].setdefault(task["id"], [])
        for u in np.linspace(0.0, 1.0, cfg["historical_grid"])[len(rows) :]:
            append_measurement(
                rows, float(u), task["mass_kg"], task["friction"], task["id"], []
            )
        archive.append(
            Source(
                task["id"],
                np.array([1.0]),
                np.array([[r["u"]] for r in rows]),
                np.array([r["quality"] for r in rows]),
                available_at=0.0,
            )
        )
    for seed in cfg["seeds"]:
        for method in cfg["methods"]:
            label = f"{seed}/{method}"
            rows = data["runs"].setdefault(label, [])
            while len(rows) < cfg["budget"]:
                start = time.perf_counter()
                u, weights = propose(method, seed, rows, cfg, archive)
                decision_seconds = time.perf_counter() - start
                append_measurement(
                    rows,
                    u,
                    scene["mass_kg"],
                    scene["friction"],
                    label,
                    weights,
                    decision_seconds,
                )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--config", default="configs/pilot_study/cube.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        run(args.config, args.output, args.binary.resolve())


if __name__ == "__main__":
    main()
