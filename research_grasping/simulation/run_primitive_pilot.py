"""Generate pre-specified exploratory primitive runs and provenance."""

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

from research_grasping.simulation import mujoco_pilot, primitive_pilot

OBJECTS = (
    {
        "id": "elongated_box",
        "family": "box",
        "geometry": {"type": "box", "size_m": [0.04, 0.06, 0.04]},
        "mjcf_size": ".02 .03 .02",
    },
    {
        "id": "vertical_cylinder",
        "family": "cylinder",
        "geometry": {
            "type": "cylinder",
            "radius_m": 0.02,
            "height_m": 0.04,
            "axis": "z",
        },
        "mjcf_size": ".02 .02",
    },
    {
        "id": "ellipsoid",
        "family": "ellipsoid",
        "geometry": {"type": "ellipsoid", "semiaxes_m": [0.02, 0.03, 0.02]},
        "mjcf_size": ".02 .03 .02",
    },
)


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = []
    runs = []
    for obj in OBJECTS:
        manifest.append(
            {
                **obj,
                "units": "m",
                "origin": "procedural_axisforge_research",
                "license": "project-authored; license not declared",
                "geometry_sha256": digest(obj["geometry"]),
            }
        )
        for offset, mass, friction in itertools.product(
            (-0.01, 0.0, 0.15), (0.05, 5.0), (0.0, 1.0)
        ):
            result = primitive_pilot.evaluate(
                object_id=obj["id"],
                offset_x_m=offset,
                mass_kg=mass,
                friction=friction,
            )
            runs.append({"object_id": obj["id"], "result": result})
    identity = primitive_pilot.evaluate() == mujoco_pilot.evaluate()
    if not identity:
        raise RuntimeError("cube control differs from original pilot")
    source = Path(primitive_pilot.__file__).read_bytes()
    output = {
        "manifest": manifest,
        "identity_control_passed": identity,
        "primitive_module_sha256": hashlib.sha256(source).hexdigest(),
        "original_module_sha256": hashlib.sha256(
            Path(mujoco_pilot.__file__).read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": runs,
    }
    payload = (json.dumps(output, indent=2) + "\n").encode("utf-8")
    if args.output.suffix == ".gz":
        args.output.write_bytes(gzip.compress(payload, mtime=0))
    else:
        args.output.write_bytes(payload)


if __name__ == "__main__":
    main()
