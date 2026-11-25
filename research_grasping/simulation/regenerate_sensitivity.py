"""Regenerate all six fixed-configuration solver/timestep trajectories."""

import argparse
import json
from pathlib import Path

from research_grasping.simulation.mujoco_pilot import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    runs = {
        f"{solver}_{step}": evaluate(solver=solver, timestep_s=step)
        for solver in ("Newton", "CG")
        for step in (0.0005, 0.001, 0.002)
    }
    args.output.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
