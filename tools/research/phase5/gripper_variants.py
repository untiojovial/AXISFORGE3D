"""Write copies of part projects with a different gripper opening.

    python tools/research/phase5/gripper_variants.py --pool POOL --out POOL120 --opening-mm 120 \
        --project ycb/Banana/cell.axisforge ...

Only the gripper's maximum opening and its two jaw limits change (each jaw
travels half the opening); fingers, force, arm and part stay the same.
"""
import argparse
import json
from pathlib import Path


def variant(src, dst, opening_m):
    doc = json.loads(Path(src).read_text())
    robot = doc['project']['robot']; gripper = robot['gripper']
    gripper['maxOpening'] = opening_m
    jaws = set(gripper['jawJointIds'])
    for joint in robot['joints']:
        if joint['id'] in jaws:
            joint['limits']['upper'] = opening_m/2
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(doc))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pool', required=True); p.add_argument('--out', required=True)
    p.add_argument('--opening-mm', type=float, required=True)
    p.add_argument('--project', action='append', required=True)
    a = p.parse_args()
    for rel in a.project:
        variant(Path(a.pool)/rel, Path(a.out)/rel, a.opening_mm/1000)
    print(len(a.project), 'projects ->', a.out)


if __name__ == '__main__':
    main()
