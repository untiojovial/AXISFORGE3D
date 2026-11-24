# Import verification

Two scripts that check the URDF importer and forward kinematics against
references that are not the product. Both need only Python 3's standard
library and a release build of the CLI (`cargo build -p axisforge-cli --release`).

## `reference_fk.py`

An independent forward-kinematics implementation that reads the original
URDF, not the imported project. It draws random joint values within limits,
runs `axisforge fk` on the imported project for the same values, and reports
the worst position and rotation difference over every link.

```sh
axisforge import-urdf robot.urdf --out robot.axisforge [--end-effector name]
python3 -I tools/verify/reference_fk.py target/release/axisforge robot.urdf robot.axisforge 10 2026
```

This reference was written alongside the importer, so a misreading of the
URDF convention shared by both would not show up. That is what the next
check is for.

## `ur5_dh_check.py`

Compares the UR5's `tool0` pose in its `base` frame with the transform built
from the Denavit-Hartenberg parameters Universal Robots publishes. It shares
nothing with the URDF. The same check runs in `cargo test` as
`crates/project-io/tests/ur5_against_published_dh.rs`.

```sh
python3 -I tools/verify/ur5_dh_check.py target/release/axisforge ur5.axisforge 10 7
```

## Results recorded on 2026-11-24

Models from the `example-robot-data` project (BSD 3-Clause), ten random
postures each:

| Robot | Link poses compared | Worst position | Worst rotation |
| --- | --- | --- | --- |
| UR5 | 110 | 4.0e-13 mm | 5.9e-14 deg |
| Kinova Jaco2 6-DOF | 130 | 2.6e-13 mm | 7.2e-14 deg |
| Unitree Z1 (gripper as end-effector joint) | 100 | 2.2e-13 mm | 4.7e-14 deg |
| UR5 against published DH | 10 tool poses | 4.7e-09 mm | 8.0e-10 deg |

The DH agreement is limited by the URDF itself, which writes pi/2 as
`1.57079632679`. The acceptance bound for phase 1 is 0.1 mm.
