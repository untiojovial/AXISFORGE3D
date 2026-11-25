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

## COLLADA meshes, recorded on 2026-11-24

The UR5's visual meshes are COLLADA files exported from CAD with Blender; its
collision meshes are STL files modelled separately. Reading both and
comparing bounding boxes checks the COLLADA reader's units, scene transforms
and indices against something it did not produce:

| UR5 link | Visual triangles | Worst bound difference |
| --- | --- | --- |
| base | 1 932 | 0.27 mm |
| shoulder | 13 488 | 0.25 mm |
| upperarm | 29 428 | 0.40 mm |
| forearm | 21 022 | 0.53 mm |
| wrist1 | 14 072 | 0.15 mm |
| wrist2 | 14 072 | 0.14 mm |
| wrist3 | 1 658 | 0.02 mm |

The differences are the two models' own: the collision meshes are coarser.
`crates/project-io/tests/collada_against_collision_meshes.rs` repeats the
check for `base` and `wrist3` within 1 mm in `cargo test`. All 11 Kinova
Jaco2 meshes also read, from 452 to 44 578 triangles, in at most 26 ms each.

## Grasping YCB objects, recorded on 2026-11-24

The 16 YCB objects of the GRASPA benchmark layouts, as published by the
`pybullet-object-models` project (visual meshes, metres, masses from its
URDFs), each added as a bare mesh with `add-object --units m --mass`, alone
on the demo table at (0.42, 0, 0.42). The demo gripper opens 80 mm; with
5 mm of clearance per jaw, parts up to 70 mm across fit. Friction 0.5.

Two questions, kept apart because they test different things:

- **Synthesis**: does the surface generator find grasps that fit the stroke
  and are in force closure? This is what a free-hand grasp planner such as
  `GraspStudio` answers. Every other stroke-fitting candidate (at most about
  170 per object) was judged in the wrench space.
- **Pick**: can the demo arm actually take it off the table? Reach,
  collision with the table and with the part, and lift all count. At most
  400 candidates examined, the viewer's practical budget.

| Object | Triangles | Fit the stroke | In force closure (judged) | Best epsilon | Picked by the demo arm |
| --- | --- | --- | --- | --- | --- |
| Banana | 15 728 | 398 | 69 / 133 | 0.061 | yes, 3 plans, 1.0 s (no before decomposition) |
| Chips can | 42 548 | 36 | 1 / 36 | 0.001 | no: 75 mm, wider than the stroke |
| Cracker box | 15 728 | 156 | 130 / 156 | 0.070 | no: 21 cm tall, grasps out of reach |
| Foam brick | 15 728 | 385 | 82 / 129 | 0.189 | yes, 3 plans, 0.8 s |
| Gelatin box | 15 728 | 317 | 74 / 159 | 0.151 | yes, 3 plans, 1.0 s |
| Hammer | 65 536 | 279 | 68 / 140 | 0.028 | no: lying flat, its grasps out of the arm's reach |
| Master chef can | 15 728 | 31 | 0 / 31 | 0 | no: 102 mm, wider than the stroke |
| Medium clamp | 65 536 | 498 | 60 / 125 | 0.079 | yes, 3 plans, 6.1 s (epsilon 0.006; no before decomposition) |
| Mustard bottle | 15 728 | 223 | 98 / 223 | 0.050 | yes, 3 plans, 0.5 s (epsilon 0.002: marginal) |
| Pear | 65 532 | 404 | 113 / 135 | 0.059 | yes, 3 plans, 2.5 s |
| Potted meat can | 15 728 | 220 | 136 / 220 | 0.116 | yes, 3 plans, 0.5 s |
| Power drill | 15 728 | 275 | 61 / 138 | 0.031 | yes, 3 plans, 0.4 s (epsilon 0.002: marginal) |
| Scissors | 15 728 | 333 | 31 / 167 | 0.028 | yes, 2 plans, 3.0 s (epsilon 0.002: marginal; no before decomposition) |
| Strawberry | 65 536 | 680 | 136 / 136 | 0.107 | yes, 2 plans, 7.9 s |
| Tennis ball | 65 536 | 576 | 144 / 144 | 0.079 | yes, 3 plans, 2.9 s |
| Tomato soup can | 15 728 | 333 | 128 / 167 | 0.139 | yes, 3 plans, 0.7 s |

Synthesis finds force-closure grasps for 15 of the 16; the one without is
the 102 mm can, which this gripper cannot hold at all. The demo arm picks
12. Before concave collision meshes were decomposed it picked 9: the
banana, clamp and scissors collided as single convex hulls, and a finger
that fits beside a banana's curve or between a clamp's jaws met the hull
instead. The four it still cannot pick fail on its own reach (the cracker
box, 21 cm tall, and the hammer lying flat) and on cans wider than the
stroke. The synthesis column is the same with and without decomposition,
within the sampling. Times are native, single-threaded.

## Convex decomposition, recorded on 2026-11-25

Space between a part's collision model and its solid, which is where a
finger can be wrongly refused; the solid is the closed mesh's volume:

| YCB mesh | Solid | One hull | Decomposed hulls | Pieces | Time |
| --- | --- | --- | --- | --- | --- |
| Hammer | 258.9 cm3 | +574.4 | +33.8 | 5 | 346 ms |
| Power drill | 580.0 cm3 | +707.2 | +73.7 | 5 | 148 ms |
| Banana | 171.4 cm3 | +93.7 | +16.7 | 3 | 128 ms |
| Medium clamp | 22.4 cm3 | +53.2 | +21.5 | 4 | 297 ms |
| Scissors | 46.6 cm3 | +82.1 | +34.8 | 4 | 138 ms |
| Mustard bottle | 611.7 cm3 | +86.9 | not cut | 1 | 26 ms |
| Tennis ball | 156.0 cm3 | +0.7 | not cut | 1 | 86 ms |
| Cracker box | 2172.7 cm3 | +132.6 | not cut | 1 | 25 ms |

The decomposed hulls still contain the whole solid: a unit test fills a
U-shaped block with 4000 random points and finds every one inside some
piece's hull, which only holds because cell corners inside the solid are
added to the pieces (cutting a block into octants would otherwise leave
its centre uncovered).

`GraspStudio` itself was not run for comparison: it needs Simox built with
its Qt and Coin3D dependencies. The phase 2 acceptance criterion, a
comparison on twenty parts, is therefore not yet met. `ycb_pick.sh` repeats
the pick column with the command line alone and no candidate limit: every
candidate of the four that fail is examined, and the result is the same
twelve.


## Phase 3: the whole cycle, proven and timed

```sh
cargo test --release -p axisforge-motion --test acceptance -- --nocapture
```

For the demo cell, the cell with a wall and the cell with two posts
(`templates::demo_cell`), and for the hand in the wall cell, the cube is
carried to (330, 210) mm and back to the start posture, with whichever
feasible grasp gives the fastest whole cycle. The test then, for
every move of every segment:

1. proves it clear again with a fresh `MotionChecker`, built from the
   segment's own allowed contacts as the test rebuilds them, not the
   planner's checkers;
2. checks 41 evenly spaced configurations with an ordinary `Scene` at the
   same margin, which must agree with the proof;

and for the cycle as a whole:

3. samples it every 4 ms and checks every joint's velocity and, by finite
   differences, acceleration against its scaled limits;
4. re-times the exported program with a controller-cycle simulation at
   8 ms, from the program's numbers alone, and compares with the planned
   cycle time (acceptance: under 2 %).

In the obstacle cells it also checks that the straight line from the lift
to above the target is blocked, so the detour was necessary.

| Cell | Gripper | Cycle | Program | Difference | Proven clearance | Closest measured |
| --- | --- | --- | --- | --- | --- | --- |
| Demo | parallel | 9.632 s | 9.632 s | 0.002 % | ≥ 1.02 mm | 1.9 mm |
| Wall | parallel | 11.240 s | 11.240 s | 0.002 % | ≥ 1.02 mm | 2.0 mm |
| Posts | parallel | 11.544 s | 11.544 s | 0.002 % | ≥ 1.01 mm | 2.0 mm |
| Wall | hand | 12.220 s | 12.220 s | 0.002 % | ≥ 1.02 mm | 1.4 mm |

The closest measured distances of 1.4 to 2.0 mm are the gaps the grasp
leaves on purpose (palm above the part, jaws beside it); the proven bound is
just above the 1 mm margin because subdivision stops as soon as a piece is
proven clear of it. The sampled-motion test in `crates/motion`
(`a_thin_pin_between_two_samples_is_not_missed`) shows what the continuous
check adds: a 3 mm pin that all six samples of a sampled check miss is
found.
