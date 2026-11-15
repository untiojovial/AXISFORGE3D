# AXISFORGE 3D

Design, analysis and **grasp planning** for robotic manipulators with one to
six axes and a parallel-jaw gripper.

> AXISFORGE 3D results are engineering approximations, not a functional-safety
> certification. Validate all motion on the real cell before operating
> machinery.

## What it does

- **Model** a robot as a tree of links and joints, with tool frames, a
  gripper, and a world of obstacles and parts. SI units throughout.
- **Kinematics**: forward kinematics, geometric Jacobian, conditioning and
  singularity detection, damped-least-squares IK with seeded restarts and
  cancellation.
- **Collision**: GJK distance between boxes, spheres, cylinders, capsules and
  convex hulls; clearance and contact reports; carried parts.
- **Grasping**: enumerate antipodal grasps of a part, score them (friction-cone
  force closure, stroke, grip force), and plan a pick — pre-grasp, grasp,
  close, lift — that the arm can reach and perform without collision. Every
  rejected grasp comes back with its reason.

## Try it

```sh
cargo test --workspace                      # the whole suite
cargo run -p axisforge-cli -- demo          # writes examples/pick_and_place_cell.axisforge
cargo run -p axisforge-cli -- plan examples/pick_and_place_cell.axisforge cube
apps/viewer/build.sh /tmp/axisforge.html    # engine as WebAssembly inside one HTML page
```

The viewer page needs the `wasm32-unknown-unknown` target
(`rustup target add wasm32-unknown-unknown`) and loads three.js from cdnjs.

## Layout

| Path | Role |
| --- | --- |
| `crates/domain` | Entities and validation. Depends on nothing but serde. |
| `crates/geometry` | Convex support functions, AABBs, GJK. |
| `crates/kinematics` | FK, Jacobian, conditioning, IK. The only place robotics maths lives. |
| `crates/collision` | Scenes and clearance checks. |
| `crates/grasping` | Grasp synthesis, quality, pick planner. |
| `crates/project-io` | Versioned project files, saved atomically. |
| `crates/service` | One JSON command interface used by every front end. |
| `crates/wasm` | The service as a zero-import WebAssembly module. |
| `apps/cli` | Command-line front end. |
| `apps/viewer` | Browser front end: draws what the engine reports, computes nothing itself. |

Dependencies point inward only: `domain` knows nothing about rendering or any
UI, and front ends never recompute a pose.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for what is and is not
built.
