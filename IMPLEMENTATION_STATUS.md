# Implementation status

Stated plainly, so nobody has to discover it.

## Built and tested

| Area | State |
| --- | --- |
| Domain model, validation, deterministic ids, templates | Done |
| Parallel-jaw gripper as ordinary links and prismatic joints | Done |
| GJK distance, AABB broad phase | Done. No penetration depth (no EPA). |
| FK, Jacobian (checked against finite differences), conditioning | Done |
| IK: damped least squares, restarts, limits, cancellation | Done. One solver. |
| Collision scenes, carried parts, exemptions | Done. Meshes collide as convex hulls. |
| Grasp synthesis for box, cylinder, capsule, sphere | Done |
| Grasp quality: force closure, stroke, grip force | Done. Two-point Coulomb model. |
| Pick planner: pre-grasp, grasp, close, lift | Done |
| Project file format, atomic save | Done |
| Service, CLI, WebAssembly build, browser viewer | Done |

## Known limits

- **Contact model.** Two point contacts with Coulomb friction. Torque about the
  closing line is not modelled; the lever arm is reported and scored against.
  The centre of mass is assumed to be the geometric centre.
- **Path checking is sampled.** The approach and lift are checked at a fixed
  number of joint-space samples, not continuously.
- **Pick only.** There is no place planning, no transport path planning between
  cells, and no trajectory timing.
- **Primitives only for grasping.** Mesh parts and multi-geometry parts are
  refused with a reason.
- **One gripper type.** Parallel jaws; no suction, no multi-finger hands.

## Not built

The original specification also calls for the items below. An earlier session
built several of them, but that work was lost with its container before it
could be pushed, and they have not been rebuilt:

- Tauri desktop shell and the React/Zustand editor (undo/redo, inspector).
- URDF import and export.
- Additional IK solvers (pseudoinverse, CCD).
- Robot Doctor diagnostics.
- ADRs, threat model, contribution docs, CI workflow, Playwright suite.
- Plugin API.

The browser viewer is a demonstration front end, not the editor: it moves
joints, edits a part's position, mass and friction, plans and replays picks.
