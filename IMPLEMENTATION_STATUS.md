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
| Grasp quality: wrench-space force closure and epsilon, stroke, grip force | Done. Contact patches with Coulomb friction; optional soft-pad torsion. |
| Planner parameters in GraspStudio vocabulary (nrGrasps, minQuality, forceClosure, friction cone) | Done. Own semantics; values not comparable one-to-one with Simox. |
| Extension registry for quality measures and candidate generators | Done, Rust crates only. |
| Pick planner: pre-grasp, grasp, close, lift | Done |
| Project file format, atomic save | Done |
| URDF import | Done. STL and OBJ meshes; COLLADA is not read. FK checked against an independent reference and against the UR5's published DH parameters. |
| Simox robot and object XML import | Done. Open Inventor (`.iv`) meshes are not read. |
| Mesh reading: binary and ASCII STL, OBJ | Done. Meshes collide as convex hulls. |
| Service, CLI, WebAssembly build, browser viewer | Done |

## Known limits

- **Contact model.** Coulomb friction at the corners of each pad's contact
  patch, friction cones approximated by pyramids. Pads are rigid unless a
  torsional friction term is given. The centre of mass is assumed to be the
  geometric centre. Nothing here has been validated against hardware.
- **Time budget.** `timeoutSeconds` works in native builds; the WebAssembly
  build has no clock and refuses it.
- **Path checking is sampled.** The approach and lift are checked at a fixed
  number of joint-space samples, not continuously.
- **Pick only.** There is no place planning, no transport path planning between
  cells, and no trajectory timing.
- **Primitives only for grasping.** Mesh parts and multi-geometry parts are
  refused with a reason.
- **One gripper type.** Parallel jaws; no suction, no multi-finger hands.
- **Unreadable meshes.** Links whose only meshes are COLLADA or Open Inventor
  are drawn as a skeleton and are not collision-checked. The engine lists
  them (`linksWithoutCollision`), the viewer's contact badge and plan verdict
  say so, and the CLI warns on `plan`.
- **Concave meshes** collide as their convex hull: extra contacts are
  possible, missed ones are not. There is no convex decomposition yet.
- **Import scope.** Robots with more than six manipulator axes are refused;
  floating and planar joints are refused; `mimic` joints become independent,
  with a warning.

## Not built

The original specification also calls for the items below. An earlier session
built several of them, but that work was lost with its container before it
could be pushed, and they have not been rebuilt:

- Tauri desktop shell and the React/Zustand editor (undo/redo, inspector).
- URDF export.
- Convex decomposition of concave meshes.
- Additional IK solvers (pseudoinverse, CCD).
- Robot Doctor diagnostics.
- ADRs, threat model, contribution docs, CI workflow, Playwright suite.
- Plugin API.

The browser viewer is a demonstration front end, not the editor: it moves
joints, edits a part's position, mass and friction, exposes the planner
parameters, plans and replays picks.

The road from here to a sellable product is in
[docs/PLAN_DE_MEJORA.md](docs/PLAN_DE_MEJORA.md).
