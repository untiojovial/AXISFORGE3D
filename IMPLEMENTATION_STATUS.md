# Implementation status

Stated plainly, so nobody has to discover it.

## Built and tested

| Area | State |
| --- | --- |
| Domain model, validation, deterministic ids, templates | Done |
| Parallel-jaw gripper as ordinary links and prismatic joints | Done |
| Hands: actors (fingers) and preshapes, as in Simox end effectors; centric three-finger gripper and three-finger hand templates | Done. Fingers close by conservative advancement (never through a thin part); a blocked phalanx stops its joint and the tip keeps curling. |
| Hand grasps: surface point, approach along the normal (and centred, deep, 25° tilted), back away, close, wrench space from the contacts | Done. GraspStudio's contact model (one point per touching pair of convex pieces). |
| GJK distance, AABB broad phase | Done. No penetration depth (no EPA). |
| FK, Jacobian (checked against finite differences), conditioning | Done |
| IK: damped least squares, restarts, limits, cancellation | Done. One solver. |
| Collision scenes, carried parts, exemptions | Done. Meshes collide as convex hulls. |
| Grasp synthesis for box, cylinder, capsule, sphere | Done |
| Grasp synthesis for any part (meshes, assembled parts): surface sampling, pads closed until contact | Done. Matches the primitive generator exactly where both apply; evaluated on 16 YCB objects (`tools/verify`). |
| Parts from a bare STL/OBJ/COLLADA file | Done. Units given or guessed and reported; mass from volume or a default, reported. |
| Grasp quality: wrench-space force closure and epsilon, stroke, grip force | Done. Contact patches with Coulomb friction; optional soft-pad torsion. |
| Planner parameters in GraspStudio vocabulary (nrGrasps, minQuality, forceClosure, friction cone) | Done. Own semantics; values not comparable one-to-one with Simox. |
| Extension registry for quality measures and candidate generators | Done, Rust crates only. |
| Pick planner: pre-grasp, grasp, close, lift | Done |
| Candidate prior from an outside model (`plan --prior`, `cycle --prior`) | Done. Scores only reorder the search; every check still runs, so verdicts never change. A grasp-transfer prior (neighbours by box size, by frozen DINOv2/DINOv3/OpenCLIP features, or both) is a research tool. On 50 new parts, under a protocol fixed beforehand, it cut the candidates examined to the first feasible grasp to about 0.4 of the default order; appearance added nothing measurable over box size (`research/phase4/grasp_prior/new_parts`). |
| Exact (continuous) segment collision checking | Done. Schwarzer–Saha–Latombe test with per-link travel bounds; certified lower bound on clearance. |
| Free motion: bidirectional RRT with shortcut smoothing | Done. Seeded, cancellable; every edge proven clear. |
| Whole pick, transport and place cycle | Done. Every feasible grasp is tried and the fastest whole cycle kept. Synchronized trapezoidal profiles within every axis' limits, stop at each waypoint; acceptance test in three cells (`crates/motion/tests/acceptance.rs`). |
| Program export | Done: URScript (`movej` with leading-axis `a`, `v`, no blending; gripper functions to fill in), sampled CSV and JSON. A controller-cycle simulation re-times the program. |
| Project file format, atomic save | Done |
| URDF import | Done. STL, OBJ and COLLADA meshes. FK checked against an independent reference and against the UR5's published DH parameters. |
| Simox robot and object XML import | Done. Open Inventor (`.iv`) meshes are not read. |
| Mesh reading: binary and ASCII STL, OBJ, COLLADA | Done. COLLADA keeps material colours and scene transforms; checked against separately modelled meshes (`tools/verify`). |
| Collision pieces | Done: a closed mesh collides as its shells, each cut into convex pieces where concave (conservative: the pieces' hulls contain the solid); an open mesh collides as one hull. |
| Service, CLI, WebAssembly build, browser viewer | Done |

## Known limits

- **Contact model.** Coulomb friction at the corners of each pad's contact
  patch, friction cones approximated by pyramids. Pads are rigid unless a
  torsional friction term is given. The centre of mass is assumed to be the
  geometric centre. Nothing here has been validated against hardware.
- **Time budget.** `timeoutSeconds` works in native builds; the WebAssembly
  build has no clock and refuses it.
- **Pick planning samples, the cycle does not.** `plan_pick` alone still
  checks the approach and lift at a fixed number of samples; the cycle
  (`plan_cycle`) proves every segment clear continuously, those included.
- **Cycle timing stops at every waypoint.** Moves are not blended, which is
  safe and exactly reproducible on a controller but slower than a blended
  path. The part keeps its orientation when it is set down. Joints without
  declared speed or acceleration limits get assumed ones, reported.
- **What may touch in a cycle.** The fingers may touch the part while they
  close and open; the part may touch the surface it leaves or is set down on
  while it is lifted and lowered. Nothing else may come within the margin.
- **Approximate decomposition.** Concave closed meshes are cut into at most
  12 convex pieces per shell; the pieces' hulls still exceed the solid
  somewhat (a fifth to a half of the original excess on the YCB tools), so
  a very tight reach into a hollow can still be refused. Open meshes are
  not decomposed.
- **Pad contact model.** The pads are flat and rigid apart from a contact
  tolerance (0.3 mm by default) standing in for their compliance; contacts
  are capped at six per pad, which can only understate quality.
- **Gripper types.** Parallel jaws and hands; no suction. A hand's fingers
  close independently and the part is assumed not to move while they do: a
  coupled centric gripper would centre the part instead. Hand contacts are
  points, so a finger lying flat on a face is credited with less torque
  than it can resist. Simox end effectors with actors import as hands; their
  opening, finger size and force are assumed and reported.
- **Unreadable meshes.** Links whose only meshes are Open Inventor (`.iv`)
  are drawn as a skeleton and are not collision-checked. The engine lists
  them (`linksWithoutCollision`), the viewer's contact badge and plan verdict
  say so, and the CLI warns on `plan`.
- **COLLADA `up_axis`** is not applied (ROS tools do not either); a file
  declaring another axis gets a warning.
- **Import scope.** Robots with more than six manipulator axes are refused;
  floating and planar joints are refused; `mimic` joints become independent,
  with a warning.

## Not built

The original specification also calls for the items below. An earlier session
built several of them, but that work was lost with its container before it
could be pushed, and they have not been rebuilt:

- Tauri desktop shell and the React/Zustand editor (undo/redo, inspector).
- URDF export.
- Open Inventor (`.iv`) mesh reading.
- Additional IK solvers (pseudoinverse, CCD).
- Robot Doctor diagnostics.
- ADRs, threat model, contribution docs, CI workflow, Playwright suite.
- Plugin API.

The browser viewer is a demonstration front end, not the editor: it moves
joints, edits a part's position, mass and friction, exposes the planner
parameters, plans and replays picks, and plans, plays and exports whole
cycles.

The road from here to a sellable product is in
[docs/PLAN_DE_MEJORA.md](docs/PLAN_DE_MEJORA.md).
