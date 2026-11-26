# Phase VII — limitations

> **Historical Phase VII document, kept as written.** Some wording here was corrected later: "pre-registered" and "hash-signed" mean a protocol frozen in Git with SHA-256 digests, not a registry preregistration or signatures. Other corrections cover P3, the fallback bound and non-significant results. See `research/phase7_extension/FINAL_CORRECTIONS_REGISTER.md` (items B, C, D, H1, H2) and the current manuscript.

Ordered by how much each one limits the conclusions. Every item is also stated in the manuscript.

## Scope of the evidence

1. **Simulation only.** No robot was used. "Held", "robust" and "load tolerant" are MuJoCo outcomes for
   an assumed mass (0.1 kg), assumed uniform density (centre of mass = the planner's centroid), assumed
   friction (μ 0.5, torsional 0.005 m) and an idealised controller. They are not real-world success rates.
2. **Kinematic feasibility is not dynamic robustness.** N_F counts the candidates a planner examines
   until one passes inverse kinematics, collision and a static wrench test. Robustness is measured
   separately, and the two are never merged.
3. **One cell, one gripper geometry.** These were fixed for the confirmation:
   - a six-axis arm and an 80 mm parallel gripper;
   - the part at one table position;
   - the 40/120 mm analyses are development-only stroke variants of the same cell, not other robots.

## Physics

4. **Uncalibrated torsional friction.** Between 6 % and 18 % of trial verdicts change across
   0.001–0.02 m. Load tolerance is the most affected.
5. **Simulator level.** The floating-gripper test and the articulated-arm pilot agree in standard
   conditions. They disagree under heavy loads, enough to reverse load-tolerance contrasts in
   development. The articulated pilot has its own simplifications:
   - the cell model's link inertias;
   - no arm collision geometry;
   - a joint-space lift;
   - a shake through the base joint;
   - computed-torque control with an exact model;
   - no joint friction, backlash or sensor noise.
6. **Collision geometry.** GSO parts use the dataset's collision models (at most 32 pieces). EGAD parts
   use CoACD with settings chosen for time on excluded objects: hull volume about 1.2 times the mesh
   volume. Contact against convex pieces is not contact against the true surface.
7. **Rotation threshold.** 15° is a defensible, a priori choice but not a calibrated one. 5/10/20° are
   reported, and the 5° threshold changes the load-tolerance ranking.

## Learning and representations

8. **Development data size.**
   - 316 parts for feasibility and 151 for dynamics.
   - Robustness labels (P(H|F)) are v2 labels (rise and slip only) for all viable candidates. v3 labels
     exist only for first grasps, because v3 was introduced after the sweep.
9. **Foundation-model features** were tested only as frozen global descriptors of grey renders (and
   patch tokens at contacts). No fine-tuning, no real images, no textures. The negative result is
   bounded to this use.
10. **Post hoc elements, declared.**
    - The L10 controls (L10p, L10r) were added after L10's result.
    - The H5 target changed from L8 to L10.
    - The primary family changed after the articulated pilot.

    All of this happened before the confirmation and is recorded in PHASE7_PROTOCOL.md II.8.

## Confirmation

11. **EGAD is a different population**: evolved shapes scaled to the gripper, not household objects.
    That makes it a demanding generalisation test, but also limits what it says about service-robot
    objects.
12. **Disclosed deviation D1.** The cells of two confirmatory objects were built (geometry only) during
    pipeline development.
13. **Underpowered secondary hypotheses.** H2 (M9 > M1 robustness) had 0.43 power at 200 viable objects;
    a non-significant result there is not evidence of no effect.
14. **Costs.**
    - Wall-clock timings are specific to this container (shared CPU, one thread).
    - The current implementation runs candidate generation twice (an external process, then the
      planner). The full online cost therefore penalises the learned orders relative to an integrated
      implementation; that estimate is reported separately.

## What the confirmation showed (added after the test; the frozen analysis is unchanged)

16. **The development robustness and load-tolerance gains did not replicate on EGAD.**
    - Robustness was −1.8 pp [−5.0, 1.4] vs the original planner. That is non-inferior only narrowly,
      and only at the pre-registered torsion: it was −6.4 at 0.001 m and −2.7 at 0.02 m, both below zero.
    - Load tolerance was −6.4 pp [−10.9, −1.8].
    - No claim of improved robustness can be made.
17. **Efficiency gains shrink with a better baseline.** The N_F ratio vs the original planner was 0.61 on
    EGAD vs 0.21 in development. The original planner's order is already efficient on gripper-scaled
    shapes.
18. **Population coverage.** 80 of 300 primary objects had no viable candidate at all. Ordering cannot help
    them, and they lower every method's population success equally.
19. **Wall-clock.** The full online cost was 1.94× the original planner's (60 objects). An integrated
    implementation was not built or measured on EGAD.
20. **Execution incident.** The container restarted once during the confirmation physics. It was resumed
    from per-object outputs with the same frozen code; objects in progress were re-simulated from
    scratch. No outcome-dependent rerun.

## External baselines

15. **No learned grasp detector was reproduced.**
    - GPD needs PCL/OpenCV and a C++ adapter to score AxisForge candidates.
    - Contact-GraspNet needs a CUDA GPU.
    - Dex-Net GQ-CNN needs TensorFlow ≤ 1.15 and Python ≤ 3.7.

    The only external comparison is the classical analytic ranking (force closure, then wrench-space
    epsilon) on a development subset. It is a reimplementation of a criterion, not a reproduction of a
    published system.

## Not done

- Calibration against hardware.
- A real robot.
- Clutter.
- Objects outside the gripper scale.
- Other grippers in the confirmation.
- Few-shot adaptation, DKL or ADKF-IFT (not justified by development, see `stage_status.json`).
