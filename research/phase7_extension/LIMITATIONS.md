# Limitations (Phase VII and extension)

Ordered by how much each one limits the conclusions. The Phase VII list (`research/phase7/limitations.md`) still
holds, apart from the items revised below; it is kept unchanged as the historical record. Every item here is
stated in the manuscript or in the supplement.

## Scope of the evidence

1. **Simulation only.** No robot was used.
   - "Held", "robust" and "load tolerant" are MuJoCo outcomes under scenarios: an assumed mass of 0.1 kg,
     uniform density, friction μ 0.5, torsional friction 0.005 m, and an idealised controller.
   - None of these parameters is calibrated against hardware.
   - They are not real-world success rates.
2. **Feasibility is not robustness.** N_F counts candidates until one passes inverse kinematics, collision and a
   static wrench test. Robustness is measured separately, and the two are never merged.
3. **One cell and one gripper.** A six-axis arm, an 80 mm parallel jaw, and the part at one table position.
   The 40 and 120 mm results are development-only stroke variants of the same cell.
4. **One object family for both confirmations.**
   - EGAD objects are evolved shapes scaled to the gripper, not household objects.
   - The second confirmation draws more cells from the same EGAD grid. It is a second sample of the same
     family, not a new domain. Neighbouring cells can resemble each other, which a cluster-bootstrap
     sensitivity analysis probes but cannot rule out.

## Physics

5. **Uncalibrated torsional friction.** Between 6 % and 18 % of verdicts change across 0.001–0.02 m. Load
   tolerance is the most affected.
6. **Simulator level.**
   - The floating-gripper and articulated-arm models agree in the standard conditions and disagree under
     heavy loads.
   - The articulated v2 run (`ARTICULATED_V2.md`) passed its negative controls. Its positive control
     (gravity off, 2 kg) was not passed as specified: 118 of 151 held.
   - It is therefore not a validated load-tolerance instrument.
7. **Collision geometry.** EGAD objects use CoACD pieces whose hull volume is about 1.2 times the mesh volume,
   so contacts are against convex pieces.
8. **The rotation threshold** of 15° was fixed a priori but is not calibrated.

## Learning and selection

9. **The second method was selected after the first confirmation.**
   - The selection rule was committed before any v3 model was fitted, and the selection used development
     data only.
   - But the questions asked (robustness across torsions) were shaped by the first confirmation's results.
   - The development advantage of the selected method (+25.2 pp in robustness at all three torsions over
     M1) is optimistic, because it was selected on the same parts.
10. **Label coverage.** The v3 labels cover 4,497 simulated candidates out of 12,585 viable ones (35.7 %).
    N_FR (candidates until the first robust one) is computed over simulated candidates only.
11. **Foundation-model features** were tested only as frozen descriptors of grey renders and patch tokens at
    contacts. The negative result is bounded to that use.
12. **The Optuna search** (exploratory, reduced budget) was not used for any selection or claim. Its absence of
    a gain is not evidence of equivalence.

## Statistics

13. **The first confirmation's P2** (robustness non-inferiority) passed the pre-specified test narrowly. It does
    not pass score-based tests (Tango, Newcombe), and changing one object's outcome reverses it. It is not
    evidence of equal robustness.
14. **Power.** The second confirmation's C2 (robustness) and C4 (time) were underpowered if the effects
    resemble the historical EGAD ones (power 0.21 and 0.54 under that scenario). A non-supported C2 or C4 is
    not evidence of inferiority or of equal cost.
15. **Rank tests versus geometric means.** The Wilcoxon results test a location shift, not the geometric-mean
    ratio, and are reported only as sensitivity analyses.

## Cost and implementation

16. **Timing is machine-specific.** It was measured in a shared cloud container (Intel Xeon at 2.30 GHz),
    with one pinned CPU and one thread.
    - During the development benchmark other processes were running on the other cores.
    - The second confirmation's timing stage was run with the other jobs stopped, and the load is recorded.
17. **The development benchmark is not confirmatory.** It uses development parts and Phase VII EGAD objects,
    which had already been evaluated.
18. **The integrated cost is not a guarantee.** The fallback bound concerns examined candidates. It bounds
    neither time, robustness, safety nor real success.

## External comparators

19. **GPD at the ranking-controlled level only.**
    - GPD's pretrained classifier ranks the shared candidates.
    - Its hand geometry differs from the 80 mm jaw.
    - Its scores are random between runs (median correlation 0.74 on 20 parts).
    - GPD's full pipeline was not run.
20. **Dex-Net 2.0 and Contact-GraspNet were not run.** The official Dex-Net weights return 404, and
    Contact-GraspNet needs a CUDA GPU. No reimplementation or unofficial weights were used.

## Process

21. **AI assistance.** The implementation, experiments, analyses and text were produced with substantial
    assistance from AI systems (`AI_USAGE.md`). The internal reviews are not independent peer review. The
    author's review and approval are pending.
22. **Declared deviations.**
    - Phase VII: D1–D4.
    - Extension: the mesh of position 300 was prepared, with no planning or simulation, before the freeze.
23. **Execution incidents.** The container restarted during the second confirmation. Stages were resumed from
    per-object outputs with the frozen code and binary; objects in progress were redone from scratch. No
    outcome-dependent rerun.

## Not done

- Calibration against hardware.
- A real robot.
- A full reproduction in a clean container (only the statistical and partial levels).
