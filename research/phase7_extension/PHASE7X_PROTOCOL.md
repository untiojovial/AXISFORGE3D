# Phase VII extension: protocol of the new independent confirmation (EGAD positions 300–499)

**Status: DRAFT, not frozen.** Sections marked ⟨selection⟩ are completed once the development selection rule
(`DEVELOPMENT_SELECTION_RULE.md`, committed before any v3 model was fitted) has been applied. The protocol is
frozen afterwards, before any object of the new set is catalogued or planned. Freezing means committing this
file, the frozen models and the code, and recording their SHA-256 and the commit in `PHASE7X_FREEZE.json`.

## 1. Relation to Phase VII

- **Phase VII stays as it is.** Its protocol, frozen models, confirmatory results and reported conclusions are
  not modified. This is an extension: a second confirmation of a method chosen afterwards on development data.
  It is not a re-run of Phase VII.
- **Same family, not a new domain.** The new objects come from the same EGAD training-set family as the Phase
  VII primary set. They are positions 300–499 of the same seeded draw.
  - `results/new_set_untouched.json` shows that none of them was built, planned, simulated or mentioned before:
    no shared cell, no shared SHA-256, no name in any file or commit.
  - They are distinct cells. Cells of the EGAD grid differ in shape complexity and grasp difficulty, but
    neighbouring cells can resemble each other.
- **What the authors had already seen.** The Phase VII EGAD results (positions 0–299) were known when this
  extension was designed. The method to confirm is selected on development data only, but knowledge of the
  EGAD results shaped which questions are asked. That is a limitation of any second confirmation, and it is
  reported as one.

## 2. Population, preparation and inclusion

- **Objects.** 200: positions 300–499 of `research/phase7/egad_selection.json`, i.e. cells `C01` … `X07`.
- **Preparation.** The frozen Phase VII rules (`tools/research/phase7/egad_prepare.py`, `build_one`):
  - the dataset authors' scale rule for an 80 mm gripper;
  - the most probable stable pose;
  - the demonstration cell, part at (0.42, 0, 0.42) m, mass 0.1 kg (assumed);
  - CoACD collision pieces with the Phase VII settings.

  The restart-safe wrapper `tools/research/phase7_extension/build_newset.py` calls that function unchanged.
  Preparation is mesh processing only. Position 300 was prepared while the wrapper was being tested, the others
  after; no planning or simulation is involved.
- **Catalogues.** Each object is catalogued once with the frozen Phase VII procedure
  (`tools/research/phase7/egad_confirm.py prepare`: every candidate of the 80 mm generator is checked with the
  native IK, collision and static-feasibility tests).
- **Statuses.**
  - *viable*: at least one feasible candidate;
  - *no viable candidate*;
  - *cell build refused*, *catalogue error*, *planner budget exhausted*: recorded and **not replaced**. No
    reserve set is opened.
- **Unit and populations.** The unit is the object.
  - Efficiency and robustness hypotheses use the viable objects (conditional rates).
  - Cost and population metrics use every catalogued object.

## 3. Methods (all frozen before cataloguing)

| Name | Definition | Models |
|---|---|---|
| M0 | the original planner order | — |
| M1 | preliminary + 2 p̃ | Phase VII frozen P(F) (`research/phase7/frozen/p_feasible.pkl`) |
| Joint-hist | p̃ · P(H\|F) · max(Q_cen, 0.001)^0.3 | Phase VII frozen P(F) and P(H\|F) (v2 labels) |
| **X** ⟨selection⟩ | the order selected in development | ⟨frozen file, hash⟩ |
| hybrid X/M0 | round robin of X and M0, skipping repeats | — |
| GPD+gate | GPD's classifier scores (adapter, commit `6327f20`), stroke gate first | GPD's own weights |

**Implementation.** Orders, first grasps and N_F come from the native planner. They are computed with
`plan --learned FILE --order MODE` and the exported, exact-bit models. The Python replay with scikit-learn is
run as a check; any mismatch is reported, not corrected. GPD orders come from the adapter, as in
`external_baselines_v2.json`.

## 4. Outcomes

- **N_F**: the number of candidates examined until the first feasible one. Success@k: N_F ≤ k for
  k = 1, 5, 10, 20, 50.
- **Robust (v3)**: the first grasp is `held_v3` in the six standard conditions at torsional friction 0.005 m
  (nominal, mass ×2, mass ×0.5, friction 0.3, pose noise 1 and 2). `held_v3` means rise ≥ 80 mm, slip < 20 mm,
  rotation ≤ 15° and contact kept; an unstable start counts as a failure. These are simulation scenarios, not
  calibrated values.
- **Robust across torsions (all3)**: robust at torsions 0.001, 0.005 and 0.02 m.
- **Load tolerant**: held at 0.5, 1 and 2 kg at 0.005 m.
- **Full cost**: wall time of a fresh planner process per object, from start to answer, including project and
  model loading. The integrated native implementation is used, one thread pinned to one CPU, modes in a seeded
  random order. Each object is run 3 times and the median is used. Planner stage timings and warm-process
  times are also reported.

## 5. Primary hypotheses ⟨selection: completed with X⟩

One-sided α = 0.025, Holm over the primary family.

| | Hypothesis | Population | Test | Supported if |
|---|---|---|---|---|
| C1 | X examines fewer candidates than M0 | viable | Wilcoxon signed-rank on log(N_F(X)/N_F(M0)), one-sided | Holm p < 0.025 and the geometric-mean ratio < 1 |
| C2 | X's v3 robustness is non-inferior to M0's, margin −5 pp | viable | **Tango score test** for a paired difference at −5 pp; Tango 95 % CI (lesson of the Phase VII P2 audit: the sample-SD z test is not used) | Holm p < 0.025 and Tango lower 95 % limit > −5 pp |
| C3 | ⟨selection⟩ | viable | ⟨selection⟩ | ⟨selection⟩ |
| C4 | the integrated implementation of X costs less time than M0 | all catalogued | Wilcoxon signed-rank on log(T(X)/T(M0)) of the per-object medians, one-sided | Holm p < 0.025 and the geometric-mean ratio < 1 |

The margin of C2 is Phase VII's. It was fixed before Phase VII's results and is not changed here.

## 6. Secondary analyses (Holm within the family; reported whatever the outcome)

- **S1.** Load tolerance, X vs M0: non-inferiority, margin −10 pp (Phase VII's margin), Tango score test.
- **S2.** X vs Joint-hist: robust v3 and load tolerance (exact McNemar, two-sided, descriptive), N_F ratio.
- **S3.** GPD+gate vs M0 and vs X: N_F ratio (Wilcoxon, two-sided), robust v3 difference (Tango CI), cost.
- **S4.** Fallback: violations of N_hybrid ≤ 2·min(N_X, N_M0), expected to be zero; hybrid N_F and robustness.
- **S5.** Torsion sensitivity: X − M0 robustness at 0.001 and 0.02 m (Tango CIs).
- **S6.** Warm-process cost ratio and planner-only cost ratio.

## 7. Sample size and power ⟨selection⟩

N = 200 is fixed by the positions that were reserved, so there is no sequential addition. The expected
viable share is 73 % (Phase VII), about 147 objects. Power by resampling, `results/power_new.json`, under the
observed, conservative and null scenarios: ⟨to be filled before freezing⟩. If the power of a primary
hypothesis is low, the protocol says so before the run, and its result is read accordingly. The margins are not
changed to gain power.

## 8. Run once

Every stage is run once, in this order:

1. build;
2. catalogue;
3. native orders;
4. physics;
5. GPD;
6. timing;
7. analysis.

The analysis script is frozen with the protocol. Results are reported as they come, negative ones included.
Crashed jobs are resumed (stages are restart-safe), not re-sampled. Timing is run with nothing else on its
pinned CPU, and the machine load is recorded. If the machine could not be quiet, that is reported.
