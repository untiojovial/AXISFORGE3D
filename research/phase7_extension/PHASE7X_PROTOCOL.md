# Phase VII extension: protocol of the new independent confirmation (EGAD positions 300–499)

**Status: final text, frozen by `PHASE7X_FREEZE.json`.** "Frozen" means that this file, the models, the code
and the compiled binary are committed in Git and their SHA-256 digests are recorded before any object of the
new set is catalogued, planned, scored or simulated. These are not third-party signatures, and this is not a
formal preregistration with an external registry. The development selection rule (`DEVELOPMENT_SELECTION_RULE.md`,
commit `49eb5bb`) was committed before any v3 model was fitted. Its result is in `results/v3_models.json`.

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
- **Chronology and a declared deviation.**
  - `DEVELOPMENT_SELECTION_RULE.md` said the protocol would be fixed "before the new set is built".
  - One object, position 300 (`egad_C01`), was prepared earlier: on 2026-11-26 between 17:59:17 and 17:59:22
    UTC, while the restart-safe build wrapper was being tested. This was mesh scaling, pose and collision
    decomposition only.
  - No object of the new set was catalogued, planned, scored by any model, simulated, or used for any selection
    before this protocol was frozen (`results/new_set_untouched.json` and the freeze checks).
  - Every other object is built only after the freeze. The deviation is declared here and in the article.
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
| **X = Joint-multi** | p̃ · P(robust at all three torsions \| F) · max(Q_cen, 0.001)^0.3. Selected by the rule: eligible (N_F 1.324 ≤ 1.10 × M1's 1.254); best y_all3 in development (0.821 vs M1 0.570; optimistic, because it was selected on the same parts) | Phase VII P(F) plus `research/phase7_extension/frozen/p_robust_all3.pkl`. The robustness model is gradient boosting (Phase V hyperparameters, seed 0) trained on y_all3 of the 4322 v3-sweep simulations of the 151 development parts. Exported exact-bit as `frozen/learned_order_x.json`. |
| hybrid X/M0 | round robin of X and M0, skipping repeats (native `--order hybrid` with X's file) | — |
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

## 5. Primary hypotheses

One-sided α = 0.025, with Holm over the four hypotheses.

| | Hypothesis | Population | Estimand | Test | Supported if |
|---|---|---|---|---|---|
| C1 | X examines fewer candidates than M0 | viable | geometric-mean ratio N_F(X)/N_F(M0) | one-sided t test on the per-object log ratios (H0: mean ≥ 0), 95 % t CI | Holm p < 0.025 and ratio < 1 |
| C2 | X's v3 robustness is non-inferior to M0's, margin −5 pp | viable | difference of paired proportions robust(X) − robust(M0) | **Tango score test** at −5 pp, Tango 95 % CI (Newcombe's hybrid score CI as sensitivity) | Holm p < 0.025 and Tango lower 95 % limit > −5 pp |
| C3 | X is non-inferior to M1 in examined candidates, ratio margin 1.25 | viable | geometric-mean ratio N_F(X)/N_F(M1) | one-sided t test of mean log ratio < log 1.25, 95 % t CI | Holm p < 0.025 and the upper 97.5 % limit < 1.25 |
| C4 | the integrated implementation of X costs less time than M0 | all catalogued (timed) objects | geometric-mean ratio of per-object median cold-process times T(X)/T(M0) | one-sided t test on the log ratios, 95 % t CI | Holm p < 0.025 and ratio < 1 |

- **C3's margin.** 1.25 is Phase VII's P3 margin, fixed before Phase VII's results. X ≠ M1, so C3 is not redundant.
  It asks whether adding the multi-torsion robustness term keeps efficiency close to the plain learned order.
- **C2's margin.** −5 pp, Phase VII's, unchanged. The Tango test replaces the sample-SD z test after the Phase VII
  P2 audit, which showed that test to be anti-conservative with few discordant pairs.
- **Ties and ratios.**
  - N_F ≥ 1 always, so log ratios are finite.
  - Ties (log ratio 0) stay in the t tests.
  - The Wilcoxon signed-rank test with zero differences dropped is reported for C1 and C4 as a rank-based
    sensitivity analysis. It tests a location shift, not the geometric mean, and it does not decide support.
- **Counts reported for every hypothesis.** Favourable, unfavourable and tied objects; and for C2, the paired
  table: both robust, X only, M0 only, neither.

## 6. Secondary analyses

**Holm within the secondary family, one-sided α = 0.025.**

- **S1.** Load tolerance (0.5, 1 and 2 kg), X vs M0: non-inferiority at −10 pp (Phase VII's margin), Tango score test.
- **S2.** Robustness across the three torsions (y_all3, the selection criterion), X > M1: exact one-sided McNemar.
- **S3.** y_all3, X > M0: exact one-sided McNemar.

**Descriptive** (no claims beyond the intervals):

- X vs Joint-hist: robust v3 and load (two-sided McNemar, Tango CIs), N_F ratio;
- X − M0 robustness at torsions 0.001 and 0.02 (Tango CIs);
- GPD+gate vs M0: N_F ratio, robust v3, load. This is a ranking-controlled comparison using GPD's pretrained
  classifier on a shared candidate catalogue, not a comparison of the full GPD and AxisForge3D pipelines;
- the fallback: violations of N_hybrid ≤ min(2 N_X − 1, 2 N_M0); expected 0;
- rates (N_F, Success@k, robust v3, all3, load) and timing percentiles (median, p90, p95, p99);
- **cluster sensitivity**: cluster-bootstrap CIs of the C1–C4 effects with clusters = the difficulty letter of the
  EGAD cell. Neighbouring cells can be similar, so this probes dependence. It does not replace the primary
  analysis.

## 7. Sample size and power

N = 200 is fixed by the reserved positions. The expected viable share is 73 % (Phase VII), about 147 objects.
Power was estimated by resampling (`power/power_egad.json`, `power/power_dev.json`; 2000 replicates, Holm,
one-sided α = 0.025) from two exploratory planning sources:

| Hypothesis | Pessimistic stand-in: EGAD Phase VII, Joint-hist | Development, X itself (optimistic) | Development, half effect |
|---|---|---|---|
| C1 | 1.00 | 1.00 | 1.00 |
| C2 | 0.21 | 0.997 | 0.53 |
| C3 | 1.00 | 0.88 | 0.34 |
| C4 | 0.54 | 1.00 | 0.73 |
| all four | 0.16 | 0.88 | 0.23 |

Type I error at the margins was ≤ 0.013 in every case.

**Reading.**

- C1 is well powered.
- C2 and C4 are underpowered if the effects look like the historical EGAD ones: the robustness gain of
  development may not transfer, and EGAD's original planner is already efficient.
- A non-supported C2 or C4 is therefore not evidence of inferiority or of equal cost. It is reported with its
  intervals.
- The margins are not changed to gain power.

## 8. Run once

Every stage is run once, in this order:

1. build;
2. catalogue;
3. GPD scoring;
4. native orders;
5. physics;
6. timing;
7. analysis.

Each stage starts by checking the freeze, including the binary's hash. Physics runs the nine conditions at
torsion 0.005 m for every method's first grasp, and at 0.001 and 0.02 m for M0, M1, Joint-hist and X. Timing
covers M0, M1, Joint-hist, X and the hybrid: three cold repetitions per object, seed 20261217.

The analysis script is frozen with the protocol. Results are reported as they come, negative ones included.
Crashed jobs are resumed (stages are restart-safe), not re-sampled. Timing is run with nothing else on its
pinned CPU, and the machine load is recorded. If the machine could not be quiet, that is reported.
