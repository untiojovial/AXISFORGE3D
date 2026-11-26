# Development selection rule for the robustness models and the final ordering (fixed before results)

Written and committed on 2026-11-26, before any v3-label model or any new fusion variant was trained or
evaluated. At that point the v3 label sweep (`tools/research/phase7_extension/v3_sweep.py`) was running,
and no model had been fitted on its output. This rule governs sections 5–7 of the extension. Everything
here is development only: the 151 dynamic development parts, out of fold, with the Phase VII folds.

## Robustness labels (section 5)

**Label.** `held_v3` in all six standard conditions:

- conditions: nominal, mass ×2, mass ×0.5, friction 0.3, pose noise 1, pose noise 2;
- `held_v3` means rise ≥ 80 mm, slip < 20 mm, rotation ≤ 15° and contact kept, with an unstable start
  counted as a failure.

The label comes from fresh v3 simulations of the development viable candidates sampled by the sweep. The
sample is at most 30 candidates per part, seeded, plus the first grasps of M0, M1, M4, M9 and Joint.
Historical v2 telemetry is not converted into v3 labels: it lacks the rotation and contact records.

**Main and multi-configuration labels.**

| Name | Definition |
|---|---|
| `y_std` (main) | the label above, at torsion 0.005 m |
| `y_tor` | `y_std` at each of the torsions 0.001, 0.005 and 0.02 m |
| `y_all3` | `y_tor` at all three torsions |
| `r_mean` | fraction held over the 18 standard trials (6 conditions × 3 torsions) |
| `r_ext` | the three extended scenarios at 0.005 m: friction 0.7, centre of mass ±5 mm |
| `y_load` | held at 0.5, 1 and 2 kg at 0.005 m |

These parameter values are scenarios. They are not calibrated against a real gripper.

## Models compared for P(H|F) (section 5.1)

**Models.**

1. Historical gradient boosting with v2 labels: Phase VII out-of-fold models, as used by M9 and Joint.
2. The same gradient boosting (`bl.fit`, Phase V hyperparameters) with `y_std` labels.
3. HistGradientBoosting with scikit-learn defaults and `y_std` labels.
4. L2 logistic regression on standardised features, C = 1.
5. A geometric MLP: `research_grasping/phase7/representation.py` architecture with the v and u blocks,
   fixed hyperparameters, seeds 0–2.
6. The same gradient boosting trained on `y_all3`: the multi-configuration model.
7. The same gradient boosting trained on `y_load`: a separate load model.

A visual MLP is not run. Phase VII found no visual gain for P(F), so there is no defensible reason to
expect one here; this is recorded as a decision, not as a result.

**Evaluation.**

- Folds: the 5 Phase VII development folds (seed 20261130). All fitting happens inside the training folds.
- Training rows: the simulated viable candidates of the training parts.
- Probability metrics, out of fold on the held-out simulated candidates: AUC, PR-AUC, Brier score,
  log-loss, ECE with 10 equal-width bins, and reliability-diagram data.
- Ordering metrics, on the held-out parts, through the fusion order: robustness of the first grasp
  (`y_std`, `y_all3`, `y_load`), N_F, N_FR (candidates until the first robust grasp, among simulated ones),
  and Success@1/5/10/20/50.

**Not a selection criterion.** No model is chosen by AUC alone.

## Orders compared (section 7)

| Group | Orders |
|---|---|
| References | M0, M1, M4, M9 (historical), Joint (historical, β per fold as in Phase VII) |
| Joint-v3 | `S = p̃ · h_v3 · max(Q_cen, 0.001)^0.3`, the historical form with h from model 2 |
| Joint-v3 with other h | the same form with h from models 3, 4 and 5 |
| Joint-multi | the same form with h from model 6 (`y_all3`) |
| Joint-load | `S = p̃ · h_std · h_load · max(Q_cen, 0.001)^0.3` |

**Combinations with `h = h_v3`.**

- arithmetic: `(p̃ + h)/2 · max(Q, 0.001)^0.3`;
- geometric: `sqrt(p̃·h) · max(Q, 0.001)^0.3`;
- lexicographic: first `h ≥ 0.5`, then by `p̃`;
- multi-objective: non-dominated sorting on (`p̃`, `h`, `Q`), ties by `p̃·h`.

**Ablations of Joint-v3.**

- without the centring term (β = 0);
- without h (this is M4-like with β = 0.3);
- β ∈ {0.1, 0.5}, reported only, never selected.

Every order puts makeable candidates first, through the planner gate (`fusion.order`).

## Selection rule (section 7), applied once to the development results

1. **Efficiency guard.** A variant is eligible only if its development N_F geometric mean is ≤ 1.10 ×
   that of M1.
2. **Primary criterion.** Robustness of the first grasp across torsions (`y_all3`), the property that failed
   in Phase VII: robustness held only at the pre-specified torsion.
3. **Candidate.** The eligible variant with the highest `y_all3` rate is the candidate.
4. **Margin over M1.** The candidate replaces M1 as the method to confirm only if its paired development
   advantage over M1 on `y_all3` is at least +2 percentage points. Otherwise **M1 is the final ordering**,
   and the report states that no robustness variant beat M1 in development.
5. **Ties.** Within 1 point on `y_all3`, the simpler variant wins: fewer models, then gradient boosting over
   MLP, then the historical form over a new combination.
6. **No further search.** The β grid and the combination list above are the whole search; nothing is added
   after results are seen. Any later addition is labelled post hoc and cannot be selected.

## What the new confirmation will compare (sections 13 and 4)

The method selected here, under the name it gets at selection, is compared on the new set with:

- M0 (original planner);
- M1;
- historical Joint (frozen Phase VII models);
- GPD+gate (secondary, external).

The hypotheses, margins and tests are fixed in the new protocol, after this selection and before the new
set is built.
