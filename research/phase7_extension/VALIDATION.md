# Validation record (Phase VII extension)

This record lists what was checked, against what, with fixed tolerances, and the results. Each row names its
results file in `research/phase7_extension/results/`. The test suites are recorded separately in
`TEST_RESULTS.md`.

## Native implementation (Rust) against the reference models (scikit-learn)

Tolerances were fixed before running: features within 1e-12 absolute, probabilities within 1e-9, and identical
decisions (the first viable grasp and N_F) for every object and mode.

| Check | Population | Result | File |
|---|---|---|---|
| Features, bounds and probabilities of the historical models (P(F), P(H\|F)) | 316 development parts and 345 EGAD objects; 786,760 candidates | maximum absolute difference 0 for each | `native_inference_validation.json` |
| Decisions, historical models: m1, m4, m9, joint, hybrid | 151 development and 255 EGAD viable objects | 406 / 406 identical per mode | same |
| The same checks for X (Joint-multi: P(F) and the all-three-torsion robustness model), modes joint and hybrid | the same populations | maximum difference 0; 406 / 406 identical per mode | `native_inference_validation_x.json` |
| The exported X model against `predict_proba` on random inputs with missing values | 200 rows | identical | `tests/phase7_extension/test_extension.py::test_exported_x_model_equals_scikit_learn` |
| The clean container: native joint and hybrid decisions on EGAD | 255 objects | 510 / 510 identical | `reproducibility/DOCKER_VALIDATION_RECORD.md` |

Each model is checked on its own. Equivalence of the historical models was not extrapolated to X.

The frozen P(F) was trained on all development parts. Development probabilities are therefore in-sample: the
check establishes equivalence, not accuracy.

## Fallback bound

**The bound.** `N_hybrid ≤ min(2 N_X − 1, 2 N_M0) ≤ 2 min(N_X, N_M0)`. It holds when:

- both orders are permutations of the same candidate set;
- feasibility does not depend on what was examined before;
- the object has at least one feasible candidate.

| Check | Result | File |
|---|---|---|
| Exhaustive over all orders of 5 candidates (Rust) | holds | `crates/grasping/src/learned.rs` tests |
| 2000 random catalogues (Rust property test), 3000 (Python reference) | holds | the same, and `tests/phase7_extension` |
| Native planner on every viable object | 151 development and 255 EGAD objects: 0 violations; worst hybrid/M0 ratio 2.0 | `fallback_audit.json` |
| Second confirmation | reported in `confirmation_x.json` (`descriptive.fallback`) | — |

**Scope.** Objects without a feasible candidate, and exhausted budgets, are reported separately. The bound
concerns examined candidates only.

## Statistics

| Check | Result | File |
|---|---|---|
| Tango's restricted MLE (closed form) against direct numerical maximisation | agree | `test_tango_restricted_mle_matches_direct_maximisation` |
| Coverage of the Tango and Newcombe intervals by simulation | near nominal | `test_tango_and_newcombe_intervals_have_near_nominal_coverage` |
| Tango 1998 and Newcombe 1998: bibliographic records | matched in Crossref (DOIs in `references_added.json`) | `references_added.json` |
| P2 audit and corrections addendum recomputed from committed per-object data, in the clean container | identical | `DOCKER_VALIDATION_RECORD.md` |
| Frozen analysis of the second confirmation on synthetic data (40 objects) | runs; counts, Holm, fallback and cluster outputs consistent | `test_confirmation_analysis_runs_on_synthetic_data` |
| Type I error of the planned tests at the margins (power simulation) | ≤ 0.013 in every scenario | `power/power_*.json` |

## Freeze and data integrity

| Check | Result | File |
|---|---|---|
| The Phase VII freeze at `de22a8a` | every recorded digest matches | `freeze_verification.json` |
| Figures quoted at the start of the extension | 14 / 14 match | `start_verification.json` |
| New set untouched before the freeze: no shared cell, SHA-256 or name in any file or commit | confirmed, apart from the declared preparation of position 300 | `new_set_untouched.json` |
| `PHASE7X_FREEZE.json`: 36 files, the planner binary and the GPD adapter | checked before and after every stage of the second confirmation | the run log and `confirmation_x.json` |

## Physics

| Check | Result | File |
|---|---|---|
| v3 criterion: Phase VII validation (rotation, contact, controls) | unchanged | `research/phase7/results/physical_validation.json`, `rotation_validation.json` |
| Articulated v2 controls: jaws open, frictionless contacts | 0 / 151 lifted for each, as expected | `articulated2_development.json` |
| Articulated v2 positive control (gravity off, 2 kg) | 118 / 151 held: **not passed as specified**, see `ARTICULATED_V2.md` | same |

## Package

| Check | Result |
|---|---|
| `tools/package/verify_package.py` on the built package: DINOv3 arrays, YCB rows, YCB geometry, manuscript files, hashes, unlisted files | 895 files, no findings |

## Not validated

- Agreement with a real robot.
- Calibration of the friction parameters.
- GPD's published results: only its classifier is used, on our candidates.
