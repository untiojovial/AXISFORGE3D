# Phase VII — protocol

Part I (development) is fixed before any Phase VII development experiment runs. Part II
(confirmation) will be appended and committed **before** the planner, the physics or any model
touches a confirmatory object. Integrity rules: those of the Phase VII request (sections 3 and 36), the
15 Phase VI principles, and no change to Phases I–VI.

## Part I — development

### I.1 Data roles

| Set | Parts | Role |
|---|---:|---|
| Frozen library (Phase V) | 116 (16 YCB, 100 GSO) | Development |
| Phase V test parts | 200 (151 viable at 80 mm) | Development (used in Phases V and VI) |
| Phase III sealed objects | 8 | Untouched |
| EGAD (Morrison et al. 2020, BSD-3) | 2 281 training-set + 49 evaluation-set meshes, downloaded 2026-11-26 | **Confirmatory candidate pool**. Before Part II, only metadata and geometric inclusion checks are allowed: no planning, physics, rendering or model inference |

Feasibility labels exist for all 316 development parts (exhaustive planner, 80 mm). Dynamic labels
(Phase VI v2 sweep) exist for the 151 viable Phase V test parts.

### I.2 Physics v3 (fixed now)

[`research_grasping/phase7/physics.py`](../../research_grasping/phase7/physics.py) is Phase VI's v2
scene (planner centre of mass, intended masses) with a corrected success criterion:

- `held_v3` requires all of:
  - rise ≥ 80 mm;
  - slip < 20 mm after the lift;
  - rotation of the part relative to the palm **from the end of the grip phase to the end** ≤ **15°**;
  - pads still touching at the end;
  - no numerical instability.
- **Why 15°**: it moves a point 80 mm from the closing axis by 20 mm, matching the existing linear
  tolerances. It was fixed before any per-method result.
- **Sensitivity only**: 5°, 10° and 20°.
- **Historical criterion**: the v2 criterion (rise and slip only) is kept and reported as such.

Conditions (unchanged from Phase VI):

- **Standard**:
  - nominal (0.1 kg, μ 0.5, torsional 0.005 m);
  - mass ×2 (0.2 kg);
  - mass ×0.5 (0.05 kg);
  - μ 0.3;
  - two pose-noise draws.
- **Stress loads**: 0.5, 1 and 2 kg.

Two summaries:

- **Robust**: holds in all six standard conditions.
- **Load tolerant**: holds at 0.5, 1 and 2 kg.

The masses are assumed, not measured, for every object.

**Torsional friction sensitivity** (MuJoCo `condim=4`, friction = [tangential μ, torsional, rolling]
with torsional in metres):

- values 0.001, 0.0025, 0.005 (default) and 0.02;
- for the first viable grasp of M0, M1, M4 and M9 on the 151 development parts, in all nine conditions.

These values are not calibrated against hardware.

### I.3 Mandatory representation experiment (DINO + MLP)

**Unit and splits.**

- 5-fold cross-validation grouped by part, seed 20261201, on all 316 development parts.
- Product-name variants are kept in the same fold (union of the Phase V name-variant pairs).
- Inside each training fold, 15 % of its groups form the inner validation set used only for early
  stopping.
- Standardisation, PCA and linear projections are fitted on the training fold only.

**Feature blocks.**

| Block | Content | Dim |
|---|---|---:|
| v(g, x, R) candidate and robot | approach xyz, \|closing\| xyz, width, relative position xyz, height, antipodal A/B, centre offset, preliminary score, makeable | 16 |
| u(x) object geometry | log bounding-box extents (3) + D2 shape distribution (32 bins) | 35 |
| d(x) object visual (global) | unit mean of the per-view CLS embeddings (Phase V descriptors): DINOv2 ViT-S/14, DINOv3 ViT-S/16, OpenCLIP ViT-B/32 (512), untrained ViT (DINOv3 architecture, seed 0) | 384/512 |

Every model receives v. A global embedding alone cannot rank the candidates of one part.

**Variants (all fixed now).**

| Id | Model | Inputs |
|---|---|---|
| L0 | Gradient boosting (Phase V settings) | v + log extents (the historical 19) |
| L1 | L2-regularised logistic regression (C = 1) | v + log extents |
| L2 | MLP | v, u |
| L2v | MLP (control: candidate only) | v |
| L3 | MLP with a linear visual branch | v, DINOv2 |
| L4 | MLP with a linear visual branch | v, DINOv3 |
| L5 | MLP | v, DINOv2 |
| L6 | MLP | v, DINOv3 |
| L7 | MLP | v, u, DINOv2 |
| L8 | MLP | v, u, DINOv3 |
| L9 | MLP | v, u, untrained ViT |
| L10 | MLP with a linear visual branch | v, u, DINOv3 |
| L11 | MLP | v, u, OpenCLIP |
| L8p | MLP (control: DINOv3 embeddings permuted across objects with one fixed permutation, seed 7) | v, u, permuted DINOv3 |

**Architecture (fixed, not tuned).**

- Branches:
  - grasp: v → 64 → 32;
  - geometry: u → 32 → k;
  - visual: d → 64 → k, or d → k for the linear branches.
- Head: concatenation → 64 → 1 logit.
- Layers and regularisation: ReLU, dropout 0.1, AdamW (lr 1e-3, weight decay 1e-4).
- Training: batch 4 096, at most 40 epochs, early stopping with patience 5 on the inner-validation loss.
- Latent size: k = 16.
- Seeds: 0, 1 and 2; per-part results are averaged over seeds.
- Pre-declared sensitivity: k ∈ {8, 32} for L2, L6 and L8.

**Losses.**

- BCE for every variant.
- Ranking loss for L2 and L8: listwise softmax cross-entropy per part, with the target uniform over the
  part's viable candidates. Parts without a viable candidate contribute only a BCE term with weight
  0.1.

**Ranking integration.** As for M1: planner score = preliminary + 2·p/max(p), stroke gate first, exact
order simulator.

**Metrics** on held-out parts:

- AUC (pooled and per-part median), log-loss and ECE;
- N_F (parts with a viable grasp);
- Success@k for k = 1, 5, 10, 20, 50, over all parts (parts without a viable grasp count as failures);
- Precision@k and Recall@k;
- for the 151 parts with dynamics: robustness of the first viable grasp (v2 standard labels) and N_FR;
- training time, parameters and inference latency.

**Development contrasts** (paired by part, bootstrap 95 % CI with 10 000 draws and seed 20261128,
Wilcoxon, Holm within each metric family):

| Contrast | Question |
|---|---|
| L8 vs L2 | Does DINOv3 add to the geometry MLP? |
| L7 vs L2 | Does DINOv2 add to the geometry MLP? |
| L8 vs L10 | Nonlinear vs linear visual branch |
| L8 vs L9 | Pretrained vs untrained ViT |
| L8 vs L8p | Real vs permuted embeddings |
| L2 vs L0 | MLP vs historical gradient boosting |
| L11 vs L2 | Does OpenCLIP add to the geometry MLP? |

A visual representation **adds value in development** only if its N_F ratio vs L2 has an upper 95 %
CI below 1 **and** it beats its permuted control.

**Robustness predictors P(H|F)** (151 parts, viable candidates, label = robust in the six standard
conditions, v2):

- Models: gradient boosting (M9 historical), MLP(v, u), MLP(v, u, DINOv3).
- Folds: Phase VI's outer folds (seed 20261130).
- Ordering: p̃·h.
- Metrics: robustness of the first viable grasp and N_FR.

**Local visual features** (run if correspondences can be computed):

- Rendering: 8 views at 224 px with the scale-normalised Phase III cameras.
- DINOv3 patch tokens (14×14 grid) bilinearly sampled at the projections of both contacts.
- Visibility: rendered depth within 3 mm of the contact depth.
- Visible views are averaged; contacts visible in no view get a zero vector and an indicator.
- PCA to 16 dimensions per contact, fitted on the training fold.
- Classical local-geometry baseline: per contact, the eigenvalue ratios (linearity, planarity,
  sphericity) of the mesh vertices within 10 mm and the angle between the local PCA normal and the
  contact normal.
- Variants: LG (v, u + local geometry), LD (v, u + local DINOv3) and LGD (both); same architecture and
  metrics.

### I.4 Fusion and robustness methods (development)

- M0, M1, M4 (α = 0.8, Q_cen) and M9 (p̃·P(H|F), gradient boosting, historical v2 label) as in Phase VI.
- **FORTIS-R Joint**: p̃ · h · Q_load^β with Q_load = Q_cen.
  - β ∈ {0.1, 0.2, 0.3, 0.5}, chosen by nested cross-validation (Phase VI folds).
  - Criterion: mean of log N_FR plus the share of first grasps that are not load tolerant, with equal
    weight on the two standardised terms. Fixed now.
- Comparators: arithmetic, geometric (the product above), lexicographic and Pareto variants of
  (p̃·h, Q_cen) as in the Phase VI grid.

### I.5 Costs

- The planner reports per-stage timings.
- Python prior construction is timed separately: features, gradient-boosting and MLP inference, render
  and DINO encoding.
- T_total = T_preparation + T_features + T_inference + T_search, from receiving the part to returning a
  grasp, for a new process and for an already-running process.
- Unit: the part.

### I.6 Few-shot, DKL and ADKF-IFT

They are run only if the development data show room for adaptation. The test is the distribution of N_F
under M1: if the planner already stops after one or two candidates for most parts, a support set
acquired during planning cannot pay back its cost. The decision and its evidence will be recorded in
`stage_status.json`.

### I.7 Cross-robot (development)

Models trained at 80 mm, applied zero-shot to the existing 40 mm and 120 mm catalogues of the 316 parts.
Compared with models trained for each gripper, with the same folds.

## Part II — confirmation (fixed before any confirmatory object is planned)

Written after every development experiment of Part I and before the planner, the physics, any model or
any renderer has touched a confirmatory EGAD object. It is committed together with
[PHASE7_MODEL_FREEZE.json](PHASE7_MODEL_FREEZE.json). Nothing below may change once the first
confirmatory catalogue is built; an error found later is handled by II.9.

### II.1 What has touched EGAD so far

- **Selection.** [egad_selection.json](egad_selection.json) (commit `c55ca8b`) fixes the seeded order
  of 599 primary cells and the 45 evaluation-set objects. It used metadata and geometric checks only.
- **Disclosed deviation D1.** The cells of primary positions 0 and 1 (L01, N23) were built while the
  preparation code was being written: scaling, stable pose, `add-object` and collision decomposition.
  Nothing was planned, simulated, rendered or predicted for them. Their geometric records (extents,
  number of collision pieces) were seen. They stay in the confirmatory set and are rebuilt with the
  final preparation code.
- **Smoke test.** Positions 500–502 (J17, J20, L08) went through the whole pipeline (build, catalogue,
  descriptors, orders, physics, articulated, timing, analysis) to test the code and choose the
  collision decomposition. **They are excluded from every confirmatory analysis**, as is every position
  ≥ 300.

### II.2 Confirmatory set

**Primary set.**

- Objects: EGAD training-set primary order positions **0–299 (300 objects)**, one mesh per grid cell
  (no near-identical siblings).
- Sample size, from development power
  ([phase7_statistical_power.json](results/phase7_statistical_power.json); paired resampling of the 151
  development parts, observed and shrunken effects):

  | Hypothesis | Power at 200 viable objects |
  |---|---|
  | P1 | ≈ 1.00 |
  | P2 | 0.99 (Holm step 0.025/3) |
  | P3 | 0.997 |
  | P2 superiority | 0.77 (0.93 at 300) |
  | S1 | 0.97 (0.73 shrunken) |
  | S3 | 0.99 (0.80 shrunken) |
  | S2 | 0.43 — underpowered, kept as historical |

- Viable fraction: 300 objects give about 200–280 viable objects if 65–95 % are viable. The viable
  fraction in the smoke test was 3/3. GSO was 0.755.

**Secondary set.** The 45 accepted EGAD evaluation-set objects, analysed with the same code and reported
separately. They never enter the primary families.

**Untouched.** The 8 sealed Phase III objects and every GSO/YCB development part.

**Inclusion and exclusion.**

- Inclusion: a watertight mesh with positive volume, already applied in the selection; failures were
  replaced by the next cell in the seeded order.
- No object is removed for any outcome.
- Every selected object enters the population metrics with one status:
  - `viable`;
  - `no viable candidate` (candidates generated, none feasible);
  - `no candidate generated`;
  - `planner budget exhausted` (exhaustive plan longer than 3 600 s);
  - `catalogue error`;
  - `cell build refused`.
- These statuses mean "AxisForge found no grasp", never "physically ungraspable".

### II.3 Preparation (frozen code: `tools/research/phase7/egad_prepare.py`)

- **Scale.** The dataset authors' rule for an 80 mm gripper.
- **Pose.** The most probable stable pose.
- **Cell.** The demo cell without parts; `add-object --units m --mass 0.1 --at 0.42 0 0.42`. The mass is
  assumed.
- **Collision pieces.** CoACD with threshold 0.1, at most 32 pieces, no preprocessing, 60 tree-search
  iterations with 10 nodes, seed 0.
  - These settings were chosen on the excluded smoke cells.
  - Threshold 0.05 without a cap gave 100–150 pieces in 200–400 s per object.
  - 32 is the cap of the GSO collision models used in development.
- **Catalogue.** The exhaustive 80 mm catalogue (`research-candidates` and an exhaustive `plan`).

### II.4 Frozen methods (hashes in PHASE7_MODEL_FREEZE.json)

**Models.**

| Symbol | Model | Training data | Label |
|---|---|---|---|
| P(F) | Gradient boosting, Phase V settings, seed 0, 19 candidate features | 316 development parts (376 690 candidates) | Exhaustive 80 mm feasibility |
| P(H\|F) | Gradient boosting, seed 0 | Viable candidates of the 151 dynamic development parts (12 585) | Phase VI v2 robustness in the six standard conditions |

p̃ = P(F)/max over the object's candidates.

**Methods.** All share the planner's stroke gate (makeable first).

| Id | Order |
|---|---|
| M0 | The original planner order |
| M1 | Preliminary + 2·p̃ |
| M4 | p̃^0.8 · Q_cen^0.2 |
| M9 | p̃ · P(H\|F) |
| **FORTIS-R Joint (final method)** | p̃ · P(H\|F) · max(Q_cen, 10⁻³)^0.3 |
| Hybrid (secondary) | Round robin of Joint and M0. Bound: N_hybrid ≤ 2·min(N_Joint, N_M0) |

- β = 0.3: the protocol I.4 criterion applied to all 151 development parts; also chosen in 4 of 5 outer
  folds.

**H5 representations.** MLPs with the Part I architecture, trained on the 316 development parts, 3 seeds,
predictions averaged, integrated as M1:

| Id | Role | Inputs |
|---|---|---|
| L2 | Geometry MLP | v, u |
| L10 | The best DINO + geometry MLP in development | v, u, DINOv3 through a linear branch |
| L10p | Control | Permuted DINOv3 |
| L10r | Control | Untrained ViT |
| L8 | Reported | Nonlinear DINOv3 branch |

- Descriptors use the Phase V recipe (16 views, 96 px, neutral grey; reproduced with cosine 1.000000
  on development parts).

**Main choices.**

- Main baseline: M0 (the original planner).
- Best learned baseline: M1.
- Main physical criterion: v3 (15°, torsion 0.005 m).
- Main operational metric: N_F, the number of candidates the planner examines until the first viable
  grasp.
- Main configuration: the 80 mm gripper of the demo cell.

### II.5 Measurements

- **N_F and first grasps.** From the exact order simulator on each object's catalogue.
- **Population metrics.** Success@B for B = 1, 5, 10, 20, 50 over all objects; an object without a viable
  candidate is a failure.
- **Physics v3.** First viable grasp of M0, M1, M4, M9 and Joint, in the nine conditions at torsion 0.005.
  - The torsion grid (0.001, 0.02) covers M0, M1 and Joint.
  - Unstable simulations count as not held.
  - The v2 criterion is also reported (historical).
  - Rotation limits 5/10/20° are sensitivity only.
- **Articulated level.** The cell's arm (`research_grasping/phase7/articulated.py`) for the M0, M1 and
  Joint first grasps of primary viable objects, in seven conditions (no pose noise). Descriptive.
- **Full online cost (H4).**
  - Objects: 60 primary viable objects drawn with seed 20261207.
  - Methods: M0, M1 and Joint, in a random order per object, with one thread and nothing else running.
  - Calls: one warm-up call per run, then four calls per object and method; the median of calls 2–4 is
    taken.
  - T_total (from receiving the object to the planner's answer):
    - M0: the planner call.
    - M1 and Joint: the `research-candidates` process + features + inference + prior file + the
      planner call.
  - The planner's own stage timings are kept.

### II.6 Hypotheses and analysis

**Unit and directions.** The unit is the object, and all comparisons are paired by object. Directions are
fixed one-sided at α = 0.025. Effect sizes always come with 95 % CIs: bootstrap with 10 000 draws, seed
20261128.

**Primary family.** Holm across P1–P3 at one-sided familywise α = 0.025, on primary viable objects.

| Id | Claim | Test | Supported only if |
|---|---|---|---|
| P1 | Joint examines fewer candidates than M0 | Geometric-mean ratio N_F(Joint)/N_F(M0); one-sided Wilcoxon signed-rank on the log ratios | Holm-adjusted p < 0.025 **and** the 95 % CI upper limit < 1 |
| P2 | Joint's v3 robustness (six standard conditions) is non-inferior to M0's, margin 5 percentage points | One-sided z test on the paired difference | Holm-adjusted p < 0.025 **and** the one-sided 97.5 % lower limit > −0.05 |
| P2-sup | Joint is more robust than M0 (fixed sequence) | Exact one-sided McNemar at 0.025/3 | Tested only if P2 holds; p < 0.025/3 **and** a positive difference |
| P3 (= H3) | Joint's N_F is non-inferior to M1's (the best learned baseline), ratio margin 1.25 | One-sided t test on log ratios against log 1.25 | Holm-adjusted p < 0.025 **and** the one-sided 97.5 % upper limit < 1.25 |

H3's "margen temporal" is expressed in examined candidates. Each candidate costs about the same planner
time, and N_F is free of timing noise. Measured time is reported by S6.

**Secondary family.** Holm across S1–S7 at one-sided 0.025; reported as secondary.

| Id | Claim |
|---|---|
| S1 (H1, historical) | M4 > M1 in load tolerance (v3) |
| S2 (H2, historical) | M9 > M1 in robustness (v3) |
| S3 | Joint > M1 in load tolerance (v3) |
| S4 | Joint > M1 in robustness (v3) |
| S5 | Joint's load tolerance (v3) is non-inferior to M0's, margin 10 pp |
| S6 (H4) | Joint's full online cost T_total is lower than M0's (ratio, Wilcoxon) |
| S7 (H5) | L10 examines fewer candidates than L2 |

- S1 and S2 use McNemar. They keep their historical meaning; the physical definition changes from v2 to
  v3, and the v2 versions are reported next to them.
- **H5** holds only if S7 is supported **and** L10 < L10p (95 % CI upper limit below 1). L10r, L8 and L2
  vs gradient boosting are reported.

**Descriptive, outside both families.**

- Population Success@B and the status counts.
- The hybrid's bound check and the number of objects with negative transfer.
- Torsion and rotation-threshold sensitivity of P2 and S3.
- The articulated replication.
- The evaluation set.
- Development vs EGAD rates (`cross_dataset_results.json`).

Not significant is never reported as equivalent. Development and confirmation results appear in
separate sections.

### II.7 Failure handling

- Every object keeps its status (II.2).
- A planner or simulator crash is retried once with the same inputs. Only a deterministic code failure
  on a frozen path is a critical error (II.9).
- A viable object without dynamics is listed in `missing_dynamics` and excluded only from the dynamic
  tests.
- Nothing is rerun to change an outcome.

### II.8 Changes from the Phase VI draft and from Part I, all decided on development data

1. **Load tolerance moved to secondary.**
   - The Phase VI draft (and a Phase VII draft) had "Joint > M1 load tolerance" as primary.
   - The development articulated pilot
     ([articulated_simulator_validation.json](results/articulated_simulator_validation.json)) agrees with
     the floating gripper on the standard conditions (agreement 0.95–0.97). It disagrees on heavy loads:
     Joint − M1 load is +11.3 pp with v3 but −3.3 [−10.6, 4.0] with the arm.
   - So load-tolerance claims depend on the simulator level and are not primary.
2. **H5 targets L10, not the pre-declared L8.**
   - L8 did not differ from L2 or from its permuted control in development.
   - L10 was the best DINO + geometry variant.
   - Its linear-branch controls (L10p, L10r) were added to the development benchmark after L10's result
     was seen (declared post hoc).
3. **The P(F) used by M1/M4/M9/Joint is refitted on all 316 development parts.** In development it was
   Phase V's library model.
4. **The collision decomposition** was chosen on the excluded smoke cells (II.3).

### II.9 Errors after the test is opened

- The method is not changed.
- An error found after opening is documented with the affected results kept.
- The record states whether a new test set is needed. Unused positions 300–499 of the frozen order are
  the reserve.
