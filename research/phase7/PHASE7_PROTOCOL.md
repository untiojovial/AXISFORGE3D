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
