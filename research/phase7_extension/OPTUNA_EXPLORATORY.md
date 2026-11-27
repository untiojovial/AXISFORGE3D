# Exploratory hyperparameter search for the feasibility MLPs (Optuna)

**Status.**

- **Scope.** Exploratory analysis on development data only.
- **No role in the confirmatory evidence.** It was not used for any selection, hypothesis, frozen model or claim.
- **Protocols.** The confirmatory protocols (`research/phase7/PHASE7_PROTOCOL.md`, `PHASE7X_PROTOCOL.md`) were
  frozen before this search and do not depend on it.
- **In the article.** It appears only as one paragraph of the supplement, in the section "Exploratory analyses that
  support no claim".
- **Review.** It was produced after the author's review of 26 November 2026 (`HUMAN_REVIEW_RECORD.md`) and needs the
  author's approval.

## 1. Question

The Phase VII representation benchmark (`research/phase7/results/mlp_benchmark.json`) trained every input family with
one fixed MLP architecture that was not tuned. An objection follows: the negative result for frozen foundation-model
features, and the gap between the MLPs and gradient boosting (L0), could be artefacts of untuned networks.

This search asks three questions, descriptively, under a modest search per family:

1. Does any MLP family examine fewer candidates than the fixed MLP with the same inputs?
2. Does any family reach L0, the model family of the final method?
3. Does the ranking of the input families change?

## 2. Design

The design was fixed in the script before running (`tools/research/phase7_extension/optuna_mlp.py`, commit `49eb5bb`,
2026-11-26 09:24 UTC).

| Element | Setting |
|---|---|
| Data | the 316 development parts (116 library, 200 Phase V test) with the feature blocks of the Phase VII benchmark: v (candidate), u (part geometry), d (frozen image features); 239 parts have at least one feasible candidate |
| Outer cross-validation | the same 5 folds, grouped by product name, as the Phase VII benchmark; the outer test fold is never seen by the search |
| Inner split | within each outer training set, one grouped hold-out of 15 % of the name groups (seed 2000 + k for fold k; 29–40 parts). It is used for the objective, for pruning and for early stopping |
| Sampler and pruner | Optuna 4.1.0, TPE (seed 20261212 + k; the first ten trials are random start-up trials, Optuna's default); MedianPruner after 5 trials and 3 epochs |
| Trials | 20 per family and outer fold: 9 × 5 × 20 = 900 trials |
| Search space | latent 8–256 (log); width 16–512 (log); 1–3 hidden layers; ReLU, GELU or SiLU; dropout 0–0.3; learning rate 1e-5–3e-3 (log); weight decay 1e-6–1e-2 (log); batch 1024, 2048, 4096 or 8192; Adam or AdamW; residual connections yes or no |
| Training during the search | at most 256 candidates per part (seeded subsample), at most 30 epochs, patience 4; binary cross-entropy |
| Objective | geometric mean of N_F on the inner hold-out's viable parts, with the M1 order (preliminary + 2 p/max, exact planner-order simulator) |
| Refit | the best configuration of each outer fold, refitted on all rows of the inner training parts (at most 60 epochs; early stopping on the same hold-out) with 5 seeds (0–4); out-of-fold predictions for all 316 parts |
| Software | PyTorch 2.9.1 (CPU, one thread per process), on 4 shared CPUs |

**Families and their fixed-architecture counterparts.** The counterparts come from the Phase VII benchmark, which
used 3 seeds and fixed settings: latent 16, dropout 0.1, learning rate 1e-3, weight decay 1e-4, batch 4096, AdamW, at
most 40 epochs, early stopping on validation loss, inner split seed 1000 + k.

| Family | Inputs | Fixed counterpart |
|---|---|---|
| candidate-only | v | L2v |
| geometry | v, u | L2 |
| dinov2 | v, DINOv2 | L5 |
| dinov3 | v, DINOv3 | L6 |
| openclip | v, OpenCLIP | none with the same inputs (L11 also has u) |
| combined | v, u, DINOv3 | L8 |
| linear | v, u, DINOv3 through a linear branch | L10 |
| random-vit | v, u, untrained ViT (control) | L9 |
| permuted | v, u, DINOv3 permuted across objects (control, the permutation of L8p) | L8p |

## 3. Budget and departures

- **Trials.** 20 per family and outer fold, against the 30–50 requested. This was a compute decision: 4 shared
  CPUs, part of the time alongside the second confirmation. It was stated in the script before running.
- **Inner validation.** One inner hold-out, not inner cross-validation.
- **Search regime.** The search trained on subsampled rows and short schedules. The refit used all rows and longer
  schedules. The selected configuration is therefore not guaranteed to be optimal at refit.
- **Docstring errors.** Two errors in the script's docstring were found after the run and corrected; the code was not
  changed:
  - the TPE seed is 20261212 + k, not 20261212;
  - the summary reads the references through `--mlp-ref`, not `--mlp-dir`.

  The run used the script as committed at `49eb5bb` (SHA-256
  `e880552987091bc736355fd984f1aead9c2f2477f735b22992a5809ff7742ab1`).
- **Conventions of the summary.** The `--summarise` step does not compute three metrics as
  `tools/research/phase7/mlp_eval.py` does:

  | Metric | `--summarise` | `mlp_eval.py` |
  |---|---|---|
  | Success@k | over the 239 viable parts | over all 316 parts |
  | ECE | 10 bins | 15 bins |
  | N_F | mean of the per-seed geometric means | geometric mean of the per-part log N_F averaged over seeds |

  The tables below use comparable values:
  - N_F in the benchmark convention, from `optuna_compare.py`;
  - the references' Success@1 rescaled to the 239 viable parts (× 316/239);
  - no ECE comparison.
- **Comparison script.** `optuna_compare.py` was written after the search had started and before any out-of-fold
  metric of the tuned models had been computed.
  - It is descriptive.
  - Holm's adjustment applies within each group of contrasts only.

## 4. Execution

The search ran from 26 November 2026, 18:43 UTC, to 27 November 2026, 14:52 UTC, on the shared 4-CPU machine.
Sources: `optuna_notes.txt` and the launch scripts in the scratch directory, and the execution record of the second
confirmation (`results/confirmation_x/execution_record.json`).

| Time (UTC) | Event |
|---|---|
| 26 Nov 18:43 | Launched with 2 parallel units on CPUs 0–2, while the second confirmation was being prepared |
| about 19:05–22:31 | Container down; the run stopped (candidate-only folds 0 and 1 complete) |
| 22:32 | Relaunched as 1 unit on CPU 3; the second confirmation used the other CPUs |
| 23:17 | Paused (SIGSTOP) for the timing stage of the second confirmation, which ran alone (`load_during_timing`) |
| 23:52 | Stopped and relaunched with 4 parallel units on CPUs 0–3; the unit in progress (candidate-only fold 4) restarted from its first trial |
| 27 Nov about 06:37 | Container restart; relaunched at 06:39 with 32 units complete |
| about 09:10 | Container restart; relaunched at 09:11 with 38 units complete |
| 14:52 | All 45 units complete |

- **Interrupted units.** Each interruption lost the units in progress. They were repeated from the first trial with
  the same seeds. A completed unit was never re-run.
- **Compute.**
  - 900 trials: 534 complete, 366 pruned, 0 failed.
  - Search: 8.8 h summed over the 45 units.
  - Refits: 225 models (45 units × 5 seeds), 511 s each on average.
  - Search and refit of the completed units: 40.8 CPU-hours in all, excluding the repeated work.
- **Interim looks.** Summaries of the families completed so far were computed during the run, from 27 November 05:41
  UTC. They changed nothing that had been fixed beforehand:
  - the search;
  - the comparison script (written at 03:57 UTC and committed in `f676448` before any metric of a tuned model had
    been computed);
  - the reading in Sect. 6, which applies the development rule of `research/phase7/PHASE7_PROTOCOL.md`.

## 5. Results

### 5.1 Trials

| Family | Trials | Complete | Pruned | Failed | Search time (h, 5 folds) | Best trial index by fold |
|---|---|---|---|---|---|---|
| candidate-only | 100 | 58 | 42 | 0 | 0.51 | 14, 10, 8, 19, 16 |
| geometry | 100 | 60 | 40 | 0 | 0.73 | 16, 3, 18, 3, 14 |
| dinov2 | 100 | 60 | 40 | 0 | 1.23 | 12, 3, 19, 11, 10 |
| dinov3 | 100 | 60 | 40 | 0 | 1.04 | 15, 3, 17, 11, 8 |
| openclip | 100 | 58 | 42 | 0 | 0.96 | 0, 11, 8, 12, 16 |
| combined | 100 | 58 | 42 | 0 | 1.20 | 16, 11, 8, 11, 10 |
| linear | 100 | 59 | 41 | 0 | 0.79 | 11, 11, 16, 10, 13 |
| random-vit | 100 | 61 | 39 | 0 | 1.21 | 0, 16, 18, 12, 17 |
| permuted | 100 | 60 | 40 | 0 | 1.15 | 11, 11, 15, 13, 17 |
| **all** | 900 | 534 | 366 | 0 | 8.8 | |

### 5.2 Out-of-fold metrics (316 parts, 239 with a feasible candidate)

| Model | Inputs | AUC (sd over seeds) | Log loss | N_F geo. mean, benchmark convention | N_F, mean of per-seed geo. means (sd) | Success@1 on the 239 viable parts | Parameters (mean) |
|---|---|---|---|---|---|---|---|
| candidate-only (tuned) | v | 0.9933 (0.0003) | 0.0461 | 1.604 | 1.604 (0.043) | 0.773 | 222,261 |
| geometry (tuned) | v, u | 0.9936 (0.0004) | 0.0451 | 1.517 | 1.519 (0.075) | 0.788 | 620,141 |
| dinov2 (tuned) | v, DINOv2 | 0.9941 (0.0003) | 0.0446 | 1.490 | 1.492 (0.071) | 0.797 | 793,921 |
| dinov3 (tuned) | v, DINOv3 | 0.9936 (0.0002) | 0.0458 | 1.516 | 1.517 (0.052) | 0.795 | 629,358 |
| openclip (tuned) | v, OpenCLIP | 0.9934 (0.0005) | 0.0460 | 1.508 | 1.509 (0.047) | 0.792 | 462,229 |
| combined (tuned) | v, u, DINOv3 | 0.9938 (0.0003) | 0.0442 | 1.526 | 1.527 (0.054) | 0.780 | 769,890 |
| linear (tuned) | v, u, DINOv3 (linear branch) | 0.9932 (0.0003) | 0.0464 | 1.499 | 1.499 (0.029) | 0.803 | 333,098 |
| random-vit (tuned) | v, u, untrained ViT | 0.9937 (0.0002) | 0.0445 | 1.464 | 1.464 (0.027) | 0.815 | 746,991 |
| permuted (tuned) | v, u, permuted DINOv3 | 0.9932 (0.0003) | 0.0464 | 1.511 | 1.512 (0.051) | 0.802 | 1,002,528 |
| L0 (reference) | gradient boosting; v + log extents | 0.9958 | 0.0370 | 1.317 | 1.316 | 0.849 | — |
| L2v (reference) | MLP; v | 0.9944 | 0.0421 | 1.491 | 1.492 | 0.801 | 5345 |
| L2 (reference) | MLP; v, u | 0.9755 | 0.2815 | 1.494 | 1.494 | 0.792 | 8049 |
| L5 (reference) | MLP; v, DINOv2 | 0.9945 | 0.0420 | 1.417 | 1.418 | 0.812 | 32049 |
| L6 (reference) | MLP; v, DINOv3 | 0.9941 | 0.0438 | 1.465 | 1.465 | 0.792 | 32049 |
| L8 (reference) | MLP; v, u, DINOv3 | 0.9661 | 0.4016 | 1.516 | 1.516 | 0.767 | 34753 |
| L9 (reference) | MLP; v, u, untrained ViT | 0.9933 | 0.0524 | 1.472 | 1.472 | 0.801 | 34753 |
| L10 (reference) | MLP; v, u, DINOv3 (linear branch) | 0.9752 | 0.2828 | 1.425 | 1.426 | 0.808 | 15233 |
| L8p (reference) | MLP; v, u, permuted DINOv3 | 0.9922 | 0.0566 | 1.518 | 1.518 | 0.787 | 34753 |

### 5.3 Paired contrasts of N_F (ratio of geometric means; < 1 means fewer examined candidates for the first model)

**tuned vs fixed**

| Contrast | Ratio [95 % CI] | Parts better / worse | Holm p |
|---|---|---|---|
| candidate-only / L2v | 1.075 [1.022, 1.134] | 30 / 53 | 0.0915 |
| geometry / L2 | 1.015 [0.956, 1.078] | 44 / 59 | 0.924 |
| dinov2 / L5 | 1.051 [1.003, 1.106] | 39 / 54 | 0.347 |
| dinov3 / L6 | 1.035 [0.992, 1.081] | 32 / 59 | 0.461 |
| combined / L8 | 1.007 [0.954, 1.063] | 50 / 51 | 1 |
| linear / L10 | 1.052 [1.003, 1.105] | 35 / 56 | 0.461 |
| random-vit / L9 | 0.995 [0.950, 1.041] | 35 / 47 | 1 |
| permuted / L8p | 0.996 [0.944, 1.050] | 39 / 48 | 1 |

**tuned vs L0**

| Contrast | Ratio [95 % CI] | Parts better / worse | Holm p |
|---|---|---|---|
| candidate-only / L0 | 1.218 [1.116, 1.335] | 20 / 57 | 0.000405 |
| geometry / L0 | 1.152 [1.064, 1.254] | 24 / 73 | 0.00117 |
| dinov2 / L0 | 1.132 [1.041, 1.235] | 23 / 68 | 0.00533 |
| dinov3 / L0 | 1.151 [1.061, 1.256] | 22 / 62 | 0.00533 |
| openclip / L0 | 1.146 [1.056, 1.248] | 22 / 63 | 0.00533 |
| combined / L0 | 1.159 [1.065, 1.265] | 23 / 65 | 0.00319 |
| linear / L0 | 1.139 [1.041, 1.249] | 24 / 62 | 0.00835 |
| random-vit / L0 | 1.112 [1.021, 1.213] | 24 / 53 | 0.017 |
| permuted / L0 | 1.148 [1.056, 1.252] | 22 / 61 | 0.00835 |

**visual vs control, tuned**

| Contrast | Ratio [95 % CI] | Parts better / worse | Holm p |
|---|---|---|---|
| dinov2 / candidate-only | 0.929 [0.883, 0.974] | 54 / 44 | 0.222 |
| dinov3 / candidate-only | 0.945 [0.903, 0.986] | 49 / 42 | 0.222 |
| openclip / candidate-only | 0.941 [0.890, 0.989] | 49 / 41 | 0.55 |
| combined / geometry | 1.006 [0.955, 1.059] | 47 / 54 | 1 |
| linear / geometry | 0.988 [0.944, 1.037] | 58 / 45 | 1 |
| combined / permuted | 1.010 [0.966, 1.056] | 47 / 49 | 1 |
| combined / random-vit | 1.042 [0.987, 1.102] | 37 / 54 | 0.49 |

### 5.4 Selected hyperparameters (45 outer-fold selections)

- layers: 1 4, 2 33, 3 8
- activation: gelu 17, relu 17, silu 11
- optimizer: adam 11, adamw 34
- residual: False 22, True 23
- batch: 1024 20, 2048 12, 4096 4, 8192 9
- latent: median 59, range 8–252
- width: median 300, range 59–509
- dropout: median 0.0671, range 0.000656–0.256
- lr: median 0.00233, range 0.000372–0.00295
- weight_decay: median 1.63e-05, range 1.01e-06–0.000551

### 5.5 Inner hold-out objective vs outer test (per family, mean over folds)

| Family | Best inner N_F during search | Inner N_F at refit (5 seeds) | Outer N_F (out of fold) |
|---|---|---|---|
| candidate-only | 1.345 | 1.363 | 1.604 |
| geometry | 1.263 | 1.329 | 1.519 |
| dinov2 | 1.213 | 1.257 | 1.492 |
| dinov3 | 1.262 | 1.277 | 1.517 |
| openclip | 1.242 | 1.291 | 1.509 |
| combined | 1.242 | 1.276 | 1.527 |
| linear | 1.243 | 1.297 | 1.499 |
| random-vit | 1.301 | 1.304 | 1.464 |
| permuted | 1.311 | 1.305 | 1.512 |

Refit: mean 511 s per model; prediction 27.28 ms per part (CPU, one thread, shared machine).

## 6. Reading

This is a description of one limited search on development data. Without a declared margin, no contrast below
tests equivalence.

1. **No tuned family examined fewer candidates than its fixed-architecture counterpart.**
   - The ratios run from 0.995 to 1.075, and no interval lies below 1.
   - For three families the interval lies above 1: candidate-only 1.075 [1.022, 1.134], DINOv2 1.051 [1.003, 1.106]
     and linear 1.052 [1.003, 1.105]. The tuned network examined more candidates than the fixed one.
   - None of the eight contrasts survives Holm's adjustment (smallest adjusted p = 0.09).
   - Reading: tuning did not improve the ordering at this budget. That is not evidence that the two are equal.
2. **No tuned family reached gradient boosting (L0).**
   - All nine examined between 11 % and 22 % more candidates than L0.
   - Every interval lies above 1, and every Holm-adjusted p is at most 0.017.
   - The Phase VII gap between the MLPs and L0 is therefore not explained by the fixed architecture, at least within
     this search.
3. **The ranking of the input families changed little, and frozen features still did not help.**
   - Tuned N_F ranged from 1.464 to 1.604, against 1.417 to 1.518 for the fixed MLPs.
   - The lowest tuned value is that of the untrained-ViT control (random-vit).
   - The development rule of `PHASE7_PROTOCOL.md` (below the geometry model with the upper CI under 1, and better
     than the permuted control), applied to the tuned families, is not met:
     - combined (v, u, DINOv3) / tuned geometry: 1.006 [0.955, 1.059];
     - combined / tuned permuted: 1.010 [0.966, 1.056];
     - linear / tuned geometry: 0.988 [0.944, 1.037];
     - combined / tuned random-vit: 1.042 [0.987, 1.102].
   - The families without the geometry block do better than the tuned candidate-only model:
     - DINOv2 0.929, DINOv3 0.945 and OpenCLIP 0.941, with unadjusted intervals below 1 and Holm p ≥ 0.22;
     - but the tuned candidate-only model is itself worse than the fixed one (1.075), and neither DINOv2 nor DINOv3 beats
       its own fixed counterpart (OpenCLIP has none).
   - No reading that the visual features help follows from this.
4. **Model properties.**
   - The selected networks are much larger than the fixed ones: 0.2 to 1.0 million parameters on average per family,
     against 5 to 35 thousand.
   - They are better calibrated than the fixed MLPs that take the geometry block: log loss 0.044–0.046, against 0.28
     for L2 and 0.40 for L8.
   - Their AUC (0.993–0.994) stays below L0's (0.9958), and better calibration did not give a better order.
   - The inner hold-out N_F of the selected configurations (1.21–1.35) was optimistic against the outer test
     (1.46–1.60). That is consistent with a noisy objective on about 30 parts.

**Summary.** The negative result for frozen foundation-model features and the advantage of gradient boosting are
**not changed by a limited hyperparameter search**. This does not show that no MLP could do better.

## 7. Limitations

- **Noisy objective.** The inner hold-out has 29–40 parts, of which roughly 22–30 are viable. The objective is
  therefore noisy:
  - 10 of the 45 selected configurations came from the first ten trials, which are TPE's random start-up trials
    (Optuna's default);
  - 15 of the 45 came from the last five trials, so the search had probably not converged at 20 trials;
  - the inner N_F of the selected configurations was optimistic against the outer test (Sect. 5.5).
- **Early stopping.** Early stopping at refit used the same hold-out that selected the configuration.
- **Budget.** 20 trials in a ten-dimensional space is a small budget. A larger search, inner cross-validation or a
  ranking loss could give different results. The Phase VII listwise ranking loss, with the fixed architecture, did
  not help (`ranking_loss_benchmark.json`).
- **Data.** Development only: 316 parts, one gripper and one planner. Nothing here was tested on the confirmation
  objects.
- **Timings.** Taken on a shared cloud machine, so they are indicative only.
- **Interrupted units.** Units interrupted by container restarts were repeated from the first trial with the same
  seeds. Their partial results were lost, so identity with the interrupted attempts was not checked.

## 8. Effect on the article

- **Hypotheses and claims.** None changes. The confirmed method uses gradient boosting. The MLPs are a development
  study, and the article states that the foundation-model result is bounded to frozen features and to this problem.
- **Supplement.** The paragraph "Hyperparameter search" summarises the outcome in one sentence and names this
  document.
- **Wording.** "Not improved by a limited search" is the accurate wording. "Equivalent" or "MLPs cannot do better"
  are not.

## 9. Files

| File | Content |
|---|---|
| `tools/research/phase7_extension/optuna_mlp.py` | search, refit and summary |
| `tools/research/phase7_extension/optuna_compare.py` | paired comparison with the references |
| `results/optuna_mlp.json` | the summary: metrics per family and seed, selected configurations, trial counts, times; the references |
| `results/optuna_contrasts.json` | the paired contrasts, N_F in the benchmark convention, viable parts |
| `results/optuna_trials.json.gz` | every trial (parameters, value, state, seconds, epochs), each fold's selection, the refit records and the SHA-256 of each checkpoint |

**Not committed.**

- **Contents.** The checkpoints (torch state dicts) and the out-of-fold predictions, about 590 MB in all (536 MB of checkpoints, 57 MB of predictions).
- **Licence.** Some are derived from DINOv3 embeddings, which are not redistributed
  (`ARTIFACT_REDISTRIBUTION_REGISTER.md`).
- **Regeneration.** Both can be regenerated with the script, from the inputs of the Phase VII benchmark.
