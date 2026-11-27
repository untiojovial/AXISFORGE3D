# External review package

This package is for an independent reviewer who wants to audit the study without the history of how it was
produced. Everything it points to is in the repository.

**Status of the reviews.**

| Review | Who | Status |
|---|---|---|
| Internal reviews and audits | AI systems (Claude Code, and ChatGPT through the author's instructions) | self-checks, **not independent peer review** |
| Human review | the author, Eduardo Pérez González, on 26 November 2026 (`HUMAN_REVIEW_RECORD.md`) | done; the author's own review, with the scope the author communicated; later material still needs the author's approval |
| Independent review | — | not done; this package is prepared for it |

The final decisions and the responsibility are the author's. The AI systems were assistants.

## 1. Commits

| Role | Commit | What it fixes |
|---|---|---|
| Phase VII development design | `8e470c0` | protocol part I, before the development experiments |
| Phase VII freeze (first confirmation) | `de22a8a` | protocol part II, frozen models, hypotheses, analysis; before any confirmatory object was planned |
| Phase VII results | `5a821dc` | the first confirmation's outputs; unchanged since |
| Start of the extension | `15fd43f` | the state audited by `results/start_verification.json` |
| Development selection rule | `49eb5bb` | committed before any v3 model was fitted |
| Extension freeze (second confirmation) | `b478289` | `PHASE7X_FREEZE.json`: protocol, model X, analysis, binary; before any new object was catalogued |
| Second confirmation results | `040cde7` | `results/confirmation_x.json`, per-object outputs, execution record |
| Final verdict | `542b274` and later documentation commits on the same branch | `FINAL_SCIENTIFIC_AUDIT.md`, `PUBLICATION_READINESS_FINAL.md` |
| Author data, declarations, literature review | `6107c4d` and the following commits (2026-11-27) | `HUMAN_REVIEW_RECORD.md`, `LITERATURE_REVIEW_FINAL.md`, `STATE_OF_THE_ART_MATRIX.md`, `NOVELTY_ASSESSMENT.md`, `REFERENCES_VERIFIED_FINAL.json` |

Verify a freeze:

- the first one: `python tools/research/phase7_extension/verify_freeze.py --commit de22a8a`;
- the second one: `load_freeze` in `confirm_x.py`, or by hand, comparing `files_sha256` with `sha256sum`.

## 2. Architecture

- **Planner** (Rust, `crates/grasping`).
  - For a part on a table it generates a deterministic catalogue of parallel-jaw candidates.
  - It puts stroke-feasible candidates first and examines them in order of a preliminary score.
  - Each candidate goes through inverse kinematics, collision checks and a static wrench test.
  - It stops at the first candidate that passes.
- **Learned order** (`crates/grasping/src/learned.rs`).
  - Gradient-boosting ensembles exported from scikit-learn with every threshold and leaf stored as its
    64-bit pattern.
  - The planner computes the same 19 candidate features and the probabilities, then orders the candidates by
    the method's score.
  - The gates and the checks are unchanged.
- **Round-robin hybrid.** It interleaves the learned order and the original one, skipping repeats.
- **Research tools** (Python, `tools/research/`, `research_grasping/`): exhaustive labelling of catalogues, the
  exact order replay, MuJoCo physics (floating gripper v3 and an articulated arm), statistics.

## 3. Mathematics

**Scores.** p̃ = P(F)/max P(F) per object.

| Method | Score |
|---|---|
| M1 | `pre + 2 p̃` |
| Joint | `p̃ · P(H\|F) · max(Q_cen, 10⁻³)^0.3` |
| X = Joint-multi | the same, with P(robust at all three torsions \| F) in place of P(H\|F) |

**Fallback bound.** With the conditions (same candidate set, order-independent feasibility, at least one
feasible candidate):

`N_hybrid ≤ min(2N_X − 1, 2N_M0) ≤ 2min(N_X, N_M0)`

The proof: by step 2k − 1 the first k candidates of X have been examined, and by step 2k those of M0.

**Estimands and tests.**

- Geometric-mean ratios of N_F and of time: one-sided t tests on per-object log ratios, with t CIs.
- Paired proportions:
  - Tango score test at the margin, with the Tango CI;
  - Newcombe hybrid score CI (method 10) as sensitivity;
  - exact McNemar for superiority.
- Holm within each family.
- The Wilcoxon signed-rank test is a sensitivity analysis only.

## 4. Data

| Data | Location | Redistributable |
|---|---|---|
| Development catalogues and labels (GSO, YCB) | regenerable; summaries in `research/phase5`–`phase7` | GSO-derived: CC BY 4.0. YCB per-object data: kept in the repository, excluded from the package |
| v3 development labels | `results/v3_models.json` (per object and per fold) | yes |
| First confirmation, per object | `research/phase7/results/confirmation_orders.json`, `confirmation_trials.json.gz` | yes (EGAD BSD-3 notice) |
| Second confirmation, per object | orders, trials and timing archives next to `results/confirmation_x.json` | yes |
| Embeddings | `research/phase4`, `research/phase5` | DINOv3-derived: DINOv3 License only, not in the package |

## 5. Hypotheses

- **First confirmation.** `research/phase7/PHASE7_HYPOTHESES.md`: P1–P3 primary; S1–S7 secondary.
- **Second confirmation.** `PHASE7X_PROTOCOL.md`: C1–C4 primary; S1–S3 secondary; descriptive analyses.
  Margins: −5 pp (C2), 1.25 (C3), −10 pp (S1).

## 6. Results

| Result | File |
|---|---|
| First confirmation | `research/phase7/results/confirmation_results.json` |
| First confirmation: corrections and P2 audit | `results/corrections_addendum.json`, `results/p2_statistical_audit.json` |
| Second confirmation | `results/confirmation_x.json` |
| Development selection | `results/v3_models.json` |
| Native implementation, benchmark, memory, ceiling, fallback | `results/native_inference_validation{,_x}.json`, `integrated_runtime_benchmark.json`, `memory_benchmark.json`, `ordering_ceiling.json`, `fallback_audit.json` |
| GPD | `results/external_baselines_v2.json`, `results/gpd_variability.json` |
| Exploratory | `results/articulated2_development.json`; Optuna: `OPTUNA_EXPLORATORY.md`, `results/optuna_mlp.json`, `results/optuna_contrasts.json`, `results/optuna_trials.json.gz` |

The summary verdicts are in `PUBLICATION_READINESS_FINAL.md`. Second confirmation in one line: C1, C2 and C3 were
supported (N_F ratio 0.449; robustness +3.3 pp, non-inferior; 0.992 against M1); C4 (time) was not (0.998); S1 was
supported; S2 and S3 were not.

## 6b. Literature and originality

- **Coverage.** `LITERATURE_REVIEW_FINAL.md` lists, family by family, what was known before the study.
- **Antecedents.** It names the works that challenge the originality claims. Learned feasibility ordering is
  precedented (Qin et al. 2026, Liu et al. 2025), and so is the round-robin bound (a parallel-execution property).
- **Assessment.** `NOVELTY_ASSESSMENT.md` grades each contribution.
- **References.** Every new or corrected reference is in `REFERENCES_VERIFIED_FINAL.json`, with the part of the paper
  that was read.

## 7. Tests

- `cargo test --workspace --release`, `cargo clippy -D warnings`, `cargo fmt --check`, `pytest tests`.
- The latest run is in `TEST_RESULTS.md`, with the skipped tests and their reasons.

## 8. Comparators

- **Internal:** the original planner (M0), the plain learned order (M1), the historical fusions (M4, M9,
  Joint).
- **External:** GPD's pretrained classifier, presented exactly as "Ranking-controlled comparison using GPD's
  pretrained classifier on a shared candidate catalogue". It is stochastic between runs.
- **Not run:** Dex-Net 2.0 (official weights return 404) and Contact-GraspNet (needs CUDA).
- **Also measured:** a classical analytic order, force closure then epsilon (Phase VII).

## 9. Errors found and corrected

`FINAL_CORRECTIONS_REGISTER.md` covers items A–H and the P2 audit:

- S6 reported for a set where it was not measured;
- wording of P3 and of non-significant results;
- the fallback bound described as a cost bound;
- denominators;
- p-values stored as 0;
- "pre-registered" used for a Git freeze;
- the fragility of P2.

The historical result files are not edited; the corrections are in addenda.

**Other errors found during the extension and fixed before any confirmatory use:**

- **Memory measurement.** `wait4` RSS inherited the parent's peak; VmHWM is used instead.
- **Benchmark loss.** A container restart lost the first benchmark; the benchmark is now restart-safe.
- **Selection training set.** The v3 model training would have included extra simulations; it was restricted
  to sweep labels, as the rule says.
- **Power-scenario bug.** The power scenarios scaled instead of shifting; this was fixed before the freeze.

## 10. Limitations and deviations

- **Limitations:** `LIMITATIONS.md`.
- **Deviations, Phase VII:** D1–D4 (supplement).
- **Deviations, extension:** the preparation of position 300 before the freeze (`PHASE7X_PROTOCOL.md` §2).
- **Execution incidents:** the container restarts listed in `confirmation_x.json`.

## 11. Commands

See `REPRODUCIBILITY.md`. The quickest audit:

```
bash reproducibility/reproduce_statistics.sh
python tools/research/phase7_extension/make_tables_x.py --out /tmp/tables && diff /tmp/tables/numbers_x.tex publication/intelligent_service_robotics/tables/numbers_x.tex
```

## 12. Files that are not redistributable, or not included

| File class | Why | How to obtain |
|---|---|---|
| EGAD, GSO, YCB meshes | third-party; not committed | the fetch scripts (`tools/fetch/README.md`) |
| Model weights (DINOv3, DINOv2, OpenCLIP, GPD) | third-party | gated (DINOv3) or pinned downloads |
| DINOv3-derived embeddings | DINOv3 License | in the repository under that licence, not in the package |
| YCB per-object data | conservative decision | regenerated with `tools/verify/ycb_pick.sh` |
| Manuscript sources | all rights reserved pending publication | in `publication/` for review only |
