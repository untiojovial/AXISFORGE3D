# Publication readiness: final verdict (updated 2026-11-27)

## Verdict: READY FOR SUBMISSION BY THE AUTHOR (approved by the author on 27 November 2026)

**Done.**

- Both confirmations were run once under frozen protocols.
- Every claim in the manuscript is backed by a committed results file.
- Negative and non-replicated results are reported.
- The literature review is complete and verified; the introduction and related work state the gap and the
  contributions against it.
- The author's data and the editorial declarations are final, as confirmed by the author on 27 November 2026.
- **Human review and approval.**
  - The author reviewed the research on 26 November 2026.
  - On 27 November the author approved the material produced afterwards (`HUMAN_REVIEW_RECORD.md`): the literature
    review, the introduction and related work, the tables, the technical-review corrections, the Optuna analysis
    and the new title.
- **Format.** The manuscript and its double-blind copy have 16 pages. The supplement is Online Resource 1
  (`ESM_1.pdf`). The cover letter is dated 27 November 2026.
- The author submits. Files and steps are in `publication/intelligent_service_robotics/submission_checklist.md`.

Nothing has been submitted, released or registered: no submission, no release, no DOI, no payment.

## 1. Verdict table

| Item | Result | Evidence | Verdict |
|---|---|---|---|
| **P1**: Joint fewer candidates than M0 (first confirmation) | 0.609 [0.516, 0.711], Holm p 6.4·10⁻⁹ | `research/phase7/results/confirmation_results.json`, `results/corrections_addendum.json` | supported |
| **P2**: Joint robustness non-inferior, −5 pp (first) | −1.8 pp. Pre-specified lower −4.9: supported narrowly. Tango p 0.040, CI [−5.5, 1.5]; Newcombe [−5.3, 1.4]; one object reverses it | `results/p2_statistical_audit.json` | supported by the frozen test; **not robust** to the procedure |
| **P3**: Joint efficiency non-inferior to M1, margin 1.25 (first) | 0.993 [0.953, 1.037] | same | supported (non-inferiority, not equality) |
| **C1**: X fewer candidates than M0 (second) | 0.449 [0.356, 0.567], Holm p 3.3·10⁻¹⁰; 54 / 4 / 93 | `results/confirmation_x.json` | supported |
| **C2**: X robustness non-inferior, −5 pp, Tango (second) | +3.3 pp [0.7, 7.5], Holm p 2.8·10⁻⁴; 5 vs 0 discordant | same | supported; superiority not claimed |
| **C3**: X efficiency non-inferior to M1 (second) | 0.992 [0.930, 1.059] | same | supported |
| **C4**: X cold-process time lower than M0 (second) | 0.998 [0.978, 1.019], Holm p 0.43; X slower on 142 / 200 | same, `results/confirmation_x_posthoc.json` | **not supported** |
| Secondary, second confirmation | S1, load non-inferior at −10 pp: −2.0 [−7.6, 3.4], supported. S2, all three torsions X > M1: 0.0, not supported. S3, X > M0: +0.7, not supported | same | the selection advantage did not replicate |
| Secondary, first confirmation | none of the seven supported; load −6.4 pp (S5) | Phase VII results | negative, reported |
| Robustness overall | non-inferiority: supported in C2, fragile in P2. No confirmed gain | — | no robustness claim beyond non-inferiority |
| Load tolerance | first −6.4 pp; second non-inferior; articulated v2 −17.2 pp (development, uncalibrated) | — | inconsistent; never primary; reported |
| Time | historical two-process 1.94×; integrated ≈ 1× (0.864 / 0.970 / 0.998) | `integrated_runtime_benchmark.json`, `confirmation_x.json` | no time saving; overhead removed |
| Fallback bound | 0 violations (406 + 151 objects); exhaustive and property tests | `fallback_audit.json`, `confirmation_x.json` | holds (candidates only) |
| Native equivalence | difference 0; identical decisions for the historical models and for X | `native_inference_validation{,_x}.json` | holds |
| GPD (ranking-controlled) | development 1.28× M0; second confirmation 6.33× M0, load −7.3 pp; stochastic | `external_baselines_v2.json`, `gpd_variability.json`, `confirmation_x.json` | descriptive; Dex-Net and Contact-GraspNet closed |
| Representations | no gain from frozen DINOv2, DINOv3 or OpenCLIP (H5 not supported) | Phase VII | bounded negative result |
| DKL, ADKF-IFT | a perfect order would save ≈1.4 ms per object | `ordering_ceiling.json`, `DKL_ADKF_DECISION.md` | line closed |
| Optuna (exploratory) | 900 trials (20 per family and fold). No tuned MLP family examined fewer candidates than its fixed counterpart (ratios 0.995–1.075) or than gradient boosting (1.11–1.22); no tuned visual family met the development rule | `OPTUNA_EXPLORATORY.md`, `results/optuna_mlp.json`, `results/optuna_contrasts.json` | exploratory only; no inclusion; not improved by a limited search (not equivalence) |
| Tests | fmt and clippy clean; 368 Rust tests, 513 Python tests pass (Python re-run on 2026-11-27); 65 skips investigated (external Phase III and IV bundles, optional numpyro) | `TEST_RESULTS.md` | pass |
| Licences | Apache-2.0 for own code (NOTICE); third-party material keeps its licences; the manuscript is not Apache; YCB conservative; no DINOv3 in the package | `LICENSE_COMPLIANCE_REPORT.md` | compliant. Open: the phase6 branch remedy. The manuscript wording window was decided by the author (not raised; public drafts disclosed as a preprint) |
| Reproducibility | statistical and partial levels in a clean container (510 / 510 decisions); full pipeline once in the development environment | `REPRODUCIBILITY.md`, `reproducibility/DOCKER_VALIDATION_RECORD.md` | partial reproduction verified; full reproduction not repeated |
| Editorial format (as relayed) | 16 pages ≤ 16 (iicol, template unchanged); abstract 235 words; 6 keywords; 52 numbered references; editable sources and PDF; supplement 11 pages | `publication/intelligent_service_robotics/` | meets the relayed limits |
| Literature and originality | verified review 2019–2026 (40 new or corrected records); M1 and the bound worded as known; gap stated as a combination not found in this search | `LITERATURE_REVIEW_FINAL.md`, `STATE_OF_THE_ART_MATRIX.md`, `NOVELTY_ASSESSMENT.md`, `REFERENCES_VERIFIED_FINAL.json` | done |
| Author data and declarations | name, affiliation, e-mail, ORCID, funding (none), competing interests (none), ethics and consents (not applicable), CRediT, AI declaration, previous work; no acknowledgements | `main.tex`, `declarations.md`, `author_information.md` | final (confirmed by the author) |
| Human review | personal review by the author, 26 November 2026; scope as communicated | `HUMAN_REVIEW_RECORD.md` | done; later material pending approval |
| Editorial status | not submitted; subscription route chosen (no APC); no option selected or paid | `JOURNAL_REQUIREMENTS_VERIFIED.md` | **pending the author** |

## 2. What the article claims, and what it does not

**It claims:**

- learned ordering reliably reduces the number of examined candidates, in two confirmations;
- adding the robustness terms does not cost efficiency beyond the margin;
- robustness was non-inferior to the original planner in the second confirmation (fragile in the first);
- the fallback bound holds;
- the integrated implementation removes the historical overhead.

**It does not claim:**

- better robustness or better load tolerance;
- faster planning;
- real-robot validity;
- that GPD's full pipeline is worse;
- that the foundation-model features are useless outside this use.

## 3. Status by category

| Category | Status |
|---|---|
| **Tasks completed** | Both confirmations; audits and corrections; native implementation and benchmarks; GPD comparison; literature review and verification; introduction, related work and comparison tables; author data and declarations; AI-use records; human-review record; manuscript (16 pages) and supplement (11 pages); reproducibility package test build |
| **Experiments completed** | Development; first confirmation; second confirmation; native validation; timing; articulated v2 (exploratory, failed a positive control); Optuna search (exploratory) |
| **Results pending** | None. The Optuna exploratory search finished on 27 November 2026 and supports no claim (`OPTUNA_EXPLORATORY.md`) |
| **Scientific limitations** | Simulation only; uncalibrated contact; one object family (EGAD) for both confirmations; one planner, cell and gripper; robustness and time gains not established (`LIMITATIONS.md`) |
| **Editorial rights questions** | None open. The DINOv3 licence copy was added to `research/axisforge-phase6-fortis-r` on 2026-11-27 (`b56f75c`, as authorised; nothing removed or rewritten). The manuscript-licence wording window (`RIGHTS.md`) was decided by the author on 2026-11-27: not raised with the editor; the cover letter discloses the public drafts as a preprint |
| **Human checks done** | The author's personal review of the research on 26 November 2026, and the approval of the 27 November material on 27 November (scope as communicated; no specific check is recorded beyond the author's statements) |
| **Human approvals still needed** | None. Zenodo deposit or DOI: declined by the author. At submission the author confirms the review model in Editorial Manager and, by submitting, the exclusive-submission sentence |
| **Not verified from this environment** | The ISR page redirects to a login here. The figure, supplementary-file and review-model rules were relayed by the author on 2026-11-27 (`JOURNAL_REQUIREMENTS_VERIFIED.md`). Still open: the review model (double-blind copies prepared) and the upload limits for PDF and ZIP |

## 4. Not to do

- No Phase VIII.
- No new hypotheses on the confirmation sets.
- No acceptance probability is given.
