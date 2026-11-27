# Final corrections register (Phase VII extension)

Started 2026-11-26 from commit `15fd43f`. Every item below changes how a committed result is *described*; none
changes a committed result. `research/phase7/results/*`, `PHASE7_PROTOCOL.md`, `PHASE7_HYPOTHESES.md` and
`PHASE7_MODEL_FREEZE.json` are not edited. The machine-readable values are in
`research/phase7_extension/results/corrections_addendum.json` (A, E, F) and
`research/phase7_extension/results/p2_statistical_audit.json` (P2). Both are produced by scripts that read
only committed files:

- `tools/research/phase7_extension/corrections_addendum.py`
- `tools/research/phase7_extension/p2_audit.py`

The starting figures were first checked against the result files
(`research/phase7_extension/results/start_verification.json`). All 14 quoted figures matched.

Status values:

- *registered*: the correct wording is fixed here and is applied when the manuscript is revised (stage 14);
- *applied*: the change is in the manuscript or document;
- *closed*: nothing to change.

## Status update (final manuscript, 2026-11-26)

Every *registered* item below is now **applied**. The item tables keep their original status lines as the record
of when each correction was registered.

| Item | Applied in |
|---|---|
| A, S6 for the evaluation set | `confirmation.tex` ("measured on 60 primary objects only; the evaluation set was not timed") |
| B, P3 wording | `confirmation.tex`, `discussion.tex`, `conclusion.tex`, abstract (rewritten), `cover_letter.md` |
| C, non-significant ≠ no effect | `confirmation.tex` (secondary results, evaluation set), `conclusion.tex`, `discussion.tex` |
| D, the bound concerns candidates | abstract, contributions, Methods ("Round-robin fallback", with its conditions), `confirmation.tex`, `confirmation_x.tex`, `discussion.tex`, `conclusion.tex` |
| E, denominators | each rate states its population; P(F) trained on all 316 development parts (`confirmation.tex`); table captions |
| F, p-values stored as 0 | full precision from `corrections_addendum.json` and `confirmation_x.json`; the macro formatter never prints 0 or 1 |
| G, margins vs signs | `confirmation.tex` (S5 and torsion sensitivity worded as specified); `confirmation_x.tex` (C2 superiority not claimed; C4 not read as equal cost) |
| H1, H2, "pre-registered" / "hash-signed" | removed from the manuscript, the cover letter and the reviewer template. The historical Phase VII documents keep their wording, with a note pointing here |
| H3, title | new title, "Data-Efficient Grasp Candidate Ordering with Learned Feasibility Priors and Bounded Fallback"; robustness is not in the title |
| H4, cost estimate | replaced by the measured native cost (Sect. 8) and C4; the historical figure is kept as history |
| H5, P2 audit | main text of the first confirmation (Tango, Newcombe, fragility) and supplement |
| H6, H7 | `research/phase7_extension/LIMITATIONS.md`; the new abstract does not describe the articulated model as a replication |

## A. S6 (H4) reported for the evaluation set

| | |
|---|---|
| Evidence | `confirmation_results.json` → `sets["evaluation set (secondary)"].secondary["S6 (H4) ..."]` is identical to the primary entry (ratio 1.9375, 95 % CI 1.836–2.035, 60 objects). None of the 60 timed objects is an evaluation-set object. |
| Cause | `tools/research/phase7/egad_confirm.py` (lines 617–620) adds S6 to every set from the same `timing.json` without filtering by set. |
| Consequence | The evaluation set appears to have a measured full online cost that was never measured. The Holm step of the evaluation-set secondary family counted 7 hypotheses instead of 6. Recomputed without S6, every Holm p of that family is still 1.0, so no decision changes. The evaluation set was reported descriptively anyway, outside both families. |
| Correction | The evaluation-set S6 is "not measured". The S6 result belongs to the primary set only: 60 seeded primary viable objects. The original file is kept; the correction is in `corrections_addendum.json` (`A_S6_in_evaluation_set`). The analysis script is not edited (it is part of the freeze); a note is added to the reproducibility documentation. |
| Status | registered (manuscript: the evaluation-set paragraph does not quote S6; checked) |

## B. P3 wording

P3 is a non-inferiority test of `N_F(Joint)/N_F(M1)` with ratio margin 1.25. The result was ratio 0.993, 95 % CI
0.953–1.037, one-sided p = 2.5·10⁻²². This supports the conclusion that Joint does not examine more than 25 % more
candidates than M1. The CI is also compatible with differences of up to about −5 % / +4 %. P3 does not show that
the two orders examine "the same number" of candidates, nor that the extra terms "cost nothing".

| Location | Current text | Corrected text |
|---|---|---|
| `confirmation.tex` (primary results) | "Joint and M1 examined the same number of candidates (ratio …)" | "Joint was non-inferior to M1 in examined candidates (ratio …; the 95 % CI excludes increases above 4 %, well inside the 1.25 margin)" |
| `discussion.tex` (first paragraph) | "adding the robustness and centring terms cost nothing in efficiency relative to the plain learned order (P3)" | "adding the robustness and centring terms did not increase the number of examined candidates beyond the pre-specified margin relative to the plain learned order (P3); the interval excludes increases above 4 %" |
| `main.tex` abstract | "and as many as the plain learned order (ratio …)" | "and was non-inferior to the plain learned order (ratio …, margin 1.25)" |
| `conclusion.tex` | "and as few as the plain learned order" | "and was non-inferior in examined candidates to the plain learned order" |
| `cover_letter.md` | "the robustness-aware variant costs nothing in efficiency" | "the robustness-aware variant is non-inferior in examined candidates to the plain learned order" |

Status: registered.

## C. Non-significant is not "no effect"

| Location | Current text | Problem | Corrected text |
|---|---|---|---|
| `confirmation.tex` (secondary) | "M4 and M1 had the same load tolerance" | S1: difference 0.0 pp, 95 % CI −2.3 to 2.3, 3 vs 3 discordant objects. Not significant; not a test of equivalence. | "M4 was not more load tolerant than M1 (0.0 pp, CI −2.3 to 2.3)" |
| same | "M9 was not more robust than M1" | correct as written | — |
| same | "Joint and M1 had the same load tolerance and robustness" | S3 +0.9 (−1.4, 3.6); S4 −0.9 (−3.2, 1.4) | "Joint was not shown to be more load tolerant or more robust than M1 (S3 …, S4 …)" |
| same (evaluation set) | "with equal robustness (80 % and 80 %)" | equal observed rates in 35 objects; not evidence of equality | "with the same observed robustness rate (80 % in both; 35 objects, not tested)" |
| `conclusion.tex` | "Frozen vision foundation-model features added nothing measurable" | H5 / S7: ratio 1.021 (0.971–1.073); not supported | "Frozen vision foundation-model features did not reduce the number of examined candidates (H5 not supported; the CI excludes reductions larger than 3 %)" |
| `main.tex` abstract | "foundation-model features gave no measurable gain" | same | "foundation-model features gave no detectable gain (H5 not supported)" |
| `discussion.tex` (foundation models) | "did not improve on a gradient-boosting model" | acceptable (no superiority claim); keep, add the CI | — |
| H1, H2 (historical) | "did not replicate" | correct: the historical one-sided superiority was not supported. It does not mean the effect is zero. | add "(not supported; the CIs include zero and small effects in either direction)" |
| MLP results (`mlp_benchmark.json`, development) | "no MLP improved on gradient boosting" | development, post-hoc; correct only as "not shown to improve" | wording check in the supplement |

Status: registered.

## D. The fallback bound is about candidates

The round-robin bound `N_hybrid ≤ 2·min(N_L, N_D)` counts examined candidates. It is a deterministic property of
the interleaving: a candidate is feasible or not independently of the order. It says nothing about:

- time: the learned order adds feature and inference time, and generation dominates the total;
- total cost;
- safety;
- the robustness of the grasp chosen, which can differ from the grasp either order would choose.

| Location | Current text | Corrected text |
|---|---|---|
| `main.tex` abstract | "with a round-robin fallback bounded at twice the original planner's cost" | "with a round-robin fallback that examines at most twice as many candidates as the original planner" |
| `main.tex` contributions | "whose cost is never more than twice that of the original order" | "which never examines more than twice as many candidates as the original order" |
| `main.tex` §Methods, paragraph title | "Fallback with a guarantee. … bounds the cost" | "Fallback with a bound on examined candidates. … bounds the number of examined candidates; it bounds neither time nor the robustness of the chosen grasp" |
| `confirmation.tex` (fallback) | "never exceeded twice the original planner's cost" | "never examined more than twice the original planner's candidates" |
| `conclusion.tex` | "never exceeded twice the original cost" | "never examined more than twice the original planner's candidates" |
| `discussion.tex` (practical guidance) | "to bound the worst case at twice the original cost" | "to bound the number of examined candidates at twice the original" |

Status: registered. Implementation check (stage 15): exhaustive and property tests, and the native implementation
on every viable object.

## E. Denominators

Recounted from the files (`corrections_addendum.json`, `E_denominators`):

| N | Population | Used for |
|---|---|---|
| 316 | development parts with labelled catalogues | feature and probability analyses; training set of the frozen P(F) (so development probabilities are in-sample) |
| 239 | development parts with at least one viable candidate | representation study (L0–L11), `mlp_benchmark.json` |
| 151 | development dynamic parts (`dev_parts.json`) | physics, fusion selection (M0–M9, Joint), manuscript `\devParts` |
| 60 | development parts timed | full online cost in development (`full_online_costs.json`) |
| 300 | primary EGAD objects | population metrics (share with a viable grasp, success within a budget) |
| 220 | primary viable objects | P1–P3, S1–S5, S7, all rates "on viable" |
| 80 | primary objects without a viable candidate | population metrics only |
| 45 / 35 | evaluation-set objects / viable | descriptive only, not part of either family |
| 60 | seeded subset of the primary viable objects | S6 (H4) full online cost |

Rule for the manuscript: each rate states its denominator. Rates "on viable" (conditional on a feasible grasp
existing) are never compared with population rates. The abstract's "151 household objects" is the dynamic
development set, which is correct for the development N_F, robustness and load figures quoted next to it. The
text must say that P(F) was trained on 316 parts.

Status: registered (manuscript check at stage 14).

## F. p-values stored as 0

`holm_p` in `confirmation_results.json` is rounded to 6 decimals, so P1 and P3 are stored as `0.0`. The unrounded
values are:

- one-sided p: P1 3.2·10⁻⁹, P3 2.5·10⁻²²;
- Holm-adjusted p: P1 6.4·10⁻⁹, P3 7.4·10⁻²²;
- the one-sided p-values were already stored unrounded, and the recomputation reproduces them exactly.

Report as "p < 10⁻⁸" and "p < 10⁻²¹", never as "p = 0".

Status: registered; values in `corrections_addendum.json` (`F_full_precision_holm_primary`).

## G. A negative difference is not a failed margin, and a failed margin is not a negative difference

- **P2** (−1.8 pp; pre-specified lower bound −4.9 pp > −5). Non-inferiority supported despite a negative point
  estimate. This is correct, but the text must not read it as "equal" robustness.
- **S5** (load, −6.4 pp, bootstrap CI −10.9 to −1.8, lower 97.5 % −11.0 < −10). Non-inferiority not shown. The
  claim "less load tolerant" rests on the unadjusted bootstrap CI excluding zero, not on the non-inferiority test.
  The text says both and must keep them apart: "the non-inferiority margin was not met; the unadjusted CI lies
  below zero".
- **Torsion sensitivity** (−6.4 pp at 0.001 m, CI −10.5 to −2.3; −2.7 pp at 0.02 m, CI −5.5 to −0.5). Both CIs
  lie below zero, and neither lower limit clears −5. "Non-inferiority holds only at the pre-specified torsion" is
  correct. The sensitivity analyses use bootstrap CIs, not the pre-specified z test; say so.
- **Rotation thresholds** (5°: −0.9, CI −6.4 to 4.6; 20°: −2.3, CI −5.0 to 0.5). Negative point estimates whose
  CIs include zero. These are not losses of robustness, and with lower limits at or below −5 they are not
  demonstrations of non-inferiority either.

Status: registered.

## H. Other contradictions and imprecisions found

| # | Location | Problem | Correction | Status |
|---|---|---|---|---|
| H1 | `main.tex` (title and l. 17, 23), `conclusion.tex`, `discussion.tex`, `confirmation.tex`, `cover_letter.md`, `reviewer_response_template.md`, `novelty_matrix.md`, `publication_readiness.md`, `scientific_review.md`, `limitations.md` | "pre-registered" / "pre-registration": the protocol was frozen and hashed in this repository, not registered with a third party | "pre-specified protocol", "protocol frozen before evaluation" | registered |
| H2 | `conclusion.tex`, `publication_readiness.md` | "hash-signed": SHA-256 digests are not signatures | "with SHA-256 digests committed before the data were opened" | registered |
| H3 | Title "Learning where to look first without losing robustness" | Implies equal robustness. P2 is narrow, method-dependent (see P2 audit below), and fails at the other torsion values. | Neutral title, e.g. "Learning where to look first in a complete grasp planner: a protocol-frozen evaluation of efficiency, robustness and cost" | registered (author decision) |
| H4 | `discussion.tex`, `confirmation.tex` §cost | "our estimate of that implementation is close to parity": an engineering estimate | Replaced by the measured cost of the native implementation (`integrated_runtime_benchmark.json`), labelled as a post-hoc development measurement; the old estimate is kept as historical | registered (after the benchmark) |
| H5 | P2 | With 12 discordant objects (4 vs 8), the pre-specified z test (sample SD) gives p = 0.0216 and lower bound −4.90 pp. Tango's score test gives p = 0.040, CI −5.45 to 1.46. Newcombe's hybrid score interval gives −5.30 to 1.44. Neither supports non-inferiority at −5 pp. | The pre-specified result stands as the confirmatory result. The manuscript reports that score-based methods for paired proportions do not support it, and that P2 is fragile: changing the Joint outcome of a single object reverses the pre-specified decision (p = 0.035 or 0.048; `fragility_pre_specified` in the P2 audit). | registered |
| H6 | `limitations.md` l. 72 | "both below zero" without the CIs | add CIs (G) | registered |
| H7 | Abstract: "replicated it with an articulated arm" | The articulated arm was a replication of the physics on 220 objects with four standard conditions, not the full protocol | "checked it with an articulated arm model" | registered |

## I. Final technical review of the manuscript (27 November 2026)

A line-by-line check of the manuscript and the supplement against the results files, the code and the frozen
protocols. No result, hypothesis, verdict or frozen file changed. The numbers are regenerated by the two table
scripts; the only new macros are counts already in the stored results (`repViable` = 239, `nxConeTiesFirst` = 92).

| # | Location | Problem | Correction |
|---|---|---|---|
| I1 | `confirmation_x.tex`, C1 | "93 ties (both orders found a viable candidate first)": in 1 of the 93 ties both orders needed the same number of candidates, more than one (`confirmation_x/orders.json.gz`) | "in 92 of them both orders' first candidate was viable" (macro `nxConeTiesFirst`) |
| I2 | `main.tex`, Methods | The robustness model of Joint was trained on the previous (v2) robustness label (`PHASE7_PROTOCOL.md`, models table); the text did not say so | The training label of each robustness model is stated |
| I3 | `main.tex`, Methods | Not stated that the fused orders (M4, M9, Joint, Joint-multi) sort the makeable candidates by the fused score alone (`fusion.order`, weight 10^6), whereas M1 adds 2·p̃ to the preliminary score | Stated |
| I4 | `main.tex`, Dynamic validation | "Robust" and "load tolerant" were not defined as "held in all" conditions, and the loads are the part's mass set to 0.5, 1 and 2 kg (`v3_sweep.py`, `physics_report.py`) | Defined explicitly |
| I5 | `main.tex`, Representations | "No visual variant beat its permuted control after Holm correction, so no visual representation met the pre-declared development rule": the rule (`PHASE7_PROTOCOL.md`) does not mention Holm, and L10 meets both conditions on unadjusted intervals, against a control added after its result (D2) | Reworded; the rule and its application (L7, L8, L11) are stated in the supplement |
| I6 | `main.tex`, Methods | $N_{FR}$ used without definition | Defined |
| I7 | `confirmation.tex`, cost | "A perfect order would save 1.4 ms": the saving is relative to Joint (`ordering_ceiling.json`, `joint_excess_candidates`) | "relative to Joint" (also in the supplement) |
| I8 | `confirmation.tex`, P2 | Bootstrap CI lower limit −5.0 printed next to the pre-specified z lower limit −4.9 without saying which is which | Both labelled |
| I9 | `confirmation.tex`, `tab_x_dev`, supplement | "Phase VII" (an internal project name) used without definition | "the first confirmation" |
| I10 | `confirmation.tex`, cost | Medians followed by "(ratio 1.938)", a geometric-mean ratio | "geometric-mean ratio" |
| I11 | `confirmation.tex`, sensitivity | "four standard conditions" with the arm, without the reason | the arm model applies no pose noise |
| I12 | supplement, GPD | "the first-ranked candidate changed": the comparison is of the first viable grasp (`gpd_variability.py`); "−21.9 points less load tolerant" (double negative) | Corrected |
| I13 | supplement, Table S1 and Fig. S3 | N_F said to be over 316 parts (it is over the 239 with a viable candidate); the Robust column uses the previous (v2) criterion on 151 parts; the dagger was not explained | Caption states the denominators, the criterion and the dagger |
| I14 | `make_tables.py`, `make_tables_x.py` | Negative values at the start of a macro or after `{[}` were typeset with a hyphen instead of a minus (for example −1.8, −6.4, −2.0) | `minus()` also converts those; regenerated |
| I15 | `confirmation_x.tex`, `discussion.tex` | Power figures not tied to their scenario; "five discordant objects" typed by hand | "with the observed effects"; macro `nxCtwoFav` |
| I16 | `confirmation_x.tex`, GPD | "as robust" for a difference of 0.0 points with a ±5-point interval | "the same robustness rate" |

To keep the 16-page limit, the model size and memory, and the reasons Dex-Net 2.0 and Contact-GraspNet could not
be run, now appear only in the supplement, which already gave them.

**For the author (not changed).** The title begins "Data-Efficient", but the study does not measure data
efficiency (no learning curve or training-set-size analysis is reported); it measures examined candidates.

## P2 statistical audit (section 11)

Recomputed from `confirmation_orders.json` and `confirmation_trials.json.gz`
(`p2_statistical_audit.json`).

Paired table on 220 primary viable objects:

- both robust: 196;
- Joint only: 4;
- M0 only: 8;
- neither: 12.

Difference −1.82 pp.

| Method | Statistic | One-sided p (H0: Δ ≤ −5 pp) | Interval | Non-inferior? |
|---|---|---|---|---|
| Pre-specified: z with the sample SD (ddof 1) | z = 2.022 | 0.0216 (Holm 0.0216) | lower 97.5 % −4.90 | yes (narrowly) |
| Wald with the multinomial MLE variance | z = 2.027 | 0.0213 | lower 97.5 % −4.89 | yes |
| Tango score test (Tango 1998) | z = 1.754 | 0.0397 | 95 % CI −5.45 to 1.46 | no |
| Newcombe hybrid score interval, method 10 (Newcombe 1998) | — | — | 95 % CI −5.30 to 1.44 | no |

Superiority, exact McNemar: 4 vs 8 discordant objects, p = 0.93.

Reproduced exactly:

- the stored difference, lower bound and one-sided p;
- the decision rule (p < 0.025 after Holm and lower bound > −5);
- the rounding of `holm_p` to 6 decimals.

Discrepancy: the confirmatory conclusion depends on the variance estimator. The score methods recommended for a
paired difference with few discordant pairs do not support it. **P2 must not be presented as evidence that the two
orders are equally robust.** The two references are cited only for the formulas. Their bibliographic details
still have to be verified before they enter `references.bib`; that is item L1 in the literature check.
