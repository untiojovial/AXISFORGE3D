# Final scientific audit (Phase VII and extension)

Written after `results/confirmation_x.json` existed. It checks every claim of the manuscript against its
results file, the statistics of both confirmations, and the residual risks.

**This audit is a self-check by the AI system that assisted with the work.** It is not independent peer review.
The author carried out a personal review of the research on 26 November 2026 (`HUMAN_REVIEW_RECORD.md`). That review
is the author's own, not independent peer review, and it does not cover material produced afterwards (the Optuna
exploratory analysis and the final literature review), which the author has to approve.

**Optuna (2026-11-27).** The exploratory search finished after this audit's main text was written. It changes no
claim: no tuned MLP family beat its fixed counterpart or gradient boosting, and no tuned visual family met the
development rule (`OPTUNA_EXPLORATORY.md`). The supplement describes it as not improved by a limited search, not as
equivalence.

**Literature (2026-11-27).** The claims in §1 were checked against the final literature review. Two claims are now
worded as known ideas, not contributions: the learned feasibility order (precedents: Qin et al. 2026, Liu et al. 2025)
and the round-robin bound (a parallel-execution property: van den Brand et al. 2024, Gomes and Selman 2001). See
`LITERATURE_REVIEW_FINAL.md` and `NOVELTY_ASSESSMENT.md`.

## 1. Claims and their evidence

| Claim in the manuscript | Evidence | Population | Status |
|---|---|---|---|
| Learned orders examine fewer candidates than the original planner | P1: ratio 0.609 [0.516, 0.711], Holm p 6.4·10⁻⁹. C1: 0.449 [0.356, 0.567], Holm p 3.3·10⁻¹⁰ | 220 and 151 viable EGAD objects | **supported twice** |
| Adding the robustness and centring terms keeps efficiency within the margin of M1 | P3: 0.993 [0.953, 1.037]. C3: 0.992 [0.930, 1.059]; margin 1.25 | same | **supported twice** (non-inferiority, not equality) |
| Robustness non-inferior to the original planner (−5 pp) | P2 pre-specified test: −1.8 pp, lower −4.9, but Tango p 0.040 and Newcombe [−5.3, 1.4] do not support it, and it is fragile to one object. C2 (Tango, pre-specified): +3.3 pp [0.7, 7.5], Holm p 2.8·10⁻⁴; still non-inferior if one more object is "M0 only" ([−0.7, 7.0], post hoc) | same | **P2 fragile; C2 supported** |
| Robustness superior to the original planner | not a hypothesis of the second confirmation. In the first, P2-sup was not supported (4 vs 8, p 0.93). The C2 interval excludes zero (5 vs 0 discordant), but the reading is fragile and post hoc | — | **not claimed** |
| Robustness across torsions better than M1 (the criterion that selected X) | S2: 0.0 pp [−5.0, 5.0], not supported; development +25.2 pp (optimistic) | 151 | **did not replicate** |
| Load tolerance | First confirmation: S5, −6.4 pp; the non-inferiority margin was not met and the unadjusted CI lies below zero. Second: S1 non-inferior at −10 pp (−2.0 [−7.6, 3.4]). Articulated v2 (development): −17.2 pp | — | mixed; reported as such; never in a primary family |
| The integrated implementation does not make planning faster | C4: 0.998 [0.978, 1.019], Holm p 0.43. X was slower on 142 of 200 objects (frozen one-sided Wilcoxon p > 0.99). Post hoc: per-object median ratio 1.022 | 200 catalogued | **not supported** (no time saving; no equivalence test) |
| The integrated implementation removes the historical two-process overhead | historical 1.94×; integrated about 1× (development 0.864, EGAD 0.970, second confirmation 0.998) | 60 + 60 + 200 | supported descriptively (benchmark not confirmatory; C4 prospective) |
| The fallback examines at most min(2N_X − 1, 2N_M0) candidates | 0 violations in 151 + 255 objects of the native validation and in all 151 viable objects of the second confirmation; exhaustive and property tests | — | **holds**; it concerns candidates, not time, robustness or safety |
| Native models are bit-exact | maximum difference 0 over 786,760 candidates; 406 / 406 identical decisions per mode, historical models and X separately | 661 objects | **holds** |
| GPD's classifier is a poor examination order here | development: 1.28× M0's candidates. Second confirmation: 6.33× [5.00, 8.02]; robustness 0.0 pp, load −7.3 pp [−14.4, −0.6] | 151 | descriptive; ranking-controlled only; GPD is stochastic between runs |
| Frozen vision foundation-model features do not help | H5 not supported (ratio 1.021 [0.971, 1.073]); development controls | — | bounded negative result |

## 2. Statistics of the second confirmation

- **Denominators.**
  - 200 objects were prepared and catalogued, with 0 preparation or catalogue failures.
  - 151 were viable and 49 had no viable candidate.
  - C1–C3 and S1–S3 use the 151 viable objects. C4 uses all 200; the non-viable objects are timed too.
- **Tests as frozen.**
  - C1, C3, C4: one-sided t tests on per-object log ratios, with t CIs (estimand: the geometric-mean ratio).
  - C2 and S1: Tango score tests at the margin, with Tango CIs; Newcombe method 10 as sensitivity.
  - S2 and S3: exact one-sided McNemar.
  - Holm within each family.
  - Wilcoxon for C1 and C4 as a rank-based sensitivity. It tests a location shift, and no decision rests on it.
- **Holm-adjusted p.**
  - Primary: C1 3.3·10⁻¹⁰, C2 2.8·10⁻⁴, C3 1.4·10⁻¹⁰, C4 0.43.
  - Secondary: S1 0.012, S2 1, S3 1.
  - No p-value is stored as 0; the full-precision values are in `confirmation_x.json`.
- **Counts reported for each hypothesis.**

  | Hypothesis | Favourable | Unfavourable | Tied |
  |---|---|---|---|
  | C1 | 54 | 4 | 93 |
  | C2 (both / X only / M0 only / neither) | 140 / 5 / 0 / 6 | | |
  | C3 | 9 | 8 | 134 |
  | C4 | 58 | 142 | 0 |
  | S1 | 6 | 9 | 136 |
  | S2 | 6 | 6 | 139 |
  | S3 | 5 | 4 | 142 |

- **Ties.** Ties stay in the t tests; Wilcoxon drops zero differences (93 for C1, 0 for C4).
- **Cluster sensitivity.** A bootstrap by EGAD difficulty letter (24 clusters) gave:
  - C1 [0.322, 0.597];
  - C2 [0.8, 6.0] pp;
  - C3 [0.924, 1.069];
  - C4 [0.975, 1.020].

  All are consistent with the primary intervals. Dependence between neighbouring cells does not change any
  conclusion.
- **Power, in hindsight.**
  - C2 and C4 had been underpowered under the pessimistic scenario.
  - C2 came out supported.
  - C4 came out not supported with a narrow interval. The data exclude time changes larger than about 2 % in
    either direction, but no equivalence margin was pre-specified, so the article does not claim equal time.
- **References.** The Tango (1998) and Newcombe (1998) records were matched in Crossref, with DOIs. The
  implementation is checked against direct maximisation of the restricted likelihood and by simulated coverage.

## 3. Protocol compliance of the second confirmation

| Item | Status |
|---|---|
| Freeze before any catalogue, plan, score or simulation | freeze commit `b478289` at 18:42:35 UTC; the first build at 18:42:59; the first catalogue at 22:44:46. Only position 300 (`egad_C01`) was prepared earlier (17:59:22, mesh only), as declared |
| Freeze checked before and after every stage | 10 checks, all passed (`confirmation_x/execution_record.json`) |
| Run once; no re-sampling; no reserve set | one run. One container restart during the build: it was resumed and completed objects were kept |
| Methods, models, binary | as frozen: native orders from the fixed binary; Python replay with 0 mismatches |
| Phase III sealed objects | not used (EGAD positions only) |
| Outcomes reported as they came | C4 and S2–S3 not supported, reported. Post-hoc descriptions are labelled and in a separate file (`confirmation_x_posthoc.json`) |
| Timing conditions | the other jobs paused; load average about 1 (the timing process); CPU 3 pinned; one thread |

## 4. Fallback guarantee

**The bound.** `N_hybrid ≤ min(2N_X − 1, 2N_M0) ≤ 2 min(N_X, N_M0)`. It holds under three conditions:

- both orders are permutations of the same candidate set;
- feasibility does not depend on the order;
- at least one candidate is feasible.

**Checks.** The conditions hold for X by construction, because the native hybrid interleaves two orders of the
same gated catalogue and the checks are stateless. 0 violations in the second confirmation.

**Scope.**

- Objects without a feasible candidate (49) and exhausted budgets (0) are reported separately.
- The bound says nothing about time, cost, robustness, safety or real success.

## 5. Issues found during this audit and how they were handled

1. **C4 needed more than the geometric mean.** The frozen estimand (geometric mean ≈ 1) and the frozen rank
   sensitivity (X slower on 142 of 200 objects) pull in different directions. Both are reported, and a
   labelled post-hoc description explains why: the learned stage adds a few milliseconds to every object, and
   the shorter search repays them on some objects only. The confirmatory decision (not supported) is
   unchanged.
2. **C2's positive interval.** It is not presented as superiority. The text states that superiority was not
   pre-specified and that the reading is fragile.
3. **X and Joint are almost the same.** X behaved almost exactly like the first confirmation's Joint. The new
   robustness model changed few first grasps (descriptive comparison: robustness +0.7 pp, load 0.0 pp, N_F
   ratio 0.996). The article says so.
4. **Articulated v2.** Its positive control was not passed as specified. It is reported as an uncalibrated
   exploratory analysis and is used for no claim.
5. **GPD is stochastic.** Its single draw is reported; the variability is quantified on 20 parts.

## 6. Residual risks

- **Simulation only.** Uncalibrated friction and simulator-level choices can reverse load-tolerance
  conclusions.
- **One object family.** Both confirmations come from EGAD, and the second one was designed knowing the first.
- **One planner, one cell, one gripper.**
- **AI involvement.** All implementation and analysis were produced with AI assistance under the author's
  direction. The author reviewed the research on 26 November 2026. No independent peer review has taken place, and
  the author's approval of the material produced after 26 November is pending.
