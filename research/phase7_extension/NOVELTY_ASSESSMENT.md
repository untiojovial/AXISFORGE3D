# Novelty assessment (2026-11-27)

This assessment rests on the final literature review (`LITERATURE_REVIEW_FINAL.md`), the matrix
(`STATE_OF_THE_ART_MATRIX.md`) and the results of both confirmations. "No precedent found" always means *in this
targeted search*. Nothing here claims that AxisForge3D is the first in the world at anything.

## 1. What came before AxisForge3D, and what this study is

### The master's thesis (2025)

Pérez González, Universidad de Zaragoza, TAZ-TFM-2025-047.

- **Problem.** Sequential Bayesian optimisation of grasps for a humanoid (iCub) hand in the Simox simulator.
- **Method.** Informed by DINO features and context, with deep-kernel Gaussian processes. Methodologies studied there
  included Bayesian optimisation, deep kernel learning, ADKF-IFT and context transfer.
- **Results.** As described in the thesis. None of them is reported in the article as new.

### This study (AxisForge3D, 2026)

It is independent later work on a different problem.

- **Problem.** The examination order of a complete parallel-jaw planner with a fixed catalogue.
- **Method.** Learned feasibility and robustness models evaluated inside the planner, a bounded fallback, and two
  frozen confirmations.
- **What it shares with the thesis.**
  - Interest in learned priors for grasping.
  - The use of foundation-model features, as one representation tested, with a negative result.
- **What it does not share.** The optimisation method, the hand, the simulator, the evaluation design and the results.

## 2. Contributions

**Types.** M methodological; Ma mathematical; E empirical; C computational; En engineering; R reproducibility.

**Defensible originality** uses this scale:

- **none**: known, used as is;
- **low**: a known idea applied in a new setting;
- **moderate**: a combination or a measurement that the search did not find elsewhere, with confirmatory evidence;
- **substantial**: a new method or result with no close antecedent. Not claimed for anything here.

| # | Contribution | Type | Closest antecedents | Difference | Evidence | Statistical result | Limitations | Defensible originality |
|---|---|---|---|---|---|---|---|---|
| 1 | Learned feasibility to order a complete planner's candidates (M1) | M, E | Qin et al. 2026; Liu et al. 2025; Jiang et al. 2022; Wells et al. 2019; Lou et al. 2020; Akinola et al. 2021 | Exact replay over exhaustively labelled catalogues; replicated on a different object population | P1 and C1 (the fusions; M1 is the reference) | P1 0.609 [0.516, 0.711]; C1 0.449 [0.356, 0.567] | simulation; one planner and gripper | **none** for the idea; **moderate** for the replicated, prospective measurement |
| 2 | Fusion of feasibility, learned dynamic robustness and centring (Joint) | M | constrained Bayesian optimisation (Gardner et al. 2014); quality plus reachability (Jiang et al. 2022); learned success evaluators (GraspGen, Matak et al. 2026) | Robustness term learned from MuJoCo lift, hold and shake trials; product inside a planner's order; β fixed by a pre-declared criterion | P2, P3 | P3 0.993 [0.953, 1.037] (efficiency non-inferior). P2 −1.8 pp, supported narrowly by the frozen test, not robust (Tango p 0.040) | robustness gain not confirmed; load tolerance −6.4 pp in the first confirmation | **low**: an engineering combination of known ideas, evaluated prospectively |
| 3 | Joint-multi: robustness across three torsional-friction values | M, E | FIRMGrasp 2026 (analytic, CVaR over friction); GraRe 2026 (minimum-friction target) | A learned multi-friction robustness term inside an order, rather than an analytic metric | C2, C3, S2 | C2 +3.3 pp [0.7, 7.5] (non-inferior); C3 0.992 [0.930, 1.059]; S2 0.0 pp [−5.0, 5.0], **not supported** (development +25.2 pp; 82.1 % vs 57.0 %) | the selection advantage did not replicate; X behaved almost like Joint | **low**. The method is not shown to be better; the non-replication is the finding |
| 4 | Physical robustness under several conditions (v3: six standard conditions, three torsions, loads) | E | ACRONYM (shaking labels); FIRMGrasp (shake and pick under adverse friction); DexGraspNet-style disturbance tests | The corrected physics changes earlier conclusions; the floating gripper and the articulated arm disagree under heavy loads | development, both confirmations, the articulated replication | 6.1–18.4 % of development verdicts change across the torsion grid; κ 0.74–0.82 in standard conditions, 0.60 at 2 kg, with reversed load contrasts | uncalibrated; articulated v2 failed a positive control | **moderate** as an empirical caution: no direct precedent found for the simulator-level reversal |
| 5 | Round-robin fallback, N_hybrid ≤ min(2N_L − 1, 2N_D) | Ma, En | step-by-step parallel simulation, factor two (van den Brand et al. 2024); algorithm portfolios (Gomes and Selman 2001); restarts (Luby et al. 1993); MPNet; PLOI | Applied to candidate examination with explicit conditions; −1 refinement from the alternation | exhaustive tests on all orders of 5 candidates; property tests; 406 + 151 objects | 0 violations | concerns candidates only, not time, robustness, safety or success | **none** as mathematics; **low** as an applied, checked safeguard |
| 6 | Bit-exact native inference in the Rust planner, with measured cost | C, En | GraRe and GraspGen report latency; Qin reports time to solution | Integrated end to end; bit-exact against the reference models; cost measured on held-out objects | native validation; benchmark; C4 | 786,760 candidates with difference 0; historical two-process 1.94×; integrated 0.864 / 0.970; C4 0.998 [0.978, 1.019], **not supported** | shared cloud container; cold processes | **low** as engineering. The empirical point that fewer examined candidates did not reduce time is a **moderate** finding |
| 7 | Two confirmations frozen before testing, on reserved objects | E, R | standard benchmark splits (GraspNet-1Billion, ACRONYM); recommendations on evaluation practice (Kress-Gazit et al. 2024) | Hypotheses, margins, models, analysis code and the planner binary frozen with SHA-256 digests; run once; negative results reported | `PHASE7_MODEL_FREEZE.json`, `PHASE7X_FREEZE.json`, execution records | — | Git freezes are not registry preregistrations; one object family; the second confirmation was designed knowing the first | **moderate** as evaluation practice for this problem; no grasp-ordering study with a frozen protocol was found |
| 8 | Ranking-controlled comparison with GPD on a shared catalogue | E | GraRe (re-ranking unchanged candidates); benchmarks with shared scenes | GPD's classifier ranks the planner's own catalogue | development, second confirmation | 1.28× and 6.33× M0's candidates; load −7.3 pp [−14.4, −0.6] | GPD is stochastic; its full pipeline is not evaluated | **low**: a controlled, descriptive comparison |
| 9 | Physical and statistical sensitivity analyses | E | — | Torsion grid, rotation thresholds, articulated arm; Tango, Newcombe and cluster bootstrap | supplement, audits | P2 fragile; C1–C4 conclusions robust to the cluster bootstrap | — | **low**; their value is in qualifying the claims |
| 10 | Negative results: frozen foundation-model features; transfer across grippers measured | E | DINO and CLIP manipulation systems (DINOBot, F3RM, LERF-TOGO, ORACLE-Grasp) | A controlled test with permuted, untrained and linear controls in an ordering problem | development; H5 in the first confirmation | H5 1.021 [0.971, 1.073], not supported. A model trained at 80 mm needed 0.94 and 1.07 times the candidates of models trained at 40 and 120 mm | one problem; frozen features only | **low to moderate**: a bounded negative result |
| 11 | Reproducibility (exact order replay, frozen models, per-object results, package with manifests) | R, En | — | Any order can be evaluated without re-planning; statistical, partial and full reproduction levels | `REPRODUCIBILITY.md`, the clean-container record | 510 / 510 decisions in a clean container | full pipeline not repeated in a clean environment | **low** (good practice) |

## 3. The central narrative, examined

**The narrative proposed.** Planners can examine many candidates before finding a viable grasp. Learning changes the
order, and the change can affect both efficiency and the physical properties of the selected grasp. AxisForge3D
studies feasibility prediction, physical robustness and ordering together, with a bounded fallback. It evaluates them
with controlled experiments, relevant comparators, the full computational cost and objects not used for selection.

**Assessment.** The narrative is justified as a description of *what was studied and how*. Two corrections are needed
so that it matches the results:

1. **It must not imply that combining the components produced better grasps.** Robustness was non-inferior, not
   better. The development gains, including the multi-friction gain of 82.1 % against 57.0 %, did not replicate.
2. **It must not imply faster planning.** The time ratio was 0.998 [0.978, 1.019].

The manuscript's introduction and conclusion were written with both corrections. The defensible message is narrow:
learned ordering reliably saves planner evaluations, and the robustness and time claims are not supported by this
study.

## 4. Claims and their limits

| Quantity | What can be claimed | What cannot |
|---|---|---|
| Examined candidates | Reduced, in two confirmations (0.61 and 0.45 of M0) | that a reduction in candidates implies a proportional reduction in time |
| End-to-end time | The two-process overhead is removed | a time saving (C4 not supported); equal time (no equivalence margin was set) |
| Standard robustness | Non-inferior in the second confirmation; fragile in the first | superiority (not pre-specified in the second; not supported in the first) |
| Multi-friction robustness | 82.1 % for Joint-multi in development, **development only** and optimistic because it was selected on the same parts | any confirmed advantage (S2 0.0 pp) |
| Load tolerance | Inconsistent: first −6.4 pp; second non-inferior; articulated v2 −17.2 pp in development | robustness to loads |
| Development vs confirmation | Development numbers are out of fold and not confirmatory | that development effects carry over |
| Simulation vs hardware | Everything is simulated | any real-robot validity; no real-robot experiment was run |

## 5. Wording rules adopted

- **Precedence.** "Known" or "precedented" where the review found antecedents (items 1, 2, 5); "not found in this
  search" otherwise.
- **Combination.** The combination is not itself presented as a contribution. The contribution is the controlled,
  prospective evidence it produced.
- **Methods not invented here.** DKL, ADKF-IFT, gradient boosting, foundation models and the parallel-execution bound
  are not presented as inventions of this study.
- **Bayesian optimisation.** The final method is not described as Bayesian optimisation. It is a learned order
  evaluated once per object.
