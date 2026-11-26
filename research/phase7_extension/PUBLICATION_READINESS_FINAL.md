# Publication readiness: final verdict (2026-11-26)

## Verdict: SCIENTIFICALLY READY, PENDING AUTHOR APPROVAL

**Done.** The evidence is complete:

- both confirmations were run once under frozen protocols;
- every claim in the manuscript is backed by a committed results file;
- negative and non-replicated results are reported;
- the manuscript meets the relayed format limits.

**Why not "ready for submission".**

- The author has not reviewed or approved the results, the analyses or the text.
- Personal and editorial data are missing (§3).

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
| Optuna (exploratory) | reduced budget (20 trials per family and fold); see `results/optuna_mlp.json` | — | exploratory only; no inclusion |
| Tests | fmt and clippy clean; 368 Rust tests, 512 Python tests pass; 65 skips investigated (external Phase III and IV bundles, optional numpyro) | `TEST_RESULTS.md` | pass |
| Licences | Apache-2.0 for own code (NOTICE); third-party material keeps its licences; the manuscript is not Apache; YCB conservative; no DINOv3 in the package | `LICENSE_COMPLIANCE_REPORT.md` | compliant. Open: the phase6 branch remedy; the editorial question on the manuscript wording window |
| Reproducibility | statistical and partial levels in a clean container (510 / 510 decisions); full pipeline once in the development environment | `REPRODUCIBILITY.md`, `reproducibility/DOCKER_VALIDATION_RECORD.md` | partial reproduction verified; full reproduction not repeated |
| Editorial format (as relayed) | 14 pages ≤ 16 (iicol, template unchanged); abstract 240 words; 6 keywords; numbered references; editable sources and PDF | `publication/intelligent_service_robotics/` | meets the relayed limits |
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

## 3. Blocking items for submission (author only)

1. Read and approve every claim, the analyses and the final text. The AI reviews are not independent peer
   review.
2. Provide or confirm:
   - e-mail and ORCID;
   - affiliation ("Independent Researcher, Zaragoza, Spain" is provisional);
   - funding, competing interests and the CRediT roles;
   - the statement of responsibility in the AI declaration.
3. Confirm the thesis record (repository URL or handle) and how it is disclosed to the editor.
4. Check the repository's visibility, and decide on:
   - the remedy for `research/axisforge-phase6-fortis-r` (DINOv3 licence copy);
   - any Zenodo deposit or DOI.
5. Consult the editor about the manuscript-licence wording window (`RIGHTS.md`), and confirm the subscription
   route at submission.

## 4. Not to do

- No Phase VIII.
- No new hypotheses on the confirmation sets.
- No acceptance probability is given.
