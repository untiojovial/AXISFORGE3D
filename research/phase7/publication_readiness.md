# Phase VII — publication readiness

> **Historical Phase VII document, kept as written.** Some wording here was corrected later: "pre-registered" and "hash-signed" mean a protocol frozen in Git with SHA-256 digests, not a registry preregistration or signatures. Other corrections cover P3, the fallback bound and non-significant results. See `research/phase7_extension/FINAL_CORRECTIONS_REGISTER.md` (items B, C, D, H1, H2) and the current manuscript.

## Classification: **CONDITIONAL READINESS**

**What is ready.** The manuscript and its package are complete and internally consistent, and the PDF
compiles from the sources. The confirmatory evidence was produced under a frozen, hash-signed protocol.

**What blocks submission.** It is not ready to submit until the conditions below are met. Several of them
can only be met by the author. Acceptance is not guaranteed and is not predicted here.

## What the article can claim (all in simulation)

**Confirmed on 300 independent EGAD objects (220 viable).**

- **P1.** A learned examination order (FORTIS-R Joint) needs fewer planner evaluations than the original
  planner: ratio 0.609 [0.516, 0.711].
- **P3.** It is as efficient as the plain learned order: 0.993 [0.953, 1.037].
- **P2, narrowly.** Its robustness is non-inferior to the original planner's at the pre-registered
  5-point margin and friction value: −1.8 pp [−5.0, 1.4]; lower 97.5 % bound −4.9.

**Not confirmed, and reported as such.**

- Robustness superiority.
- The historical H1 and H2.
- Load-tolerance gains (Joint was worse than the original planner).
- A wall-clock saving (H4: 1.94× slower in the current implementation).
- Any value of foundation-model features (H5).

**Development-only contributions.**

- Physics corrections and the evidence that load-tolerance conclusions depend on friction and simulator
  level.
- The articulated pilot.
- The fallback bound.
- The representation study with controls.

## Conditions before submission (closed list)

| # | Condition | Who | Notes |
|---|---|---|---|
| 1 | Provide e-mail, ORCID and confirmed affiliation; complete funding and competing interests | Author | `author_information.md`, `declarations.md` |
| 2 | Read the journal's current guidelines (login-protected during preparation) and adapt: article type, length, abstract, declarations, AI policy, blind review | Author | `submission_checklist.md` |
| 3 | Decide the repository visibility and add a root `LICENSE` (Apache-2.0 is declared but the file is missing) | Author | `code_availability.md` |
| 4 | Decide on redistribution of DINOv3-derived artefacts and verify the YCB and OpenCLIP/LAION weight licences | Author | `PHASE7_DATA_MANIFEST.json` |
| 5 | If the journal requires archived data or code: authorise a release (and a DOI) of the final commit with the catalogues and collision pieces | Author | Not created |
| 6 | Read and approve every claim, the AI-use statement and the cover letter | Author | `ai_usage_disclosure.md`, `cover_letter.md` |
| 7 | Decide how to declare the overlap with the master's thesis | Author | `cover_letter.md` |

## Recommended, not required (scientific strength)

These would answer the most likely reviewer requests. Each would be **exploratory**: none may change the
frozen confirmatory results.

| # | Recommendation | Effort | Notes |
|---|---|---|---|
| R1 | One learned external baseline: GPD at the ranking-controlled level | Moderate | CPU only; dependencies installable; needs a C++ adapter. `external_baselines.json` |
| R2 | An exact (or Newcombe-type) paired interval for P2, as a check of the z-based bound | Small | — |
| R3 | An integrated implementation of the prior inside the planner, to measure the real wall-clock cost | Moderate | — |
| R4 | v3 robustness labels for all viable development candidates, then a fresh test on reserve positions 300–499, to separate label and population explanations for the non-replication | Large | — |
| R5 | Real-robot experiments and friction calibration | Hardware | Outside this repository |

## Verified state of the repository at classification time

- **Tests.**
  - cargo: 360 passed, 0 failed.
  - clippy: 0 warnings.
  - fmt: clean.
  - pytest: 500 passed, 65 skipped (skipped tests are not counted as passed).
  - One regression (a determinism test that compared wall-clock timings) was found and fixed; see
    PHASE7_REPOSITORY_AUDIT.md §6.
- Protocol and freeze: commit `de22a8a`. Confirmation results: commit `5a821dc` and later.
- No Phase I–VI result or protocol was modified. No history rewrite, no force push.
