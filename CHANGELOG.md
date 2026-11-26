# Changelog

Research history of AxisForge3D's learned grasp-candidate ordering. Engineering history is in Git. Every
result listed here stays in the repository unchanged, negative ones included.

## Phase VII extension (2026-11-26)

**Corrections.**

- `research/phase7_extension/FINAL_CORRECTIONS_REGISTER.md` corrects issues A–H in how the Phase VII results
  were described:
  - S6 appeared in the evaluation set although it was never measured there;
  - the wording of P3;
  - non-significant results read as no effect;
  - the fallback bound described as a bound on cost;
  - denominators;
  - p-values stored as 0;
  - "pre-registered" used where the protocol was only frozen in Git.
- P2 audit: score-based paired methods (Tango, Newcombe) do not support the historical non-inferiority, and a
  single changed outcome reverses it.

**Native implementation.**

- The learned order is now computed inside the Rust planner, so candidates are generated once.
- It is bit-exact against scikit-learn, with identical decisions, for the historical models and for the new
  model X.
- Integrated cost against the original planner (cold-process time):
  - development: 0.86×;
  - EGAD objects already evaluated in Phase VII: 0.97×, with a CI that includes 1.
  - The learned models add about 0.8 MiB of memory.

**External comparators.**

- GPD, ranking-controlled: GPD's classifier orders the shared candidate catalogue, using the unmodified
  library. GPD is stochastic by design (random shadow sampling), and its run-to-run variability is reported.
- Dex-Net 2.0 could not be run: all seven official weight links return 404.
- Contact-GraspNet could not be run: it needs a CUDA GPU.

**Development selection.** The rule was committed before any model was fitted. It selected **Joint-multi**: the
historical Joint form with a robustness model trained on robustness at all three torsions.

**DKL, ADKF-IFT and few-shot adaptation.** Not pursued: a perfect order would save at most about 1.4 ms per
object.

**New confirmation.** It ran once on EGAD positions 300–499, which had never been used. It was frozen in
`PHASE7X_FREEZE.json` (commit `b478289`) before any object was catalogued or planned, and the freeze was
checked before and after every stage. Results: `research/phase7_extension/results/confirmation_x.json`.

- Objects: 200 prepared and catalogued; 151 viable; 49 without a viable candidate.
- C1 supported: Joint-multi examined 0.449 [0.356, 0.567] times the original planner's candidates.
- C2 supported: robustness non-inferior, +3.3 pp (Tango [0.7, 7.5]).
- C3 supported: non-inferior to M1, 0.992 [0.930, 1.059].
- C4 not supported: cold-process time ratio 0.998 [0.978, 1.019]; slower on 142 of 200 objects.
- S1 supported: load tolerance non-inferior at −10 pp.
- S2 and S3 not supported: the multi-torsion robustness advantage that selected the method did not
  replicate.

**Exploratory.**

- Articulated v2 (development, uncalibrated): Joint less load tolerant than M0, −17.2 pp. The positive
  control was not passed as specified.
- Optuna MLP search: reduced budget, no inclusion in any claim.

**Manuscript.** Restructured for ISR:

- iicol, 14 pages, 240-word abstract, 6 keywords;
- new title: "Data-Efficient Grasp Candidate Ordering with Learned Feasibility Priors and Bounded Fallback";
- Statements and Declarations;
- LLM use described in Methods.

**Verdict.** `research/phase7_extension/PUBLICATION_READINESS_FINAL.md`: scientifically ready, pending author
approval. Not submitted.

**Licensing.**

- Apache-2.0 `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `LICENSES/`.
- YCB and OpenCLIP blockers resolved.
- DINOv3 embeddings excluded from the new package.
- The manuscript is excluded from Apache-2.0.

**Reproducibility.** A local package with inclusion and exclusion manifests; a slim validation Docker image;
a clean-environment statistical reproduction.

## Phase VII (2026-11-26)

- Confirmatory evaluation on 300 EGAD objects, under a protocol frozen with SHA-256 digests.
- P1 supported: Joint examined 0.61× the candidates of the original planner.
- P2 supported narrowly by the pre-specified test. The extension audit shows it is fragile.
- P3 supported: efficiency non-inferior to M1.
- None of the seven secondary hypotheses supported.
- Full online cost of the two-process implementation: 1.94×.
- Frozen vision-model features gave no detectable gain.

## Phase VI and earlier

- Phase VI: the FORTIS-R robustness mechanism and the simple fusion orders (M0–M8), tuned on development.
- Phase V: labelled catalogues, the exact ordering simulator, baselines and adaptive transfer, a MuJoCo dynamic
  validation.
- Phase IV: frozen vision foundation-model features (DINOv2, DINOv3, OpenCLIP) and the visual grasp prior.
- Phases I–III: the planner, collision and motion stack, and the first datasets.
