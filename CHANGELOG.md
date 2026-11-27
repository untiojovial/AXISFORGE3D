# Changelog

Research history of AxisForge3D's learned grasp-candidate ordering. Engineering history is in Git. Every
result listed here stays in the repository unchanged, negative ones included.

## Editorial closing (2026-11-27)

**Author data and declarations** (confirmed by the author):

- Eduardo Pérez González, Independent Researcher, Zaragoza, Spain;
- e-mail eduperez2000g@gmail.com; ORCID 0009-0007-0303-8839;
- no funding; no competing interests;
- ethics and consents not applicable; materials not applicable;
- CRediT roles;
- the generative-AI declaration, built from the author's base text;
- no acknowledgements.

These are in `main.tex`, `declarations.md`, `author_information.md`, `CITATION.cff` and the draft Zenodo metadata.

**Human review.** The author's personal review of the research on 26 November 2026 is recorded in
`research/phase7_extension/HUMAN_REVIEW_RECORD.md`, with its scope as communicated. Material produced afterwards needs
the author's approval.

**Literature review.**

- `LITERATURE_REVIEW_FINAL.md`, `STATE_OF_THE_ART_MATRIX.md`, `NOVELTY_ASSESSMENT.md` and
  `REFERENCES_VERIFIED_FINAL.json`.
- The closest precedents are Qin et al. (RA-L 2026; replaces the earlier preprint entry), Liu et al. (2025 preprint)
  and GraRe (2026 preprint).
- The round-robin bound is now presented as a known parallel-execution property (van den Brand et al., SODA 2024;
  Gomes and Selman 2001), applied and checked.

**Manuscript.**

- New introduction: research questions RQ1–RQ4, and contributions with their evidence and outcomes.
- Related work by family, with a comparison table (Table 1) and an extended table in the supplement (Table S5).
- The Methods paragraph on AI use was revised.
- 16 pages. For length, the first confirmation's contrast figure moved to the supplement (Fig. S6). The conclusion was
  shortened.

**Final technical review** (`FINAL_CORRECTIONS_REGISTER.md`, section I). Sixteen wording and reporting corrections
found by checking the text against the results, the code and the protocols, among them:

- the C1 ties;
- the training label of Joint's robustness model;
- how the fused orders use the prior;
- the definitions of robust and load tolerant;
- the description of the visual-value rule;
- the reference order of the ordering ceiling;
- the denominators of Table S1;
- typographic minus signs in the generated numbers.

**Figure quality control (2026-11-27).**

- All eight figures were checked: margins, fonts, type size, line width, fit and collisions.
- Fixes: thin lines in Fig. 2; label collisions in S6; jargon and spacing in S1; units in S2 and S6; the
  denominator in S3; type size in S4; one font (Liberation Sans) in the supplementary figures.
- Details are in `FINAL_CORRECTIONS_REGISTER.md`, section K.

**Figures in the article (2026-11-27, at the author's request).** Fig. 1 (method overview, TikZ) and Fig. 2
(confirmatory effects of both confirmations, pgfplots, data from `make_figures_x.py`). Helvetica 8 pt at the column
width. Repeated text was shortened to keep 16 pages (`FINAL_CORRECTIONS_REGISTER.md`, section J).

**Submission preparation (2026-11-27).**

- **Approval.** The author approved the material of 27 November (`HUMAN_REVIEW_RECORD.md`).
- **Cover letter.** Dated 27 November 2026; code and results of the submitted version at `10e4eb0`.
- **Journal guidelines** (relayed by the author):
  - the supplement is Online Resource 1 (`ESM_1.pdf`) and is cited that way;
  - a double-blind copy of the manuscript and of Online Resource 1 (`main_anonymous.tex`,
    `supplementary_anonymous.tex`), with a title page;
  - a few phrases shortened to keep 16 pages.
- **DINOv3 licence.** The DINOv3 licence copy and a notice were added to `research/axisforge-phase6-fortis-r`
  (`b56f75c`, fast-forward, as authorised).
- **Zenodo or DOI.** Declined by the author.
- **Not done.** Nothing was submitted, released or paid.

**Optuna (exploratory)** finished on 2026-11-27 (`research/phase7_extension/OPTUNA_EXPLORATORY.md`): 900 trials, 45
outer-fold selections, 225 refitted models. No tuned MLP family examined fewer candidates than its fixed counterpart
(0.995–1.075) or than gradient boosting (1.11–1.22); no tuned visual family met the development rule. It supports no
claim and changes none. The supplement summarises it in one paragraph. The script's docstring was corrected (TPE seed
20261212 + k; `--mlp-ref`); the code that ran is unchanged.

**Title** (approved by the author on 2026-11-27): "Learned Grasp Candidate Ordering with Feasibility Priors and
Bounded Fallback: Two Prospective Confirmations in Simulation". It replaces "Data-Efficient ...": the study measures
examined candidates, not data efficiency.

The manuscript stays at 16 pages.

**No change** to any result, protocol, freeze or analysis.

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
