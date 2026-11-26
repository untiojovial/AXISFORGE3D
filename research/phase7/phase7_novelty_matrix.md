# Phase VII — novelty matrix

> **Historical Phase VII document, kept as written.** Some wording here was corrected later: "pre-registered" and "hash-signed" mean a protocol frozen in Git with SHA-256 digests, not a registry preregistration or signatures. Other corrections cover P3, the fallback bound and non-significant results. See `research/phase7_extension/FINAL_CORRECTIONS_REGISTER.md` (items B, C, D, H1, H2) and the current manuscript.

**Rule.** Compare only with verified works ([phase7_literature_review.md](phase7_literature_review.md)).
The search was targeted (about 40 queries), not systematic. "No direct precedent found" means *in this
search*, never "new to the world".

| Element | Closest verified precedent | What this work adds | Status |
|---|---|---|---|
| Learned feasibility classifier that orders a planner's candidates (M1) | Lou et al. 2020; Akinola et al. 2021; Qin et al. 2025 | Nothing essential in the method. The evaluation replays the **exact** order arithmetic of a validated native planner over exhaustively labelled catalogues (316 development parts, EGAD confirmation) | **Not new** |
| Product of utility and feasibility/robustness probabilities (M4, M9, FORTIS-R Joint) | Gardner et al. 2014; Hernández-Lobato et al. 2015 (constrained BO) | Applied as an examination order inside a complete planner, with a centring term that restores what the learned order removed | **Not new as an idea**; an engineering combination evaluated under a pre-registered protocol |
| Measured tension between planning speed and dynamic robustness | Weisz and Allen 2012 (score vs pose-error robustness) | The conflict is between **reaching feasibility quickly** and **dynamic robustness**, measured with corrected physics (v3), a torsional-friction grid and an articulated replication | Empirical contribution; **no direct precedent found** |
| Simulator-level dependence of load-tolerance conclusions | None found | The floating gripper and the articulated arm agree in standard conditions (κ 0.74–0.82) but reverse heavy-load contrasts | Empirical caution; no precedent found; development evidence only |
| Worst-case bound by interleaving with the original order (N ≤ 2·min) | Round-robin and portfolio arguments are textbook; no grasp-specific precedent was searched | A deterministic guarantee against negative transfer, at almost no mean cost in development | **Not claimed as new**; a safety mechanism |
| N_F, N_FR and Success@B under a stroke gate, population-level with non-viable objects | Not found | Joint measure of planning cost and dynamic validation per object | Methodological proposal; no precedent found |
| Frozen vision features (DINOv2/v3, OpenCLIP) for this ordering problem | Caron 2021; Oquab; Siméoni 2025; Radford 2021 (the models) | Controlled test with permuted, untrained and linear controls and local patch features. **No added value** in development under the pre-declared rule | Bounded negative result; not a method |
| ADKF-IFT / DKL for grasping (thesis line) | Chen et al. 2023; Wilson et al. 2016 | Not run. Development shows a 1.30-candidate ceiling, which leaves no room for adaptation | No new evidence |
| Exact order simulator of the native planner | Not searched specifically | Lets any prior be evaluated without replanning; 60/60 encoded-prior agreement | Tool; no novelty claim |

## How to read this for the manuscript

The defensible contribution is empirical and methodological, not "a new grasping algorithm":

1. A pre-registered, independently confirmed evaluation (EGAD, frozen before opening) of a simple fusion
   that keeps a learned prior's efficiency while protecting dynamic robustness.
2. The corrections behind it, and their effect:
   - masses;
   - centre of mass;
   - rotation criterion;
   - torsion grid;
   - articulated pilot.
3. Bounded negative results: foundation-model features, ranking loss, few-shot adaptation, and the
   full-cost question (H4).
