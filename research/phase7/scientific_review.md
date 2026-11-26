# Phase VII — simulated internal review (four perspectives)

**Not peer review.** These four reviews were written by the same AI system that ran the study, as a
structured self-check of the manuscript (`publication/intelligent_service_robotics/`) and the repository.
They do not replace independent review.

**How to read them.**

- Each review lists major and minor issues.
- Each issue carries one of two labels:
  - **[addressed]**, with the place where it is handled;
  - **[open]**, with what would close it.
- Numbers refer to `research/phase7/results/`.

---

## Review 1 — Machine learning

**Summary.** The paper evaluates learned examination orders for a grasp planner. It also tests frozen
vision foundation-model features against controls. The ML side is careful:

- grouped folds;
- three seeds;
- permuted, untrained and linear controls;
- a pre-declared value rule.

The negative representation result is credible.

**Major issues**

1. *Distribution shift of P(H|F).* The robustness model was trained on 151 GSO parts with v2 labels
   (rise and slip only). It was applied to EGAD shapes judged with v3. The non-replication of the
   robustness gain may be a label or population mismatch, not a property of the fusion.
   - **[addressed]** The Discussion states both explanations and says the study cannot separate them.
   - **[open]** Retrain P(H|F) on v3 labels for all viable candidates (a v3 sweep), then test on fresh
     reserve objects (positions 300–499).
2. *Calibration and AUC vs ranking.* AUC is high for every model (0.97–0.996), yet N_F differs by
   15–40 %. A high AUC does not mean an optimal ranking.
   - **[addressed]** The paper reports N_F, S@k and ECE, and ranks by N_F.
3. *Post hoc L10 controls.* H5 was retargeted to L10 after development, and its controls were added
   afterwards.
   - **[addressed]** Declared in the protocol (II.8), in the manuscript and in the table footnote. H5 then
     failed on EGAD anyway (1.021 [0.971, 1.073]).

**Minor issues**

- MLP architecture not tuned. **[addressed]** Fixed a priori; the k = 8/32 sensitivity changes N_F by
  less than 4 %.
- A listwise loss only for L2 and L8. **[addressed]** Reported as such.
- No fine-tuning of the vision encoders. **[addressed]** Limitations: the negative result is bounded to
  frozen descriptors of grey renders.

## Review 2 — Robotics

**Summary.** It is useful to show that an order which saves planner evaluations changes which grasp is
executed. The physics corrections and the articulated replication are a plus. The main weakness is the
absence of real-robot evidence and of learned external baselines.

**Major issues**

1. *No real robot.*
   - **[open, cannot be closed here]** Stated in the title area, abstract, limitations and conclusion.
     Any real-robot validation is future work and needs hardware.
2. *Uncalibrated contact parameters decide the conclusions.* The robustness non-inferiority fails at
   torsion 0.001 m and 0.02 m (−6.4 and −2.7 pp).
   - **[addressed]** Reported in the confirmation section and the abstract.
   - **[open]** Calibrate torsional friction against measured pad–object pairs.
3. *External baselines.* GPD, Contact-GraspNet and Dex-Net were not run. Reviewers in this venue will
   ask for at least one.
   - **[open]** The requirements and a closed action list are in `results/external_baselines.json`.
     GPD is CPU-only and its dependencies are installable, so it is the cheapest to add.
   - The analytic epsilon order was run on 40 development parts.
4. *Wall-clock cost.* The learned orders are slower end to end (1.94×).
   - **[addressed]** Reported as H4 failing. The integrated estimate (≈ 1.07 in development) is labelled
     an engineering estimate.
   - **[open]** Implement the prior inside the planner and measure it.

**Minor issues**

- The articulated model has no arm collision geometry and uses an ideal controller. **[addressed]**
  Limitations.
- EGAD objects are not household objects. **[addressed]** Stated. The "service robotics" relevance rests
  on the GSO development set.
- The 40/120 mm analyses are not other robots. **[addressed]** Stated.

## Review 3 — Statistics

**Summary.** The pre-registration and freeze are well executed, with hashes committed before opening.
Holm families, paired tests, non-inferiority with bounds, and CIs are all present. The reporting of
non-replicated results is appropriate.

**Major issues**

1. *P2 is borderline.* The lower 97.5 % bound is −4.9 pp against a −5 pp margin, and the point estimate
   is negative. Calling P2 "supported" is correct under the protocol, but it must not be read as
   "equally robust".
   - **[addressed]** The text says "narrowly", reports that superiority failed in the unfavourable
     direction, and gives the torsion sensitivity.
2. *The z test for paired binary non-inferiority.* With 220 objects and few discordant pairs, the normal
   approximation is acceptable but not exact.
   - **[open, minor]** Report an exact or Newcombe-type interval for the paired difference as a check.
     The bootstrap CI ([−5.0, 1.4]) is already reported and agrees.
3. *P3 uses a t test on log ratios with many ties.* 195 of 220 objects have identical N_F.
   - **[addressed in part]** The bootstrap CI ([0.953, 1.037]) and the counts (15 fewer, 10 more, 195
     ties) are reported. The conclusion does not depend on the test choice.
4. *Changes to the family after development.* Load tolerance was moved to secondary, and H5 was
   retargeted.
   - **[addressed]** Both were made before opening, are declared, and are justified by development
     evidence. The confirmation then agreed with the reason for the move.

**Minor issues**

- Holm-adjusted p-values below 1e-4 are printed as "< 10⁻⁴". **[addressed]**
- Power: H2 was underpowered. **[addressed]** Stated.
- Development and confirmation are kept in separate sections. **[addressed]**

## Review 4 — Reproducibility

**Summary.**

- **Strengths.** Every number in the manuscript is a generated macro. Figures come from scripts. Seeds,
  hashes and commands are recorded.
- **Weaknesses.** Large intermediate artefacts are not archived, third-party licences are unresolved,
  and there is no archival release.

**Major issues**

1. *Catalogues and per-candidate simulation logs are not in the repository.*
   - **[addressed in part]** The per-object confirmation outputs (orders, trials, articulated, timing)
     are committed. The catalogues are regenerable.
   - **[open]** Archive them in a release; this needs the author's authorisation.
2. *Licences.*
   - No root `LICENSE` file (Apache-2.0 is declared).
   - The redistribution of DINOv3-derived artefacts and the LAION weights licence are unresolved.
   - **[open]** These are the author's decisions (listed in `publication_readiness.md`).
3. *Determinism.*
   - CoACD output may vary across versions. The collision pieces used are not archived, only counted.
   - **[open]** Archive the EGAD collision pieces with the release.
4. *Container restart during the confirmation.*
   - **[addressed]** Documented in the supplement. Per-object outputs make the resume safe.

**Minor issues**

- The Springer template is not redistributed. **[addressed]** Download source and zip hash recorded.
- Pickled models: their hashes are checked before loading, and users are told to verify them.
  **[addressed]**

---

## Consolidated actions before submission

All are listed in `publication_readiness.md`.

1. Author details and declarations.
2. The journal's current guidelines.
3. Licences and repository visibility.
4. An archival release if the journal requires one.
5. Optionally, one learned external baseline (GPD) and an exact paired interval for P2. Any such
   addition is exploratory and must not change the frozen results.
