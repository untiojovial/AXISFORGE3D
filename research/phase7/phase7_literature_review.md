# Phase VII — literature review (verified references only)

Every work cited here is in [references_verified.json](references_verified.json). Each was checked during
Phase VI or VII against a primary or bibliographic source: publisher, proceedings, arXiv, DBLP or an
institutional repository.

- **Search method.** Targeted, not systematic: about 40 queries over Phases VI and VII. No claim of
  world-wide novelty follows from it (see [phase7_novelty_matrix.md](phase7_novelty_matrix.md)).
- **Not cited.** Works that could not be verified (e.g. "Tfaily et al.", the acronym "POGPE").

## 1. Grasp synthesis: where this work sits

- **Surveys.** Bohg et al. (T-RO 2014) split data-driven grasp synthesis by object familiarity. Newbury
  et al. (T-RO 2023) review 85 deep 6-DoF grasp-synthesis papers and identify sampling-based methods as
  one of the four main families.
- **Our problem is the sampling-based family's evaluation stage.** A classical planner generates a
  catalogue of parallel-jaw candidates. It then examines them one by one: inverse kinematics, collision,
  static wrench quality. It stops at the first one that passes.
- **What is learned.** We learn only the **order of examination**. Nothing is removed from the
  catalogue, so the planner stays complete: it still returns a feasible grasp whenever one exists.
- **On sampling.** Eppner, Mousavian and Fox (ISRR 2019) show that common sampling schemes miss parts of
  the grasp space. Our catalogues are the planner's own deterministic generator, and coverage is not
  studied here.

## 2. Learning feasibility or reachability to save planning work

This is the closest family, and its core idea is not new.

| Work | What it learns |
|---|---|
| Lou, Yang and Choi (ICRA 2020) | Reachability as a binary label of the grasp pose |
| Akinola et al. (IROS 2021) | Reachability tied to inverse-kinematics existence |
| Qin et al. (arXiv 2025) | From the planner's own data, to speed up grasp choice in pick-and-place planning |

Our M1 is this idea applied to a validated planner. It is a gradient-boosting classifier on 19 candidate
features, trained on exhaustively labelled catalogues. The contribution is evaluation under control:
every ordering is replayed exactly through the planner's own order arithmetic, validated against 7 588
real planner runs in Phase V.

## 3. Grasp quality, robustness and the speed–robustness tension

- **Analytic quality.** Ferrari and Canny (ICRA 1992) introduced the largest-ball wrench-space quality.
  Montana (IJRR 1988) gave the kinematics of contact. Miller and Allen (RAM 2004) built GraspIt! for
  grasp analysis.
- **Score vs robustness.** Weisz and Allen (ICRA 2012) showed that the best-scored grasps are not always
  the most robust to pose error. Nogueira et al. (IROS 2016) favoured safe regions under input
  uncertainty.
- **Robustness labels.** Dex-Net 2.0 (Mahler et al., RSS 2017) uses analytic robustness as a training
  label.
- **Our version of the tension.** Learning only to be fast (M1) moves the planner away from centred
  grasps, and dynamic robustness can suffer. FORTIS-R Joint multiplies three terms:
  - the feasibility prior;
  - a learned robustness probability P(H|F);
  - a centring term.

  Product fusion follows the constrained Bayesian-optimisation intuition: Gardner et al. (ICML 2014) and
  Hernández-Lobato et al. (ICML 2015) weight utility by the probability of satisfying a constraint.

## 4. Learned grasp detectors (external baselines)

- **GPD** (ten Pas et al., IJRR 2017): point-cloud candidates with a CNN classifier.
- **Contact-GraspNet** (Sundermeyer et al., ICRA 2021).
- **GraspNet-1Billion** (Fang et al., CVPR 2020).
- **Dex-Net 2.0** (§3).
- **AdaGrasp** (Xu et al., ICRA 2021): learns gripper-aware policies, relevant to our cross-gripper
  analysis.

None of these was reproduced in Phase VII. Their real requirements and the blockers are in
[external_baselines.json](results/external_baselines.json). Our analytic baseline is not a
reproduction of any of them.

## 5. Vision foundation models

- **The models.** DINO (Caron et al., ICCV 2021), DINOv2 (Oquab et al., TMLR), DINOv3 (Siméoni et al.,
  arXiv 2025), CLIP (Radford et al., ICML 2021) and OpenCLIP weights (Cherti et al., CVPR 2023).
- **How we use them.** Frozen encoders, unit-mean CLS embeddings of 16 grey renders. We also sample
  DINOv3 patch tokens at the contacts.
- **What we found.** In this problem they did not add measurable value over geometry, against permuted
  and untrained-ViT controls ([mlp_benchmark.json](results/mlp_benchmark.json)). Learned 3D-shape
  baseline: the D2 shape distribution (Osada et al., TOG 2002).
- **Task transfer.** Taskonomy (Zamir et al., CVPR 2018) and TaskWeb (Kim et al., EMNLP 2023) study when
  transfer helps. Expert selection by geometry (GES-UniGrasp, Xu et al., arXiv 2025) is a related use of
  shape for choosing among strategies. Kazanskii (arXiv 2026, peer review not verified) studies
  connectivity-aware exploration of grasp spaces.

## 6. Deep kernels, few-shot adaptation and Bayesian optimisation

These are the line of the author's master's thesis (Pérez González, Universidad de Zaragoza, 2025):

- GP-UCB (Srinivas et al., ICML 2010);
- deep kernel learning (Wilson et al., AISTATS 2016);
- ADKF-IFT (Chen, Tripp and Hernández-Lobato, ICLR 2023);
- transfer across BO tasks: RGPE (Feurer et al., arXiv 2018) and products of GP experts (Schilling et al.,
  ECML PKDD 2016);
- one-shot grasp transfer (Kopicki et al., IJRR 2016).

**Phase VII did not run them.** The order learned without adaptation already finds a viable grasp at the
first candidate in 87 % of development parts. The ceiling is 1.30 examined candidates (about 5 ms of
planner time) against a fixed cost of about 0.57 s. So a support set gathered during planning cannot pay
for itself ([few_shot_analysis.json](results/few_shot_analysis.json)).

## 7. Learning to rank and calibration

- **Ranking loss.** The listwise loss tested is ListNet's cross entropy over top-one probabilities (Cao et
  al., ICML 2007).
- **Calibration.** ECE follows Guo et al. (ICML 2017).

## 8. Simulation and geometry tools

| Tool | Reference |
|---|---|
| Physics | MuJoCo (Todorov, Erez and Tassa, IROS 2012) |
| Collision pieces for EGAD | CoACD (Wei et al., ACM TOG 2022) |
| Models | scikit-learn (Pedregosa et al., JMLR 2011), PyTorch (Paszke et al., NeurIPS 2019) |

## 9. Datasets

| Dataset | Reference | Licence | Role here |
|---|---|---|---|
| Google Scanned Objects | Downs et al., ICRA 2022 | CC BY 4.0 | Development |
| EGAD | Morrison, Corke and Leitner, RA-L 2020 | BSD-3 | Confirmation. Shapes evolved for diversity of grasp difficulty and geometric complexity, so a deliberately different distribution from household scans |
| Objaverse | Deitke et al., arXiv 2022 | Per-object licences | Considered and not used: licences vary per object |

## 10. Statistics

- Paired percentile bootstrap (Efron, Annals of Statistics 1979).
- Exact McNemar test (McNemar, Psychometrika 1947).
- Wilcoxon signed-rank test (Wilcoxon, Biometrics Bulletin 1945).
- Holm's step-down procedure (Holm, Scand. J. Statist. 1979).

## Corrections carried from Phase VI (the thesis bibliography)

The thesis citation errors listed in `research/phase6/phase6_literature_review.md` §9 remain:

- DINO's arXiv identifier;
- Montana's year;
- Srinivas et al.'s venue;
- the expansion of ADKF-IFT.

The manuscript cites only the verified forms.
