# Final literature review (2026-11-27)

This review supports the introduction and the related-work section of the manuscript. It covers what was known
before AxisForge3D and what the study adds. The per-work comparison is in `STATE_OF_THE_ART_MATRIX.md`, the
assessment of each contribution in `NOVELTY_ASSESSMENT.md`, and the verification of every new reference in
`REFERENCES_VERIFIED_FINAL.json`.

## 0. Method

**Sources.** The search used web search and, for every candidate work, these primary records:

- Crossref (DOI, title, authors, venue, volume, issue, pages);
- the arXiv API (versions, comments, journal references);
- OpenAlex (open-access locations);
- the authors' or institutions' repositories (arXiv full texts, Zaguán).

IEEE Xplore, SpringerLink and ScienceDirect could not be read directly from this environment: they returned paywalls,
a 403 or a login redirect. For papers published there, the record comes from Crossref and the content from the
abstract or from the arXiv version, as the register states for each one.

**Queries.** About 35 targeted queries, plus Crossref queries over *Intelligent Service Robotics* (ISSN 1861-2784,
2023–2026) for grasping, manipulation, pick and place and planning. The topics were:

- grasp candidate ordering;
- learned feasibility and reachability;
- robust and multi-friction grasp selection;
- learning to rank grasp candidates;
- learned planning with fallbacks and guarantees;
- algorithms with predictions;
- negative transfer;
- Bayesian optimisation for grasping;
- deep kernels and meta-learning;
- foundation models for grasping;
- evaluation practice in robot learning;
- bin-picking pipelines that check ranked candidates in order.

**What the search can and cannot support.** The search was targeted, not systematic. "Not found" means not found in
this search; it is never a claim of world-wide novelty.

**Verification rules.**

- No reference is cited without a primary record.
- Preprints are marked "not peer reviewed".
- A claim about a paper rests only on the part of the paper that was read, and the register says which part that is.
- A capability a paper does not report is recorded as "not reported in the version read", not as absent.

## 1. Families of related work

For each family: the problem, the method, the results, the stated limits, how it differs from AxisForge3D, and what
it leaves open.

### 1.1 Grasp synthesis and sample-and-evaluate detectors

**Background works.**

- **Surveys.** Bohg et al. 2014 and Newbury et al. 2023.
- **Detectors that sample candidates and score them with a learned network.** Dex-Net 2.0 (Mahler et al. 2017),
  GPD (ten Pas et al. 2017), Contact-GraspNet (Sundermeyer et al. 2021) and GraspGen (Murali et al., ICRA 2026).
  GraspGen uses a diffusion generator plus an "on-generator" discriminator that scores and filters the sampled
  grasps.
- **Shared data.** GraspNet-1Billion and ACRONYM (Eppner et al. 2021) provide shared scenes and simulated labels.

**Problem and results.** These works decide which grasps to generate and how likely each is to succeed. They report
success rates on benchmarks and on real robots.

**Difference from AxisForge3D.**

- These works change the candidate set and its scores.
- AxisForge3D keeps the planner's deterministic catalogue complete and changes only the examination order. Every
  feasible grasp therefore remains reachable.

**Open question taken up here.** What changes when only the order of a complete planner is learned, rather than the
generator?

### 1.2 Learned feasibility and reachability to save planning work (closest family)

**Works and what they report.**

| Work | Problem | Method | Reported result | Stated limits |
|---|---|---|---|---|
| Wells et al., RA-L 2019 | task and motion planning | feasibility classifier as a heuristic for task-plan search | fewer motion-planning calls (secondary description) | not read in full here |
| Lou et al., ICRA 2020; Akinola et al., IROS 2021 | 6-DoF grasping, dynamic grasping | learned reachability of grasp poses | earlier phases' review | — |
| Jiang et al., Front. Neurorobot. 2022 | suction bin picking | U-Net predicting suction quality and reachability | "learning the reachability helps to reduce motion planning computation time" | suction only |
| **Qin et al., RA-L 2026** (arXiv 2506.15920 read) | pick and place: grasps feasible at both poses | energy-based model over the candidate set | Table I: lower time to a solution than the analytical method that checks IK and collision for every candidate (e.g. 1.7 vs 5.7 with 57 candidates), but more candidates attempted before a feasible motion plan (18.2 vs 8.5), fewer than random selection (23.6); simulation and real-world settings | similarly shaped objects; failure after 50 attempts |
| **Liu et al., arXiv 2025** (full text read) | grasp–place candidates for budget-aware pick and place | MLP predicting path-wise IK feasibility and transit collision; rank, then plan, under the same IK gate and planner | Succ@1 47 → 53 %, failures within three calls 26 → 13 %; fewer planner calls | one cuboid, side-face grasps, a fixed waypoint template; preprint |
| Ansari et al., arXiv 2026 | motion feasibility in clutter | MLP, volumetric CNN and point-cloud transformer classifiers; a 2.7 M-label benchmark | AUROC 0.996 on novel objects; faster than sampling-based planners | preprint |

**What this family establishes.** Learned feasibility reduces examination work or its time. Qin et al. also show the trade-off this study measures: their learned selection was faster than exhaustive analytical checking but needed more downstream attempts. This is the idea behind AxisForge3D's
M1, and the article says so: M1 is not presented as new.

**Difference from AxisForge3D.** None of the versions read:

- puts the dynamic robustness of the returned grasp in the ranking;
- bounds the examination work when the order transfers badly;
- reports end-to-end time with the learned inference integrated into a complete planner, measured on a population
  not used for design;
- reports a protocol frozen before testing.

**Open questions taken up here.**

- What the faster order does to the returned grasp (RQ2).
- Whether the gain replicates on a different object population (RQ1).
- Whether fewer candidates mean less time (RQ3).

### 1.3 Ranking for quality and robustness; dynamic and multi-friction robustness

**Analytic quality.**

- Ferrari and Canny 1992, and the kinematics of contact (Montana 1988).
- Robustness to pose error (Weisz and Allen 2012).
- Safe regions under input noise (Nogueira et al. 2016).

**FIRMGrasp** (Enwerem et al., arXiv 2026, full text read).

- *Finding.* 53 % of nominally force-closed hand grasps lose closure in the adverse-friction tail.
- *Method.* A CVaR-based margin, with a probabilistic certificate of force closure.
- *Evaluation.* Shake and pick tests in MuJoCo; hardware validation is left to future work.

**GraRe** (Yuan et al., arXiv 2026, full text read).

- *Method.* It re-ranks the unchanged candidates of five frozen 6-DoF detectors with a Transformer quality model.
  The training target is the minimum friction coefficient needed for force closure.
- *Results.* Gains of up to 13.56 AP on GraspNet-1Billion; 0.5–0.84 s of added latency on an RTX 2060; real-robot
  experiments.

**Matak et al.** (RA-L 2026).

- In generator–evaluator–planner pipelines, falling back to lower-ranked grasps after a planning failure lowers
  success.
- They plan to several targets and evaluate success at the planned configuration.

**Constrained Bayesian optimisation** (Gardner et al. 2014) weights a utility by the probability of satisfying
constraints.

**Difference from AxisForge3D.**

- The robustness term is learned from dynamic MuJoCo trials (lift, hold, shake), at one or three torsional-friction
  values, and multiplies a feasibility prior inside a complete planner's order.
- GraRe's quality is analytic, its target is detection AP, and it has no feasibility component.
- FIRMGrasp is an analytic metric, not a learned order.

**Open question taken up here.** Does a learned robustness term change the returned grasp enough to matter on new
objects without giving back efficiency? The answer in this study: robustness was non-inferior, but the development
gains did not replicate.

### 1.4 Learning-based acceleration of planning, fallbacks and guarantees

**Learned acceleration.** Learned sampling distributions (Ichter et al. 2018) and learned visual feasibility
heuristics (Driess et al. 2020).

**Learned components paired with a guarantee.**

- MPNet (Qureshi et al., T-RO 2021) combines a neural planner with classical sample-based planners to obtain
  worst-case guarantees.
- PLOI (Silver et al., AAAI 2021) enlarges its learned object set incrementally. Its Lemma 1 proves completeness.
- Policy-guided heuristic search (Orseau and Lelis, AAAI 2021) bounds the search effort in terms of the quality of
  the policy.

**Algorithms with predictions.**

- Robustness to bad predictions is a design goal (Mitzenmacher and Vassilvitskii, CACM 2022; Lykouris and
  Vassilvitskii, JACM 2021).
- van den Brand et al. (SODA 2024; arXiv full text read): for static algorithms, "simulate two algorithms – one with
  predictions, and another one with best known worst-case guarantees – step by step, in parallel, and stop whenever
  one of these algorithms stops". This costs only a factor-of-two slowdown per instance.
- The same principle underlies algorithm portfolios (Gomes and Selman, AIJ 2001) and restart schedules (Luby et al.
  1993).

**Difference from AxisForge3D.** The round-robin hybrid is this principle applied to candidate examination. The
refinement to min(2N_L − 1, 2N_D) follows directly from alternating L first. The article:

- states the three conditions under which the bound holds in a grasp planner;
- checks it on every object (0 violations);
- reports its average cost;
- presents it as a known property, applied and checked, not as a new theorem.

### 1.5 Negative transfer

- Wang et al. (CVPR 2019) define negative transfer and how to avoid it.
- Zhang et al. (IEEE/CAA JAS 2023) survey the methods.

**In AxisForge3D.**

- Negative transfer appears as objects on which the learned order is much worse than the original one. In
  development, M1 and Joint were worse on a few parts, sometimes by a lot. The fallback bounds that loss in examined
  candidates.
- The cross-gripper analysis measures transfer at the gripper level. A model trained at 80 mm and applied at 40 and
  120 mm needed 0.94 and 1.07 times the candidates of models trained for those grippers.

### 1.6 Bayesian optimisation, Gaussian processes, DKL, ADKF-IFT and meta-learning

**Works.**

- **Bayesian optimisation for grasping.** Unscented Bayesian optimisation for safe grasps (Nogueira et al. 2016;
  Castanheira et al. 2018) and Bayesian optimisation on a real sensorized compliant hand (Lechuz-Sierra et al., RA-L
  2024).
- **Kernels and acquisition.** GP-UCB (Srinivas et al. 2010) and deep kernel learning (Wilson et al. 2016).
- **Transfer and meta-learning.** Few-shot Bayesian optimisation with meta-learned deep-kernel surrogates (Wistuba
  and Grabocka, ICLR 2021) and ADKF-IFT (Chen et al., ICLR 2023).
- **The author's thesis** (Pérez González, 2025; Zaguán record read). Sequential Bayesian optimisation of grasps for
  a humanoid hand (iCub) in Simox, informed by DINO features and context, with deep-kernel Gaussian processes.

**Difference from AxisForge3D.**

- The article's method is not sequential Bayesian optimisation. It is a fixed learned order over a complete
  catalogue, evaluated once per object.
- DKL and ADKF-IFT were studied: an ordering-ceiling analysis showed that a perfect order saves about 1.1–1.4 ms per
  object.
- They were not run, and no claim is made about them. They are cited so that the thesis line and the decision not to
  pursue it are explicit.

### 1.7 Foundation models for grasping

**Encoders.** DINO, DINOv2, DINOv3, CLIP and OpenCLIP provide generic features.

**Manipulation systems that use them.**

- Retrieval and alignment: DINOBot, ICRA 2024.
- Language-guided grasping: F3RM and LERF-TOGO, CoRL 2023.
- Zero-shot, affordance-aligned grasp selection with large multimodal models: **ORACLE-Grasp** (Giuili, Atari and
  Sintov, *Intelligent Service Robotics* 19(3), 2026). It discretises the image into candidate regions, reasons over
  them with a large multimodal model, refines the choice with depth, and uses early stopping for efficiency.
- A survey: Firoozi et al., IJRR 2025.

**Difference from AxisForge3D.** These systems use semantics to decide where to grasp. AxisForge3D asks a narrower
question: do frozen features help order a geometric catalogue? With permuted, untrained and linear controls, they did
not (H5 not supported; ratio 1.021 [0.971, 1.073]). This is a bounded negative result.

**ORACLE-Grasp and this study.** ORACLE-Grasp matters for the target journal because both select among candidates
for unknown objects. They differ in three ways:

- generation and semantic selection versus ordering a fixed geometric catalogue;
- images versus a planner with kinematics and physics;
- no frozen confirmatory protocol.

### 1.8 Evaluation practice

- Kress-Gazit et al. (arXiv 2024) argue that robot-learning evaluations should report the number of trials, the
  conditions, several metrics and statistics.
- Shared benchmarks (GraspNet-1Billion, ACRONYM, FetchBench through GraspGen) fix scenes and candidates. GraRe uses
  unchanged candidates, which is a controlled comparison on a shared catalogue.

**In AxisForge3D.**

- The GPD comparison is ranking-controlled on the same catalogue.
- Both confirmations were frozen before testing, on EGAD objects: shapes evolved for diversity, a different
  distribution from the household scans used for development.
- Non-replicated results are reported.

### 1.9 *Intelligent Service Robotics*, 2023–2026

Crossref lists these grasping and manipulation articles:

- 6-DoF grasp pose estimation from instance reconstruction (Han et al. 2023, `10.1007/s11370-023-00489-z`);
- reinforcement-learning grasping of moving objects (2023);
- cooperative push-grasping in clutter with deep RL (Shen et al. 2025);
- pixel-level multi-object planar grasp detection (Liu et al. 2025);
- a gripper with a frictional self-locking mechanism (2025);
- RL for precision grasping (2026);
- an obstacle-avoidance path-planning method for robot grasping (Li et al. 2026);
- visual–tactile stabilised grasping (2026);
- five-fingered posture planning (2026);
- kitchen manipulation with foundation models (Jeon et al. 2026);
- **ORACLE-Grasp** (2026).

Only ORACLE-Grasp is close enough to cite: it selects grasps for unknown objects with an explicit efficiency
mechanism. The others are about detection, control, hardware or path planning. Their abstracts were not compared in
depth; they are listed here so that the journal's recent grasping work was considered.

## 2. Works that could challenge the originality claims

| Work | What it shares with AxisForge3D | Consequence for the claims |
|---|---|---|
| Qin et al. 2026 | learned feasibility over a candidate set; time to solution measured against analytical checking; real robot | M1-style learned ordering is **not new**; the article says so |
| Liu et al. 2025 (preprint) | rank-then-plan over a fixed pool under the same gate and planner | the same; the "fixed pool, same gate" design is precedented |
| GraRe 2026 (preprint) | learned re-ranking of unchanged candidates; latency reported; Novel split | "controlled comparison on a shared catalogue" is precedented; AxisForge3D's comparison with GPD is not a new method |
| Jiang et al. 2022 | learned quality combined with learned reachability | fusing quality with feasibility is precedented |
| van den Brand et al. 2024; Gomes and Selman 2001 | parallel execution bounds the slowdown at two | the bound is a known property |
| MPNet 2021; PLOI 2021 | a learned component with a fallback that keeps a guarantee | fallbacks for learned guidance are standard |
| FIRMGrasp 2026 (preprint) | robustness across friction values | multi-friction robustness is precedented, as an analytic metric |

## 3. The combination question

Is there earlier work that combines all of the following? The answer recorded for each element is what the search
found.

| # | Element | Precedents found | In one work together with the others? |
|---|---|---|---|
| 1 | Learned order over a fixed candidate catalogue | Qin 2026; Liu 2025; GraRe 2026 | — |
| 2 | Geometric and kinematic feasibility prediction | Qin; Liu; Jiang; Akinola; Lou; Ansari | Qin and Liu combine 1 and 2 |
| 3 | Prediction of physical robustness | GraspGen and Matak (learned success); FIRMGrasp (analytic) | not with 1 and 2 in the versions read |
| 4 | Robustness over several friction values | FIRMGrasp (analytic); GraRe (minimum-friction target) | not as a learned term inside an order |
| 5 | Fusion of feasibility, robustness and centring | Jiang (quality and reachability); constrained Bayesian optimisation (product of utility and probabilities) | not with a dynamic, multi-friction robustness term |
| 6 | Fallback with a bound on examined candidates | algorithms with predictions; portfolios; MPNet; PLOI | not in a grasp planner, in the versions read |
| 7 | Native inference inside the planner | GraspGen and GraRe report latency; Jiang reports planning time | not with an end-to-end comparison on held-out objects |
| 8 | Confirmatory evaluation on reserved objects | none with a protocol frozen before testing | — |
| 9 | Full computational cost | GraRe (latency); Qin (time to solution) | — |

**Conclusion.** Each element has precedents. The search found no work that combines them in one complete planner
and tests the combination prospectively. That is a statement about this search.

The article therefore does not claim that combining known components is itself a scientific contribution. It claims
the knowledge the controlled combination produced:

- the efficiency gain replicates;
- the robustness gains selected in development do not;
- fewer examined candidates do not mean less time in this planner;
- load-tolerance conclusions depend on simulator choices.

## 4. Evaluated, studied, or left for future work

| Method | Status in the article |
|---|---|
| Gradient boosting feasibility and robustness models; Joint; Joint-multi; round-robin hybrid | evaluated (development and two confirmations) |
| Logistic regression, MLPs, DINOv2, DINOv3 and OpenCLIP features, untrained ViT, permuted controls, contact patch tokens, listwise ranking loss | evaluated in development (negative) |
| GPD's pretrained classifier | evaluated, ranking-controlled, on the same catalogue |
| Dex-Net 2.0, Contact-GraspNet | not run: the official weights return 404, and CUDA is required (`EXTERNAL_BASELINES_STATUS.md`) |
| DKL, ADKF-IFT, few-shot adaptation | studied through the ordering ceiling; not run (`DKL_ADKF_DECISION.md`) |
| Sequential Bayesian optimisation | the thesis line; not part of this study |
| Optuna-tuned MLPs | exploratory, reduced budget; supports no claim |
| Real robot, calibrated contact | future work |

## 5. Antecedents of the mathematical bound

**The property.** Alternating L₁, D₁, L₂, D₂, … and skipping repeats examines L's first feasible candidate by step
2N_L − 1 and D's by step 2N_D. It follows from the definition. Run-in-parallel arguments with a factor of two are
classical: algorithm portfolios, Luby-style restarts, and the step-by-step simulation in van den Brand et al. 2024.

**What the article adds.**

- The conditions under which the bound holds in a grasp planner: the same candidate set, feasibility independent of
  the order, and at least one feasible candidate.
- The check on 406 + 151 objects.
- The measured average cost.
- The statement of what the bound does not cover: time, robustness, safety and real success.

## 6. Structural examples relayed by the author (§8)

The three articles are Bernal-Agustín, Lujano-Rojas, Artal-Sevil and Dufo-López (Universidad de Zaragoza):

- Energy Conversion and Management 372 (2027) 122284;
- Sustainable Energy, Grids and Networks 47 (2026) 102455;
- Applied Energy 424 (2026) 128497.

**What could be read.** The ScienceDirect pages returned HTTP 403. The open-access (CC BY-NC 4.0) copies in Zaguán
(records 181107, 177678 and 177235) were downloaded and read for structure only. The text of the second one
extracted poorly, so only its section list and its limitations section were read.

**Structure observed.**

- **Introduction.** The context is built up with the cited works woven into the argument, one stream at a time (in
  *Applied Energy*, four streams). Each stream says what it establishes and then what it leaves out ("However, …").
- **Gap.** It is stated as a combination: few studies combine all elements.
- **Comparison table.** A feature table (✓, partial, –) compares representative studies with "this work".
- **Then:** an explicit hypothesis (*Energy Conversion and Management*), a positioning subsection, numbered or
  bulleted contributions (three or four), and the paper's organisation.
- **Results.** Followed by validation and sensitivity analyses, a separate limitations section enumerated "First …
  Finally", and data availability with a Zenodo DOI.

**What was applied.**

- The introduction now goes from the problem to the open questions, explicit research questions (RQ1–RQ4) and
  contributions that state their evidence and outcome.
- The related work goes family by family, each with what it establishes and what it leaves open.
- A feature table compares the closest works with this study (main text, plus the extended version in Supplementary
  Table S5).
- The gap is stated as a combination, with the caveat that it concerns this search.

**What was not taken.** No text was copied, and no reference was taken from these articles: they are about energy
systems.

## 7. Bibliographic control

- **Register.** Every new or corrected entry is in `REFERENCES_VERIFIED_FINAL.json`, with what was read and the
  claim it supports.
- **Corrections.**
  - Qin et al.: arXiv preprint replaced by RA-L 11(2):2026–2033, 2026.
  - Akinola et al.: full authors and pages.
  - Contact-GraspNet: pages.
  - The thesis: Zaguán URL and type.
- **Preprints.** Marked "not peer reviewed": Liu 2025, Ansari 2026, GraRe 2026, FIRMGrasp 2026, Kress-Gazit 2024.
- **Long author lists.** Truncated to three authors and "et al." for entries with more than seven authors, as earlier
  entries already were, for length.
- **Removed from the main text for length (kept in this review).** Driess 2020, Lou 2021 (CARP), Ichter 2018, Orseau
  and Lelis 2021, Luby 1993, Shanthi and Hermans 2024, Noseworthy 2024, F3RM, LERF-TOGO, DINOBot, Castanheira 2018,
  Wistuba and Grabocka 2021, Zhang 2023, Hernández-Lobato 2015, Montana 1988, Miller and Allen 2004, Eppner 2019.
  - Some of these have entries in `references.bib`.
  - Bibtex prints only cited entries.
  - The supplement prints its own list.
- **Checks.** No duplicate keys; no citation without an entry (checked 2026-11-27).

## 8. Sources that could not be accessed

- ScienceDirect pages (403). Open copies in Zaguán were used instead.
- The ISR submission guidelines (login redirect). The requirements relayed on 2026-11-26 are kept as such.
- Paywalled full texts on IEEE Xplore and SpringerLink. Claims rest on the abstracts or on the arXiv versions.
