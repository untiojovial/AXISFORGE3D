# State-of-the-art matrix (2026-11-27)

This is the full comparison behind Table 1 of the article and Supplementary Table S5. Each cell records what the
version of the paper read here reports. The last column names that version and the part read.

**Symbols.**

| Symbol | Meaning |
|---|---|
| ✓ | reported |
| ∘ | partly (the notes say how). Under Dyn, ∘ means a success or quality score, or robustness to pose noise only, rather than a robustness measure |
| – | not reported in the version read; not evidence that the capability is absent |
| ? | not verified in this review |
| n/a | not applicable |

**Numbers.** No numbers are compared across rows: datasets, budgets, physics settings and hardware differ.

**Columns.**

| Abbreviation | Meaning |
|---|---|
| Gen | candidate generation |
| Ord | learned ordering of a candidate set |
| Feas | learned feasibility (kinematic or geometric) |
| Dyn | physical or dynamic robustness in the ranking |
| MFr | several friction values |
| FB | fallback |
| Guar | mathematical guarantee |
| BO | Bayesian optimisation |
| DKL | deep kernel learning |
| Trf | transfer of knowledge across objects, tasks or grippers |
| Time | time including the learned inference |
| Unseen | evaluation on objects not used for design |
| Sim | simulation |
| Real | real robot |
| Code | public code or data |

| Work (year) | Method | Gen | Ord | Feas | Dyn | MFr | FB | Guar | BO | DKL | Trf | Time | Unseen | Sim | Real | Code | Version read and where |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jiang et al. (2022), Front. Neurorobot. | suction graspability and reachability U-Net | pixel-wise map | ∘ (map ranks pixels) | ✓ | ∘ (analytic suction quality) | – | – | – | – | – | ? | ✓ (motion-planning time reduced) | ? | ✓ | ? | ? | arXiv 2111.02571: abstract, introduction |
| Qin et al. (2026), RA-L 11(2) | energy-based model for shared grasps | precomputed grasp set | ✓ | ✓ (IK and collision at both poses) | – | – | – (failure after 50 attempts) | – | – | – | ✓ (unseen grasps, similar shapes) | ✓ (Table I, row T: faster than analytical checking; row N: more attempts than the filtered set) | ✓ | ✓ | ✓ (simulation and real-world settings) | ? | arXiv 2506.15920 (pre-publication version): abstract, Sect. comparison with analytical methods, Table I |
| Liu et al. (2025), arXiv | dual-output MLP; rank, then plan | precomputed grasp–place pool | ✓ | ✓ (path IK, collision) | – (physics-free labels) | – | – | – | – | – | – | ∘ (planner calls; Succ@K) | ∘ (held-out split; one object) | ✓ (PhysX execution) | – | ? | arXiv 2512.18922: contributions, Table 1, discussion |
| Ansari et al. (2026), arXiv | MLP, CNN and point-cloud-transformer feasibility classifiers | labelled grasps in scenes | – | ✓ | – | – | – | – | – | – | ✓ (novel objects) | ∘ (prediction time vs planners) | ✓ | ✓ | ? | ? | arXiv 2606.26700: abstract |
| GraRe (Yuan et al., 2026), arXiv | Transformer quality model; re-ranks frozen detectors | candidates of 5 detectors | ✓ | – | ∘ (learned analytic quality) | ∘ (target: minimum friction for force closure) | – | – | – | – | ✓ (5 detectors, Novel split) | ✓ (latency 0.5–0.84 s added) | ✓ | – (dataset of real scenes) | ✓ | ✓ (project page) | arXiv 2608.00946: contributions, training objective, latency, limitations |
| Matak et al. (2026), RA-L 11(9) | plan to many targets, then evaluate | learned generator (multi-finger) | ∘ (learned evaluator ranks) | – (planned, not learned) | ∘ (learned success likelihood) | – | ∘ (next-best fallback in the baseline; no bound) | – | – | – | ✓ (novel environments) | ∘ (planning time discussed) | ✓ | ✓ | ✓ | ? | arXiv 2509.07162: abstract, experiments |
| GraspGen (Murali et al., 2026), ICRA | diffusion generator with on-generator discriminator | diffusion samples | ∘ (discriminator scores and filters) | – | ∘ (simulated success labels) | – | – | – | – | – | ✓ (grippers, objects) | ✓ (inference time) | ✓ | ✓ | ✓ | ✓ (code, 53 M-grasp dataset) | arXiv 2507.13097: abstract, method |
| FIRMGrasp (Enwerem et al., 2026), arXiv | CVaR force-closure margin under friction uncertainty | grasp sets (hands) | ∘ (analytic ranking) | – | ✓ (analytic margin; shake and pick tests) | ✓ | – | ✓ (probability of force closure ≥ β) | – | – | – | – | ? | ✓ (MuJoCo) | – (left to future work) | ? | arXiv 2607.25049: abstract, results, future work |
| MPNet (Qureshi et al., 2021), T-RO 37(1) | neural motion planner plus classical planner | n/a | n/a | n/a | n/a | n/a | ✓ | ✓ (worst-case guarantees of the hybrid) | – | – | ✓ (unseen environments) | ✓ | ✓ | ✓ | ? | ? | arXiv 1907.06013: abstract |
| PLOI (Silver et al., 2021), AAAI | GNN object importance, incremental relaxation | n/a | n/a | n/a | n/a | n/a | ✓ | ✓ (Lemma 1: complete) | – | – | ✓ | ✓ | ✓ | ✓ | – | ✓ (code link) | arXiv 2009.05613: method, Lemma 1 |
| van den Brand et al. (2024), SODA | algorithms with predictions | n/a | n/a | n/a | n/a | n/a | ✓ | ✓ (factor two, per instance) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | arXiv 2307.09961: introduction |
| Nogueira et al. (2016), IROS | unscented Bayesian optimisation | BO queries | n/a | – | ∘ (robustness to pose noise) | – | – | – | ✓ | – | – | ? | ? | ✓ | – | ? | Phase VII register; abstract |
| Lechuz-Sierra et al. (2024), RA-L 9(11) | Bayesian optimisation, sensorized compliant hand | BO queries | n/a | – | ∘ (grasp metrics under noise) | – | – | – | ✓ | – | – | ? | ✓ (unknown objects) | ? | ✓ | ? | arXiv 2410.18237: abstract |
| ADKF-IFT (Chen et al., 2023), ICLR | meta-learned adaptive deep-kernel GP | n/a (molecules) | n/a | n/a | n/a | n/a | n/a | – | ? | ✓ | ✓ | n/a | ✓ | n/a | n/a | ? | Phase VII register |
| Pérez González (2025), master's thesis | BO informed by DINO features and context | BO queries | n/a | – | ? | – | – | – | ✓ | ✓ | ✓ (context) | ? | ? | ✓ (Simox) | – | – | Zaguán record 152263: abstract, outline |
| ORACLE-Grasp (Giuili et al., 2026), ISR 19(3) | large multimodal model as a semantic oracle | image candidate regions | ∘ (reasons over regions; early stopping) | – | – | – | – | – | – | – | ✓ (zero-shot) | ∘ (early stopping for efficiency) | ✓ | – (images) | ? | ? | Crossref abstract |
| **AxisForge3D, final method** | gradient boosting P(F)·P(robust\|F)·Q_cen^0.3; round robin with the original order | fixed deterministic catalogue of the planner | ✓ | ✓ (IK, collision, wrench) | ✓ (learned from MuJoCo lift, hold and shake) | ✓ (three torsional-friction values; mass and pose-noise conditions) | ✓ | ∘ (bound on examined candidates only) | – | – | ∘ (development to EGAD shift; cross-gripper ratios 0.94 and 1.07) | ✓ (integrated, bit-exact) | ✓ (two frozen confirmations) | ✓ | no | ✓ | this repository |
| **AxisForge3D, development only** | DINOv2, DINOv3 and OpenCLIP features, MLPs, ranking loss; DKL and ADKF-IFT studied | same | ✓ | ✓ | ✓ | ✓ | ✓ | ∘ | – (thesis only) | ∘ (ceiling analysis; not run) | ∘ | ✓ | ✓ | ✓ | no | ✓ | this repository |

## Notes on the distinctions

- **AxisForge3D's historical components** were studied during development and are not used by the final method:
  - the foundation-model features (no gain against their controls);
  - the MLPs and the listwise ranking loss;
  - DKL and ADKF-IFT, which were not run because of the ordering ceiling;
  - Bayesian optimisation, which belongs to the thesis.
- **The fallback** is marked ∘ under Guar for AxisForge3D because the bound concerns the number of examined
  candidates. It says nothing about time, robustness, safety or real success.
- **The robustness term** is learned from simulation that is not calibrated against hardware. In the article, ✓ under
  Dyn and MFr means the term is present and evaluated. It does not mean that it helped: the development gains did not
  replicate.
