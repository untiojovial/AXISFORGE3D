# External comparators: what was run, what could not be, and why (2026-11-26)

## GPD: run (ranking-controlled level)

- **Source.** https://github.com/atenpas/gpd at commit `6327f20eabfcba41a05fdd2e2ba408153dc2e958`
  (2021-07-15), BSD-2-Clause.
- **Build.** Built here from source, unmodified (Release, without Caffe, OpenVINO or OpenCV DNN). It uses GPD's
  own Eigen implementation of its LeNet classifier and the shipped 15-channel weights.
- **Adapter.** `tools/research/phase7_extension/gpd_adapter/gpd_score_candidates.cpp` is AxisForge3D code built
  with GPD's compiler flags. It turns every AxisForge candidate into a GPD hand (frame: approach, closing,
  approach × closing), builds GPD's 15-channel image from a two-camera point cloud, and classifies it with GPD's
  network.
- **What it measures.** This is a **ranking-controlled** comparison:
  - same objects, candidates, gripper gate and planner checks;
  - GPD only decides the examination order.

  It is **not** GPD's full pipeline. GPD neither samples the candidates nor uses its own hand geometry for
  collision and approach checks. GPD's image geometry (outer diameter, depth and height of the hand volume:
  0.10, 0.06 and 0.02 m) is the one its network was trained on, not the 80 mm AxisForge jaw. The adapter must
  not be presented as a reproduction of GPD's published results.
- **Presentation (fixed wording).** "Ranking-controlled comparison using GPD's pretrained classifier on a shared
  candidate catalogue." Used in the protocol, the freeze record, the article and every results file.
- **Cost.**
  - Measured on one thread for the whole catalogue: the median per development part is 22.6 s, plus 0.7 s
    to render two depth images (`external_baselines_v2.json`, cost block).
  - The adapter now prints GPD's own stage timings. On 20 development parts the medians were 0.006 s for cloud
    preprocessing, 9.07 s for image generation and 9.95 s for classification, plus 0.48 s of rendering
    (`gpd_variability.json`; a different run with less machine load).
  - The learned order costs about 10 ms per object.
- **GPD is stochastic.** `HandSet::shadowVoxelsToPoints` (`src/gpd/candidate/hand_set.cpp`, lines 191–192)
  seeds its generator from `std::random_device`, so two runs on the same cloud and candidates give different
  images and scores. We did not modify the library. On 20 development parts scored twice
  (`gpd_variability.json`):
  - the median Pearson correlation of scores between runs was 0.743 (minimum 0.675);
  - the first-ranked candidate changed in all 20 parts;
  - the geometric-mean N_F was 19.23 and 23.44 in the two runs.

  Every GPD result is therefore one draw. The development and confirmation runs are single runs, reported as
  they are.
- **Results.**
  - Development: `results/external_baselines_v2.json`, 151 parts with v3 physics of GPD's first grasps.
    - GPD+gate N_F geometric mean 7.73 (M0 6.07, Joint 1.29);
    - robust 89.4 % (M0 80.1 %, Joint 89.4 %);
    - load tolerant 37.1 % (M0 58.9 %, Joint 56.3 %).
  - New confirmation: GPD+gate is a descriptive arm of `results/confirmation_x.json` on the same viable objects
    as the other methods.
  - Phase VII EGAD objects: scoring stopped after 21 of 255 objects and is reported as **not completed**. No
    result is drawn from it.
- **Full-pipeline level.** Not run. Letting GPD sample its own candidates would also require running the
  AxisForge planner's IK and collision checks on foreign candidates. That is a different experiment, and it is
  not needed to answer the ordering question.

## Dex-Net 2.0 (GQ-CNN): blocked, model weights unavailable at the official source

- **Code.** https://github.com/BerkeleyAutomation/gqcnn. The licence permits "educational, research, and
  not-for-profit purposes" (header of the official scripts).
- **Weights.** The official download script (`scripts/downloads/models/download_models.sh`) points to seven
  `berkeley.box.com/shared/static/...` archives: GQCNN-2.0, 2.1, 3.0, 4.0-PJ, 4.0-SUCTION, FC-GQCNN-4.0-PJ and
  FC-GQCNN-4.0-SUCTION. On 2026-11-26 **all seven returned HTTP 404**.
- **Environment.** `requirements/cpu_requirements.txt` pins `tensorflow<=1.15.0`, which means Python ≤ 3.7.
- **Decision.** Without the published weights, Dex-Net 2.0 could only be retrained. That would need its
  synthetic training dataset and would test a re-implementation, not the published model. **Not run.** Weights
  from unofficial mirrors were not used, because their provenance cannot be verified.

## Contact-GraspNet: blocked, needs a CUDA GPU

- **Code.** https://github.com/NVlabs/contact_graspnet. The README says "tested with python 3.7, tensorflow 2.2,
  CUDA 11.1". The environment file pins `tensorflow-gpu==2.2.0`, `cudatoolkit=10.1` and `cudnn=7.6`. The
  PointNet++ operators are custom TensorFlow CUDA ops.
- **Weights.** Published as a Google Drive folder (README).
- **This environment.** No GPU (`nvidia-smi` absent, no `/dev/nvidia*`). The official code cannot run on CPU
  without rewriting its CUDA operators, and that would no longer be the published method. **Not run.**

## Status of the comparator line (closed)

- GPD+gate is kept, at the ranking-controlled level only.
- Dex-Net 2.0 and Contact-GraspNet are closed. No weights of doubtful provenance were used, and nothing was
  reimplemented.

## Studied but not counted as baselines

The classical analytic order (force closure and the epsilon quality of Ferrari and Canny) was run in Phase VII
(`research/phase7/results/external_baselines.json`) and stays as a measured internal-analytic comparator.
Reading documentation does not count as a completed baseline.
