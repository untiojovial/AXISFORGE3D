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
- **Cost.** Measured on one thread: the median per development part is 22.6 s for preprocessing and
  classification of the whole catalogue, plus 0.7 s to render two depth images.
- **Results.** `results/external_baselines_v2.json`, written once the v3 physics of GPD's first grasps is complete.
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

## Studied but not counted as baselines

The classical analytic order (force closure and the epsilon quality of Ferrari and Canny) was run in Phase VII
(`research/phase7/results/external_baselines.json`) and stays as a measured internal-analytic comparator.
Reading documentation does not count as a completed baseline.
