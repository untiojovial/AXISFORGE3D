# Third-party notices

AxisForge3D's own source code, documentation and results are licensed under the Apache License 2.0 (`LICENSE`).
That licence covers only material the AxisForge3D author has the rights to. The datasets, models, weights,
robot models and libraries below are **not** relicensed: each keeps its own terms. The machine-readable
register, with sources, versions, evidence, hashes and the files affected, is
`research/phase7_extension/third_party_manifest.json`. The decision for each committed artefact is in
`research/phase7_extension/ARTIFACT_REDISTRIBUTION_REGISTER.md`. All entries were verified on 2026-11-26.

## Datasets (meshes are not redistributed)

**YCB Object and Model Set.** Calli, B. et al. (2015), *The YCB object and model set*. Data licensed under
Creative Commons Attribution 4.0 International (CC BY 4.0, `LICENSES/CC-BY-4.0.txt`). The licence is stated on
the official distribution page (https://ycb-benchmarks.s3.amazonaws.com/index.html) and in the AWS Open Data
Registry (https://registry.opendata.aws/ycb-benchmarks/); YCB code is MIT.

The 16 meshes used here are the YCB-Video models of the PoseCNN project
(https://rse-lab.cs.washington.edu/projects/posecnn/), as packaged by `pybullet-object-models`
(https://github.com/eleramp/pybullet-object-models, LGPL-2.1, commit
`57cc4007607f6bcbb0a5364ccab7636faf7a91e4`). Changes made here: each object is placed in the demonstration cell
and given the mass of the package URDF. Its geometry is not modified. The repository contains numerical data
derived from these objects (bounds, grasp candidates, descriptors, statistics); it contains no mesh and no image
of them.

**Google Scanned Objects.** Downs, L. et al. (2022), *Google Scanned Objects: a high-quality dataset of 3D
scanned household items*. The MuJoCo adaptation by K. Zakka
(https://github.com/kevinzakka/mujoco_scanned_objects, commit `6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c`) is
licensed CC BY 4.0 (OBJ, PNG) and MIT (MJCF). Changes made here:

- grey renders without texture (`research/phase4/embedding_experiment/development-images.zip`, 22 objects,
  distributed under CC BY 4.0);
- convex collision pieces;
- placement in the demonstration cell.

The repository contains no mesh.

**EGAD.** Morrison, D., Corke, P., Leitner, J. (2020), *EGAD! An Evolved Grasping Analysis Dataset for
diversity and reproducibility in robotic manipulation*. BSD-3-Clause, copyright D. Morrison, ACRV/QUT 2020
(`LICENSES/BSD-3-Clause-EGAD.txt`). Changes made here: rescaling by the authors' rule for an 80 mm gripper, a
stable resting pose and CoACD collision pieces. The repository contains names, hashes and derived statistics
only.

## Models and weights (not redistributed)

**DINOv3** (Meta). Governed by the DINOv3 License (`LICENSES/DINOv3-License.md`), which is not Apache-2.0. The
weights are not in the repository. The DINOv3 embeddings and the values derived from them, listed in the
manifest, are distributed only under the DINOv3 License and with this copy of it. Publications using them must
acknowledge the use of DINO Materials (clause 1.b.ii). The licence includes trade-control and use restrictions
(clauses 1.b.iii–1.b.v) that bind every recipient.

**This applies to every commit and branch.** Every DINOv3-derived file in any commit or branch of this repository
is governed by the DINOv3 License; this includes the commits before the licence copy was added (`75bec22` to
`fd982a9`) and the branch `research/axisforge-phase6-fortis-r`, whose head has no copy. The copy is
`LICENSES/DINOv3-License.md` on the working branches. Scope and remedies are in
`research/phase7_extension/results/history_licence_audit.json`. The reproducibility package
(`axisforge3d-research-reproducibility`) contains no DINOv3 embeddings.

**DINOv2** (Meta). Oquab, M. et al. (2023). The weights are Apache-2.0 according to the model card. Only
embeddings are included.

**OpenCLIP** ViT-B-32-quickgelu trained on LAION-400M, epoch 31. Ilharco, G. et al., OpenCLIP. The code is MIT.
The model card of the checkpoint's Hugging Face mirror (`timm/vit_base_patch32_clip_224.laion400m_e31`)
declares the weights MIT. The checkpoint used is `vit_b_32-quickgelu-laion400m_e31-d867053b.pt`, SHA-256
`d867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e`. It is fetched, not redistributed, by
`research_grasping/phase3/fetch_openclip.py`. The training data (LAION-400M, Schuhmann, C. et al. 2021) has its
own terms and is not used directly. Only embeddings are included.

**GPD.** ten Pas, A., Gualtieri, M., Saenko, K., Platt, R. (2017), *Grasp Pose Detection in Point Clouds*.
BSD-2-Clause, copyright 2018 Andreas ten Pas; the classifier weights are in the same repository (commit
`6327f20eabfcba41a05fdd2e2ba408153dc2e958`). GPD is built locally, unmodified, and is not redistributed. The
adapter in `tools/research/phase7_extension/gpd_adapter` is AxisForge3D code.

## Robot model (redistributed)

**UR5** from `example-robot-data` (Gepetto). BSD-3-Clause, copyright LAAS-CNRS, University of Edinburgh and
INRIA. The licence sits next to the files in `apps/viewer/samples/ur5/` and
`crates/project-io/tests/data/ur5/`, and a copy is in `LICENSES/BSD-3-Clause-example-robot-data.txt`.

## Software dependencies (not redistributed)

- **Rust.** The 69 crates in `Cargo.lock` declare MIT, Apache-2.0, Zlib, Unlicense or Unicode-3.0 licences, alone
  or in combination.
- **Python.** The packages used declare BSD, MIT, Apache-2.0, PSF-based or MIT-CMU licences: NumPy, SciPy,
  scikit-learn, PyTorch, torchvision, open_clip_torch, timm, transformers, safetensors, MuJoCo, trimesh, CoACD,
  Matplotlib, GPyTorch, BoTorch, Optuna, pytest, huggingface_hub and Pillow.
- **Viewer.** It loads three.js 0.150.1 (MIT) from a CDN at run time.

The full audit is in `research/phase7_extension/results/dependency_licences.json`, produced by
`tools/licenses/audit_dependencies.py`. A binary release would have to bundle the notices of the crates it
links.
