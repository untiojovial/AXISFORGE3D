# Licence and editorial sources checked (2026-11-26)

These pages were read during the Phase VII extension. Only what each page states is recorded here; anything
inferred is marked as such. The licence register and the editorial checklist cite this file.

| Source | URL read | Result |
|---|---|---|
| GPD | local clone of https://github.com/atenpas/gpd at commit `6327f20eabfcba41a05fdd2e2ba408153dc2e958` (2021-07-15), `LICENSE.md` | **BSD 2-Clause License**, "Copyright (c) 2018, Andreas ten Pas". The LeNet weights (`models/lenet/15channels/params/*.bin`) are in the same repository. No separate licence for them was found in the repository (inference: the repository licence covers them; to be confirmed by the author before redistribution). |
| EGAD | `research/phase7/EGAD_LICENSE.txt` (from the dataset's GitHub repository) | BSD-3-Clause, copyright D. Morrison, ACRV/QUT 2020 |
| OpenCLIP code | https://raw.githubusercontent.com/mlfoundations/open_clip/main/LICENSE | MIT ("Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman, ...") |
| OpenCLIP weights (ViT-B-32, LAION-400M) | https://github.com/mlfoundations/open_clip README; `src/open_clip/pretrained.py` (main); https://huggingface.co/timm/vit_base_patch32_clip_224.laion400m_e31 and `…_e32` model cards | The README is silent on the weights. `pretrained.py` maps `ViT-B-32-quickgelu` / `laion400m_e31` to `vit_b_32-quickgelu-laion400m_e31-d867053b.pt` and to the Hugging Face mirror `timm/vit_base_patch32_clip_224.laion400m_e31`, whose card declares `license: mit` (as does e32's). The checkpoint used here is e31 (SHA-256 `d867053b…`, matching the file name). **Status: resolved (MIT).** The direction to the model cards came from the author (via ChatGPT) and was confirmed here. |
| DINOv3 | https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md ("DINOv3 License", last updated 19 Aug 2025) | See the clauses below. |
| YCB Object and Model Set | https://www.ycbbenchmarks.com/ (HTTP 500); then https://ycb-benchmarks.s3.amazonaws.com/index.html and https://registry.opendata.aws/ycb-benchmarks/ (alternatives relayed by the author, read here) | The S3 index page says "License for the data set: Creative Commons Attribution 4.0 International (CC BY 4.0)" and "License for the code: MIT license". The AWS registry entry says `"license" : "Creative Commons Attribution 4.0 International (CC BY 4.0)"`. **Status: resolved.** Provenance of the meshes used: `pybullet-object-models` (LGPL-2.1 repository) @ `57cc400`, reproduced exactly (see LICENSE_COMPLIANCE_REPORT.md). |
| Intelligent Service Robotics submission guidelines | https://link.springer.com/journal/11370/submission-guidelines (and www.springer.com, and the journal page) | 303 redirect to the Springer login from this environment. The requirements were read from the official page by ChatGPT on 2026-11-26 and relayed by the author; they are recorded as such in `publication/intelligent_service_robotics/JOURNAL_REQUIREMENTS_VERIFIED.md`. |

## DINOv3 License clauses (verbatim)

- **Definition.** "DINO Materials" means, collectively, Documentation and the models, software and
  algorithms, including machine-learning model code, trained model weights, inference-enabling code,
  training-enabling code, fine-tuning enabling code, and other elements of the foregoing distributed by
  Meta and made available under this Agreement.
- **1.a, grant.** "… to use, reproduce, distribute, copy, create derivative works of, and make modifications
  to the DINO Materials."
- **1.b.i, redistribution.** "Distribution of DINO Materials, and any derivative works thereof, are subject
  to the terms of this Agreement. If you distribute or make the DINO Materials, or any derivative works
  thereof, available to a third party, you may only do so under the terms of this Agreement and you shall
  provide a copy of this Agreement with any such DINO Materials."
- **1.b.ii, acknowledgement in publications.** "If you submit for publication the results of research you
  perform on, using, or otherwise in connection with DINO Materials, you must acknowledge the use of DINO
  Materials in your publication."
- **1.b.iii and 1.b.v, compliance and trade controls.** Use must comply with Trade Control Laws. It excludes
  ITAR activities and end uses prohibited by Trade Controls, "including those related to military or warfare
  purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal
  weapons".
- **1.b.iv, reverse engineering.** "Your use of the DINO Materials will not involve or encourage others to
  reverse engineer, decompile or discover the underlying components of the DINO Materials."
- **5.a, ownership of your derivatives.** "… with respect to any derivative works and modifications of the
  DINO Materials that are made by you, as between you and Meta, you are and will be the owner of such
  derivative works and modifications."
- **6, termination.** "Upon termination of this Agreement, you shall delete and cease use of the DINO
  Materials."

The licence does not say whether embeddings computed with the model are "derivative works". This decides
whether the committed DINOv3 descriptors (`research/phase5/results/descriptors.npz`) must be distributed
with a copy of the licence. That is a legal question for the author (decision list, item L-DINOv3). The
manuscript must acknowledge the use of DINOv3 (1.b.ii) regardless of how that question is answered.
