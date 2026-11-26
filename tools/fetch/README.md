# Fetching external data and models (nothing third-party is redistributed)

Every external input is fetched from its publisher. Where possible, it is pinned and checked by hash. Put every
download in its own new, empty directory, and read the files only as data.

| Input | How | Pin / check | Licence |
|---|---|---|---|
| YCB objects (16, GRASPA layouts) | `tools/verify/ycb_pick.sh target/release/axisforge DIR` | `pybullet-object-models` commit `57cc4007607f6bcbb0a5364ccab7636faf7a91e4`; reproduces the Phase IV objects exactly (world bounds and candidate counts) | YCB: CC BY 4.0; packaging repository: LGPL-2.1 |
| Google Scanned Objects (Phases III–VI) | `python -m research_grasping.phase3.dataset_reproduce --root DIR`; the Phase IV/V package scripts in `tools/research/phase4/new_parts.py` | `kevinzakka/mujoco_scanned_objects` commit `6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c`; locked manifests with SHA-256 | CC BY 4.0 (meshes), MIT (MJCF) |
| EGAD (Phase VII and the extension) | download the training and evaluation archives from the dataset authors, then `python -I tools/research/phase7/egad_prepare.py select/build ...` | `research/phase7/egad_archives.sha256`; per-mesh SHA-256 in `research/phase7/egad_selection.json` | BSD-3-Clause |
| OpenCLIP ViT-B-32-quickgelu LAION-400M e31 | `python -m research_grasping.phase3.fetch_openclip DIR` | URL `.../v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt`, 605 159 845 bytes, SHA-256 `d867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e` | MIT (model card); LAION-400M terms apply to the training data |
| DINOv2 ViT-S/14 | Hugging Face `facebook/dinov2-small`, at the revision recorded in each descriptor file's `provenance` | weight hashes recorded in the provenance | Apache-2.0 |
| DINOv3 ViT-S/16 | the user's own licensed download (gated by Meta); then `tools/research/phase4/extract_frozen_embeddings.py` | weight hash recorded in the provenance of `research/phase4/grasp_prior/dinov3.npz` | DINOv3 License; accepting it is the user's step |
| GPD | `git clone https://github.com/atenpas/gpd` then `git checkout 6327f20eabfcba41a05fdd2e2ba408153dc2e958`; build with CMake (PCL, OpenCV, Eigen), then build `tools/research/phase7_extension/gpd_adapter` with GPD's compiler flags | commit hash | BSD-2-Clause |

After a fetch, `python tools/licenses/build_manifest.py` re-hashes every affected committed file and every
licence text in `LICENSES/`. Run it to confirm that nothing committed has changed.
