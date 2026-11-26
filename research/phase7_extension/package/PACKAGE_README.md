# axisforge3d-research-reproducibility

This is a local research package built from the AxisForge3D repository by `tools/package/build_package.py`.
It has **not been published**: there is no release, no Zenodo record and no DOI. Publishing requires the
author's final approval.

## Contents

The repository's tracked files at the commit named in `MANIFEST_INCLUDED.json`, with the following omitted
(each one listed with its reason in `MANIFEST_EXCLUDED.json`):

- **DINOv3.** No DINOv3 embedding or value derived from DINOv3 embeddings. Mixed files are cleaned: for example,
  `research/phase5/results/descriptors.npz` is included without its `dinov3*` arrays.
- **YCB.** No per-object descriptor, geometry or grasp pose of the 16 YCB objects. Their rows and records are
  removed from mixed files, and files that cannot be separated are excluded. Aggregate statistics and object
  identifiers stay.
- **The manuscript and submission material** (`publication/`). All rights are reserved pending publication.

## Licences

- Code, documentation and results of the author: Apache-2.0 (`LICENSE`, `NOTICE`).
- Third-party material keeps its own licence (`THIRD_PARTY_NOTICES.md`, `LICENSES/`). Each file's licence is
  listed in `MANIFEST_INCLUDED.json`.

## Regenerating what is excluded

- **DINOv3 embeddings.** Use `tools/research/phase4/extract_frozen_embeddings.py` with your own licensed copy of
  `timm/vit_small_patch16_dinov3.lvd1689m`. The revision and SHA-256 are in each descriptor file's provenance.
  The DINOv3 License applies.
- **YCB objects and their derived data.** Run `tools/verify/ycb_pick.sh` (pinned to
  pybullet-object-models@57cc400), then the Phase IV tools listed in `research/phase4/grasp_prior/README.md`.
- **Other datasets and weights.** See `tools/fetch/README.md`, which gives the pinned sources and hashes.

## Reproduction levels

| Level | Command | Needs |
|---|---|---|
| Statistical | `bash reproducibility/reproduce_statistics.sh` | Python with the packages pinned in `reproducibility/requirements-lock.txt` |
| Partial planner | `cargo build --release` then `python tools/research/phase7_extension/native_validation.py` on a sample | the datasets (`tools/fetch/README.md`) |
| Full experimental | the per-phase pipelines in `REPRODUCIBILITY.md` | the datasets, weights and compute |
