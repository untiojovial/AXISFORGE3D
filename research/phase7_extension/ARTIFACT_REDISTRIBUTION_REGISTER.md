# Artifact redistribution register (2026-11-26)

This register decides, for each class of committed artefact, which licence it is offered under and whether it
belongs in the reproducibility package `axisforge3d-research-reproducibility` (built locally; any Zenodo
deposit or DOI needs the author's authorisation). Nothing here rewrites Git history: everything listed is already committed on this branch and on
`research/axisforge-phase7-final-publication`, and stays there. Component details and file hashes are in
`third_party_manifest.json`.

| Class | Examples | Offered under | In the public package | Regeneration |
|---|---|---|---|---|
| Own source code, scripts, tests, technical documentation | `crates/`, `apps/`, `tools/`, `research_grasping/`, `docs/` | Apache-2.0 (Copyright 2026 Eduardo Pérez González, `NOTICE`) | yes | — |
| Manuscript and submission material | `publication/` | **not Apache**: all rights reserved pending publication (`RIGHTS.md`) | **no** | LaTeX sources in the repository |
| Own experimental results (statistics, orders, outcomes, timings, frozen models trained on own labels) | `research/phase*/results/*.json`, `research/phase7/frozen/*`, `research/phase7_extension/native/learned_order_phase7.json` | Apache-2.0, with dataset attribution (THIRD_PARTY_NOTICES.md) | yes | the scripts named in each result's note |
| Embeddings and values derived from DINOv3 | the 8 DINOv3 `.npz` files, `new_parts/library.json.gz`, `scale/test/test-library.json.gz`, `research/phase5/near_duplicates.json` (DINOv3 cosine flags), the DINOv3 columns of `research/phase5/results/descriptors.npz` | **DINOv3 License** (copy in `LICENSES/DINOv3-License.md`), never Apache-2.0 | **no** (author decision): excluded from the main package; mixed files are cleaned of `dinov3*` arrays and fields; the historical originals stay in the repository under the DINOv3 License | `tools/research/phase4/extract_frozen_embeddings.py` with the user's own licensed DINOv3 weights (gated download); renders as in research/phase4 |
| Embeddings from DINOv2 and OpenCLIP, untrained-ViT controls | `dinov2*.npz`, `openclip*.npz`, `vit-random*.npz`, `openclip_development.zip` | Apache-2.0 (own outputs) with model attribution | yes | `fetch_openclip.py` (pinned URL, size, SHA-256); DINOv2 from its pinned Hugging Face revision |
| Renders of GSO objects | `research/phase4/embedding_experiment/development-images.zip` | CC BY 4.0 (adaptation of CC BY 4.0 models), attribution in THIRD_PARTY_NOTICES.md | yes, marked CC BY 4.0 | research/phase3 dataset tools from the pinned GSO commit |
| Numerical data derived from YCB objects | YCB rows of `research/phase4/grasp_prior/*.npz` and `library.json`, `research/phase5/results/descriptors.npz` | Apache-2.0 for the numbers, with YCB attribution | **aggregated statistics, tables, conclusions and identifiers only** (author decision, conservative): per-object descriptors, geometry and individual YCB rows are excluded from the package | `tools/verify/ycb_pick.sh` (now pinned to commit `57cc400…`; reproduction checked, see below) |
| Names, hashes and statistics of EGAD objects | `research/phase7/*`, `research/phase7_extension/*` | Apache-2.0 for the numbers, BSD-3 notice for EGAD | yes | `tools/research/phase7/egad_prepare.py` and `egad_confirm.py prepare` from the hashed archives |
| UR5 robot model | `apps/viewer/samples/ur5/`, `crates/project-io/tests/data/ur5/` | BSD-3-Clause (its own LICENSE files) | yes | — |
| Third-party meshes, images of YCB, model weights | — | — | **not included** (none committed) | fetch scripts listed in `tools/fetch/README.md` |

## Notes

- **Mixed file `research/phase5/results/descriptors.npz`.** It holds DINOv2, DINOv3, OpenCLIP, untrained-ViT and
  D2 descriptors in one file. Because it contains DINOv3 embeddings, the repository file is offered only under
  the DINOv3 License. The package carries a copy produced by `tools/package/build_package.py`: the `dinov3*`
  arrays and the YCB rows are dropped, and the copy's SHA-256 is in `MANIFEST_INCLUDED.json`. The package was
  built locally and verified (895 files, no findings). Nothing was uploaded.
- **Public history.** Excluding files from the package does not remove them from the Git history or from
  existing copies. The DINOv3 history audit is in `results/history_licence_audit.json` and
  `LICENSE_COMPLIANCE_REPORT.md`.
- **D-YCB.** The YCB licence (CC BY 4.0) and the exact provenance were verified. The 16 meshes from
  `pybullet-object-models` @ `57cc400` reproduce the Phase IV objects exactly: identical world bounds and
  identical candidate counts. One question remains open: whether the LGPL-2.1 terms of the packaging repository
  attach to numerical data derived from its meshes. No mesh or image is redistributed. Status: compliant with
  attribution, with this residual point listed for the author.
- **DINOv3.** Offering the embeddings under the DINOv3 License, with a copy of it, satisfies clause 1.b.i whether
  or not the embeddings are derivative works. This register therefore needs no legal opinion on that question,
  and the embeddings are not relicensed.
