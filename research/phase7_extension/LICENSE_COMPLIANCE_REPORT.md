# Licence compliance report (2026-11-26)

## What was done

- **Licence for own code.** The author authorised Apache-2.0 for the code they hold the rights to (instruction of
  2026-11-26).
  - `LICENSE` now holds the official Apache License 2.0 text, fetched from
    https://www.apache.org/licenses/LICENSE-2.0.txt (SHA-256 `cfc7749b…bc523d30`).
  - `Cargo.toml` (workspace `license = "Apache-2.0"`, inherited by every crate) and `pyproject.toml`
    (`license = "Apache-2.0"`) already declared it and are unchanged.
- **Third-party material.** Third-party material is not relicensed. The licence texts kept in `LICENSES/` are:
  - `Apache-2.0.txt`;
  - `CC-BY-4.0.txt`, the official legal code;
  - `DINOv3-License.md`, the official text, last updated 2025-08-19;
  - `BSD-3-Clause-EGAD.txt`;
  - `BSD-3-Clause-example-robot-data.txt`.
- **Notices and registers.**
  - `THIRD_PARTY_NOTICES.md`: attribution, licence, the modifications made, and what is or is not
    redistributed.
  - `research/phase7_extension/third_party_manifest.json`: the machine-readable register, built by
    `tools/licenses/build_manifest.py`. For each component it records name, version, source, licence,
    evidence, verification date, redistribution terms, attribution, the affected files with their SHA-256, and
    status.
  - `ARTIFACT_REDISTRIBUTION_REGISTER.md`: the licence and package decision for each class of committed
    artefact.
- **Dependency audit.** `tools/licenses/audit_dependencies.py` produced
  `results/dependency_licences.json`. All 69 Rust crates and all audited Python packages declare permissive
  licences; none needs review.
- **Fetch guide.** `tools/fetch/README.md` lists how each external input is fetched and pinned.
  `tools/verify/ycb_pick.sh` is now pinned to a commit.

## Status by component

| Component | Licence | Status | Evidence |
|---|---|---|---|
| Own code, technical docs, own results | Apache-2.0 (Copyright 2026 Eduardo Pérez González) | compliant | `LICENSE`, `NOTICE`, author authorisation |
| Manuscript (`publication/`) | all rights reserved pending publication | excluded from Apache | `RIGHTS.md` |
| YCB (16 objects) | CC BY 4.0 (data), MIT (code); packaging repository LGPL-2.1 | **resolved** (licence and exact provenance); residual D-YCB | S3 index and AWS registry read 2026-11-26; geometry reproduced from commit `57cc400` (bounds identical to 0.0 mm, identical candidate counts) |
| GSO | CC BY 4.0 / MIT | compliant with attribution | pinned adaptation README (Phase III) |
| EGAD | BSD-3-Clause | compliant | licence file |
| DINOv3 embeddings | DINOv3 License | compliant under the stricter reading; never Apache | official licence text; separate archive in the package proposal |
| DINOv2 embeddings | Apache-2.0 weights | compliant | model card (Phase VII audit) |
| OpenCLIP e31 embeddings | MIT weights and code | **resolved**: checkpoint identified as e31 and its card verified | `pretrained.py` mapping; HF card `license: mit`; checkpoint SHA-256 |
| GPD | BSD-2-Clause | compliant (not redistributed) | repository LICENSE.md |
| UR5 (example-robot-data) | BSD-3-Clause | compliant | licence beside the files |
| three.js | MIT | compliant (CDN, not vendored) | — |
| Rust and Python dependencies | permissive | compliant (not redistributed) | audit output |

## Author decisions applied (instruction "final author decisions", 2026-11-26)

| Item | Decision | What was done |
|---|---|---|
| Own code | Apache-2.0, "Copyright 2026 Eduardo Pérez González" | `NOTICE`; `Cargo.toml` `authors`; `pyproject.toml` `authors`; README licence section. Third-party rights are not attributed to the author. |
| Licences by material | code, data, third-party material and manuscript separated | code, technical documentation and own results: Apache-2.0. Third-party material: its own licence (`THIRD_PARTY_NOTICES.md`, `LICENSES/`). Manuscript and submission material (`publication/`): **not** Apache, all rights reserved pending publication (`publication/intelligent_service_robotics/RIGHTS.md`). |
| YCB (conservative) | keep aggregated statistics, tables, conclusions and identifiers; exclude per-object descriptors, geometry and individual data from the new package | `tools/package/build_package.py` drops YCB rows from `.npz` and JSON, and excludes files it cannot separate; `verify_package.py` finds no YCB row. The repository keeps the historical files. `pybullet-object-models` stays pinned at `57cc400` for regeneration (`tools/verify/ycb_pick.sh`); attribution kept. |
| DINOv3 | no embeddings in the main package | Mixed files (for example `research/phase5/results/descriptors.npz`) are cleaned in the package (the `dinov3*` arrays dropped); DINOv3 folders are excluded except their README, summary, reproduction and index files. The historical originals stay in the repository under the DINOv3 License. |
| OpenCLIP e31 | MIT evidence; weights not redistributed; reproducible download | model card `license: mit`; checkpoint SHA-256 pinned; `fetch_openclip.py`. |
| Manuscript | not Apache; previously granted licence checked; no revocation claimed | `RIGHTS.md` records the statement history: no licence at `5b08627`; ambiguous README wording ("own code, documentation and results … Apache") from `fd982a9` until the correction; excluded afterwards. The window lasted 40 minutes (`fd982a9` to `cbab6c5`). The author decided on 2026-11-27 not to raise it with the editor and to disclose the public drafts as a preprint; nothing is claimed to be revoked. |

## Public-history audit (DINOv3)

`tools/licenses/audit_history.py` → `results/history_licence_audit.json`.

**Window without the licence copy.**

- DINOv3-derived embeddings entered the history at `75bec22` (2026-11-25).
- The licence copy (`LICENSES/DINOv3-License.md`) was added at `fd982a9` (2026-11-26), 24 commits later.
- In that window the folder README described the licence terms and the duty to cite, but did not include the
  licence text.

**Branches.**

- `research/axisforge-phase6-fortis-r` still has 12 DINOv3-derived files at its head (`4c3e8af`) and no
  licence copy.
- The working branches have the copy.

**What was done (no rewrite).**

- A notice in `README.md` and `THIRD_PARTY_NOTICES.md`: every DINOv3-derived file in any commit or branch is
  governed by the DINOv3 License, whose copy is in `LICENSES/`.
- No history rewrite and no force push.

**Proposed, pending the author's authorisation.** Either:

- a fast-forward commit on `research/axisforge-phase6-fortis-r` adding the licence copy and the notice; or
- archiving or deleting that stale branch through the hosting service.

Neither removes the copies already in existing clones or forks.

**Scope of the package exclusion.** Excluding the embeddings from the new package does **not** remove them from
the Git history, nor from any copy already made.

## Reproducibility package (local only)

`axisforge3d-research-reproducibility` is built by `tools/package/build_package.py` from `git ls-tree` at a
given commit.

- It contains `MANIFEST_INCLUDED.json` and `MANIFEST_EXCLUDED.json`, with a reason and a licence for each
  file, plus SHA-256 digests, regeneration scripts and instructions.
- `tools/package/verify_package.py` scans it independently for DINOv3 arrays, YCB rows, YCB geometry,
  manuscript files, hashes and unlisted files.
- Draft Zenodo metadata: `research/phase7_extension/package/zenodo_metadata.json`.
- **No release, upload or DOI was made.**

## Still open (author decisions)

1. **D-release.** Any release, Zenodo deposit or DOI. None was created.
2. **D-phase6-branch.** The remedy for `research/axisforge-phase6-fortis-r` (above).
3. **D-manuscript-window.** Decided by the author on 2026-11-27: the ambiguous wording between `fd982a9` and
   `cbab6c5` (40 minutes) is not raised with the editor. The public drafts are disclosed as a preprint in the cover
   letter, and the history stays in `RIGHTS.md`.
4. **D-YCB residual.** Whether LGPL-2.1 terms of `pybullet-object-models` attach to derived numbers. It is
   moot for the new package, which excludes per-object YCB data. It remains relevant only to the historical
   files in the repository.

## Superseded list (before the author decisions)


1. **D-YCB.** Whether the LGPL-2.1 terms of `pybullet-object-models` attach to numerical data derived from its
   meshes. No YCB mesh or image is redistributed; attribution is complete. The options are to keep the YCB-derived
   numbers in the public package, or to exclude them and rely on the pinned regeneration script.
2. **D-DINOv3-package.** Whether to publish the DINOv3-derived archive at all. If published, it goes only under the
   DINOv3 License with its copy and the acknowledgement. Trade-control terms bind every recipient.
3. **D-name.** The copyright holder's name for the Apache-2.0 notice. It is currently "the AxisForge3D author";
   no name is invented.
4. **D-release.** Any release, Zenodo deposit or DOI. None was created.

All of these are legal or personal decisions. Everything that could be done automatically, such as licence texts,
notices, registers, hashes, the audit and pinning, has been done.
