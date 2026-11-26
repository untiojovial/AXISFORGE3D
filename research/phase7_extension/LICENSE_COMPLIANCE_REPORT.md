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
| Own code, docs, results | Apache-2.0 | compliant | `LICENSE`, author authorisation |
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

## Not resolved here (closed list of author decisions)

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
