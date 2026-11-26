# Code availability (draft)

**Repository.**

- GitHub `untiojovial/Colegio-Rumareda`.
- Branches: `claude/new-session-nsx8rp` and `research/axisforge-phase7-final-publication`.
- Contents:
  - the planner (Rust workspace `crates/`, `apps/cli`);
  - the native learned ordering (`crates/grasping/src/learned.rs`);
  - the research tools and analysis scripts (`tools/research/phase7/`, `tools/research/phase7_extension/`,
    `research_grasping/`);
  - the tests.
- **Visibility.** Whether the repository is public or private must be checked by the author before
  submission.

**Licence.**

- The author's own code, technical documentation and experimental results are under the Apache License 2.0
  (`LICENSE`; `NOTICE`: "Copyright 2026 Eduardo Pérez González").
- Third-party components keep their own licences: `THIRD_PARTY_NOTICES.md`, `LICENSES/`.
- The manuscript and the submission material in `publication/` are **not** under Apache-2.0 (`RIGHTS.md`).

**Archival copy.** No release, Zenodo deposit or DOI has been created; any of these needs the author's
authorisation.

- A local reproducibility package (`axisforge3d-research-reproducibility`) can be built from any commit with
  `tools/package/build_package.py`. It comes with inclusion and exclusion manifests, and draft Zenodo
  metadata is in `research/phase7_extension/package/zenodo_metadata.json`.
- The commits that matter are:
  - the Phase VII freeze, `de22a8a`;
  - the extension freeze, `b478289` (`PHASE7X_FREEZE.json`);
  - the final commit of the submission.

**What each part reproduces.**

| What | Entry point |
|---|---|
| Development experiments (Phase VII) | `tools/research/phase7/*.py` |
| First confirmation | `tools/research/phase7/egad_prepare.py`, `egad_confirm.py` |
| Native integration, benchmark, fallback bound | `tools/research/phase7_extension/{export_models,native_validation,integrated_benchmark,memory_benchmark,ordering_ceiling,fallback_audit}.py` |
| v3 labels and selection | `v3_sweep.py`, `extra_physics.py`, `v3_models.py`, `freeze_x.py` |
| GPD (ranking-controlled) | `gpd_adapter/`, `gpd_baseline.py`, `gpd_evaluate.py`, `gpd_variability.py` |
| Second confirmation | `build_newset.py`, `confirm_x.py` (prepare, orders, physics, timing, analyse) |
| Tables and figures | `tools/research/phase7/make_tables.py`, `make_figures.py`, `tools/research/phase7_extension/make_tables_x.py` |
| Clean-environment validation | `reproducibility/Dockerfile.repro`, `docker_validate.sh` |

**Third-party weights.** None are redistributed.

- DINOv3: gated, under its own licence.
- OpenCLIP ViT-B-32 LAION-400M e31: MIT according to its model card. It is fetched by a pinned script and
  checked by SHA-256.
- GPD's classifier weights: shipped with the official GPD source, which is fetched at a pinned commit.
