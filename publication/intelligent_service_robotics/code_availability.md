# Code availability (draft)

**Repository.** All code is in the project repository (GitHub `untiojovial/Colegio-Rumareda`). This
covers the planner (Rust workspace `crates/`, `apps/cli`), the research tools and the analysis scripts
(`tools/research/phase7/`, `research_grasping/`). **Its visibility (public or private) must be checked by
the author before submission.**

**Licence.** `Cargo.toml` and `pyproject.toml` declare **Apache-2.0**, but the repository root has no
`LICENSE` file. Adding it is the author's decision.

**Archival copy.** No Zenodo release or DOI has been created; that needs the author's authorisation. If
the journal requires an archived copy, create one from the commit named in
`research/phase7/PHASE7_MODEL_FREEZE.json` and the final Phase VII commit.

**What each part reproduces.**

| What | Entry point |
|---|---|
| Development experiments | `tools/research/phase7/{mlp_benchmark,mlp_eval,physics_dev,joint_v3,physics_report,articulated_pilot,negative_transfer,cross_robot,few_shot_analysis,power}.py` |
| Confirmation | `tools/research/phase7/egad_prepare.py` and `egad_confirm.py` (stages freeze, prepare, describe, apply, physics, articulated, timing, analyse) |
| Tables and figures | `tools/research/phase7/make_tables.py`, `make_figures.py` |
| Costs | `tools/research/phase7/full_online_cost.py` and `egad_confirm.py timing` |

**Third-party model weights.** None are redistributed. DINOv3 is used under its own licence. The OpenCLIP
LAION-400M weights have a licence that was not verified. Hashes are in
`research/phase7/PHASE7_DATA_MANIFEST.json`.
