# Reproducibility of the Phase VII extension

How to reproduce every result of the extension. Read it with `research/phase7/reproducibility.md`, which covers
Phase VII.

**Levels of reproduction.**

| Level | What it reproduces | Needs | Status |
|---|---|---|---|
| **Statistical** | every reported statistic, recomputed from the committed per-object data | the repository; Python with `reproducibility/requirements-lock.txt` | done in a clean container (`reproducibility/DOCKER_VALIDATION_RECORD.md`) and in the development environment |
| **Partial** | planner decisions with the frozen models on prepared objects (native equivalence, the fallback bound) | the repository; Rust; the prepared EGAD pool, regenerated from the public EGAD meshes | done in the container for 255 EGAD objects (510 of 510 decisions identical) |
| **Full** | catalogues, MuJoCo physics, GPD scores, timing, both confirmations | everything above, plus MuJoCo, CoACD, the GPD build and hours of CPU | done once, in the development environment, with the commands below; not repeated in a clean environment |

The rule for every run: one thread per process (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`), seeds as recorded
in each result file, Python run with `-I` on downloaded data.

## 1. Statistical reproduction (minutes)

```
bash reproducibility/reproduce_statistics.sh            # P2 audit, corrections addendum, figure checks, tests
docker build --network host -f reproducibility/Dockerfile.repro -t axisforge-repro .
docker run --rm -v EGAD_POOL:/data/pool:ro axisforge-repro bash reproducibility/docker_validate.sh
```

For the second confirmation, the analysis can be re-run from the committed per-object outputs:

```
python tools/research/phase7_extension/confirm_x.py analyse --pool NEW_POOL --out-dir OUT \
    --freeze research/phase7_extension/PHASE7X_FREEZE.json --out confirmation_x.json
```

## 2. Inputs that are fetched, not committed

| Input | Source and pin | Script |
|---|---|---|
| EGAD meshes | official archives, SHA-256 in `research/phase7/egad_selection.json` | `tools/research/phase7/egad_prepare.py` |
| GSO parts | pinned GSO commit (Phase III) | research/phase3 dataset tools |
| YCB meshes | `pybullet-object-models` at commit `57cc400` | `tools/verify/ycb_pick.sh` |
| OpenCLIP e31 weights | pinned URL, size and SHA-256 | `research_grasping/phase3/fetch_openclip.py` |
| DINOv3 weights | gated; the user's own licensed copy | `tools/research/phase4/extract_frozen_embeddings.py` |
| GPD | `atenpas/gpd` at commit `6327f20`, built unmodified | `tools/research/phase7_extension/gpd_baseline.py` |

## 3. Extension pipeline (in order)

| Step | Command (see each script's docstring for every argument) | Output |
|---|---|---|
| Start check | `verify_start.py`, `verify_freeze.py --commit de22a8a` | `results/start_verification.json`, `results/freeze_verification.json` |
| Corrections and P2 audit | `corrections_addendum.py`, `p2_audit.py` | `results/corrections_addendum.json`, `results/p2_statistical_audit.json` |
| Native export | `export_models.py` (Phase VII models; `freeze_x.py` for X) | `native/learned_order_phase7.json`, `frozen/learned_order_x.json` |
| Native equivalence | `native_validation.py` (`--robustness-pkl`, `--modes` for X) | `results/native_inference_validation{,_x}.json` |
| Fallback bound | `fallback_audit.py` | `results/fallback_audit.json` |
| Benchmark | `integrated_benchmark.py`, `memory_benchmark.py`, `ordering_ceiling.py` | `results/integrated_runtime_benchmark.json`, `results/memory_benchmark.json`, `results/ordering_ceiling.json` |
| v3 labels | `v3_sweep.py`, `v3_models.py needed`, `extra_physics.py`, `v3_models.py analyse` | `results/v3_models.json` |
| Freeze of X | `freeze_x.py`, `power_new.py`, `make_freeze_x.py` | `frozen/*`, `power/*`, `PHASE7X_FREEZE.json` |
| GPD | `gpd_baseline.py score`, `gpd_evaluate.py`, `gpd_variability.py` | `results/external_baselines_v2.json`, `results/gpd_variability.json` |
| Articulated v2 | `articulated2_dev.py run` / `report` | `results/articulated2_development.json` |
| Optuna (exploratory) | `optuna_mlp.py`, then `--summarise` | `results/optuna_mlp.json` |
| Second confirmation | `build_newset.py`; `confirm_x.py prepare`; `gpd_baseline.py score`; `confirm_x.py orders`, `physics`, `timing`, `analyse` | `results/confirmation_x.json` and the per-object outputs |
| Manuscript numbers | `make_tables_x.py --out publication/intelligent_service_robotics/tables` | `numbers_x.tex`, `tab_x_*.tex` |
| Package | `tools/package/build_package.py`, `verify_package.py` | `axisforge3d-research-reproducibility/` (local) |

## 4. The second confirmation

- **Freeze.** Every stage uses the fixed planner binary whose SHA-256 is in `PHASE7X_FREEZE.json`. The run
  wrapper checks the freeze (36 files, the binary and the GPD adapter) before and after each stage.
- **Restarts.** The stages are restart-safe: one file per completed object. A crashed stage is resumed with the
  same inputs, never re-sampled. Container restarts are listed in the execution record of
  `results/confirmation_x.json`.
- **Timing.** Timing is run once, with the machine as quiet as possible: the other jobs are stopped, and the
  load is recorded per object.

## 5. Known limits of reproducibility

- **GPD** is stochastic (`std::random_device` in its shadow computation). Re-running it gives a different draw
  (`gpd_variability.json`).
- **Timing** depends on the hardware and the machine load. The benchmark records the CPU model, the pinned CPU,
  the thread settings and the processes running at the start.
- **The full Docker image** (`reproducibility/Dockerfile`) was not built here: 9.8 GB of disk with the `vfs`
  storage driver. The slim image covers the statistical and partial levels only.
- **Third-party data.** YCB per-object data and DINOv3 embeddings are not in the reproducibility package. They
  are regenerated with the scripts above.
