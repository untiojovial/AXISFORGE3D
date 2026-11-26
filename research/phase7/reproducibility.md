# Phase VII — reproducibility

## 1. Environment

| Component | Version or setting |
|---|---|
| Machine | Linux container, 4 CPU cores, no GPU. All Phase VII runs are CPU-only; one thread per process (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`) |
| Python | 3.12 virtual environment, Phase IV constraints (`research/constraints-linux-cpu-py312.txt`) |
| Python packages | numpy, scipy, scikit-learn, torch (CPU), timm, open_clip_torch, transformers, mujoco, trimesh, coacd. Exact versions of the frozen models are in `PHASE7_MODEL_FREEZE.json` (`environment`) |
| Rust | `cargo build --release` of the workspace; the binary is `target/release/axisforge` |

## 2. Inputs that are not in the repository

| Input | How to obtain it | Check |
|---|---|---|
| GSO and YCB catalogues of the 316 development parts | `tools/research/phase5/build_catalogues.py` on the Phase V pool (`research/phase5/README.md`) | Catalogue hashes are in `PHASE7_MODEL_FREEZE.json` (`training_data.catalogue_sha256`) |
| GSO collision models | Phase V `fetch_collision.py` | — |
| EGAD archives | CSIRO Data Access Portal (public, BSD-3) | `research/phase7/egad_archives.sha256` |
| DINOv3 ViT-S/16 (timm copy) | Per its licence | Weights and config hashes in `PHASE7_DATA_MANIFEST.json` |
| Untrained ViT checkpoints | Generated in Phase V | — |
| OpenCLIP checkpoint | Per its licence | Hash in `PHASE7_DATA_MANIFEST.json` |

## 3. Development

Run in this order. Each script's docstring has its full command line.

1. `tools/research/phase7/verify_history.py` → `results/phase6_verification.json`
2. `tools/research/phase7/physics_dev.py` → `results/physics_dev.json.gz`; then `joint_v3.py` →
   `results/physics_dev_with_joint.json.gz`
3. `tools/research/phase7/robustness_bench.py` → `results/robustness_benchmark.json`
4. `tools/research/phase7/local_features.py`, `mlp_benchmark.py` (3 seeds, `--only` to split), then
   `mlp_eval.py` → `results/mlp_benchmark.json`, `linear_projection_benchmark.json`,
   `visual_ablation.json`, `ranking_loss_benchmark.json`, `local_features_benchmark.json`
5. `tools/research/phase7/physics_report.py` → `results/physical_validation.json`,
   `rotation_validation.json`, `friction_sensitivity.json`
6. `tools/research/phase7/articulated_pilot.py run` and `report` →
   `results/articulated_simulator_validation.json`
7. `tools/research/phase7/negative_transfer.py`, `cross_robot.py`, `few_shot_analysis.py`, `power.py`
   → the corresponding JSON files
8. `tools/research/phase7/analytic_baseline.py collect` and `evaluate` → `results/external_baselines.json`
   (epsilon part)
9. `tools/research/phase7/full_online_cost.py` → `results/full_online_costs.json` (quiet machine)

## 4. Confirmation (order fixed by the protocol, part II)

```bash
python -I tools/research/phase7/egad_prepare.py build --egad EGAD_DIR --selection research/phase7/egad_selection.json \
    --binary target/release/axisforge --out POOL --jobs 4 --positions 0 300 --evaluation-set
python tools/research/phase7/egad_confirm.py freeze ...        # produces the files whose hashes are in PHASE7_MODEL_FREEZE.json
python tools/research/phase7/egad_confirm.py prepare --pool POOL --binary target/release/axisforge --out-dir CONF --jobs 4
python tools/research/phase7/egad_confirm.py describe --pool POOL --out-dir CONF --models MODELS --jobs 4
python tools/research/phase7/egad_confirm.py apply --frozen FROZEN --out-dir CONF
python tools/research/phase7/egad_confirm.py physics --pool POOL --out-dir CONF --jobs 4
python tools/research/phase7/egad_confirm.py articulated --pool POOL --out-dir CONF --binary target/release/axisforge --jobs 4
python tools/research/phase7/egad_confirm.py timing --pool POOL --out-dir CONF --binary target/release/axisforge --frozen FROZEN   # quiet machine
python tools/research/phase7/egad_confirm.py analyse --out-dir CONF --results research/phase7/results
python tools/research/phase7/cross_dataset.py ...              # development vs EGAD summary
```

- **Checking the frozen models.** `apply` and `timing` refuse to run if any frozen model file differs
  from its hash.
- **Regenerating the MLPs.** The MLPs with visual inputs are not in the repository. Regenerate them with
  `freeze` and compare their hashes.
- **Gradient-boosting models.** They and the geometry-only MLPs are in `research/phase7/frozen/`. They
  are Python pickles: load them only after checking the SHA-256.

## 5. Manuscript

```bash
python tools/research/phase7/make_tables.py --results research/phase7/results --out publication/intelligent_service_robotics/tables
python tools/research/phase7/make_figures.py --results research/phase7/results --out publication/intelligent_service_robotics/figures
cd publication/intelligent_service_robotics && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The Springer Nature template class is not redistributed. Its download source and the hash of the zip
used are in `publication/intelligent_service_robotics/README.md`.

## 6. Determinism and expected variation

**Deterministic given the same library versions.**

- Gradient boosting (seed 0) and catalogues.
- Order replay.
- MuJoCo trials: same inputs, same build.
- Bootstrap: seed 20261128.

**Single-thread deterministic.** MLP training on CPU with fixed seeds.

**Not deterministic.** Wall-clock timings vary with the machine and its load. They are reported as
percentiles over repeated calls, and the absolute values are specific to this container.

**CoACD.** It is seeded, but its output can change across library versions. The collision pieces of each
confirmatory object are kept in the run directory and listed with their counts in `built.json`.
