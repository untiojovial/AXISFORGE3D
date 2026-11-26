# Fase VI — reproducibilidad

## Entorno

Python 3.12 con los extras `phase3`/`test` de `pyproject.toml`:

- MuJoCo 3.3.7;
- scikit-learn;
- scipy;
- trimesh.

También hace falta Rust estable para el planificador (`axisforge`). Todo corre en CPU; aquí, 4 núcleos.

## Entradas

Se regeneran con los scripts de la Fase V; no se versionan por tamaño:

| Entrada | Cómo se obtiene |
|---|---|
| Catálogos de 80 mm (`cat80/`) | `tools/research/phase5/build_catalogues.py` |
| `store.json` (cajas y descriptores) | `tools/research/phase5/build_store.py` |
| Piezas de colisión GSO (`COLLISION_DIR`) | `tools/research/phase5/fetch_collision.py`, revisión `6ff8d275…` |
| Proyectos de celda (`POOL`) | Los de la Fase V |

## Orden de ejecución

```bash
P=research/phase6; T=tools/research/phase6
# A. Auditoría
python $T/audit.py --catalogues cat80 --store store.json --library research/phase5/library_parts.txt \
  --dev $P/dev_parts.json --features-out $P/results/feature_audit.json \
  --manifest-out $P/results/phase6_data_manifest.json
python $T/physics_controls.py --dev $P/dev_parts.json --catalogues cat80 --collision COLLISION_DIR \
  --projects POOL --store store.json --phase5-dynamics research/phase5/results/dynamics.json.gz \
  --out $P/results/physics_validation.json --sample 60
# Probabilidades (reproduce las posiciones de la Fase V; si no coinciden, se detiene)
python $T/priors.py --catalogues cat80 --store store.json --library research/phase5/library_parts.txt \
  --dev $P/dev_parts.json --phase5-rows research/phase5/results/test_rows.json.gz --out priors.npz
# Dinámica v2: pasada 1 (robustez de todos los viables), primeros agarres, pasada 2 (9 condiciones), unión
python $T/sweep.py robust --dev $P/dev_parts.json --catalogues cat80 --collision COLLISION_DIR \
  --projects POOL --store store.json --cache cache --jobs 4
python $T/first_grasps.py --catalogues cat80 --dev $P/dev_parts.json --priors priors.npz \
  --phase5-dynamics research/phase5/results/dynamics.json.gz --out first_grasps.json
python $T/sweep.py full --labels first_grasps.json --dev $P/dev_parts.json --catalogues cat80 \
  --collision COLLISION_DIR --projects POOL --store store.json --cache cache --jobs 4
python $T/sweep.py merge --dev $P/dev_parts.json --cache cache --out $P/results/sweep.json.gz
# B, C, D
python $T/mechanism.py --catalogues cat80 --store store.json --library research/phase5/library_parts.txt \
  --dev $P/dev_parts.json --phase5-dynamics research/phase5/results/dynamics.json.gz --priors priors.npz \
  --sweep $P/results/sweep.json.gz --out $P/results/robustness_mechanism.json \
  --ranking-out $P/results/ranking_quality_analysis.json
python $T/fusion_bench.py --catalogues cat80 --dev $P/dev_parts.json --priors priors.npz \
  --sweep $P/results/sweep.json.gz --protocol $P/protocol.json --out $P/results/fusion_benchmark.json
python $T/calibration.py --catalogues cat80 --library research/phase5/library_parts.txt \
  --dev $P/dev_parts.json --priors priors.npz --out $P/results/calibration_results.json
python $T/power.py --fusion $P/results/fusion_benchmark.json --out $P/results/statistical_power.json
python $T/feature_ablation.py --catalogues cat80 --store store.json --library research/phase5/library_parts.txt \
  --dev $P/dev_parts.json --sweep $P/results/sweep.json.gz --out $P/results/feature_ablation.json
python $T/infeasibility.py --catalogues cat80 --store store.json --library research/phase5/library_parts.txt \
  --rows research/phase5/results/test_rows.json.gz --out $P/results/infeasibility_benchmark.json
# Etapa D: P(H|F). Si pide primeros agarres sin cargas, simularlos y volver a unir:
python $T/conditional.py --catalogues cat80 --store store.json --dev $P/dev_parts.json --priors priors.npz \
  --sweep $P/results/sweep.json.gz --fusion $P/results/fusion_benchmark.json \
  --out $P/results/robustness_prediction.json --manifest-out $P/results/dynamic_training_manifest.json
python $T/sweep.py full --labels $P/results/missing_first_grasps.json ... --cache cache_supp
python $T/sweep.py merge --dev $P/dev_parts.json --cache cache --supplement cache_supp --out $P/results/sweep.json.gz
python $T/negative_transfer.py --fusion $P/results/fusion_benchmark.json \
  --conditional $P/results/robustness_prediction.json --out $P/results/negative_transfer_analysis.json
# Etapa G: coste online, un hilo y la máquina sin otra carga
RAYON_NUM_THREADS=1 python $T/online_cost.py --binary target/release/axisforge --dev $P/dev_parts.json \
  --catalogues cat80 --store store.json --projects POOL --priors priors.npz \
  --fusion $P/results/fusion_benchmark.json --out $P/results/cost_profile_online.json --parts 60
```

El suplemento (26 primeros agarres de P(F)·P(H|F)) solo añade condiciones de carga a agarres que no
eran primeros en ningún otro orden. Se comprobó que, tras la unión, `fusion_benchmark.json` y
`robustness_mechanism.json` salen idénticos byte a byte.

## Semillas

| Uso | Semilla |
|---|---|
| Clasificador | 0, como en la Fase V |
| Pliegues de la biblioteca para la calibración y la importancia por permutación | 20261129 |
| Pliegues externos de la fusión | 20261130 |
| Bootstrap | 20261128, 10 000 réplicas |
| Ruidos de pose | 1 y 2 |

MuJoCo es determinista en la misma máquina. En el control de paso y solver, el veredicto cambia en 0–2 de
180 ensayos ([physics_validation.json](results/physics_validation.json)).

## Tests

```bash
python -m pytest tests/phase6 -q        # fusión, filtro, condiciones, centro de masas
cargo test --workspace --release        # 359
python -m pytest tests -q               # 463 + los de la Fase VI; 65 omitidos (datos externos)
```
