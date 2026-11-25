# DINOv2 preentrenado sobre el mismo panel

Estado: **EJECUTADO** con el checkpoint oficial, en CPU, sobre los 22 objetos
de desarrollo autorizados. Los ocho objetos de test siguen sellados.

El bloqueo anterior era de red: desde la máquina de la ejecución de Codex,
Hugging Face y `dl.fbaipublicfiles.com` devolvían HTTP 403. Desde esta máquina
la API oficial respondió y se descargó el modelo con
`tools/research/phase4/download_models.py`, sin espejos ni tokens.

| Campo | Valor |
|---|---|
| Modelo | `facebook/dinov2-small` (ViT-S/14), licencia Apache-2.0 |
| Revisión oficial | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| `model.safetensors` | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` (coincide con el digest LFS publicado) |
| `config.json` | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |
| `preprocessor_config.json` | `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828` |
| Procesado | `BitImageProcessor`: lado corto 256, recorte central 224, rejilla 16×16 de parches |
| Entorno | Python 3.12, PyTorch 2.9.1+cpu, Transformers 4.57.3, lote 16 |
| Tiempo | 38 s para las 352 vistas en CPU (4 hilos) |

Reproducibilidad: dos extracciones en directorios distintos dieron los 22
ficheros de características byte a byte idénticos
([reproduction.json](reproduction.json)). El ZIP del bundle (129 MB) supera el
límite de ficheros de GitHub y no se versiona: se regenera en unos 40 s y su
índice está en [bundle_index.json](bundle_index.json). Los vectores globales
por vista (22 × 16 × 384) sí se versionan en
[global_embeddings.npz](global_embeddings.npz), para el planificador.

## Resultado en el panel emparejado

Mismas imágenes, mismos candidatos, mismas trayectorias y mismo control
geométrico que OpenCLIP ([comparación](../foundation_comparison.json),
[análisis](analysis.json), [figura](dinov2_panel.png)):

| Características | Modelo | AUC contextual | AUC adaptativo | MAE ε | Brier F1 |
|---|---|---:|---:|---:|---:|
| Geometría | — | .04170 | .04170 | .01378 | .03176 |
| Global | OpenCLIP | .04170 | .04170 | .01350 | .03231 |
| Global | DINOv2 | .05217 | .04170 | .01359 | .03254 |
| Local | OpenCLIP | .05217 | .05217 | .01335 | .03204 |
| Local | DINOv2 | .05217 | .05217 | .01367 | .03212 |
| Global + local | DINOv2 | .05217 | .05217 | .01382 | .03272 |

Lectura descriptiva, no confirmatoria: las señales locales de DINOv2 producen
el mismo cambio de AUC adaptativo que las de OpenCLIP (+.01047 frente a
geometría); el vector global de DINOv2 mejora la política contextual donde el
de OpenCLIP no, pero no la adaptativa; el error de ε es algo peor que con
OpenCLIP local y el Brier de factibilidad empeora ligeramente respecto a
geometría, como ya ocurría con OpenCLIP. Son seis objetos de validación, tres
semillas y muchos empates; no hay base para declarar un modelo superior.

## En el planificador de agarres

Los mismos pesos describen piezas nuevas para el prior visual de `axisforge plan --prior`
([prior de agarre](../../grasp_prior/README.md)).

## Reproducir

```bash
python tools/research/phase4/download_models.py --model dinov2 \
  --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 --out /ruta/dinov2-small
PYTHONPATH=. python tools/research/phase4/run_experiments.py \
  --job research/phase4/embedding_experiment/development-images.zip \
  --work /ruta/trabajo --bundle /ruta/dinov2_development.zip \
  --checkpoint /ruta/dinov2-small --model dinov2 \
  --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 --device cpu --batch-size 16 \
  --weights-sha256 ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1 \
  --config-sha256 1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d \
  --processor-sha256 14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828
PYTHONPATH=. python tools/research/phase4/import_embedding_bundle.py \
  --job research/phase4/embedding_experiment/development-images.zip \
  --bundle /ruta/dinov2_development.zip --destination /ruta/importado \
  --analysis-out /ruta/analysis.json \
  --expected-revision ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --expected-checkpoint-sha256 ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1 \
  --expected-processor-sha256 14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828
```

Las vistas derivan de Google Scanned Objects (Downs et al., 2022), adaptación
MuJoCo de Kevin Zakka, CC BY 4.0. Los pesos no se redistribuyen.
