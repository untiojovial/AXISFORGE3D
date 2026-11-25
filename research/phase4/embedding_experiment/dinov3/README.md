# DINOv3 preentrenado sobre el mismo panel

Estado: **EJECUTADO** en CPU sobre los 22 objetos de desarrollo autorizados. Los ocho objetos de test
siguen sellados.

Los pesos oficiales de Meta piden un formulario de acceso, y el repositorio de Hugging Face de Meta,
una aprobación manual. Se usa la copia que **timm** redistribuye bajo la licencia DINOv3, sin formulario.
El usuario aceptó esa licencia antes de usarla: hay que citar DINOv3 al publicar, cumplir los controles de
exportación y no usarlo con fines militares, de armas, nucleares ni de espionaje. La licencia va junto a
los pesos y no se redistribuyen.

| Campo | Valor |
|---|---|
| Modelo | `timm/vit_small_patch16_dinov3.lvd1689m` (DINOv3 ViT-S/16 destilado, LVD-1689M) |
| Revisión | `3bf4720a82ec2066db88137180ff1f83a675cef0` |
| `model.safetensors` | `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040` (coincide con el digest LFS publicado) |
| `config.json` | `8eb1eeb3a94a3faddfcdd9ae693607a9cd9c8b91c34381578f7f0a48ce454eef` (fija el preprocesado, por eso hace de hash del procesador) |
| Procesado | Redimensionado bicúbico a 256 × 256, normalización ImageNet, rejilla 16 × 16, 4 tokens de registro |
| Vector global | Token CLS tras la normalización final |
| RoPE | Periodos truncados a bfloat16 para igualar al original (nota de la ficha de timm) |
| Entorno | Python 3.12, PyTorch 2.9.1+cpu, timm 1.0.31, lote 16, unos 5 s por objeto en CPU |

Reproducibilidad: dos extracciones en directorios distintos dieron los 22 ficheros de características
byte a byte idénticos ([reproduction.json](reproduction.json)). El ZIP (129 MB) no se versiona; su índice
está en [bundle_index.json](bundle_index.json) y los vectores globales en
[global_embeddings.npz](global_embeddings.npz).

## Resultado en el panel emparejado

Mismas imágenes, candidatos, trayectorias y control geométrico que OpenCLIP y DINOv2
([comparación de los tres](../foundation_comparison.json), [análisis](analysis.json),
[figura](dinov3_panel.png)):

| Características | Modelo | AUC contextual | AUC adaptativo | MAE ε | Brier F1 |
|---|---|---:|---:|---:|---:|
| Geometría | — | .04170 | .04170 | .01378 | .03176 |
| Global | OpenCLIP | .04170 | .04170 | .01350 | .03231 |
| Global | DINOv2 | .05217 | .04170 | .01359 | .03254 |
| Global | DINOv3 | .04170 | .04170 | .01351 | .03264 |
| Local | OpenCLIP | .05217 | .05217 | .01335 | .03204 |
| Local | DINOv2 | .05217 | .05217 | .01367 | .03212 |
| Local | DINOv3 | .04518 | .04518 | **.01326** | .03274 |
| Global + local | DINOv3 | .04518 | .04518 | .01345 | .03322 |

Lectura descriptiva: con señales locales, DINOv3 mejora el AUC adaptativo menos que OpenCLIP y DINOv2
(+.00348 frente a +.01047 sobre geometría). Da el menor error de ε de los tres, pero empeora algo el
Brier de factibilidad. Son seis objetos de validación con muchos empates; como con DINOv2, no hay base
para declarar un modelo superior.

## Reproducir

```bash
# pesos: https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m, revisión fijada arriba
# (model.safetensors y config.json en /ruta/dinov3-timm; LICENSE.md incluido)
PYTHONPATH=. python tools/research/phase4/run_experiments.py \
  --job research/phase4/embedding_experiment/development-images.zip \
  --work /ruta/trabajo --bundle /ruta/dinov3_development.zip \
  --checkpoint /ruta/dinov3-timm --model dinov3-timm \
  --revision 3bf4720a82ec2066db88137180ff1f83a675cef0 --device cpu --batch-size 16 \
  --weights-sha256 2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040 \
  --config-sha256 8eb1eeb3a94a3faddfcdd9ae693607a9cd9c8b91c34381578f7f0a48ce454eef
PYTHONPATH=. python tools/research/phase4/import_embedding_bundle.py \
  --job research/phase4/embedding_experiment/development-images.zip \
  --bundle /ruta/dinov3_development.zip --destination /ruta/importado --analysis-out /ruta/analysis.json \
  --expected-revision 3bf4720a82ec2066db88137180ff1f83a675cef0 \
  --expected-checkpoint-sha256 2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040 \
  --expected-processor-sha256 8eb1eeb3a94a3faddfcdd9ae693607a9cd9c8b91c34381578f7f0a48ce454eef
```

DINOv3: Siméoni et al., 2025, Meta AI Research, licencia DINOv3. Vistas derivadas de Google Scanned
Objects (Downs et al., 2022), adaptación MuJoCo de Kevin Zakka, CC BY 4.0.
