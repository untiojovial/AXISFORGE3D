# Protocolo: renders que conservan la escala, en 50 piezas nuevas

Fijado **antes** de resolver ninguna pieza de prueba. La hipótesis nace del análisis de la prueba anterior
([piezas nuevas](../new_parts/README.md)): las vistas de la Fase III ponen la cámara a 1,5 veces la diagonal
de la caja, así que todas las piezas ocupan lo mismo en la imagen. Por eso las 66 piezas ya vistas (16 YCB y
las 50 de la semilla 20261126) son ahora de desarrollo, y la prueba usa piezas que no ha visto nada.

## Render que conserva la escala

Misma órbita de 16 vistas a 25° y mismas intrínsecas, pero con la cámara **a 0,6 m del centro de la caja
para todas las piezas** y 224 px de lado. Una pieza el doble de grande ocupa el doble en la imagen; la más
pequeña del desarrollo (fresa, 8 cm de diagonal) ocupa unos 30 px y la mayor (martillo, 36 cm) cabe en el
encuadre. Código: `render_views(..., distance=0.6, size=224)` en
[visual_prior.py](../../../../research_grasping/phase4/visual_prior.py), con una prueba que comprueba que una
pieza el doble de grande ocupa unas cuatro veces más píxeles. Los descriptores llevan el sufijo `-metric`.

## Desarrollo (66 piezas conocidas, 49 con agarre viable)

Con estas vistas, el parecido visual ya depende del tamaño: la correlación entre el coseno y la diferencia
de volumen pasa de −0,09 (DINOv3 normalizado) a −0,22 (DINOv3), −0,25 (DINOv2) y −0,41 (ViT sin
entrenar). Pero frente a las vecinas por tamaño de caja no hay mejora clara
([evaluación](development/evaluation.json), [análisis](development/analysis.json)):

| Contraste (hasta el 1.er viable) | Cociente | IC 95 % |
|---|---:|---|
| DINOv3 con escala ÷ tamaño | 0,91 | 0,68–1,22 |
| DINOv3 con escala ÷ DINOv3 normalizado | 1,10 | 0,89–1,42 |
| DINOv3 con escala + tamaño ÷ tamaño | 0,94 | 0,70–1,26 |
| Ídem hasta el 3.er viable | 0,83 | 0,67–1,04 |

La previsión honesta, por el desarrollo, es que en la prueba tampoco se demuestre una mejora.

## Piezas de prueba

- Misma fuente, revisión y regla que la vez anterior ([new_parts.py](../../../../tools/research/phase4/new_parts.py)),
  con la semilla **20261127**. Se excluyen los 30 objetos de la Fase III y las 50 piezas de la primera prueba.
- Resultado: 50 aceptadas de 214 examinadas, ninguna rechazada por la CLI y ninguna compartida con conjuntos
  anteriores ([test_parts/new_parts.json](test_parts/new_parts.json)). Sus vistas y descriptores se calcularon
  antes de este commit; eso no mira ningún resultado de agarre.

## Prueba

- Biblioteca: las 66 de desarrollo y las 50 de prueba, cada una resuelta entera (116 piezas).
- Se evalúan las piezas de prueba con algún agarre viable; cada una con el prior de las otras 115.
- Métrica, parámetros, bootstrap, Wilcoxon y Holm idénticos a la prueba anterior
  ([grasp_prior_stats.py](../../../../tools/research/phase4/grasp_prior_stats.py)).
- **Principales** (hasta el 1.er viable, Holm entre los dos):
  1. `visual:dinov3-metric` frente a `box`: ¿un descriptor visual que ve el tamaño supera al tamaño solo?
  2. `visual:dinov3-metric` frente a `visual:dinov3`: ¿conservar la escala mejora el prior visual?
- Se declara mejora solo si el intervalo queda entero por debajo de 1 y el p de Holm es menor que 0,05.
- **Secundarios** (descriptivos, hasta el 1.º y el 3.º): la lista exacta está en [protocol.json](protocol.json).
  Incluyen la fusión con tamaño, DINOv2, el ViT sin entrenar como control y todo frente al orden por defecto.

## Órdenes

```bash
T="python tools/research/phase4/visual_prior.py"
$T views --part ... --size 224 --distance 0.6 --out metric-views.zip   # y sin --distance, las normalizadas
$T encode --views metric-views.zip --model dinov3-timm --label dinov3-metric ...   # ídem dinov2, ViT aleatorio
$T library --binary target/release/axisforge --part ... (las 50 de prueba) --descriptors ... --out test-library.json
# biblioteca de prueba = desarrollo (new_parts/library.json.gz + descriptores -metric) + las 50 de prueba
$T evaluate --library library.json --binary target/release/axisforge --query ... (las 50) \
  --model dinov3-metric --model dinov2-metric --model vit-random-metric --model dinov3 --model dinov2 \
  --model vit-random --methods visual fusion --random-seeds 10 --jobs 4 --out evaluation.json
python tools/research/phase4/grasp_prior_stats.py --evaluation evaluation.json \
  --primary visual:dinov3-metric box --primary visual:dinov3-metric visual:dinov3 --secondary ... --out analysis.json
```

La biblioteca de desarrollo con los descriptores `-metric` tiene SHA-256
`1358cb0607dd0a7275f2d007a046ff35d07fa37f5f2d13c6768daedffa5367d2`. Se reconstruye con `add-descriptors`
a partir de [../new_parts/library.json.gz](../new_parts/library.json.gz) y los ficheros de
[development/](development/).
