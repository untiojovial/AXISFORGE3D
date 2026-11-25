# Protocolo: prior visual + tamaño en 50 piezas nuevas

Fijado **antes** de resolver ninguna pieza nueva, y subido en el mismo commit que la selección. Las piezas
YCB son el conjunto de desarrollo: sirvieron para elegir la combinación y no cuentan en la prueba.

## Piezas

- Fuente: Google Scanned Objects, adaptación MuJoCo de Kevin Zakka, revisión
  `6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c` (la misma de la Fase III), 1030 modelos.
- Se excluyen los 30 objetos del manifiesto de la Fase III (train, validation y los 8 de test sellados).
  Nada de la prueba se ha visto antes, y el test sellado sigue sin tocar.
- Se barajan los demás con la semilla 20261126 y se recorren en orden. Se queda una pieza si su caja
  cabe en la pinza y el brazo la alcanza: lado menor de 5 a 70 mm, lado mayor de 250 mm como mucho y
  altura de 200 mm como mucho. La selección acaba al llegar a 50. Solo se miran nombres y cajas, nunca
  resultados de agarre.
- Cada pieza va sola en la mesa de la demo: `add-object --units m --mass 0.1 --at 0.42 0 0.42`.
- Resultado: 50 aceptadas de 133 examinadas, ninguna rechazada por la CLI ([new_parts.json](new_parts.json),
  con el hash de cada malla y de cada proyecto). Herramienta:
  [new_parts.py](../../../../tools/research/phase4/new_parts.py).

## Prueba

- Biblioteca: las 16 piezas YCB y las 50 nuevas, cada una resuelta entera.
- Se evalúan las piezas nuevas con algún agarre viable. Para cada una, su prior sale de las otras 65
  piezas (se deja fuera una cada vez). Las que no tienen ningún agarre viable se cuentan y se excluyen
  de los cocientes, igual que en desarrollo.
- Métrica: candidatos examinados por el planificador nativo hasta el primer agarre viable (principal) y
  hasta el tercero (secundaria, en piezas con tres o más).
- Parámetros sin cambios: 3 vecinas, peso 2, y las escalas de la firma de agarre. Orden de vistas,
  render y modelos como en desarrollo.

## Combinación elegida en desarrollo

Se probaron dos combinaciones en las 12 piezas YCB con agarre viable
([development_combined.json](../development_combined.json)):

- **fusion**: las 3 mejores por la suma del puesto visual y el puesto por tamaño de caja.
- **gate**: las 6 más cercanas por tamaño y, de ellas, las 3 más parecidas.

Hasta el primer agarre viable, ninguna mejoró al tamaño solo (cocientes frente al orden por defecto
entre 0,58 y 0,70, frente a 0,33 del tamaño). Hasta el tercero, quedaron al nivel de la visual sola
(0,48 a 0,51). Se fija **fusion** porque no tiene parámetros y fue algo mejor hasta el primer agarre.
Se fija **DINOv3** como modelo principal porque fue el mejor con fusion (0,575). La biblioteca de
desarrollo es [library.json](../library.json) más los descriptores [dinov3.npz](../dinov3.npz) (SHA-256
`08e483742e1dde35fa64e83778d8f7f2f8b03b9c192d67d120c735b404c587b0`; se reconstruye con `library` y
los cuatro ficheros de descriptores). La previsión honesta,
por el desarrollo, es que la combinación no supere al tamaño solo.

## Contrastes y decisión

Análisis con [grasp_prior_stats.py](../../../../tools/research/phase4/grasp_prior_stats.py), sin cambios
después de este commit. La lista exacta está en [protocol.json](protocol.json):

- **Principales** (hasta el primer viable, Holm entre los dos):
  1. `fusion:dinov3` frente a `box`: ¿añadir la apariencia al tamaño ayuda?
  2. `fusion:dinov3` frente a `visual:dinov3`: ¿añadir el tamaño a la apariencia ayuda?
- Cada contraste da la media geométrica del cociente de candidatos examinados, un intervalo bootstrap
  del 95 % por piezas (10 000 remuestreos, semilla 20261126), las piezas mejor/igual/peor y una prueba
  de rangos con signo de Wilcoxon sobre el logaritmo del cociente (sin las diferencias cero).
- Se declara que la combinación ayuda solo si el intervalo queda entero por debajo de 1 y el p de Holm
  es menor que 0,05. Si no, se informa como no demostrado, con el intervalo.
- **Secundarios** (descriptivos, hasta el primero y el tercero): todas las ordenaciones frente al orden
  por defecto; `box` y `pooled` frente al defecto; `fusion:dinov2`, `fusion:openclip` y `gate:dinov3`
  frente a `box`; `visual:dinov2`, `visual:openclip` y `visual:dinov3` frente a `box`;
  `visual:dinov3` frente a `visual:vit-random` y `fusion:dinov3` frente a `fusion:vit-random` (control de
  preentrenamiento); DINOv3 frente a DINOv2 y OpenCLIP en visual.

## Órdenes

```bash
T="python tools/research/phase4/visual_prior.py"
$T views --part NOMBRE NOMBRE/cell.axisforge ... --out new-views.zip       # las 50 nuevas
$T encode --views new-views.zip --model dinov2|openclip|dinov3-timm ...     # mismos pesos y hashes que en desarrollo
$T library --binary target/release/axisforge --part ... (16 YCB + 50 nuevas) --descriptors ... --out library.json
$T evaluate --library library.json --binary target/release/axisforge --query NOMBRE ... (las 50) \
  --model dinov2 --model openclip --model dinov3 --model vit-random --methods visual fusion gate \
  --random-seeds 10 --jobs 4 --out evaluation.json
python tools/research/phase4/grasp_prior_stats.py --evaluation evaluation.json \
  --primary fusion:dinov3 box --primary fusion:dinov3 visual:dinov3 --secondary ... --out analysis.json
```

DINOv3 es la copia que redistribuye timm (`timm/vit_small_patch16_dinov3.lvd1689m`, revisión
`3bf4720a82ec2066db88137180ff1f83a675cef0`, pesos `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`)
bajo la licencia DINOv3, que el usuario aceptó. Los periodos RoPE se truncan a bfloat16 para igualar al
original, según la ficha del modelo.
