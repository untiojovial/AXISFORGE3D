# Respuesta a la revisión de la Fase IV

Respuesta punto por punto a la revisión («revisión mayor»). Cada cambio está en la Fase V, con el protocolo
fijado antes de ver los datos de prueba ([protocolo](PROTOCOL.md), commit `ad9e1d2`;
[resultados](README.md)).

## Corrección previa

La revisión marca como pendiente la RQ4 (renders que conservan la escala). Ya estaba probada antes de la
revisión, con protocolo previo y 50 piezas nuevas ([informe](../phase4/grasp_prior/scale/README.md)):
DINOv3 con escala ÷ tamaño dio 1,24 [0,85–1,85], no demostrado. En la Fase V, con 151 piezas, DINOv3 con
escala ÷ tamaño da 1,07 [0,88–1,28]. Frente a DINOv3 normalizado da 0,87 [0,77–0,99], pero sin corregir;
con Holm no se sostiene.

## Objeciones mayores

**1. No demostrar diferencia no es demostrar equivalencia.** De acuerdo. Ahora hay pruebas de equivalencia
con margen fijado: equivalentes si el IC del 95 % queda dentro de [0,8; 1,25], es decir, dos pruebas
unilaterales al 2,5 %, más conservadoras que el TOST habitual al 5 %. El margen se justifica con un cálculo
de tamaño muestral hecho con la variabilidad de la Fase IV:

| Margen | Piezas viables necesarias |
|---|---:|
| ±10 % | ~950 |
| ±20 % | ~260 |
| ±25 % | ~170 |

GSO no da más que las 200 usadas; con la regla de tamaño se agotó prácticamente el catálogo. Resultado:
AFAT con y sin DINOv3 es **equivalente**, 1,01 [0,96–1,07]; el intervalo cabe de hecho en ±7 %. El texto
ya no dice que el preentrenamiento sea inútil. Dice lo que se mide:

- buscar vecinas con DINOv3 es mucho mejor que con el mismo ViT sin entrenar (0,32 [0,26–0,40]);
- pero dentro del mejor sistema DINOv3 no aporta nada mayor que el margen.

**2. Las piezas sin agarre viable.** Se incluyen:

- Curvas de éxito frente a presupuesto (1, 5, 10, 20, 50 y 100 candidatos) sobre las 200 piezas; las que no
  tienen agarre cuentan como fallo.
- Las causas de inviabilidad por pinza:
  - 40 mm: carrera en las 142;
  - 80 mm: carrera en 41 y alcance en 8;
  - 120 mm: alcance en 27 y carrera en 16.
- El coste de buscar en vano: entre 0,8 y 1,6 s, igual para todos los órdenes.
- Se dice explícitamente que no tener agarre viable entre los candidatos generados no prueba que la pieza no
  se pueda agarrar.

**3. Un conjunto de prueba independiente de la biblioteca.** Hechos los tres experimentos pedidos:

- **B**: biblioteca congelada de 116 piezas y 200 piezas nuevas que nunca entran en ella.
- **A**: biblioteca solo YCB y prueba con GSO.
- **C**: la biblioteca excluye la categoría oficial de la pieza (Gazebo Fuel). Cubre 5 categorías con 133
  piezas viables a 80 mm.

Además:

- **Auditoría de casi duplicados**: 52 variantes de producto por nombre y 74 piezas parecidas por forma y
  apariencia.
- **Sensibilidad**: los cuatro contrastes principales dan la misma decisión sin las variantes y sin todas
  las marcadas.
- **Hallazgo nuevo**: en A, las vecinas DINOv3 sacadas solo de YCB empeoran el orden por defecto (1,44
  [1,09–1,91]), una transferencia negativa entre datasets. El clasificador aprendido aguanta (0,28).

**4. El orden por defecto como baseline débil.** Se añaden:

- orden aleatorio real dentro del filtro de carrera (20 semillas);
- cuatro heurísticas geométricas (centro, apertura, antipodalidad, aproximación vertical);
- el descriptor 3D métrico D2 (Osada et al., 2002), en lugar de FPFH;
- un **clasificador aprendido de candidatos** entrenado con los catálogos de la biblioteca.

El orden aleatorio resulta 3,3 veces peor que el defecto, así que el defecto no es arbitrario. Es la
heurística de calidad del planificador. Pero el clasificador aprendido lo reduce a 0,21. Contact-GraspNet
no se pudo ejecutar aquí (TensorFlow con operadores CUDA propios, sin GPU); el clasificador aprendido hace
de comparador aprendido con la misma información.

**5. La novedad.** De acuerdo con la reformulación propuesta. La contribución es medir, con controles,
cuándo un modelo fundacional justifica su uso frente a priors geométricos y aprendidos bajo restricciones
cinemáticas. Para eso se construyó **AFAT**: el clasificador más las puntuaciones de cuatro fuentes de
transferencia y el perfil de similitud de la pieza, que hace de peso w(x). Se comparó con su ablación sin
imagen. Resultado:

- AFAT no mejora al clasificador: 1,06 [0,99–1,14].
- DINOv3 no aporta dentro de AFAT.
- Las fuentes de transferencia, que aportaban por sí solas, quedan absorbidas por el aprendizaje sobre
  catálogos.

Es un resultado que acota el valor de los modelos fundacionales en este problema, no un sistema nuevo que
gane a todo. La optimización bayesiana de la Fase IV no se reivindica: los resultados son de priorización.

**6. Viable no es agarrar.** Se distinguen cuatro niveles:

1. Viable en el planificador (alcance, colisiones, cierre de fuerzas).
2. Estático (calidad en el espacio de llaves).
3. Dinámico en MuJoCo: cerrar, levantar, sostener y sacudir, con 6 condiciones de masa, fricción y pose;
   3 613 simulaciones.
4. Robot real: no disponible.

En el nivel 3:

- El primer agarre de cada método sostiene entre el 89 y el 95 % de las veces.
- Aguanta las 6 condiciones solo entre el 49 y el 64 %.
- **Exploratorio**: los priors eligen agarres menos robustos que el orden por defecto (McNemar p = 0,0002
  para el aprendido), sobre todo con masa doble. Encontrar antes tiene un coste de robustez. Queda como
  limitación y como línea siguiente: combinar viabilidad y calidad.

## Otros puntos

| Petición | Respuesta |
|---|---|
| ≥100 piezas nuevas con análisis de potencia | 200 piezas (151 viables a 80 mm); cálculo de tamaño en el protocolo |
| Corregir multiplicidad o declarar exploratorio | Holm entre los tres principales de superioridad y en toda la familia de 464 secundarios; la dinámica comparativa se declara exploratoria |
| Presupuestos de 1, 5, 10, 20 y 50 | Hechos, más 100, con IC bootstrap ([análisis](results/analysis.json)) |
| Normalizado frente a escala | Hecho (RQ4) |
| ViT aleatorio con varias inicializaciones | 5 semillas, combinadas por pieza |
| Transferencia adaptativa con vuelta atrás | AFAT; tasa de transferencia negativa: 7 % AFAT y 6 % el aprendido, frente a 24 % DINOv3 y 15 % tamaño |
| Pinzas de 40, 80 y 120 mm | Hechas; las conclusiones se mantienen (tabla en [resultados](README.md)) |
| Tiempo, memoria y coste de la biblioteca | [Costes](results/costs.json): ~4 s por pieza para resolverla entera a 80 mm, 120 kB por catálogo |
| Muestra efectiva frente a 1 846 planificaciones | La unidad es la pieza; con biblioteca congelada, las piezas de prueba son independientes dada la biblioteca |
| Comparar con métodos publicados | No hecho (ver objeción 4) |
| Otro dataset independiente | Parcial: YCB→GSO (A). EGAD devolvió HTTP 403 y ShapeNet/Objaverse piden cuenta o tienen licencias mixtas |
| Categorías desconocidas | Hecho (C) |
| Robot real | No disponible |
| Publicar código, protocolo y datos | En el repositorio: código, protocolos, selecciones con hashes, filas, análisis, descriptores; las mallas y pesos se regeneran con scripts fijados |

## Lo que sigue abierto

- **Robot real**: el modelo dinámico de almohadillas aisladas no lo sustituye.
- **Comparación con métodos publicados**, al menos uno, en una máquina con GPU.
- **Margen de equivalencia más fino**: necesita un dataset mayor que GSO.
- **Robustez**: incorporar la calidad o la robustez dinámica al clasificador y medir de nuevo el compromiso.
- **Manos de varios dedos** (el planificador ya las tiene), donde la forma local podría pesar más que el
  tamaño.
