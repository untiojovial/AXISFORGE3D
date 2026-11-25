# Fase V — protocolo

Fijado **antes** de resolver ninguna pieza de prueba. Responde a la revisión de la Fase IV («revisión
mayor»): cada objeción tiene aquí un elemento de diseño. Lo que no se puede hacer en este entorno se dice
al final. La lista exacta de contrastes está en [protocol.json](protocol.json).

## Qué cambia respecto a la Fase IV

| Objeción | Respuesta en el diseño |
|---|---|
| 1. No demostrar diferencia no es demostrar equivalencia | Pruebas de equivalencia con margen fijado: equivalentes si el IC bootstrap del 95 % del cociente queda dentro de [0,8; 1,25] (dos pruebas unilaterales al 2,5 %). Se justifica abajo |
| 2. Se excluían las piezas sin agarre viable | Curvas de éxito frente a presupuesto (1, 5, 10, 20, 50 y 100 candidatos) sobre **todas** las piezas de prueba (las que no tienen agarre cuentan como fallo); coste y causas de inviabilidad de cada pieza |
| 3. Sin conjunto de test independiente de la biblioteca | **B**: biblioteca congelada de 116 piezas, prueba con 200 piezas nuevas que nunca entran en la biblioteca. **A**: biblioteca solo YCB (entre datasets). **C**: la biblioteca excluye la categoría oficial de la pieza (Gazebo Fuel). Auditoría de casi duplicados con análisis de sensibilidad |
| 4. El orden por defecto es un baseline débil | Orden aleatorio real (20 semillas), cuatro heurísticas geométricas (centro, apertura, antipodalidad, aproximación vertical), descriptor 3D métrico D2 (Osada et al., 2002) y un **clasificador aprendido de candidatos** entrenado con los catálogos de la biblioteca |
| 5. La novedad no es transferir agarres | La pregunta es cuándo un modelo fundacional justifica su uso frente a priors geométricos y aprendidos bajo restricciones cinemáticas. **AFAT** combina las fuentes según la pieza, y su ablación sin imagen lo mide directamente |
| 6. Viable no es agarrar | Prueba dinámica en MuJoCo de los agarres encontrados: cerrar, levantar, sostener y sacudir, con 6 perturbaciones de masa, fricción y pose. Se distinguen 4 niveles: viable en el planificador, estable en estática, estable en dinámica y robot real (no disponible) |
| Muestra efectiva | La unidad es la pieza. Las piezas de prueba no están en la biblioteca, así que sus resultados son independientes dada la biblioteca congelada |
| Contrastes secundarios sin corregir | Holm sobre toda la familia de secundarios; además, se dan los p sin corregir |
| Varias inicializaciones del ViT aleatorio | 5 inicializaciones (semillas 0–4) con la arquitectura de DINOv3 ViT-S/16, combinadas por pieza |
| Varias pinzas | Aperturas de 40, 80 y 120 mm (misma mano y fuerza) |
| Costes | Tiempo de render, codificación, resolución exhaustiva y evaluación; tamaño de la biblioteca |

## Datos

- **Biblioteca congelada (116 piezas)**: 16 YCB, 50 GSO (semilla 20261126) y 50 GSO (semilla 20261127),
  todas de la Fase IV. Cada una está resuelta entera con las tres pinzas.
- **Prueba (200 piezas)**: GSO, revisión `6ff8d275…`, semilla **20261128**, misma regla de tamaño, excluyendo
  los 30 objetos de la Fase III y las 100 de la Fase IV. Hubo que examinar 858 modelos para reunirlas: es
  prácticamente todo lo que quedaba en el catálogo. Selección: [test_parts.json](test_parts.json).
- **Categorías**: las oficiales de Gazebo Fuel, en [categories.json](categories.json). En la prueba:
  Consumer Goods 88, sin categoría 49, Bottles and Cans and Cups 21, Toys 21, Media Cases 12, Action
  Figures 5 y otras 4 con una pieza. El experimento C usa las categorías con al menos 5 piezas de prueba.

## Cómo se mide

- **Simulador exacto del orden del planificador.** Para cada candidato, el planificador exporta su
  puntuación preliminar y su filtro de carrera. Con eso, el orden de examen bajo cualquier prior se calcula
  sin ejecutarlo. Reproduce **6 788 de 6 788** ejecuciones reales registradas en la Fase IV, con 0
  discrepancias ([validación](simulator_validation.json)).
- **Confirmación con el planificador real**: en la prueba B con 80 mm, los métodos default, box, learned y
  afat se ejecutan también con `axisforge plan --prior`, y deben coincidir con el simulador.
- **Métrica principal**: candidatos examinados hasta el primer agarre viable, en las piezas de prueba con
  alguno; experimento B, pinza de 80 mm. Secundarias: hasta el tercero, curvas de presupuesto, tasa de
  transferencia negativa (fracción de piezas en que un método examina más candidatos que el orden por
  defecto) y resultados dinámicos.

## Métodos

| Familia | Métodos |
|---|---|
| Sin conocimiento | orden por defecto; orden aleatorio dentro del filtro (20 semillas) |
| Heurísticas | `com`, `narrow`, `antipodal`, `topdown` |
| Aprendido | `learned`: gradient boosting (scikit-learn, semilla 0) sobre 19 rasgos por candidato, entrenado con todos los candidatos de la biblioteca |
| Vecinas por objeto | `box` (tamaño); `pooled` (todas); `random` (10 semillas); `visual:` con DINOv2, DINOv3, DINOv3 con escala, D2 y ViT aleatorio ×5; `fusion:` (puesto visual + puesto de tamaño) con los mismos descriptores |
| Adaptativo | `afat`: el mismo clasificador más las puntuaciones de cuatro fuentes (box, pooled, DINOv3, D2) y el perfil de similitud de la pieza; entrenado con las fuentes de cada pieza de la biblioteca calculadas sin ella |
| Ablación | `afat-novis`: AFAT sin DINOv3 (fuentes box, pooled y D2) |
| Techo | `self`: los agarres viables de la propia pieza |

Todos los parámetros son los de la Fase IV (3 vecinas, peso 2, escalas de firma). Ninguno se ajusta con las
piezas de prueba. El desarrollo con validación cruzada de 5 pliegues sobre la biblioteca está en
[development.json.gz](development.json.gz).

## Contrastes principales

Experimento B, 80 mm, hasta el primer viable. Holm entre los tres de superioridad.

| | Contraste | Tipo | Pregunta |
|---|---|---|---|
| P1 | `learned` ÷ `box` | superioridad | RQ1: ¿un clasificador aprendido de candidatos supera a transferir de piezas de tamaño parecido? |
| P2 | `afat` ÷ `learned` | superioridad | RQ5: ¿la combinación adaptativa mejora al clasificador? |
| P3 | `afat` ÷ `afat-novis` | equivalencia | RQ2/RQ3: dentro del mejor sistema, ¿aporta DINOv3 menos de un 25 %? |
| P4 | `visual:dinov3` ÷ `visual:vit-random` | superioridad | RQ3: ¿el preentrenamiento supera al mismo ViT sin entrenar? |

- **Superioridad**: p de Wilcoxon ajustado por Holm menor que 0,05 e IC del 95 % que excluye el 1.
- **Equivalencia**: IC del 95 % dentro de [0,8; 1,25].

**Por qué ±25 %.** Con la variabilidad de la Fase IV (desviación del logaritmo del cociente ≈ 1,0), demostrar
equivalencia al ±10 % exige unas 950 piezas con agarre viable, al ±20 % unas 260 y al ±25 % unas 170. El
catálogo GSO no da más de las 200 elegidas. Además, la diferencia entre transferir y el orden por defecto
ronda el 60 %, así que ±25 % es menos de la mitad del efecto que importa. Un margen más fino no es alcanzable
con este dataset, y se dirá así.

**Previsión honesta por el desarrollo**: P1 favorable a `learned` (0,48 frente a box). P2 y P3 con efectos
pequeños (0,167 frente a 0,179 y frente a 0,172). P4 favorable al preentrenamiento (DINOv3 0,39 frente al
ViT aleatorio 1,06, ambos respecto al orden por defecto).

## Secundarios

Todos los contrastes de [protocol.json](protocol.json), en los experimentos A, B y C, con las pinzas de 40,
80 y 120 mm y hasta el primer y el tercer viable, ajustados por Holm como una sola familia. Además:
presupuestos, transferencia negativa, causas de inviabilidad y sensibilidad sin los casi duplicados.

## Casi duplicados

Auditoría hecha antes de resolver nada ([near_duplicates.json](near_duplicates.json)). Cada pieza de prueba se
compara con todas las de la biblioteca:

- **Variante del mismo producto**: índice de Jaccard de las palabras del nombre ≥ 0,5 (colores, tallas,
  ediciones). Marca **52** piezas de prueba. La sensibilidad principal repite los contrastes principales
  sin ellas.
- **Parecido**: coseno D2 ≥ 0,99 y coseno DINOv3 ≥ 0,90. Marca otras **74**, porque GSO tiene muchas cajas y
  frascos semejantes. La segunda sensibilidad excluye las 126.

## Validación dinámica

Experimento B, 80 mm, métodos default, box, learned, afat y DINOv3 visual. Para el primer agarre viable de
cada uno, las 6 pruebas de [dynamics.py](../../research_grasping/phase5/dynamics.py):

- nominal (0,1 kg, μ 0,5);
- masa ×2 y ×0,5;
- μ 0,3;
- dos ruidos de pose (±3 mm, ±3°).

Resultados:

- **Sostenido**: sube al menos 80 mm y se desliza menos de 20 mm respecto a la palma.
- **Robusto**: las 6 pruebas sostienen.
- **Candidatos hasta el primer agarre que sostiene**: siguiendo el orden de cada método, hasta 20 agarres viables.

Es un modelo de almohadillas aisladas: no incluye brazo ni palma, que el planificador ya comprobó.

## Lo que no se puede hacer aquí

| Pedido | Motivo |
|---|---|
| Robot real | No hay hardware en este entorno |
| Contact-GraspNet u otro método publicado | Necesita TensorFlow con operadores CUDA propios y un checkpoint externo; aquí solo hay CPU. Se compara con un clasificador aprendido propio entrenado con los mismos datos |
| Un tercer dataset independiente | EGAD (QUT) devolvió HTTP 403; ShapeNet y Objaverse piden cuenta o tienen licencias mixtas. Se usan YCB y GSO, y A mide la transferencia entre ambos |
| Margen de equivalencia de ±10 % | Haría falta del orden de 950 piezas viables; GSO no las tiene |
