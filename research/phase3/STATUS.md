# Estado de los entregables de fase III

Los estados distinguen la evidencia alcanzada en el alcance señalado. La partición final de ocho objetos sigue retenida. El protocolo científico y sus puertas se detallan en [PROTOCOL.md](PROTOCOL.md).

| Nº | Entregable | Estado | Evidencia y límite |
|---:|---|---|---|
| 1 | Auditoría del software de agarre | PARTIALLY_VALIDATED | Revisión de generación, wrench, IK, rechazo y pose por ID. Prueba nativa adicional de tres dedos y mano con 18 candidatos. La dinámica solo se ensayó con dos dedos. |
| 2 | Arquitectura integrada | VALIDATED | Ruta reproducida desde archivo de malla y rasgos visuales a catálogo nativo, selección BO, evaluación, registro y visor. |
| 3 | 20–30 objetos públicos | VALIDATED | 30 mallas GSO distintas por hash e identidad, con 16/6/8 por partición. |
| 4 | Licencias y origen | PARTIALLY_VALIDATED | Revisión y autor fijados; la adaptación identifica OBJ/PNG como CC BY 4.0. Faltan comprobaciones independientes de escala y posibles duplicados semánticos. |
| 5 | Importación y conversión | PARTIALLY_VALIDATED | 30 OBJ convertidos a STL con hashes comprobados; el importador acepta OBJ, STL, PLY y GLB con unidades explícitas. Se documentan límites de unidades y topología. |
| 6 | Geometría visual y de colisión | PARTIALLY_VALIDATED | STL visuales y proxies oficiales en sondeo físico. Solo comprobación local de discrepancia en las poses seleccionadas. |
| 7 | Render multivista | VALIDATED | 16 cámaras calibradas por objeto de desarrollo; pruebas de reproyección, orden de profundidad y transformación. |
| 8 | RGB-D | VALIDATED | 352 vistas de desarrollo con profundidad, normales, máscara y calibración. Variantes gris y textura. |
| 9 | Nubes de puntos | VALIDATED | 342.005 muestras métricas de superficie en 22 objetos. Exportación por vista y colores. |
| 10 | Embeddings preentrenados reales | VALIDATED | OpenCLIP ViT-B/32 con pesos verificados de sus autores, global y tokens de patch. |
| 11 | Comparativa entre modelos fundacionales | BLOCKED | Servidores oficiales de DINOv2 devolvieron HTTP 403. DINOv3 requiere acceso autorizado. Falta una segunda familia de pesos verificada para la comparativa. |
| 12 | Similitud visual | VALIDATED | Matrices de 1/4/8/16 vistas, comparación de apariencia y 60 consultas de validación. |
| 13 | Similitud geométrica | PARTIALLY_VALIDATED | Distancia de dimensiones, descriptores locales y evaluación predictiva. Falta estudio general de correspondencias entre familias. |
| 14 | Similitud funcional | PARTIALLY_VALIDATED | Predicción de ε y factibilidad desde experiencias históricas, con asociación de tokens visibles. Faltan etiquetas dinámicas independientes suficientes. |
| 15 | Integración con el motor | VALIDATED | Pose SE(3), apertura, contactos propuestos y F1/F2 bajo demanda mediante CLI nativo. |
| 16 | Dinámica MuJoCo | PARTIALLY_VALIDATED | Seis poses de tres objetos con objeto libre, dedos y palma del proyecto, mesa, controles y perturbaciones. Falta brazo dinámico completo y validación física. |
| 17 | Optimización de poses | PARTIALLY_VALIDATED | Selección secuencial entre poses 3D de un catálogo finito; falta refinamiento continuo de SE(3). |
| 18 | ACT-BO | PARTIALLY_VALIDATED | Selección causal, pesos por evidencia y controles. La evidencia actual no establece beneficio del gate. |
| 19 | DKL multimodal | PARTIALLY_VALIDATED | Entrenado en 160 mediciones y comparado en seis objetos. Calibración insuficiente para promoverlo a adquisición. |
| 20 | Baselines competitivos | PARTIALLY_VALIDATED | Nueve políticas en los paneles principales de 10 y 20 consultas, con controles visual, geométrico y funcional suplementarios. ICM de correlación fija y variante RGPE documentados. Faltan comparadores DINO y baselines completos. |
| 21 | Optimización multifidelidad | PENDING | Existen diagnósticos de pares F2/F3 y costes. Seis pares condicionados son insuficientes para justificar B15. |
| 22 | Benchmark de transferencia | PARTIALLY_VALIDATED | Historial y semillas compartidos, 16 fuentes y seis piezas de desarrollo. |
| 23 | Ablaciones | PARTIALLY_VALIDATED | Puerta de evidencia, pesos iguales y comparación de rasgos geométricos, globales y locales. |
| 24 | Objetos no vistos | PARTIALLY_VALIDATED | Las seis identidades de validación son distintas de las 16 históricas. Test final y generalización entre familias pendientes. |
| 25 | Estadística | PARTIALLY_VALIDATED | Resultados descriptivos por objeto y semilla, costes y límites del diseño. Falta potencia para contrastes confirmatorios. |
| 26 | Visualización | VALIDATED | Visor AxisForge, CSV, curvas y matriz de similitud. Se verificó en Chromium la lectura de un informe real y la exportación de sus diez filas. |
| 27 | Reproducibilidad | PARTIALLY_VALIDATED | Locks, comandos, pruebas, historial, reanudación y registros. El hardware, los UUID y el tiempo medido influyen en la repetición. |
| 28 | Auditoría independiente | PARTIALLY_VALIDATED | Revisiones internas de robótica, representación y metodología con objeciones abiertas. |
| 29 | Novedad | PARTIALLY_VALIDATED | Evaluación de precedentes primarios; contribución publicable sin demostrar. |
| 30 | Manuscrito | PENDING | La evidencia no justifica un manuscrito confirmatorio. |

**Milestones.** M0 cuenta con referencia de software reproducida. M1–M7 disponen de resultados con el alcance descrito. M8 carece de calibración multifidelidad suficiente. M9 incluye pilotos exploratorios. M10 continúa cerrado. M11 incluye revisiones internas y mantiene objeciones de validez. M12 queda pendiente. La rama no recibe una puntuación agregada ni una declaración de superioridad.
