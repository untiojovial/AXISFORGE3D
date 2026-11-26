# Fase VI — FORTIS-R: viabilidad aprendida y robustez física

**Pregunta.** ¿Pueden los priors de viabilidad aprendidos acelerar la planificación de agarres sin perder
robustez física cuando las propiedades del objeto son inciertas y el presupuesto de cálculo es limitado?

**Respuesta corta** (desarrollo, no confirmación):

1. **Con la incertidumbre que pretendía el protocolo** (masa ×0,5–×2, μ 0,3, ruido de pose), el prior
   aprendido **no** pierde robustez: 90,7 % frente al 84,1 % del orden por defecto.
2. **La pérdida que publicó la Fase V era con 2 kg**, no con masa ×2. Era un error de la Fase V
   ([fe de erratas](ERRATUM_PHASE5_DYNAMICS.md)).
3. **Con cargas grandes la pérdida es real** (48 % frente a 65 % con 2 kg). Se explica por el brazo de
   palanca: el prior anula el término de centrado del planificador.
4. **Una fusión simple la recupera casi entera** sin perder velocidad. M4 = p̃^0,8·Q_cen^0,2, elegida
   igual en los 5 pliegues, da una tolerancia a la carga de 60 % frente a 48 %.
5. **Un P(H|F) aprendido** lleva la robustez estándar a 96 %.
6. **En segundos** el ahorro es menor que en candidatos: −24 % por llamada, frente a −80 % de candidatos.

Protocolo fijado antes de los resultados de fusión: [PHASE6_PROTOCOL.md](PHASE6_PROTOCOL.md) (commit
`f1f0891`). Auditoría: [PHASE6_REPOSITORY_AUDIT.md](PHASE6_REPOSITORY_AUDIT.md).

## Datos

- **Desarrollo**: 151 piezas viables a 80 mm, que fueron la prueba de la Fase V. Ya no son un conjunto
  confirmatorio.
- **Biblioteca congelada**: 116 piezas, solo para entrenar.
- **Los 8 objetos sellados de la Fase III** no se han abierto.
- **Física v2** ([physics.py](../../research_grasping/phase6/physics.py)): centro de masas del
  planificador y masas correctas. Los controles pasan ([physics_validation.json](results/physics_validation.json)).
- **Barrido dinámico** ([sweep.json.gz](results/sweep.json.gz)), 80 246 simulaciones:
  - 12 585 viables con las 6 condiciones estándar;
  - los 1 008 primeros agarres de todos los órdenes, además, con 0,5, 1 y 2 kg.

## Etapa B — por qué se perdía robustez

[robustness_mechanism.json](results/robustness_mechanism.json),
[ranking_quality_analysis.json](results/ranking_quality_analysis.json),
[feature_ablation.json](results/feature_ablation.json).

| Hallazgo | Evidencia (151 piezas; IC bootstrap por pieza) |
|---|---|
| En la Fase V, la pérdida venía de la condición de 2 kg | Sostienen con 2 kg: aprendido 74, defecto 98. En nominal: 143 y 134 |
| La «robustez 6/6» de la Fase V era «sostiene 2 kg» | Aprendido: 0,490 y 0,490. Defecto: 0,642 y 0,649 |
| El brazo de palanca predice sostener 2 kg | AUC con validación cruzada por pieza de 0,86 (solo el brazo de palanca) y 0,95 (los 19 rasgos). Mediana de 33 mm en los que sostienen y 67 mm en los que caen |
| El aprendido elige brazos más largos sin buscarlos | +7,4 mm [4,7; 10,3]; término de centrado −0,098 [−0,132; −0,066]. Su probabilidad no se relaciona con el brazo de palanca entre los viables (Spearman mediana 0,04) |
| Cuánto explica el brazo de palanca | Predice −9,3 puntos [−12,5; −6,5] de los −15,2 observados [−23,2; −8,0] |
| Por qué la viabilidad empuja hacia arriba | En la biblioteca, la tasa de viables pasa de 0,05 % en el cuarto inferior de la pieza a 17 % en el superior |
| **Con las masas correctas (v2)** | Robusto en las 6 condiciones estándar: aprendido **0,907**, defecto 0,841 (+6,6 [0,0; 13,2], 18 frente a 8, p = 0,076). Con 2 kg: 0,483 frente a 0,649 (−16,6 [−23,8; −9,3], p = 2·10⁻⁵). Con 1 kg: 0,73 frente a 0,76. Con 0,5 kg: 0,87 frente a 0,88 |
| Ablación del clasificador | Ningún grupo de rasgos es imprescindible (AUC de 0,988 a 0,993). Sin antipodalidad, N_F empeora un 13 % [4; 24 %] |

## Etapa C — fusión simple

[fusion_benchmark.json](results/fusion_benchmark.json). Validación cruzada anidada por pieza; M3–M8, en el
pliegue externo.

| Método | N_F (media geom.) | Robusto (6 estándar) | N_FR | Tolerante a carga (0,5/1/2 kg) |
|---|---:|---:|---:|---:|
| M0 defecto | 6,07 | 0,841 | 6,33 | **0,649** |
| Aleatorio dentro del filtro (ref.) | 7,15 (1,18× M0 [0,91; 1,52]) | — | — | — |
| M1 aprendido (Fase V) | 1,25 | 0,907 | 1,33 | 0,483 |
| M2 AFAT (Fase V) | 1,33 | 0,907 | 1,40 | 0,497 |
| M3 suma | 1,28 | 0,921 | 1,36 | 0,477 |
| **M4 producto** (α 0,8, Q_cen; 5/5 pliegues) | **1,23** | 0,927 | **1,28** | **0,603** |
| M5 lexicográfico | 1,37 | 0,821 | 1,61 | 0,543 |
| M6 filtro de calidad | 1,25 | 0,907 | 1,34 | 0,431 |
| M7 Borda | 1,30 | 0,927 | 1,31 | 0,543 |
| M8 Pareto | 1,22 | 0,921 | 1,30 | 0,450 |
| M9 P(F)·P(H\|F) (etapa D) | 1,29 | **0,960** | 1,28 | 0,464 |

**Regla fijada:**

- La cumplen M3, M4, M6, M7 y M8; M5 no.
- No significa que la fusión «recupere» nada en las condiciones estándar, porque M1 ya era no inferior a
  M0.

**Secundaria prevista en el protocolo (cargas):** M4 frente a M1 en tolerancia a la carga:

- +11,9 puntos [6,6; 17,9], 20 frente a 2, Holm p = 0,001;
- N_F 0,98× [0,91; 1,04];
- frente al defecto, −4,6 puntos [−10,6; 1,3], sin diferencia demostrada.

**Etapa D:**

- **P(H|F)**: el centrado separa poco los robustos (AUC 0,57 < 0,70), así que la puerta se abrió. Se
  aprendió P(H|F) con la misma validación cruzada (AUC fuera de pliegue 0,956;
  [robustness_prediction.json](results/robustness_prediction.json)). Frente a M1: +5,3 puntos de
  robustez estándar [2,0; 9,3], 8 frente a 0, p = 0,008, con N_F 1,03× [0,98; 1,08]. Con cargas no mejora.
- **Calibración** ([calibration_results.json](results/calibration_results.json)): ECE 0,010 (0,005 con
  Platt).
- **Inviabilidad temprana** ([infeasibility_benchmark.json](results/infeasibility_benchmark.json)):
  ninguna pieza se demuestra inviable solo con el filtro; 13 de 49 se estiman sin perder ninguna viable.

**Transferencia negativa** ([negative_transfer_analysis.json](results/negative_transfer_analysis.json)).
Proporción de piezas en las que el método es más lento que el defecto, pierde robustez o pierde
tolerancia a la carga:

| Método | Más lento | Pierde robustez | Pierde tolerancia a la carga |
|---|---:|---:|---:|
| M1 | 6 % | 5 % | 20 % |
| M4 | 5 % | 5 % | 9 % |
| M9 | 7 % | 2 % | 24 % |

## Etapa G (parcial) — coste real y potencia

**Coste real** ([cost_profile_online.json](results/cost_profile_online.json)): 60 piezas, un hilo,
`axisforge plan --nr-grasps 1`, sin discrepancias con el simulador.

- Mediana por llamada: 0,56 s (M0), 0,46 s (M1) y 0,44 s (M4). Es un ahorro del 24 %.
- El coste fijo (arranque, carga y generación) ronda 0,57 s; cada candidato examinado añade ~4 ms.
- M4 se pasa al planificador con un prior que codifica el orden con peso 5, el máximo que admite el
  planificador. El protocolo decía «peso 10⁶», que no se puede pasar al planificador; el orden es
  idéntico.

**Potencia** ([statistical_power.json](results/statistical_power.json)): para confirmar con objetos nuevos
al 80 % hacen falta:

- ~105 piezas para la tolerancia a la carga de M4 frente a M1;
- ~71 para la no inferioridad de la robustez a ±5 puntos;
- ~38 para la no inferioridad de N_F (margen 1,25).

## Tabla para el verificador

| Área | Resultado | Evidencia | Commit | Limitación |
|---|---|---|---|---|
| Auditoría de la Fase V | Cifras verificadas: 359 Rust, clippy 0, 463 → 472 Python, 7 588 coincidencias del simulador, 3 613 simulaciones | [auditoría](PHASE6_REPOSITORY_AUDIT.md) | `f1f0891` | El orden de git no prueba cuándo se vieron los datos |
| Fe de erratas de la Fase V | Masas 2/0,5 kg en lugar de ×2/×0,5; centro de masas desplazado; orden aleatorio fuera del filtro | [fe de erratas](ERRATUM_PHASE5_DYNAMICS.md) | `f1f0891` | Ficheros de la Fase V sin modificar |
| Rasgos y fuga | 19 rasgos de la generación; ninguno de cinemática inversa, colisiones ni veredicto | [feature_audit.json](results/feature_audit.json) | `f1f0891` | Altura específica de la celda |
| Física | Controles con respuesta conocida superados; sensible a la fricción torsional | [physics_validation.json](results/physics_validation.json) | `f1f0891` | Sin calibrar con hardware |
| Mecanismo | La pérdida era con 2 kg, por el brazo de palanca, porque el prior anula el centrado | [robustness_mechanism.json](results/robustness_mechanism.json) | este commit | Explicativo |
| Masas correctas | El aprendido no pierde robustez estándar (+6,6 puntos); sí con 2 kg (−16,6) | ídem, B5 | este commit | Modelo de almohadillas |
| Fusión simple | M4 recupera tolerancia a la carga (+11,9 frente a M1) a igual velocidad | [fusion_benchmark.json](results/fusion_benchmark.json) | este commit | Desarrollo; secundaria |
| P(H\|F) aprendido | +5,3 puntos de robustez estándar frente a M1 a igual velocidad | [robustness_prediction.json](results/robustness_prediction.json) | este commit | Desarrollo; no mejora con cargas |
| Calibración | ECE 0,010 → 0,005 (Platt) | [calibration_results.json](results/calibration_results.json) | `74d81a4` | Una celda |
| Inviabilidad temprana | 0 demostrables; 13/49 estimadas sin perder viables | [infeasibility_benchmark.json](results/infeasibility_benchmark.json) | `74d81a4` | «Inviable» = sin viable entre los generados |
| Coste online | −24 % de tiempo con −80 % de candidatos | [cost_profile_online.json](results/cost_profile_online.json) | este commit | 60 piezas; sin tiempo por etapa |
| Potencia | ~105 / 71 / 38 piezas nuevas según la hipótesis | [statistical_power.json](results/statistical_power.json) | este commit | Aproximación normal |
| Literatura y novedad | El clasificador de viabilidad no es nuevo; el compromiso entre velocidad y robustez, posible aportación empírica | [revisión](phase6_literature_review.md), [matriz](phase6_novelty_matrix.md) | `74d81a4` | Búsqueda dirigida, no sistemática |
| TFM | Referencias con errores y una posible fuga en el vector de contexto; no se reutiliza como evidencia | [revisión científica](scientific_review.md) | `74d81a4` | Sin el código del TFM |

## Listas

**Hecho:**

- auditoría y fe de erratas;
- física v2 con controles;
- barrido dinámico completo;
- mecanismo B1–B5;
- ablación;
- fusiones M0–M8 con validación cruzada anidada;
- P(H|F) aprendido;
- calibración;
- inviabilidad temprana;
- transferencia negativa;
- coste online (60 piezas);
- potencia;
- revisión bibliográfica verificada;
- matriz de novedad;
- registro de datasets.

**No hecho, con la razón:**

- ADKF-IFT, DKL, MLP y rasgos locales de DINO: la puerta F no se abrió.
- Simulador articulado: la puerta H no se abrió.
- Orden por coste y multifidelidad: el planificador no da el tiempo por etapa.
- Fuentes complementarias, incertidumbre y fuera de distribución: no justificados con los datos de hoy.

**Bloqueado:**

- baselines externos (sin GPU);
- tercer dataset y test confirmatorio (sin objetos nuevos descargados ni protocolo nuevo congelado);
- robot real.

**Siguiente paso recomendado:**

1. Congelar el protocolo confirmatorio de [phase6_go_no_go.md](phase6_go_no_go.md) (H1–H3).
2. Reunir de 140 a 180 objetos de EGAD/Objaverse con licencia compatible.
3. Calibrar la fricción torsional.

## Líneas de manuscrito

- **A (evidencia actual)**: «Learned feasibility priors in kinematically constrained grasp planning:
  where speed costs robustness». Fase V más esta fase, con la fe de erratas, como estudio controlado
  sobre GSO. La contribución es la medición y el mecanismo, no un método nuevo.
- **B (metodología FORTIS-R)**: N_FR como medida conjunta, fusiones simples frente a P(H|F) y el diseño
  confirmatorio. Necesita el test nuevo.

Ver también: [limitaciones](limitations.md), [reproducibilidad](reproducibility.md),
[preguntas](PHASE6_RESEARCH_QUESTIONS.md) y [revisión científica](scientific_review.md).
