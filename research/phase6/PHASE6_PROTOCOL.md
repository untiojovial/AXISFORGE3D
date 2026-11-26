# Fase VI (FORTIS-R) — protocolo

**Pregunta central.** ¿Pueden los priors de viabilidad aprendidos acelerar la planificación de agarres
sin perder robustez física cuando las propiedades del objeto son inciertas y el presupuesto de cálculo
es limitado?

Este protocolo se fija **antes** de calcular el resultado de cualquier método de fusión. Lo que ya se
conoce al fijarlo está en [PHASE6_REPOSITORY_AUDIT.md](PHASE6_REPOSITORY_AUDIT.md):

- la fe de erratas de la dinámica de la Fase V;
- la auditoría de rasgos;
- los primeros datos exploratorios de la etapa B, que se dan como tales.

La parte que se puede leer con una máquina está en [protocol.json](protocol.json).

## 1. Datos y papel de cada conjunto

| Conjunto | Piezas | Papel en la Fase VI |
|---|---:|---|
| Biblioteca congelada | 116 (16 YCB, 100 GSO) | Entrena el clasificador de la Fase V (semilla 0), sin cambios |
| Desarrollo (piloto) | 151 viables a 80 mm, de las 200 de prueba de la Fase V | **Desarrollo**. Fueron la prueba confirmatoria de la Fase V; aquí se usan para explicar y ajustar, y nunca más como prueba confirmatoria |
| Sellados de la Fase III | 8 objetos (`split = test` en [manifest.lock.json](../phase3/dataset/manifest.lock.json)) | No se abren |
| Confirmatorio | Ninguno disponible | Solo se diseña (§7); GSO quedó agotado en la Fase V |

La lista exacta de desarrollo está en [dev_parts.json](dev_parts.json).

## 2. Modelo físico (dinámica v2)

Es [physics.py](../../research_grasping/phase6/physics.py): el ensayo de la Fase V (cerrar, levantar,
sostener y sacudir; solo almohadillas) con dos cambios.

1. **Centro de masas**: se pone donde lo pone el planificador. Se recupera del catálogo con un residuo
   menor de 1 µm. En la Fase V se usaba la media de los vértices de colisión, que difiere 6,4 mm en la
   pieza mediana y hasta 51 mm.
2. **Masas correctas**: en la Fase V, «masa ×2» y «masa ×0,5» eran en realidad 2 kg y 0,5 kg (fe de
   erratas en la auditoría).

**Condiciones estándar** (definen la robustez):

- nominal (0,1 kg, μ 0,5);
- masa ×2 (0,2 kg);
- masa ×0,5 (0,05 kg);
- μ 0,3;
- ruido de pose 1 y ruido de pose 2 (±3 mm, ±3°).

**Cargas de estrés**: 0,5 kg, 1 kg y 2 kg. Se informan aparte. 0,5 kg y 2 kg son las cargas que la
Fase V aplicó de hecho.

**Definiciones:**

- **Sostenido**: sube al menos 80 mm y se desliza menos de 20 mm (como en la Fase V).
- **Robusto**: sostenido en las 6 condiciones estándar.
- **Tolerante a la carga**: sostenido con 0,5, 1 y 2 kg.

Los controles físicos ([physics_validation.json](results/physics_validation.json)) se ejecutan antes que
cualquier método. Cada viable de cada pieza de desarrollo se simula en las 9 condiciones (`sweep.py`).
Así el agarre que elige cualquier orden se consulta en la tabla, sin simular otra vez.

## 3. Etapa B — explicar la pérdida de robustez (explicativa)

| Pregunta | Análisis |
|---|---|
| B1. ¿Qué condición y qué modo de fallo producen la pérdida? | Fallos por condición: caída (subida < 80 mm), deslizamiento (≥ 20 mm) y giro respecto a la palma; primeros agarres de la Fase V con v1 y con v2 |
| B2. ¿Qué rasgo del agarre la explica? | AUC por rasgo y regresión logística con validación cruzada por pieza: sostenido ~ rasgos del candidato. Hipótesis previa (de la exploración): el brazo de palanca, la distancia del centro de masas a la línea de cierre (`center_offset`) |
| B3. ¿El efecto del método pasa por ese rasgo? | Diferencia de robustez entre el aprendido y el defecto, en bruto y estratificada por terciles de `center_offset`; predicción de la robustez con un modelo que solo usa el rasgo |
| B4. ¿Por qué el aprendido elige brazos de palanca mayores? | En los catálogos de la biblioteca: tasa de viabilidad por bin de `center_offset` y de altura; importancia por permutación de los 19 rasgos del clasificador |
| B5. ¿Sobrevive con las masas correctas? | Robustez de los primeros agarres de los métodos de la Fase V con v2 y las 6 condiciones estándar, y con las cargas de estrés |

La etapa B se declara explicativa y exploratoria. No hay contraste de hipótesis confirmatorio.

## 4. Etapa C — fusión simple (desarrollo)

### Variables

- **p̃**: probabilidad del clasificador de la Fase V dividida por la máxima de la pieza (el prior de la
  Fase V).
- **Q̃**: calidad normalizada por pieza (mín–máx sobre todos sus candidatos), con dos definiciones fijas:
  - **Q_pre**: la puntuación preliminar del planificador (carrera, fuerza, centrado y aproximación).
  - **Q_cen**: el término de centrado, 1 − `center_offset`/45 mm, recortado a [0, 1].

Las dos se conocen antes de planificar y salen de la generación de candidatos.

### Cómo pasa una puntuación fusionada al planificador

- La puntuación fusionada **s** se normaliza a [0, 1] por pieza.
- Se pasa como prior **(s + 2·filtro)/3** con peso 10⁶ (`--prior` con `prior_weight`).
- El filtro de carrera y fuerza va dentro del propio prior. Así sigue delante: con peso 10⁶, el +10 del
  planificador ya no lo garantizaría. Este es el error que tuvo el orden aleatorio de la Fase V
  ([fe de erratas](ERRATUM_PHASE5_DYNAMICS.md), §3).
- La preliminar solo desempata.
- El orden de examen se calcula con el simulador exacto validado en la Fase V.

### Métodos

| Id | Método | Hiperparámetros (rejilla fijada) |
|---|---|---|
| M0 | Orden por defecto del planificador | — |
| M1 | Clasificador aprendido de la Fase V (preliminar + 2·p̃) | — |
| M2 | AFAT de la Fase V | — |
| Ref | Orden aleatorio dentro del filtro (20 semillas; solo para N_F) | — |
| M3 | Suma ponderada α·p̃ + (1−α)·Q̃ | α ∈ {0; 0,1; …; 1} × Q ∈ {pre, cen} |
| M4 | Producto p̃^α · Q̃^(1−α), con Q̃ ≥ 10⁻³ | igual que M3 |
| M5 | Lexicográfico: primero los de p̃ ≥ θ ordenados por Q̃; después el resto por p̃ | θ ∈ {0,3; 0,5; 0,7; 0,9} × Q |
| M6 | Filtro de calidad con vuelta atrás: primero los de Q̃ ≥ τ ordenados por p̃; después el resto por p̃ | τ ∈ {0,3; 0,5; 0,7; 0,9} × Q |
| M7 | Agregación de rangos (Borda ponderado) α·rango(p̃) + (1−α)·rango(Q̃) | α ∈ {0,1; …; 0,9} × Q |
| M8 | Pareto: frentes no dominados en (p̃, Q̃), cada frente ordenado por p̃ | Q ∈ {pre, cen} |

### Medidas por pieza

- **N_F**: candidatos examinados hasta el primer viable (la medida de la Fase V).
- **R**: si el primer viable es robusto.
- **N_FR**: candidatos hasta el primer viable **y** robusto. Es la medida de ajuste. Modela un sistema que
  prueba en simulación cada viable antes de ejecutarlo. Las piezas sin ningún viable robusto quedan fuera
  de N_FR, y se dice cuántas son.
- **Secundarias**:
  - tolerancia a la carga del primer viable;
  - sostenido nominal;
  - giro en la condición nominal.

### Ajuste sin fuga

- Validación cruzada **anidada** por pieza: 5 pliegues externos con semilla 20261130.
- En cada pliegue de entrenamiento se elige, dentro de cada familia (M3–M8), la combinación de la rejilla
  que minimiza la media de log N_FR. El tipo de Q forma parte de la rejilla.
- Esa elección se evalúa en el pliegue externo que quedó fuera.
- Las estimaciones de M3–M8 son las de los pliegues externos. M0–M2 no tienen nada que ajustar.
- Además se dan las curvas completas de la rejilla, como descripción. No sirven para elegir.

### Contrastes

Son de desarrollo. Hay una familia por medida y se corrige con Holm dentro de cada una. Cada familia
M3–M8 se compara con M0 y con M1:

- **N_F y N_FR**: cociente de medias geométricas por pieza, IC bootstrap del 95 % (10 000 réplicas,
  semilla 20261128) y Wilcoxon.
- **R**: diferencia de proporciones pareada, con IC bootstrap y McNemar exacto.

### Regla de decisión fijada: ¿una fusión simple recupera la robustez?

Una familia **cumple** si, en la validación cruzada externa, se dan las tres condiciones:

1. **Robustez no inferior a M0**: el límite inferior del IC del 95 % de (R_familia − R_M0) es mayor que
   −5 puntos porcentuales.
2. **Sigue siendo más rápida que M0**: el límite superior del IC del cociente N_F frente a M0 es menor
   que 1.
3. **Conserva la aceleración de M1**: el límite superior del IC del cociente N_F frente a M1 es como
   mucho 1,25, el margen de la Fase V.

**Cómo se interpreta:**

- Si ninguna cumple pero alguna mejora N_FR frente a M1, se dice así, sin más.
- Si el IC de R es demasiado ancho para decidir la condición 1, el resultado es «no demostrado».
- Si con las masas correctas M1 ya no pierde robustez frente a M0 (B5), la pregunta se traslada a las
  cargas de estrés. Esas se informan como secundarias.

## 5. Etapas D–G: solo si los datos lo justifican

Las puertas de decisión están en [phase6_go_no_go.md](phase6_go_no_go.md). Resumen:

| Etapa | Se ejecuta si… |
|---|---|
| Calibración de p_F (Brier, NLL, ECE; Platt e isotónica) | Siempre: es barata y la necesita cualquier producto P(F)·P(H\|F) |
| Modelo condicional P(H\|F) aprendido | Si ninguna fusión simple cumple la regla de §4, o si Q_cen explica poco la robustez (AUC < 0,70) |
| Rasgos locales de DINO, MLP, DKL, ADKF-IFT | Solo si un modelo geométrico simple deja una mejora grande y demostrable por explicar. ADKF-IFT necesita además tareas con consultas de soporte, que este problema no tiene hoy |
| Simulador articulado, costes online, baselines externos, tercer dataset, test confirmatorio | Según los recursos. Lo que no se pueda hacer aquí se documenta como bloqueo |

## 6. Integridad

- **Los 15 principios** de la petición se aplican sin excepción. En particular:
  - no se usa ningún dato confirmatorio;
  - la medida de ajuste y la regla de decisión quedan fijadas aquí;
  - las piezas, no los candidatos ni las perturbaciones, son la unidad de análisis.
- **Desfavorables**: todos los resultados se publican, también los desfavorables.
- **Fases anteriores**: no se reescriben. La fe de erratas de la Fase V vive en la Fase VI y no cambia
  sus ficheros.

## 7. Diseño confirmatorio (no ejecutado)

Está en [statistical_power.json](results/statistical_power.json) y en
[dataset_access_register.md](dataset_access_register.md):

- tamaño muestral para la regla de §4, con la variabilidad observada en desarrollo;
- fuentes posibles de objetos nuevos y su estado de acceso;
- qué se congelaría antes de abrirlas.

Ningún objeto de un conjunto confirmatorio se descarga ni se evalúa en esta fase.
