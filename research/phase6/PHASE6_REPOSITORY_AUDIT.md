# Fase VI — auditoría del repositorio (etapa A)

Fecha: 2026-11-26. Punto de partida: `11f6fb7`. Ficheros de datos: [feature_audit.json](results/feature_audit.json),
[phase6_data_manifest.json](results/phase6_data_manifest.json) y [physics_validation.json](results/physics_validation.json).

## 1. Estado de git

- Rama de trabajo: `claude/new-session-nsx8rp`, en `11f6fb7`.
- Remoto comprobado con `git fetch`: no hay commits posteriores a `11f6fb7`.
- Otras ramas remotas: `integration/axisforge-phase2`, `research/axisforge-phase3` y `research/axisforge-phase4`.
- La petición propone la rama `research/axisforge-phase6-fortis-r` «si resulta compatible». Encaja con la
  convención de las fases 3 y 4. Sin embargo, esta sesión solo puede desarrollar en
  `claude/new-session-nsx8rp`. Por eso los commits se hacen en esa rama y el mismo historial se publica
  también con el nombre pedido, sin reescribir nada (§10).

## 2. Cifras de la Fase V, verificadas

| Cifra declarada | Verificada | Fuente |
|---|---|---|
| 359 tests Rust | 359 pasan, 0 fallan (27 binarios, `cargo test --workspace --release`) | ejecución de hoy |
| Clippy, 0 avisos | 0 (`cargo clippy --workspace --all-targets --release -- -D warnings`) | ejecución de hoy |
| 463 tests Python, 65 omitidos | 463 pasan y 65 se omiten (§8) | ejecución de hoy |
| 7 588 coincidencias del simulador | 6 788 + 800, 0 discrepancias | [simulator_validation.json](../phase5/simulator_validation.json), [planner_confirmation.json](../phase5/results/planner_confirmation.json) |
| 116 piezas de biblioteca | 116 | [library_parts.txt](../phase5/library_parts.txt) |
| 200 piezas de prueba | 200 | [test_rows.json.gz](../phase5/results/test_rows.json.gz) |
| 151 viables y 49 no viables a 80 mm | 151 / 49 | ídem; lista en [dev_parts.json](dev_parts.json) |
| 3 613 simulaciones dinámicas | 3 613 | [dynamics.json.gz](../phase5/results/dynamics.json.gz) |
| Masas ×2 y ×0,5 en la dinámica | **No**: 2 kg y 0,5 kg | [fe de erratas](ERRATUM_PHASE5_DYNAMICS.md) |
| Centro de masas del simulador = el del planificador | **No**: difieren 6,4 mm en la mediana, hasta 51 mm | ídem |
| 8 objetos sellados de la Fase III sin usar | Ninguno está en la biblioteca ni en la prueba de la Fase V | [manifest.lock.json](../phase3/dataset/manifest.lock.json) |

## 3. Orden protocolo → resultados

| Protocolo | Resultados | ¿El commit de resultados toca el protocolo? |
|---|---|---|
| `75bec22` (19:59) | `08bd728` (20:20) | No. Cambia `frozen_models.py` (hash del procesador en la procedencia) y `visual_prior.py` (lectura de .gz) |
| `df48486` (20:59) | `2a0b8f6` (21:16) | No. Cambia `visual_prior.py` (subcomando para unir bibliotecas) |
| `ad9e1d2` (23:14) | `11f6fb7` (23:52) | No |

Ningún commit de resultados modifica un protocolo, una métrica ni un margen. Los cambios de código son de
entrada/salida y procedencia, no del método.

Límite de esta verificación: el orden de git prueba qué se publicó antes, no cuándo se vieron los datos.
No hubo prerregistro con un tercero. La fe de erratas muestra además una **desviación del protocolo de la
Fase V que nadie detectó**: el protocolo decía ×2 y ×0,5, y el código simuló 2 y 0,5 kg.

## 4. Los 19 rasgos del clasificador

Todos salen de la generación de candidatos (`axisforge research-candidates`, 0,4 s por pieza para todos
sus candidatos) y de la caja envolvente de la pieza. Ninguno usa cinemática inversa, colisiones ni el
veredicto del planificador. El veredicto exhaustivo es solo la etiqueta de entrenamiento, y solo en piezas
de la biblioteca.

| Rasgo | Fuente | Coste | Disponible antes de planificar | Riesgo de leakage |
|---|---|---|---|---|
| approach_x, _y, _z | Dirección de aproximación del candidato (generador) | Ninguno extra | Sí | Bajo |
| closing_x, _y, _z | \|dirección de cierre\| (generador) | Ninguno extra | Sí | Bajo |
| width | Distancia entre contactos (generador) | Ninguno extra | Sí | Bajo |
| rel_x, _y, _z | Centro del candidato relativo a la caja de la pieza | Ninguno extra | Sí | Bajo. rel_z codifica la altura sobre la mesa de **esta** celda |
| height | Altura del centro sobre el fondo de la caja | Ninguno extra | Sí | Bajo; igual que rel_z |
| log_extent_x, _y, _z | Tamaño de la caja de la pieza (igual para todos sus candidatos) | Ninguno extra | Sí | Bajo |
| antipodal_a, _b | \|normal · cierre\| en cada contacto | Ninguno extra | Sí | Bajo |
| centre_offset | Distancia del centro de masas del planificador a la línea de cierre (brazo de palanca) | Ninguno extra | Sí | Bajo. Supone densidad uniforme |
| preliminary | Puntuación preliminar del planificador | Ninguno extra | Sí | Bajo. Usa la masa (0,1 kg) y la fricción supuestas de la celda |
| makeable | Filtro de carrera y fuerza | Ninguno extra | Sí | Bajo. Es el propio filtro del planificador y usa la masa supuesta |

Lo que el clasificador usa de verdad, medido en las 151 piezas de desarrollo (AUC de viabilidad 0,993;
caída de AUC al permutar cada rasgo):

- **makeable** 0,069, **approach_x** 0,058, **height** 0,034, **approach_z** 0,028 y **closing_z** 0,011.
- El resto queda por debajo de 0,006.
- **centre_offset**: 0,0002. Prácticamente no lo usa. Por sí solo apenas separa viables de no viables
  (AUC 0,51 en la biblioteca y 0,57 en desarrollo).

**Riesgos que sí existen:**

- **A nivel de objeto**: 52 variantes de producto y 74 piezas parecidas entre las 200 de prueba de la
  Fase V. La sensibilidad de la Fase V mantuvo todas las decisiones sin ellas.
- **A nivel de celda**: todas las piezas comparten mesa, robot y pose. La altura transporta alcance
  específico de esta celda, y no hay ninguna prueba con otra celda.

## 5. Costes

De [costs.json](../phase5/results/costs.json), con la pinza de 80 mm:

- generación de candidatos: 0,37 s por pieza (mediana);
- resolución exhaustiva: 3,7 s;
- render: 5,3 s;
- codificación DINOv3: ~0,8 s;
- 59 órdenes: 0,5 s;
- entrenamiento del clasificador: ~80 s, una vez;
- ensayo dinámico: ~0,16 s.

**Límite**: la medida principal cuenta **candidatos**, no segundos. Un candidato que cae en el filtro
cuesta mucho menos que uno que llega a la cinemática inversa y a las colisiones. Por eso N_F no es tiempo
de planificación. La etapa G (coste real online) no se ha hecho (§9).

## 6. Modelo físico

| | Fase V (v1) | Fase VI (v2) |
|---|---|---|
| Qué simula | Almohadillas de 25 × 45 mm sobre una palma dinámica soldada a un objetivo cinemático; sin brazo ni palma con colisión | Igual |
| Contactos | Rígidos (solref 0,002), conos elípticos, fricción torsional 0,005 m (condim 4), multiccd | Igual |
| Centro de masas | Media de vértices de colisión | El del planificador |
| Inercia | Caja envolvente | Igual |
| Masas | 0,1 kg nominal, 2 kg y 0,5 kg (etiquetadas ×2 y ×0,5) | 0,1, 0,2 y 0,05 kg, más cargas de estrés de 0,5, 1 y 2 kg |
| Sostenido | Sube ≥ 80 mm y se desliza < 20 mm | Igual, más el giro respecto a la palma, el contacto final y la penetración |

### Controles ([physics_validation.json](results/physics_validation.json))

Con un cubo de 40 mm salvo donde se indica.

| Control | Resultado | Veredicto |
|---|---|---|
| Caída libre de 0,3 s | Error del 0,33 % respecto a ½gt² | Pasa |
| Giro libre de par durante 2 s | Deriva del momento angular de 0,02 % y de la energía de 0,02 % | Pasa |
| Almohadillas abiertas | No sube ni toca | Pasa |
| Almohadillas abiertas en 60 agarres reales | 0 sostenidos | Pasa |
| Fuerza insuficiente | Umbral simulado de 1,25 N por almohadilla; Coulomb da entre 0,98 N (solo peso) y 1,26 N (con las aceleraciones de subida y sacudida) | Pasa |
| Fricción baja (5 N por almohadilla) | Umbral simulado μ = 0,12; Coulomb da entre 0,098 y 0,126 | Pasa |
| Cubo centrado con las 6 condiciones | 6/6 sostenido | Pasa |
| Barra de 160 mm agarrada a d del centro | Sostiene hasta 0,8 kg con d = 60 mm; cae con 1,6 kg | Informativo |
| Paso de 0,5 ms en lugar de 1 ms | 178 de 180 veredictos iguales | Pasa |
| Solver CG / PGS en lugar de Newton | 180 / 178 de 180 | Pasa |
| Fricción torsional 0,001 / 0,02 en lugar de 0,005 | 163 / 160 de 180 | **Sensible** |
| Centro de masas v1 frente a v2 | 176 de 180 | Ver fe de erratas |
| Penetración durante el ensayo | Mediana de 0,27 mm, máximo de 1,1 mm | Aceptable |

La conclusión de robustez depende sobre todo de la **fricción torsional**, que no está calibrada con
almohadillas reales. Con cargas grandes es el parámetro que manda (sección B de los resultados).

## 7. Bloqueos históricos

| Bloqueo | Causa | Estado |
|---|---|---|
| Robot real | No hay hardware | Abierto |
| Contact-GraspNet, GPD y Dex-Net | Necesitan GPU, TensorFlow/CUDA propios o checkpoints externos; aquí solo hay CPU | Abierto (etapa G) |
| EGAD | HTTP 403 | Abierto |
| ShapeNet y Objaverse | Cuenta o licencias por objeto | Sin comprobar en esta fase ([registro](dataset_access_register.md)) |
| DINOv3 oficial | Formulario de Meta | Resuelto con la copia de timm; licencia aceptada por el usuario |
| API de GitHub (403 por alcance) | Alcance del token | Resuelto con un clon sin blobs |
| GSO agotado | La regla de tamaño dejó ~200 piezas nuevas de 858 | Impide un test confirmatorio nuevo con GSO |

## 8. Tests omitidos y avisos F401

**65 tests omitidos**, todos por datos o pesos que no están en el repositorio. Ninguno por fallo.

| Causa | Tests |
|---|---:|
| Paquetes de desarrollo externos de las fases III/IV (`AXISFORGE_PHASE3_DATA`, activos preparados) | 57 |
| Entradas selladas de la Fase III (ausentes a propósito) | 4 |
| Pesos autorizados (checkpoint de OpenCLIP, activos oficiales de DINO) | 3 |
| Dependencia opcional `numpyro` | 1 |

Por fichero: 13 de `tests/phase3`, 51 de `tests/phase4` y 1 de `tests/test_research_backends.py`.

**23 avisos F401** (importación sin usar):

- Todos están en código de las fases III y IV (commits `5326e4a`, `3ac8292`, `25e1171`, `d51bc20`,
  `5c3c623` y `a2fc364`), que no tocan las fases V ni VI.
- No se corrigen aquí para no cambiar código de fases anteriores.
- El código de las fases V y VI pasa `ruff --select F` sin errores.

## 9. Lo que la auditoría no cubre

- El coste online real (tiempo por candidato y por etapa de rechazo).
- La calibración del modelo dinámico con hardware.
- Cualquier dato confirmatorio nuevo.

Todo eso queda en las etapas F y G.

## 10. Ramas

Los commits de la Fase VI se hacen sobre `claude/new-session-nsx8rp` y se publican también como
`research/axisforge-phase6-fortis-r` apuntando al mismo commit. No se reescribe historial y las fases I–V
quedan intactas.
