# AxisForgeBO y agarre robótico · Fase II

> **Fase V (2026-11-26).** Respuesta a la revisión de la Fase IV, con protocolo fijado antes de los datos,
> 200 piezas nuevas, tres experimentos de generalización, tres pinzas y validación dinámica en MuJoCo:
> [resultados](phase5/README.md) · [protocolo](phase5/PROTOCOL.md) · [respuesta a la revisión](phase5/RESPONSE.md) ·
> [esquema del manuscrito](phase5/MANUSCRIPT.md). Lo que sigue describe la Fase II.

**Resultado provisional.** La biblioteca y el simulador quedaron integrados en un merge de dos padres, con una referencia CPU probada. Se implementó un mecanismo de transferencia que actualiza la confianza en cada tarea histórica al observar resultados en la tarea actual. Los bancos sintéticos y los ensayos en MuJoCo proporcionan diagnósticos. La ventaja científica en objetos nuevos con pesos fundacionales preentrenados sigue sin demostrar. La rama de investigación corresponde a AxisForgeBO 0.2.0; el merge de referencia conserva la versión 0.1.0. [Integración](GIT_INTEGRATION.md) · [Auditoría independiente](audit/PHASE_II_INITIAL_REVIEW.md)

El paquete de la fase II pasó **196 pruebas Python** desde una instalación limpia con todos los extras; el simulador integrado pasó **353 pruebas Rust**. El núcleo alcanzó 95,26 % de líneas y 88,66 % de ramas. El nuevo paquete de investigación quedó en 65,15 % y 57,84 %. [Verificación](REPRODUCIBILITY.md)

**Qué se puede reproducir.** En el escenario sintético adversario con semillas 40–47, la ponderación contextual fija obtuvo un área media de 0,6843, el GP sin historial 0,8621, la transferencia adaptativa 0,8621 y la variante de RGPE 0,8627. La fuente histórica invierte el signo del objetivo. Esta igualdad redondeada con el GP no demuestra inmunidad general a la transferencia negativa. En el cubo de AxisForge3D, con tres semillas y diez intentos por método, el área media de la mejor elevación normalizada fue 0,7974 con transferencia y 0,7954 con el GP. La primera política tenía acceso adicional a 18 evaluaciones históricas del mismo cubo en condiciones físicas diferentes. [Banco sintético](results/transfer/BASELINES.md) · [Piloto del cubo](results/cube/README.md)

En 36 condiciones dinámicas adicionales, la caja alargada cumplió el criterio de sostén en 2 de 12 y el cilindro y el elipsoide en 0 de 12 cada uno. Son tres primitivas prescritas en la misma pinza. Su alcance es diagnóstico. [Procedimiento y resultados](experiments/primitive_pilot_results.md)

**Equipo y aprobación.** La sesión principal coordinó cinco instancias independientes GPT-6.1-Terra para BO, representaciones, simulación, auditoría y metodología. La plataforma permite seleccionar el modelo de los subagentes, pero no verificar ni cambiar a petición la identidad exacta del modelo que ya ejecuta la coordinación. El auditor tuvo capacidad de veto y no implementó los componentes sometidos a su revisión. No se alcanzó una calificación científica global de 95/100. El trabajo se presenta como rama de investigación. [Dictámenes](audit/PHASE_II_INITIAL_REVIEW.md)

## Entrega por hitos

| Hito | Estado | Evidencia y dependencia |
|---|---|---|
| M0 integración | VALIDATED en su alcance técnico CPU; BLOCKED para aprobación científica global | Referencia `2ab53e0`, 353 pruebas Rust y 163 pruebas del wheel de referencia; permanecen requisitos científicos de la fase I. |
| M1 viabilidad | PARTIALLY VALIDATED | Antecedentes, hipótesis y metodología revisados; acceso incompleto a fuentes y diseño final sin congelar. |
| M2 reproducción | PARTIALLY VALIDATED | Contraste automatizado de ocho máximos publicados y doce posiciones en figuras. Código, mallas, particiones y checkpoint del autor no recuperados. |
| M3 modelos y geometría | PARTIALLY VALIDATED | Carga offline, tokens, RGB-D y descriptores probados en CPU. Faltan checkpoints preentrenados accesibles y evaluación de afinidad funcional en objetos. |
| M4 física | PARTIALLY VALIDATED | Cierre/elevación en un modelo dinámico reducido, fallos deliberados y sensibilidad de solver. Faltan validación externa del modelo, mano histórica y diversidad representativa. |
| M5 transferencia | PARTIALLY VALIDATED | Mezcla prequential, incertidumbre total, EI y pruebas sintéticas. Faltan comparaciones visuales y robustez fuera de la población simulada. |
| M6 pilotos | PARTIALLY VALIDATED | Pilotos de integración, estadística, respuesta física y transferencia con límites documentados. El piloto con 20–30 objetos diversos sigue pendiente. |
| M7 confirmación | BLOCKED | Falta población de objetos y particiones auditadas, pesos, comparación visual y potencia reales. No se abrió un test final. |
| M8 validación externa | PARTIALLY VALIDATED solo en sensibilidad simulada | Se midieron variaciones físicas y numéricas. Faltan validación externa y hardware. |
| M9 manuscrito | BLOCKED | Resultados insuficientes para un artículo sobre transferencia visual entre objetos. No hubo envío ni revisión académica externa. |

## Inventario de los 28 entregables

| Nº | Entregable | Clasificación y acceso |
|---:|---|---|
| 1 | Biblioteca consolidada y versionada | VALIDATED dentro del alcance técnico CPU. [Informe](GIT_INTEGRATION.md) |
| 2 | Informe de integración Git | VALIDATED. [Historial y pruebas](GIT_INTEGRATION.md) |
| 3 | Auditoría actualizada | PARTIALLY VALIDATED. [Registro independiente](audit/PHASE_II_INITIAL_REVIEW.md) |
| 4 | Revisión bibliográfica | PARTIALLY VALIDATED. [Síntesis](RELATED_WORK.md) |
| 5 | Matriz del estado del arte | PARTIALLY VALIDATED. [Matriz y límites de acceso](LITERATURE_MATRIX.md) |
| 6 | Contribución científica | PARTIALLY VALIDATED como pregunta. [Viabilidad](RESEARCH_GAP.md) y [novedad](NOVELTY_ASSESSMENT.md) |
| 7 | Protocolo experimental | PARTIALLY VALIDATED como diseño. [Protocolo](EXPERIMENTAL_PROTOCOL.md) e [hipótesis](HYPOTHESES.md) |
| 8 | Reconstrucción del TFM | PARTIALLY VALIDATED. [Matriz de ambigüedades](legacy/RECONSTRUCTION.md) |
| 9 | Pipeline de modelos fundacionales | PARTIALLY VALIDATED con pesos de prueba. [Representaciones](methodology/REPRESENTATIONS.md) |
| 10 | Representaciones 3D | PARTIALLY VALIDATED para descriptores, proyección y malla; Uni3D/Sonata sin ejecutar. [Geometría](methodology/REPRESENTATIONS.md) |
| 11 | Entorno de agarre | PARTIALLY VALIDATED para pilotos CPU. [Física](methodology/PHYSICS.md) |
| 12 | ACT-BO | PARTIALLY VALIDATED en CPU con funciones sintéticas y búsqueda lateral. [Formulación](methodology/TRANSFER.md) |
| 13 | Baselines | PARTIALLY VALIDATED. Aleatorio, Sobol, GP, contexto fijo, mezcla, RGPE e ICM restringidos; B3–B5 y B7 aprendido comparables aún pendientes. [Comparación](results/transfer/BASELINES.md) |
| 14 | Benchmarks sintéticos | VALIDATED para repetición y agregación de los pilotos especificados. [Resultados](results/transfer/README.md) |
| 15 | Manifiestos de datos | PARTIALLY VALIDATED. Particiones y archivos finales pendientes. [Control de identidad y disponibilidad](methodology/DATA_SPLITS.md) |
| 16 | Experimentos piloto | PARTIALLY VALIDATED. [Cubo](results/cube/README.md) y [controles físicos](methodology/PHYSICS.md) |
| 17 | Experimentos científicos finales | BLOCKED por requisitos de M7. |
| 18 | Ablaciones | PARTIALLY VALIDATED para historia/evidencia sintéticas; ablaciones visuales pendientes. [Comparadores](results/transfer/README.md) |
| 19 | Análisis estadístico | PARTIALLY VALIDATED para diseño, cobertura y potencia hipotéticos. [Simulación](results/SYNTHETIC_POWER.md) y [alternativa acotada](results/BOUNDED_POWER.md) |
| 20 | Transferencia negativa | PARTIALLY VALIDATED en familias sintéticas. [Escenario adversario](results/transfer/BASELINES.md) |
| 21 | Generalización | PENDING para objetos y categorías no vistos. |
| 22 | Informe de reproducibilidad | PARTIALLY VALIDATED. [Reproducción y entorno](REPRODUCIBILITY.md) |
| 23 | Manuscrito LaTeX | BLOCKED por ausencia de evidencia final. |
| 24 | Figuras y tablas | VALIDATED para el piloto del cubo. [Figura y tabla](results/cube/README.md) |
| 25 | Material suplementario | PARTIALLY VALIDATED como registros y configuraciones de pilotos. [Datos](results/) |
| 26 | Informes independientes | PARTIALLY VALIDATED. Hay revisiones técnicas y de diseño; los tres dictámenes internos de un manuscrito quedan pendientes. [Registro](audit/PHASE_II_INITIAL_REVIEW.md) |
| 27 | Guía de reproducción | PARTIALLY VALIDATED en Linux CPU. [Comandos](REPRODUCIBILITY.md) |
| 28 | Propuesta de publicación | PARTIALLY VALIDATED. RA-L es el destino temático provisional; convocatoria y requisitos vigentes sin comprobar. [Criterios](PUBLICATION.md) |
