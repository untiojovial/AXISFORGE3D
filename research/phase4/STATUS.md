# Fase IV — estado de la investigación

> La revisión de esta fase se responde en la [Fase V](../phase5/README.md) ([respuesta punto por punto](../phase5/RESPONSE.md)).

**Desarrollo ejecutado. Evaluación confirmatoria cerrada.** La evaluación independiente acepta diagnósticos exploratorios sobre los 22 objetos autorizados. La recuperación de agarres y la superioridad de la transferencia siguen sin demostrarse. Los ocho objetos de test permanecen reservados. Este informe distingue una implementación que funciona, un resultado de simulación y una hipótesis validada.

| Hito | Estado y evidencia disponible |
|---|---|
| M0 · Auditoría inicial | Recontados 160 intentos históricos, 3 viables, y 256 pares distintos en validación, 8 viables. Revisados motivos nativos, puerta y calibración. Algunas causas físicas siguen abiertas. [Auditoría](INITIAL_AUDIT.md) |
| M1 · Modelos congelados | OpenCLIP, DINOv2 y DINOv3 (copia de timm, licencia DINOv3) ejecutados en CPU y reproducidos byte a byte; DINOv2 verificado también a través del notebook. Prueba del notebook en Colab/GPU pendiente. [Visión](vision/README.md) · [DINOv2](embedding_experiment/dinov2/README.md) · [DINOv3](embedding_experiment/dinov3/README.md) |
| M2 · Pipeline visual | Implementados grid y afinidad específicos del procesador, sellos por vista, pruebas de oclusión y asociación de regiones. OpenCLIP, DINOv2 y DINOv3 medidos en el mismo panel, sin base para declarar uno superior. Integrado en el planificador nativo como prior de agarre. En dos pruebas con protocolo fijado sobre 50 piezas nuevas cada una, el prior ahorra más de la mitad de candidatos frente al orden por defecto, pero la apariencia no mejora al tamaño solo, ni con renders normalizados ni con renders que conservan la escala. [Panel](embedding_experiment/README.md) · [Prior de agarre](grasp_prior/README.md) · [Piezas nuevas](grasp_prior/new_parts/README.md) · [Escala](grasp_prior/scale/README.md) |
| M3 · Factibilidad | Catálogos de 48/128/256/512 y evaluación real en 128. En validación 4/150 viables con selección estratificada, 7/150 con uniforme. En un segundo panel prospectivo hubo 0/72 en cada brazo. La mejora geométrica no se tradujo en recuperación de F1. [Candidatos](candidates/README.md) |
| M4 · Archivo histórico | Dos estrategias en 16 objetos train y prefijos 10/25/50. La memoria estratificada de 800 consultas contiene 17 éxitos y 783 rechazos. Los costes y tipos de rechazo se conservan. La historia sigue siendo poco informativa para ciertos objetos. [Resultados](candidates/results.json) |
| M5 · Transferencia V2 | 144 decisiones de la puerta original bajo prefijos iguales, una distinta. Comparaciones pareadas sobre 6 objetos y 2 semillas. El signo y tamaño dependen de la referencia y la historia. No se ha validado superioridad. [Pilotos, baselines y figura](transfer/PILOT.md) |
| M6 · Incertidumbre | Validación por objeto dentro de train. DKL de semilla 11 alcanza cobertura del 95,6 % tras recalibrar y ensancha sus intervalos 4,6 veces. La densidad predictiva favorece a Matérn en esta muestra. [Calibración](methodology/CALIBRATION_RESULTS.md) |
| M7 · Física | 14 objetos train, 84 poses, 71 nominales y 13 discrepancias de proxy. Cinco sostenes brutos, cuatro tras asentamiento. Hay penetraciones y controles numéricamente inválidos. El UR5 completo solo se evaluó en seguimiento libre. [Piloto y controles](robotics/PHYSICAL_RESULTS.md) |
| M8 · Multifidelidad | Calculadas dependencias, ranking y cascadas en el efector aislado. F2 frente a sostén elegible tiene Pearson 0,201 y Spearman 0,146 en el subconjunto observable. No hay validación de una política adaptativa, no inferioridad ni ahorro de coste total. [Análisis](methodology/MULTIFIDELITY_RESULTS.md) |
| M9 · Desarrollo científico | Baselines, ablaciones y controles negativos ejecutados sobre paneles emparejados y con trazas. El número de objetos, los empates y la dependencia de la historia limitan la inferencia. [Visión](embedding_experiment/README.md) · [Transferencia](transfer/PILOT.md) |
| M10 · Protocolo | Borrador con unidad independiente, AUC, baselines, hipótesis candidatas, potencia e información autorizada. Falta tamaño defendible y dictamen de aprobación. [Borrador](methodology/CONFIRMATORY_DRAFT.md) |
| M11 · Confirmación | Bloqueado. El modo confirmatorio falla antes de acceder al dataset. |
| M12 · Manuscrito | Aplazado hasta que haya resultados y protocolo suficientes. Evaluación provisional de novedad y limitaciones disponible. [Antecedentes](methodology/NOVELTY.md) |

## Inventario de los 28 entregables

| Entregables solicitados | Resultado |
|---|---|
| 1. Auditoría | Recuento verificable de fase III. [Referencia](verification/reference.json) |
| 2–3. Colab y descarga | Notebook con código e imágenes incluidos y descarga oficial. CPU comprobada. [Notebook](../../notebooks/AxisForge_Phase4_Frozen_Foundation_Models.ipynb) |
| 4–6. DINOv2, DINOv3 y adaptador | DINOv2 oficial y DINOv3 (copia de timm) ejecutados y reproducidos. [Adaptador](../../research_grasping/phase4/frozen_models.py) · [DINOv2](embedding_experiment/dinov2/README.md) · [DINOv3](embedding_experiment/dinov3/README.md) |
| 7–8. Transporte y jobs | Exportación, importación, reintento, límites, validación y estado por objeto. [Automatización](automation/README.md) |
| 9. Comparación de modelos | OpenCLIP, DINOv2 y DINOv3 medidos con el mismo control geométrico, control aleatorio y señales globales/locales. En el prior de agarre con piezas nuevas, ninguno supera a un ViT sin entrenar ni a vecinas por tamaño de caja. [Tabla generada](embedding_experiment/foundation_comparison.json) · [Prior de agarre](grasp_prior/README.md) |
| 10–11. Factibilidad y generador | Nuevo selector optativo, control original, dos ubicaciones de escena y dos intervenciones medidas. [Resultados y condiciones](candidates/README.md) |
| 12–15. BO con factibilidad, archivo, diagnóstico de transferencia y ACT-BO V2 | Implementados y medidos. Hay controles target, contextuales y RGPE, memoria con rechazos y piloto V2b. La evidencia sobre la ventaja es inconclusa. [Resultados](transfer/PILOT.md) |
| 16. DKL | Estimación de variación entre objetos, constante y GP de referencia, recalibración anidada y anchura de intervalos. [Informe](methodology/CALIBRATION_RESULTS.md) |
| 17–19. Integración física, dinámica multiobjeto y multifidelidad | Ejecutadas pruebas del efector aislado, seguimiento UR5 sin contacto y análisis condicional de cascadas. Quedan pendientes el agarre dinámico integrado y una política prospectiva multifidelidad. [Física](robotics/PHYSICAL_RESULTS.md) · [Fidelidades](methodology/MULTIFIDELITY_RESULTS.md) |
| 20–21. Baselines y ablaciones | Incluye aleatorio, Sobol proyectado, GP objetivo/contextual, ICM fijo, RGPE, mezclas con/sin evidencia, controles visuales y presupuestos de memoria. ICM usa parámetros fijos. [Trazas](transfer/PILOT.md) |
| 22. Automatización | DAG declarativo, jobs independientes, locks, caché, trazas y límite confirmatorio. [Reproducción](REPRODUCIBILITY.md) |
| 23–24. Calidad y reproducción | Pasaron 502 pruebas Python y 355 Rust. El paquete de fase IV alcanzó 96,35 % de líneas y 85,42 % de ramas. Limitaciones de hardware, reproducibilidad de artefactos y costes documentados. [Cobertura](verification/COVERAGE.md) · [Reproducción](REPRODUCIBILITY.md) |
| 25–28. Protocolo, auditoría, novedad y artículo | Borrador metodológico y revisión independiente realizados. Novedad, aprobación confirmatoria y manuscrito pendientes. [Acta](audit/REVIEW.md) · [Registro de bloqueos](BLOCKER_REGISTER.md) |
