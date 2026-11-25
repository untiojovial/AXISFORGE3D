# Auditoría independiente global — AxisForgeBO

Fecha: 2026-11-25. Auditor: cuarto agente independiente, sin modificaciones de implementación; las pruebas adversariales propias están en `tests/test_scientific_audit.py`. **Estado: BLOQUEADO**, no aprobado. El registro de revisiones e incidencias está en `AUDIT.md`. El presente informe se actualiza con las últimas verificaciones; no atribuye evidencia a ejecuciones no realizadas.

## Evidencia y límites

La tesis suministrada tiene 71 páginas y SHA-256 `3b44e30ba7a476b9c50a7ff146db5061a2aa1761db173485611afd65d4b57cb1`. Su texto y figuras seleccionadas se han contrastado con los documentos de investigación. Las discrepancias documentadas incluyen el significado de ADFK/IFT, la entrada de 387 componentes frente a la media escalar de métricas, el riesgo no demostrado de fuga temporal, y la ausencia de índices de partición verificables. No hay evidencia para afirmar reproducción fiel ni para inferir que el código original cometió fuga. El inventario `THESIS_SOURCES.md` reconoce fuentes primarias de la bibliografía aún no leídas; la investigación exhaustiva y el estado del arte completo no están demostrados.

El wheel definitivo se instaló en un entorno limpio. El auditor verificó el manifiesto de fuentes SHA-256 `7c00f9be60e1c607d100d23cd7557cc94d46289b8b4146c0f41ade8348e18f67` y ejecutó `python -I -m coverage run --branch --source=/tmp/axisforge-bo-clean/lib/python3.12/site-packages/axisforge_bo -m pytest -q --import-mode=importlib`: **163 aprobados, 0 fallidos, 2 avisos**, 19,08 s. Los avisos corresponden al fallback Python de qLogEHVI por caché de compilación de solo lectura y a una advertencia de estandarización con datos constantes. Cobertura independiente de todo el paquete instalado: **1.948/2.045 sentencias (95,26 %)** y **687/776 ramas (88,53 %)**; supera los umbrales de 90 %/85 % en este alcance medido, sin probar que cada algoritmo individual cumpla esos porcentajes. AUD-008 queda resuelta para el wheel/manifiesto comprobados.

El auditor comprobó las 960 filas escalares: 240 grupos problema/semilla × 4 métodos, 16 observaciones por fila y cuatro observaciones iniciales iguales para los métodos de cada grupo. Recalculó las diferencias emparejadas y los intervalos bootstrap de `paired.csv` a partir de los datos crudos: coinciden. Las 240 trayectorias nativas y sklearn coinciden en la configuración de RBF fija; esa comprobación es una referencia de implementación, no una prueba de superioridad general. El archivo avanzado de desarrollo contiene 55 filas con cinco semillas por configuración. Una cohorte confirmatoria adicional tiene 330 filas y 30 semillas nuevas (301–330); ZDT1 y DTLZ2 comparten cuatro puntos iniciales entre métodos en los 30 grupos. El auditor recalculó las diferencias medias de hipervolumen, tasa de violación y regret de fidelidad: coinciden con el informe redondeado. El experimento contextual contiene 180 filas, 30 semillas por combinación y ocho evaluaciones de tarea actual. Los escenarios avanzados iniciales de cinco semillas son exploratorios; la cohorte de 30 semillas es evidencia adicional para esos problemas específicos. Las mediciones temporales y de memoria no demuestran un límite general de escalabilidad.

El auditor escribió ocho pruebas propias que incluyen comparaciones analíticas de factibilidad, CVaR y valor de información, hipervolumen con BoTorch, LogEI, casos extremos y retención del mejor resultado tras un reinicio. Estas pruebas forman parte de la suite completa del wheel definitivo, que pasó tras los cambios. Las incidencias P1 de overflow, mutación de resultados, política de ruido y procedencia WASM fueron corregidas y reexaminadas individualmente en `AUDIT.md`, sin aprobar por ello el sistema entero.

## Matriz y puntuación

| Categoría | Máximo | Estado / evidencia | Puntos |
|---|---:|---|---:|
| Corrección matemática y científica | 25 | Referencias analíticas y bibliotecas externas para subconjuntos; algoritmos y escenarios avanzados no auditados exhaustivamente. | No evaluado |
| Arquitectura y extensibilidad | 15 | Interfaces, estados y adaptador revisados; wheel definitivo y regresión aislada comprobados. | No evaluado |
| Pruebas y robustez | 15 | 163/163 pruebas aprobadas en wheel limpio; 95,26 % de sentencias y 88,53 % de ramas en el paquete completo. | No evaluado |
| Reproducibilidad científica | 15 | Agregaciones escalares verificadas e instalación aislada comprobada; la coordinación informa de repetición de los benchmarks, sin repetición independiente de todas sus ejecuciones. | No evaluado |
| Rendimiento y estabilidad | 10 | Benchmarks con 30 semillas confirmatorias para escenarios concretos; falta caracterizar escalado de aproximaciones variacionales y aceleradores. | No evaluado |
| Integración AxisForge3D | 10 | Hay pruebas dirigidas de operación real del servicio WASM; no son equivalentes a integración arbitraria de agarres ni validación física. | No evaluado |
| Documentación y mantenibilidad | 10 | Trazabilidad y reportes disponibles; el manifiesto y wheel fueron comprobados. | No evaluado |
| **Total** | **100** | No es válido convertir ausencia de evidencia en puntos. | **No evaluado** |

No se ha demostrado el mínimo 95/100 ni el 90 % por categoría. La cobertura medida del paquete completo supera los objetivos cuantitativos de líneas y ramas. Las puntuaciones no sustituyen a los controles obligatorios. La evaluación debe reabrirse cuando exista evidencia de los requisitos pendientes.

## Bloqueos para la aprobación completa

1. **AUD-008, resuelta:** wheel definitivo reinstalado y probado de forma aislada; el manifiesto de fuentes coincide.
2. **Corrección verificada:** las ramas semánticas de error del evaluador WASM conservan coste y hash; la suite final aislada incluye las pruebas pertinentes.
3. La coordinación registró la repetición sobre el wheel congelado de 960/960 trayectorias escalares y 330/330 avanzadas (sin tiempos), además de 180 combinaciones contextuales; el auditor comprobó independientemente las agregaciones pero no repitió cada ejecución. La evaluación en GPU permanece no evaluada. SAAS requiere diagnóstico de convergencia y mezcla de cadenas, más allá de la prueba breve de integración; los modelos variacionales requieren estudios de escalado.
4. La investigación obligatoria de fuentes sigue incompleta: por ejemplo, los artículos originales de DINO, DINOv2 y Swersky no pudieron leerse en esta sesión. Los pesos, código y particiones del TFM tampoco están disponibles para resolver sus ambigüedades; la reproducción robótica pertenece a una fase posterior y no es una condición añadida a esta fase.
5. Completar las fuentes y validaciones aún pendientes, ejecutar las regresiones afectadas tras cambios y someter el resultado a nueva auditoría independiente. No aprobar con P0/P1 abiertos o criterios obligatorios no verificados.
