# Protocolo experimental mejorado

**Estado: PENDING, sujeto a revisión independiente y congelación antes de test.** Este protocolo es independiente del Legacy TFM: no reconstruye ni modifica sus detalles. La reproducción exacta, las fuentes y sus parámetros desconocidos se documentan en [la reconstrucción histórica](legacy/RECONSTRUCTION.md); cualquier reconstrucción aproximada se etiquetará como tal, con resultados separados. Nada de este documento prueba que el simulador, los métodos o los datos estén disponibles o validados.

## Población y particiones

Crear manifest versionado con identificadores y hashes de activos, procedencia, licencia, categorías, familias geométricas y procedimiento de inclusión; identificar duplicados y variantes casi idénticas antes de dividir. Separar por identidad/familia de objeto entrenamiento, validación y test retenido; reservar categorías desconocidas cuando sea factible. Los objetos y categorías, proporciones y reglas exactas son desconocidos y deben fijarse antes de congelar. La población objetivo será explícitamente la de objetos elegibles y condiciones simuladas del manifest, no todos los objetos reales.

Entrenar normalización, PCA, recuperación, calibración, hiperparámetros y selección de modelos únicamente en entrenamiento; elegir configuración y umbrales en validación o piloto separado. Nunca usar el test final para decisiones de diseño, potencia o ajuste. Antes de cada evaluación t solo se permite imagen/geometría observada, datos históricos autorizados y resultados del objeto actual hasta t−1. Registrar procedencia y tiempo de disponibilidad de cada característica, incluidos contextos derivados de medias de agarre. Prohibida una media calculada con futuras evaluaciones. Comprobar duplicados y contaminación conocida de preentrenamiento; datos de preentrenamiento desconocidos limitan la afirmación de novedad de objeto.

Experimento primario: memoria construida exclusivamente con tareas de entrenamiento y congelada; idéntica para métodos autorizados, sin datos de test. Toda secuencia online/precuencial será experimento separado: predefinir orden, revelar y actualizar solo después de evaluar cada tarea; registrar memoria y hashes por paso. Si se comparte una memoria u orden entre objetos, no presumir independencia: replicar historias independientes y analizar por bloques de historia, o limitar inferencia condicional a la historia fija.

## Comparadores y equidad

| ID | Comparador y requisito operativo |
|---|---|
| B0 | búsqueda aleatoria en dominio factible |
| B1 | secuencia Sobol con scrambling reproducible |
| B2 | GP-BO solo target, sin contexto ni historia |
| B3 | TFM original si recuperado; en otro caso reconstrucción identificada y separada, sin llamarla reproducción exacta |
| B4 | GP-BO contextual con DINOv2 congelado |
| B5 | GP-BO contextual con DINOv3 congelado |
| B6 | BO con descriptores geométricos solamente |
| B7 | GP multitarea con datos históricos autorizados |
| B8 | RW-GP ensemble u otro método robusto publicado, especificado antes del congelado |
| B9 | transferencia indiscriminada con historia autorizada |
| B10 | selección de transferencia solo por similitud visual |
| B11 | ACT-BO propuesto |

La tabla especifica comparaciones deseadas, no implementaciones verificadas. Definir versiones, kernels, adquisición, restricciones, entrenamiento y ajustes en configuraciones versionadas. Ningún método debe recibir observaciones ocultas o más evaluaciones del objetivo. Igualar dominio, simulador, función objetivo, restricciones, presupuesto total y distribución de inicialización; compartir exactamente puntos iniciales factibles por objeto/semilla cuando sea compatible, incluyendo su coste en el presupuesto. Otros costes de preparación, datos históricos, entrenamiento, extracción, inferencia y memoria deben contabilizarse por separado. Las diferencias inevitables en conocimiento previo se explicitan; los generadores de candidatos externos no se presentan como optimizadores completos sin evaluación equiparable. Incluir B3 únicamente si es ejecutable y comparable; de otro modo consignar bloqueo sin sustituir evidencia.

## Ejecución

Depurar con piloto de objetos distintos de test; objetivo provisional de 30 semillas independientes por objeto, sujeto a potencia y coste; evaluar 10, 20, 50 y 100 llamadas totales cuando sea viable, registrando trayectorias completas. Congelar el presupuesto primario antes de test. Generar una tabla de semillas explícita por objeto y réplica, común a métodos para inicialización y perturbaciones exógenas; flujos de aleatoriedad internos derivados de identificadores de método y réplica. Registrar si el pareamiento no se puede mantener. Repetir órdenes y memorias independientes para experimentos secuenciales. El objeto es unidad de generalización; semillas no son objetos adicionales.

Métricas físicas, dominio, unidades, validez del simulador y rango de calidad deben establecerse y verificarse antes de recopilar resultados finales. Manipulabilidad no equivale a estabilidad de agarre. Registrar calidad best-so-far, elevación/éxito, estabilidad, colisiones, restricciones, costes, calibración, degradación y fallos. No inventar óptimos para regret; identificar referencias aproximadas. Si solo hay simulación, conclusiones limitadas a dicha simulación.

Ablaciones predefinidas cuando sean implementables: quitar DINOv3, global, local, geometría, memoria, DKL, gate, evidencia predictiva, incertidumbre, fallback, contexto dinámico y warm start. Comparar con configuración completa, igual presupuesto e información; documentar componentes no activos y dependencias. Sensibilidad: dimensión latente, historia, tareas recuperadas, similitud, vistas, kernel, adquisición, orden, presupuesto e inicialización. Robustez por cambios visuales, oclusión/cámara/ruido/profundidad, geometría/pose/escala, fricción/masa/centro de gravedad, percepción/cinemática y efector; especificar magnitudes y semillas en validación. Evitar afirmar robustez general a partir de una perturbación.

## Trazabilidad y gates

Por ejecución registrar commit, hash de configuración, dataset y splits, objeto, categorías, orden e historia, semilla y flujos, versiones/checkpoints/licencias, hardware, presupuesto, observaciones, calidad, tiempo, errores y recursos. Conservar fallos y desviaciones en manifiesto inmutable. Gates sin evidencia: literatura/novedad, reproducción legacy, modelos, física, ausencia de leakage, matemática, equidad, estadística, instalación limpia y auditoría académica son **PENDING**. Ninguno se aprueba por cobertura o puntuación agregada; auditor independiente revisa evidencia, corrige y reevalúa.
