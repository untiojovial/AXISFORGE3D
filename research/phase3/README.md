# AxisForge · Fase III

Esta rama extiende la fase II desde `e4e4f70`. El estudio usa 30 mallas públicas de Google Scanned Objects. Se fijaron 16 objetos de entrenamiento, seis de validación y ocho de test antes de medir los resultados. El test permanece cerrado.

## Evidencia de desarrollo

El primer piloto comparó nueve políticas con tres semillas y diez consultas por política en dos objetos de validación. Dos objetos adicionales aportaron el archivo histórico. De 540 consultas contabilizadas por los métodos, 77 obtuvieron un plan nativo viable y 463 fueron rechazadas por el planificador. Los métodos pueden consultar el mismo candidato. El resultado es una medida de planificación F1 y calidad de wrench F2. El éxito de elevación y mantenimiento bajo dinámica MuJoCo requiere una evaluación F3 independiente.

Se renderizaron 352 vistas RGB-D sobre 22 objetos y se extrajeron embeddings con OpenCLIP ViT-B/32 preentrenado. El visor AxisForge permite cargar el informe JSON, recorrer candidatos y contactos propuestos, inspeccionar la predicción anterior a cada observación y exportar la trayectoria. En el panel **Experimentos de agarre** puede abrirse el [ejemplo del piloto](pilot/viewer-example.json).

- [Manifiesto congelado](pilot/manifest.prepilot.json) y [reproducción de los objetos](dataset/README.md)
- [Visión y pesos](vision/README.md), [matriz de similitud visual](vision/similarity_16.png)
- [Trazabilidad robótica](robot/TRACEABILITY.md), [resultados nativos](robot/native_dev_results.json)
- [Protocolo](PROTOCOL.md), [comparadores](bo/IMPLEMENTATION.md), [resultados del piloto inicial](bo/results/smoke/summary.csv)
- [Auditoría interna](audit/REVIEW.md), [evaluación de novedad](NOVELTY_ASSESSMENT.md)
