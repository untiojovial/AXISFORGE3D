# Bloqueos y condiciones para resolverlos

| Área | Evidencia de esta ejecución | Qué permitiría cerrar la cuestión |
|---|---|---|
| DINOv2 preentrenado | **Resuelto.** Desde otra máquina, la API oficial respondió: checkpoint `facebook/dinov2-small` descargado con revisión y hashes fijados, extracción del panel en CPU reproducida byte a byte y notebook verificado de principio a fin. [Resultados](embedding_experiment/dinov2/README.md) | — |
| DINOv3 preentrenado | **Resuelto.** Se usa la copia que timm redistribuye bajo la licencia DINOv3 (aceptada por el usuario), con revisión y hashes fijados. Panel ejecutado y reproducido byte a byte; usado en el prior de agarre. El checkpoint de Meta en Hugging Face sigue pidiendo aprobación manual. [Resultados](embedding_experiment/dinov3/README.md) | — |
| GPU y Colab | Este entorno solo tiene CPU. Se verificaron celdas sin red, extracción real local y dos ejecuciones idénticas de OpenCLIP. | Ejecutar el notebook en un runtime real de Colab o el runner en CUDA y comparar procedencia, dimensiones y tolerancias contra CPU. |
| Recuperación de F1 | Apertura compatible más frecuente y pocos éxitos. La validación emparejada siguió concentrada en pocos objetos. | Comprobar en un conjunto de desarrollo adicional si predictores de alcance/colisión o nuevos generadores conservan su ventaja y diversidad. |
| Transferencia y calibración | La puerta casi nunca cambia decisiones. Seis objetos de validación y muchos empates. DKL recalibrado ensancha los intervalos. | Mejor memoria positiva, decisión adicional que tenga valor fuera de train y más objetos independientes para evaluar incertidumbre. |
| Física y multifidelidad | Sensibilidad a proxies, penetración y controles numéricamente inválidos. El UR5 articulado se ha evaluado en seguimiento libre. | Validar geometría de colisión, contacto, masa, centro de gravedad y dinámica de la mano y brazo ensamblados. Repetir sobre un panel con suficientes positivos fiables. |
| M10 y manuscrito | La unidad independiente es el objeto. Se desconoce la potencia para un efecto relevante y faltan validaciones. | Fijar población, tamaño, métricas y umbrales, inmovilizar artefactos y obtener dictamen independiente. Ocho objetos de test siguen reservados. |

[Acta de revisión](audit/REVIEW.md) · [Fuentes y acceso a modelos](vision/README.md) · [Borrador metodológico](methodology/CONFIRMATORY_DRAFT.md)
