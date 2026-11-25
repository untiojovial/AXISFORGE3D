# Bloqueos y condiciones para resolverlos

| Área | Evidencia de esta ejecución | Qué permitiría cerrar la cuestión |
|---|---|---|
| DINOv2 preentrenado | **Resuelto.** Desde otra máquina, la API oficial respondió: checkpoint `facebook/dinov2-small` descargado con revisión y hashes fijados, extracción del panel en CPU reproducida byte a byte y notebook verificado de principio a fin. [Resultados](embedding_experiment/dinov2/README.md) | — |
| DINOv3 preentrenado | El checkpoint requiere que Meta apruebe la cuenta que lo descarga; sin ella, la descarga oficial devuelve 401. La ruta completa (procesador, registros, rejilla 14×14) se probó con un ViT-S/16 de pesos aleatorios. | La persona con acceso aprobado ejecuta el notebook de Colab con su secreto `HF_TOKEN`: devuelve el bundle del panel y, con un `views.zip`, los descriptores para el [prior de agarre](grasp_prior/README.md). También puede ejecutarse aquí si se guarda su token mediante el formulario seguro del entorno. |
| GPU y Colab | Este entorno solo tiene CPU. Se verificaron celdas sin red, extracción real local y dos ejecuciones idénticas de OpenCLIP. | Ejecutar el notebook en un runtime real de Colab o el runner en CUDA y comparar procedencia, dimensiones y tolerancias contra CPU. |
| Recuperación de F1 | Apertura compatible más frecuente y pocos éxitos. La validación emparejada siguió concentrada en pocos objetos. | Comprobar en un conjunto de desarrollo adicional si predictores de alcance/colisión o nuevos generadores conservan su ventaja y diversidad. |
| Transferencia y calibración | La puerta casi nunca cambia decisiones. Seis objetos de validación y muchos empates. DKL recalibrado ensancha los intervalos. | Mejor memoria positiva, decisión adicional que tenga valor fuera de train y más objetos independientes para evaluar incertidumbre. |
| Física y multifidelidad | Sensibilidad a proxies, penetración y controles numéricamente inválidos. El UR5 articulado se ha evaluado en seguimiento libre. | Validar geometría de colisión, contacto, masa, centro de gravedad y dinámica de la mano y brazo ensamblados. Repetir sobre un panel con suficientes positivos fiables. |
| M10 y manuscrito | La unidad independiente es el objeto. Se desconoce la potencia para un efecto relevante y faltan validaciones. | Fijar población, tamaño, métricas y umbrales, inmovilizar artefactos y obtener dictamen independiente. Ocho objetos de test siguen reservados. |

[Acta de revisión](audit/REVIEW.md) · [Fuentes y acceso a modelos](vision/README.md) · [Borrador metodológico](methodology/CONFIRMATORY_DRAFT.md)
