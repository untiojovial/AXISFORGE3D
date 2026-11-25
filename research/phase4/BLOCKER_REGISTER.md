# Bloqueos y condiciones para resolverlos

| Área | Evidencia de esta ejecución | Qué permitiría cerrar la cuestión |
|---|---|---|
| DINOv2 preentrenado | Las rutas oficiales de Hugging Face y Meta devolvieron HTTP 403, incluida la comprobación con red elevada. | Ejecutar el notebook o runner en una máquina que pueda descargar el checkpoint oficial. Importar con revisión y hashes independientes y reproducir sus salidas. |
| DINOv3 preentrenado | El checkpoint requiere acceso de Meta para el usuario que lo descarga. Tampoco hubo acceso de red al recurso. | Persona autorizada con acceso concedido, token privado en el entorno y las mismas comprobaciones. |
| GPU y Colab | Este entorno solo tiene CPU. Se verificaron celdas sin red, extracción real local y dos ejecuciones idénticas de OpenCLIP. | Ejecutar el notebook en un runtime real de Colab o el runner en CUDA y comparar procedencia, dimensiones y tolerancias contra CPU. |
| Recuperación de F1 | Apertura compatible más frecuente y pocos éxitos. La validación emparejada siguió concentrada en pocos objetos. | Comprobar en un conjunto de desarrollo adicional si predictores de alcance/colisión o nuevos generadores conservan su ventaja y diversidad. |
| Transferencia y calibración | La puerta casi nunca cambia decisiones. Seis objetos de validación y muchos empates. DKL recalibrado ensancha los intervalos. | Mejor memoria positiva, decisión adicional que tenga valor fuera de train y más objetos independientes para evaluar incertidumbre. |
| Física y multifidelidad | Sensibilidad a proxies, penetración y controles numéricamente inválidos. El UR5 articulado se ha evaluado en seguimiento libre. | Validar geometría de colisión, contacto, masa, centro de gravedad y dinámica de la mano y brazo ensamblados. Repetir sobre un panel con suficientes positivos fiables. |
| M10 y manuscrito | La unidad independiente es el objeto. Se desconoce la potencia para un efecto relevante y faltan validaciones. | Fijar población, tamaño, métricas y umbrales, inmovilizar artefactos y obtener dictamen independiente. Ocho objetos de test siguen reservados. |

[Acta de revisión](audit/REVIEW.md) · [Fuentes y acceso a modelos](vision/README.md) · [Borrador metodológico](methodology/CONFIRMATORY_DRAFT.md)
