# Plan de desarrollo y puertas de decisión

Se trabaja sobre los 22 objetos autorizados. Los seis objetivos de desarrollo ya influyeron en decisiones de fases anteriores. Cada comparación mantiene el mismo catálogo y semillas entre los métodos que compara. Se contabilizan evaluaciones históricas, reutilizadas y nuevas. F1, ε analítica y estabilidad dinámica son tres resultados distintos.

1. Recontar las trazas históricas. Bloquear selección de candidatos y presupuesto antes de medir el nuevo archivo, comparar catálogo previo y estratificado. La fase IV partió del protocolo `aedb2c8` y fijó los IDs en `fbe8fc8`.
2. Verificar imágenes y calibración, congelar pesos y comprobar la integración de cada modelo en su propia geometría de tokens. Ejecutar el mismo trabajo portable en CPU o GPU y validar los resultados recibidos.
3. Separar factibilidad de calidad y medir transferencia con presupuestos de memoria comunes. Analizar calibración de DKL y clasificadores por objeto. La cobertura nominal, el ancho de los intervalos y su coste predictivo se informan conjuntamente.
4. Ensayar estabilidad en simulación con controles físicos. Examinar si F1 y F2 justifican una cascada de decisión respecto a F3, con desglose de casos y coste.
5. Revisión independiente y corrección. El auditor controla la autorización del test final. Una fase confirmatoria requiere población y tamaño defendibles, métrica, pérdidas, umbrales y código inmutables aprobados antes del primer resultado final.

Se reactivaron los seis agentes especialistas reales de la sesión anterior en sus funciones de optimización, visión, robótica, auditoría independiente, metodología y datos/automatización. El PI de la sesión coordinó el trabajo. El selector del modelo del PI no está expuesto en esta ejecución. Las decisiones y objeciones están en [el registro del auditor](audit/REVIEW.md).
