# Piloto exploratorio con RGPE e ICM fijo

Estado: implementado, pruebas locales y repetición local superadas; el auditor independiente repitió los 112 registros y las ocho pruebas. M5 continúa sin aprobación científica global. Diseño en `research/methodology/TRANSFER_BASELINES.md` aceptado para implementación por el auditor, condicionado a revisión de código y datos. Configuración fijada antes de ejecutar: `configs/transfer/synthetic_baselines.json`. Semillas 40–47, distintas del piloto previo. Sin física ni embeddings; estas cifras no son resultados confirmatorios. El objetivo usa ruido pero la métrica es **oracle sintética**, best-so-far de la función latente en puntos elegidos, que nunca llega al optimizador. Área es media de 12 valores en escala [-1,1]; valores mayores son mejores. Resultados resumidos de `synthetic_baselines.json`:

| Escenario | Método | Área media | Diferencia media pareada vs target | Perjudicadas / 8 |
|---|---|---:|---:|---:|
| related | random | .7702 | -.0918 | 8 |
| related | sobol | .8238 | -.0383 | 8 |
| related | target | .8621 | 0 | 0 |
| related | context | .8692 | .0072 | 1 |
| related | adaptive | .8694 | .0074 | 0 |
| related | rgpe | .8689 | .0068 | 1 |
| related | icm_fixed | .8681 | .0061 | 3 |
| adversarial | random | .7702 | -.0918 | 8 |
| adversarial | sobol | .8238 | -.0383 | 8 |
| adversarial | target | .8621 | 0 | 0 |
| adversarial | context | .6843 | -.1777 | 8 |
| adversarial | adaptive | .8621 | .0000 | 0 |
| adversarial | rgpe | .8627 | .0007 | 4 |
| adversarial | icm_fixed | .8524 | -.0097 | 5 |

Las diferencias pequeñas y ocho semillas no justifican conclusiones de superioridad. Los dos escenarios comparten por construcción funciones objetivo y semillas; fuentes adversarias invierten su signo. Se comparte una única secuencia Sobol: sus primeros tres puntos inicializan todos los métodos, y Sobol usa el resto de la misma secuencia. Los BO comparten conjunto de candidatos por paso y ruido por orden de evaluación. La fuente completa y los registros por tarea se almacenan en JSON. Ningún método de búsqueda utiliza el valor oracle.

RGPE sigue cálculo de ranking mediante muestras conjuntas fuente y modelos target leave-one-out del [tutorial BoTorch e95ccc2](https://github.com/pytorch/botorch/blob/e95ccc2ab52c3bd0af20ab3fb487198bd4969e5d/tutorials/meta_learning_with_rgpe/meta_learning_with_rgpe.ipynb); 256 muestras normales MC independientes por modelo y prefijo, semilla `seed+40000+n`, desempate primer índice (fuentes antes de target), e independencia entre expertos al sumar. A diferencia del tutorial, usa GP de hiperparámetros fijos, ruido conocido, EI analítica con incumbente observado en vez de qNEI, y peso target 1 con menos de dos datos. Por tanto es una variante inspirada en tutorial y no su reproducción exacta. El GP multitarea ICM usa correlación entre tareas fijada a 0.5; no equivale a B7 aprendido completo. Las fuentes no se filtran dentro de estos dos baselines específicos del piloto: el generador crea las fuentes válidas antes de la tarea; no usar como interfaz de archivo físico.

Tiempo total medio por trayectoria, CPU e hilos BLAS fijados a 1, aproximadamente: target .005 s, adaptive .009 s, RGPE .039 s, ICM .003 s; solo mediciones locales aproximadas. Costes de generación de fuentes excluidos para todos; random/Sobol no ajustan fuentes. Tiempos incluyen entrenamiento e inferencia y no son una evaluación de escalado. Dos ejecuciones locales reprodujeron los 112 registros en todos los campos salvo duración. Ocho pruebas locales cubren componentes, PSD, cálculo EI, temporalidad y fallo de ajuste; no constituyen validación independiente. Comando: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research_grasping.contextual_transfer.baseline_pilot --output /tmp/baselines.json`.
