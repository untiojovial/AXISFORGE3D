# Resultados de benchmarks · CPU

Ocho funciones escalares, cuatro métodos y 30 semillas emparejadas produjeron 960 trayectorias de 16 evaluaciones. Los cuatro puntos iniciales coinciden dentro de cada pareja. Las dos implementaciones de EI recibieron las mismas alternativas por paso. Las diferencias de regret son nativo menos comparador; un valor negativo favorece a EI. Los corchetes indican intervalos bootstrap percentiles del 95 % para la **media de la diferencia emparejada**, sin corrección por comparaciones múltiples.

| Función | Regret final medio EI | Diferencia frente a aleatorio [IC] | Diferencia frente a Sobol [IC] | RMSE predictivo | NLL gaussiana media | Cobertura marginal 95 % |
|---|---:|---:|---:|---:|---:|---:|
| branin | 0.7382 | -1.73 [-2.67, -0.854] | -3.62 [-4.96, -2.37] | 28.39 | 5.099 | 70.3% |
| hartmann3 | 0.2428 | -0.713 [-0.908, -0.535] | -0.521 [-0.667, -0.388] | 0.7441 | 1.129 | 94.7% |
| hartmann6 | 1.387 | -0.983 [-1.23, -0.729] | -0.7 [-0.927, -0.47] | 0.5876 | 1.022 | 96.8% |
| sphere | 0.5565 | -4.22 [-5.37, -3.12] | -4.5 [-5.84, -3.26] | 10.36 | 3.55 | 95.8% |
| ackley | 10.73 | -5.93 [-7.58, -4.29] | -5.7 [-7.39, -3.85] | 3.49 | 8.356 | 74.8% |
| rosenbrock | 11.71 | -35.7 [-50.4, -23.1] | -29.3 [-44.5, -15.7] | 920 | 11.42 | 78.1% |
| rastrigin | 22.9 | -1.74 [-5.56, 1.9] | -2.33 [-6.62, 2.3] | 32.1 | 12.41 | 56.2% |
| levy | 2.088 | -1.9 [-2.65, -1.22] | -1.57 [-2.52, -0.664] | 25.94 | 6.233 | 71.4% |

Los recorridos de EI nativo y de scikit-learn coinciden para estas 240 combinaciones de semilla y función. Ambas implementaciones usan el mismo modelo especificado; el acuerdo comprueba las operaciones numéricas y la selección de candidatos. En Rastrigin los dos intervalos incluyen cero. La calibración varía por función. Una buena selección de candidatos no asegura por sí sola buenos intervalos predictivos.

Cada línea del fichero crudo conserva coordenadas, valores, curva best-so-far, regret acumulado, primer cruce del umbral, tiempo y pico de memoria Python. Las columnas de predicción se calculan en 32 puntos adicionales de diagnóstico que el optimizador nunca observa. [Trayectorias escalares](results/scalar.jsonl.gz) · [Diferencias emparejadas](results/paired.csv) · [Protocolo](BENCHMARK_PROTOCOL.md)

## Escenarios avanzados

Se conservó la cohorte inicial de cinco semillas. Una segunda cohorte de 30 semillas nuevas repite exactamente los métodos y presupuestos, con 330 registros repartidos entre las siguientes comparaciones. Para restricciones se informa la tasa media de evaluaciones realmente infractoras, incluyendo la inicialización. Se comparan recomendaciones al mismo coste en multifidelidad, aunque el número de evaluaciones difiera.

| Escenario y magnitud | Método frente a comparador | Media método | Media comparador | Diferencia emparejada [IC 95 %] |
|---|---|---:|---:|---:|
| ZDT1 · hipervolumen ↑ | qlog_ehvi / random | 9.726 | 8.319 | 1.407 [1.027, 1.787] |
| ZDT1 · hipervolumen ↑ | qlog_ehvi / sobol | 9.726 | 8.749 | 0.9777 [0.6494, 1.262] |
| DTLZ2 · hipervolumen ↑ | qlog_ehvi / random | 7.588 | 7.289 | 0.2991 [0.1449, 0.4564] |
| DTLZ2 · hipervolumen ↑ | qlog_ehvi / sobol | 7.588 | 7.499 | 0.08934 [-0.02976, 0.2083] |
| Restricción · tasa de violaciones ↓ | feasible_ei / random | 0.2444 | 0.45 | -0.2056 [-0.2528, -0.1611] |
| Fidelidad alta · regret ↓ | kg_per_cost / random | 0.003188 | 0.01861 | -0.01543 [-0.0275, -0.005008] |
| Perturbaciones · CVaR superior 90 % ↓ | decisión robusta / nominal | 0.161 | 0.6117 | -0.4506 [-0.4551, -0.4461] |
| Sphere 10D · regret ↓ | BO local / aleatorio | 11.25 | 40.5 | -29.25 [-33.07, -25.5] |

Las dos series multiobjetivo usan lotes de dos; el área depende de la referencia declarada por problema. En el problema restringido se usa un incumbente observado con ruido y un modelo independiente de factibilidad. Los resultados no constituyen una garantía de seguridad. La KG discreta ajusta un GP conjunto sobre coordenada y fidelidad; el GP específico multifidelidad de BoTorch se comprueba en la suite por separado. CVaR se evalúa con perturbaciones nuevas y un archivo de decisiones espaciadas. La variante de BO local se compara con el mismo presupuesto de 25 evaluaciones. [Cohorte avanzada inicial](results/advanced.jsonl) · [Confirmación](results/advanced_confirmatory.jsonl) · [BO local](results/local_sphere_confirmatory.json)

## Transferencia entre tareas

Treinta semillas, dos familias de historias y tres métodos generaron 180 trayectorias. Cada historia contiene cuatro observaciones previas y el presupuesto de la tarea nueva es de ocho evaluaciones. Para el método con historia frente al arranque en frío, la diferencia media de regret terminal fue:

- related: 6.361e-06 [-8.686e-07, 1.669e-05].
- adversarial: 0.0009126 [8.834e-06, 0.002692].

En este ensayo la transferencia adversaria empeoró el regret medio; en historias relacionadas el intervalo incluye cero. La variante permutada cambia etiquetas por observación dentro de una tarea fuente. La desviación entre semillas, el error predictivo y la incertidumbre posterior se calculan a niveles distintos. [Datos, tiempos y ablación](../../experiments/contextual_transfer_results/raw.csv) · [Resumen](../../experiments/contextual_transfer_results/summary.json)

## Escalado del GP exacto

Medianas de tres procesos independientes, Linux x86_64, f64, seis dimensiones y un hilo BLAS. RSS es el máximo del proceso completo en MiB; incluye intérprete y bibliotecas. El tiempo de predicción corresponde a marginales, y excluye el inicio del proceso.

| Observaciones | Consultas | Ajuste (ms) | Predicción (ms) | RSS pico (MiB) |
|---:|---:|---:|---:|---:|
| 32 | 128 | 0.35 | 0.37 | 61.2 |
| 32 | 1024 | 0.35 | 4.98 | 59.9 |
| 128 | 128 | 1.88 | 1.26 | 61.2 |
| 128 | 1024 | 1.53 | 9.06 | 67.9 |
| 512 | 128 | 22.74 | 5.80 | 72.7 |
| 512 | 1024 | 22.03 | 47.34 | 92.2 |

La factorización exacta crece cúbicamente en el número de observaciones. Estas medidas son específicas de esta máquina; las variantes variacionales y el entrenamiento en GPU requieren estudios de escalado propios. [Mediciones por proceso](results/scaling.json)

Para reproducir la cohorte confirmatoria pasa `--confirmatory` a `tools/bo/advanced_benchmark.py` y a `tools/benchmarks/local_sphere.py`. Para reconstruir este informe desde los registros conservados ejecuta `python tools/bo/summarize.py`.
