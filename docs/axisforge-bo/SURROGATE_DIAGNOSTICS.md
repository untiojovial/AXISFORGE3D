# Diagnósticos suplementarios del paquete congelado

El auditor pidió evidencia adicional sobre el muestreo SAAS y el coste de la aproximación variacional. Se registró el protocolo antes de ejecutar ambos ensayos. Se usó el mismo wheel que pasó las 163 pruebas, en Linux x86_64 y CPU, con un hilo de BLAS. [Protocolo](BENCHMARK_PROTOCOL.md) · [Ejecutable](../../tools/bo/diagnose_surrogates.py)

## SAAS con muestreo NUTS

Cuatro cadenas independientes utilizaron las mismas 16 observaciones sintéticas en seis dimensiones, 256 iteraciones de calentamiento y 256 extracciones retenidas por cadena. Para las nueve magnitudes monitorizadas, el mayor R-hat dividido fue 1.0151 y el menor tamaño muestral efectivo fue 321.6. Se alcanzaron los criterios preespecificados (R-hat ≤ 1.05, ESS ≥ 100) en **este ajuste**. Los diagnósticos se calcularon sobre las muestras conservadas con NumPyro. La interfaz BoTorch utilizada no devuelve el registro completo por transición NUTS, por lo que estos resultados no cuantifican divergencias ni certifican mezcla en otras tareas. [Extracciones, semillas y estadísticos](results/saas_chains.json)

## GP variacional

Se usaron 16 puntos inductores y 25 pasos de optimización por ajuste, con seis dimensiones y 128 puntos de diagnóstico por caso. Las cifras son medianas de tres procesos independientes. RSS incluye las bibliotecas y el entorno Python. El tamaño del entrenamiento modifica la muestra observada y la solución aprendida; las diferencias de RMSE tampoco demuestran un efecto causal aislado del tamaño.

| Observaciones | Tiempo de ajuste (s) | Predicción (ms) | RSS pico (MiB) | RMSE |
|---:|---:|---:|---:|---:|
| 32 | 0.797 | 1.73 | 513.8 | 0.6687 |
| 128 | 0.798 | 1.84 | 513.1 | 0.5380 |
| 512 | 0.743 | 1.80 | 514.7 | 0.4966 |

[Mediciones por proceso](results/variational_scaling.json). Se verificó el entorno con PyTorch 2.9.1+cpu (0 dispositivos CUDA) y JAX (un dispositivo CPU).
