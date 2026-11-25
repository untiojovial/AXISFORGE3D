# Pruebas y cobertura

**502 pruebas Python y 355 pruebas Rust pasaron.** Se ejecutó pytest con los activos de desarrollo reales, el checkpoint OpenCLIP y el oráculo nativo. La suite Rust del workspace se ejecutó con Cargo.lock fijado. Todas las pruebas del conjunto medido terminaron, sin omisiones. Se verificó también el formato Rust.

| Ámbito de medición Python | Líneas ejecutables | Ramas |
|---|---:|---:|
| Biblioteca axisforge_bo | 1948/2045 (95.26 %) | 688/776 (88.66 %) |
| Paquete de investigación fase III completo | 1760/1876 (93.82 %) | 588/674 (87.24 %) |
| Paquete de investigación fase IV completo | 2375/2465 (96.35 %) | 820/960 (85.42 %) |
| Todo research_grasping (incluye III y IV) | 4849/5397 (89.85 %) | 1582/1928 (82.05 %) |

El objetivo solicitado es 90 % en líneas y 85 % en ramas. Los paquetes completos de fase III y fase IV alcanzan ambos umbrales en esta suite.

La medición usa coverage.py 7.17.5 con ramas e instrumentación de subprocesos creados por las pruebas. Cuenta el código ejecutado por pytest, incluidos sus entrypoints. Las simulaciones y entrenamientos ejecutados individualmente fuera de las pruebas no se suman al numerador. Se dan paquetes completos para evitar que unos pocos módulos bien cubiertos oculten el resto. Las carpetas de resultados, notebooks, scripts de herramientas y crates Rust quedan fuera de este denominador Python. Los recuentos de las pruebas Rust se presentan aparte. [Numeradores y denominadores por archivo](coverage.json) · [Comandos](../REPRODUCIBILITY.md)

En la referencia de fase III se habían medido 62,74 % de líneas y 50,59 % de ramas para ese paquete. La medición nueva incorpora más regresiones con activos reales y cobertura de subprocess. También cambió el conjunto de pruebas, de modo que la diferencia no aísla el efecto de un solo cambio.

Quedaron registradas tres advertencias de pytest en este entorno. Una proviene del plugin anyio, otra de la compilación opcional de la extensión qLogEHVI de BoTorch en una ruta de solo lectura y otra de un caso de validación de estandarización. MuJoCo también emite un aviso QVEL durante la prueba que introduce deliberadamente un estado inválido. Los 57 controles de fricción cero numéricamente inválidos en los datos de investigación son un hallazgo distinto del estado de la suite.

Las pruebas de Colab reproducen el notebook con CPU y con un sustituto de la interfaz de descarga. Faltan validación externa sobre Colab/CUDA e inferencia de DINO con pesos oficiales. La cobertura no certifica fidelidad física ni superioridad científica.

[Salida de pytest](pytest-output.txt) · [Salida de Cargo](cargo-test-output.txt) · [Resumen y hashes](suite_summary.json)
