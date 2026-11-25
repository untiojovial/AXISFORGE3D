# Regresión y reproducibilidad · 25 de noviembre de 2026

Se instaló el wheel de AxisForgeBO 0.1.0 en un entorno virtual de CPython 3.12 sin acceso a los paquetes de la instalación principal. `pip check` terminó correctamente. El wheel ensayado tenía SHA-256 `55e96a129a73965ca58ec461123fd0a432f2c7d822edad6b7149ede98c979095`; reconstruir con otros metadatos o marcas de tiempo puede cambiar los bytes. Los 26 archivos Python y JavaScript distribuidos coincidieron por SHA-256 con el árbol fuente ensayado. [Manifiesto de fuentes y pruebas](results/sources.sha256) · [Versiones de dependencias](constraints-linux-cpu-py312.txt)

| Comprobación | Resultado medido |
|---|---|
| Python en modo aislado (`-I`), pytest con `--import-mode=importlib`, cobertura medida sobre site-packages | 163 aprobadas, 0 fallidas, 0 omitidas |
| Líneas ejecutables de todo el paquete Python | 1948 / 2045 = 95.26 % |
| Ramas del paquete Python | 687 / 776 = 88.53 % |
| Regresión de AxisForge3D, `cargo test --workspace --locked --offline -j 4` | 337 aprobadas, 0 fallidas, 0 ignoradas |
| Casos escalares, segunda ejecución con la instalación congelada | 960 / 960 registros con los mismos puntos, pérdidas, diagnósticos y métricas. Se exceptúan tiempo y memoria. La tabla de diferencias y sus intervalos se reprodujo byte por byte. |
| Casos avanzados confirmatorios, segunda ejecución | 330 / 330 registros con datos científicos idénticos, excluyendo tiempo |
| Historial contextual | 180 combinaciones repetidas; resultados por semilla, protocolo y huellas se conservan en experiments/contextual_transfer_results |
| Ejemplos | Se ejecutaron ask_tell.py y multimodal_bo.py con la instalación aislada |
| Integración | Pruebas contra el WASM de AxisForge3D y su proceso Node, incluidas respuesta válida, configuración articular inválida y rutas de fallo |

Los avisos de la ejecución aislada se debieron a la caché no escribible de una extensión C++ opcional de qLogEHVI (se usó el cálculo Python de BoTorch) y a la estandarización de una salida constante en una prueba. La ejecución corresponde a CPU. El flujo de GitHub Actions queda configurado en el repositorio; la ejecución remota de CI no se verificó en esta sesión.

Para repetir la comprobación aislada, construye un wheel con `python -m pip wheel --no-deps . -w /tmp/axisforge-dist`, instálalo con sus extras en un entorno limpio con las restricciones de la guía y ejecuta desde la raíz del repositorio `python -I -m pytest --import-mode=importlib tests`. Para comprobar los archivos del árbol ejecuta `sha256sum -c docs/axisforge-bo/results/sources.sha256`. Disponibilidad del binario WASM, Node y extras científicos determinan qué pruebas pueden ejecutarse en una máquina nueva.

La cobertura cuantifica código ejecutado; los resultados por algoritmo, sus hipótesis y el alcance de la revisión independiente se encuentran en el [informe de benchmarks](BENCHMARK_REPORT.md) y la [auditoría global](FINAL_AUDIT.md).
