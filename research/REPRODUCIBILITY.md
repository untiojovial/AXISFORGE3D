# Reproducir los pilotos CPU

Los comandos se ejecutan en Linux desde la raíz de la rama. El wheel incluye la biblioteca general y el espacio de nombres de investigación. Los experimentos versionados, sus fixtures y el CLI Rust se distribuyen en el repositorio. El paquete mínimo instala NumPy y SciPy. Para reproducir el conjunto completo con CPython 3.12 se utilizan las dos listas de restricciones. La receta de CPU requiere acceso a PyPI y al índice de PyTorch.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
constraints=docs/axisforge-bo/constraints-linux-cpu-py312.txt
extra=research/constraints-linux-cpu-py312.txt
python -m pip install -c "$constraints" 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python -m pip install -c "$constraints" -c "$extra" '.[test,research,mcmc,vision,grasping]'
python -m pip check
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/axisforge-matplotlib XDG_CACHE_HOME=/tmp/axisforge-cache
rustup target add wasm32-unknown-unknown
cargo build --locked -p axisforge-cli
cargo build --locked -p axisforge-wasm --target wasm32-unknown-unknown --profile wasm-release
cargo test --workspace --locked
python -I -m pytest --import-mode=importlib -q tests
```

La guía requiere Node en PATH para las pruebas del WASM. Los paquetes de investigación se probaron en CPU x86_64. Las imágenes del TFM, los pesos de DINO, los datasets de terceros y las credenciales se mantienen fuera de Git.

## Reconstrucción, modelos y física

```bash
python tools/research/check_legacy.py "/ruta/al/TAZ-TFM-2025-047 (2).pdf"
python tools/research/check_ur5_mesh.py
python -m research_grasping.contextual_transfer.pilot --output /tmp/transfer.json
python -m research_grasping.contextual_transfer.baseline_pilot --output /tmp/comparadores.json
python -m research_grasping.simulation.regenerate_sensitivity /tmp/sensibilidad.json
python -m research_grasping.simulation.run_primitive_pilot /tmp/primitivas.json.gz
python -m research_grasping.experiments.cube --binary target/debug/axisforge --output /tmp/cubo.json
python -m research_grasping.experiments.plot_cube /tmp/cubo.json /tmp/figuras-cubo
python -m research_grasping.statistics.simulate_design --output /tmp/potencia.json
python -m research_grasping.statistics.simulate_bounded --output /tmp/acotada.json
```

El cubo genera el proyecto a través del CLI, evalúa un candidato de su plan, mide los 18 intentos de las fuentes y ejecuta 120 intentos objetivo. El JSON se reemplaza de forma atómica tras cada decisión. Al reanudar, se valida la identidad del binario, la escena, el plan, la configuración, las versiones y los archivos de implementación. Los datos completados recuperan el mismo prefijo. Un error de infraestructura invalida ese ensayo y el registro conserva el intento fallido. El tiempo de pared depende de la carga. Los pesos de la mezcla se registran antes de revelar cada observación.

Con pesos preentrenados ya autorizados, la entrada offline `python -m research_grasping.representations.extract --help` explica los argumentos para una o varias vistas del **mismo objeto**. Deben suministrarse imágenes RGB, un directorio con un único `model.safetensors` y `config.json`, hashes verificados independientemente y una receta de preprocesado acorde con el checkpoint. La configuración de ejemplo `configs/representations/rgb_pilot.json` es una receta de ensayo. El informe identifica cada vista, su orden y la clave del vector agregado, que se guarda en la caché. Distingue carga, calentamiento e inferencia medida; una lectura de caché no se contabiliza como inferencia. En CUDA la herramienta comprueba disponibilidad y sincroniza el dispositivo. Las pruebas locales de arquitectura usan pesos aleatorios diminutos. Para comparar CPU/GPU hay que usar los mismos pesos, imágenes, receta, versiones y cachés separadas, registrar tiempo y memoria, y contrastar tolerancias del resultado.

## Evidencia de repetición

El merge de referencia se comprobó con una rueda reconstruida e instalada en un entorno aislado existente. Se creó también un entorno virtual nuevo para el piloto MuJoCo con versiones fijadas de NumPy y SciPy. Se reprodujeron las trayectorias escalares de los dos pilotos de transferencia (96 y 112 registros), las posiciones y respuestas de las 138 evaluaciones del cubo y las seis trazas de sensibilidad. Las comparaciones numéricas excluyeron tiempos de ejecución, versiones de artefactos y metadatos de procedencia cuando diferían. Los recuentos de trayectorias de fase I siguen documentados por separado. [Integración](GIT_INTEGRATION.md) · [Revisión independiente](audit/PHASE_II_INITIAL_REVIEW.md)

La fase II no incorpora datos de test reservados. Para una réplica exacta del TFM siguen faltando los activos y configuraciones originales. La red del entorno restringe el acceso a varias editoriales, repositorios académicos y servidores de pesos. Se guardó una propuesta de nueve dominios científicos en el borrador de configuración para revisión del propietario. Guardar el borrador no activa el acceso ni concede derechos sobre DINOv3.

## Verificación final desde instalación limpia

Con el código de `1c076c2` se construyó la rueda 0.2.0 (SHA-256 `9e657b51dfee5c3d1edea28d3addfef3f2f6001a7d6fe9918d653e3b031f69b5`). Se creó un entorno virtual vacío, se instalaron PyTorch CPU y todos los extras desde la rueda con las restricciones anteriores y se ejecutó `pip check`. El intérprete aislado con `pytest --import-mode=importlib` ejecutó **196 pruebas, cero fallos, cero omisiones y dos avisos**. El código Rust es el del merge probado con **353 pruebas**. La cobertura de pytest en el paquete general fue de 1.948/2.045 sentencias y 688/776 ramas (95,26 % y 88,66 %). En el paquete nuevo de investigación fue de 615/944 y 155/268 (65,15 % y 57,84 %). La cobertura informada no incorpora la ejecución separada de los experimentos. La máquina tenía una cuota de cuatro CPU y 32 GiB de memoria. No se ejecutó ninguna prueba en GPU. [Registro de verificación](results/validation.json)

También se reejecutaron con el wheel final instalado en ese entorno vacío los dos bancos de simulación estadística. Coinciden **666/666 filas** del diseño de potencia y **20/20 filas** de la alternativa acotada, además de la configuración y metadatos de versión. Estas repeticiones verifican el cálculo con los supuestos del generador. La potencia para agarres de objetos nuevos continúa sin estimar.
