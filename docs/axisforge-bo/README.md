# AxisForgeBO

AxisForgeBO selecciona candidatos para una función costosa que evalúa otro proceso, una biblioteca o una persona. El núcleo funciona en CPU con NumPy y SciPy. PyTorch, BoTorch, GPyTorch y NumPyro aportan modelos y adquisiciones opcionales. El proyecto Rust y el paquete Python tienen ciclos de compilación independientes.

## Instalación

Desde la raíz del repositorio con Python 3.11 o posterior:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python examples/ask_tell.py
```

Para reproducir **el conjunto completo validado en Linux x86_64, CPython 3.12, CPU**, ejecuta los comandos siguientes en ese entorno virtual. El fichero de restricciones fija las dependencias de ejecución y prueba. El entorno virtual debe tener acceso a los índices de PyTorch y PyPI.

```sh
constraints=docs/axisforge-bo/constraints-linux-cpu-py312.txt
python -m pip install 'torch==2.9.1+cpu' -c "$constraints" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python -m pip install -c "$constraints" -e '.[test,research,mcmc]'
python -m pip check
```

## Evaluación externa, contexto y reanudación

```python
from axisforge_bo import BayesianOptimizer, Box, ExactGP, StationaryKernel

opt = BayesianOptimizer(
    Box([(-2, 2)]), ExactGP(StationaryKernel(length_scale=0.25)),
    context_dimension=1, seed=7, max_evaluations=20,
)
job = opt.ask(context=[0.3])
loss = (job.x[0] - job.context[0]) ** 2  # Sustituir por el evaluador.
opt.tell(job, loss)
mean, latent_variance = opt.predict([0.1], context=[0.3])
best_observation = opt.best(context=[0.3])
```

- `ask` reserva un identificador y fija el contexto. `tell` consume esa reserva. `fail(job, status, detail)` registra un fallo terminal e impide convertirlo accidentalmente en una medición. `add_observation` incorpora datos históricos con una decisión y un contexto conocidos. El presupuesto incluye evaluaciones importadas, fallidas y reservadas.
- El sentido por defecto es minimizar. `maximize=True` cambia el sentido público conservando las unidades de la observación. La predicción devuelve **varianza latente**. Para compararla con una medición ruidosa hay que añadir su varianza de observación. El GP nativo tiene media a priori cero y parámetros fijos; para observaciones con una escala muy grande, transforma los objetivos con estadísticas de entrenamiento o utiliza el backend con estandarización.
- Para varias evaluaciones simultáneas usa `ask_batch(count, context)` y devuelve los resultados en cualquier orden. El optimizador secuencial aplica la heurística *constant liar* a reservas pendientes cuando dispone de observaciones de ese contexto. El backend opcional permite puntuar lotes completos con adquisiciones conjuntas. Las reservas y el bloqueo corresponden a un objeto Python dentro de un proceso.
- `save_checkpoint(path)` conserva las reservas, observaciones, estados terminales, configuración y estado aleatorio mediante sustitución atómica. `BayesianOptimizer.load_checkpoint(path)` restaura el flujo. La reconstrucción cubre el GP nativo con kernel estacionario. Los callbacks y modelos arbitrarios requieren un protocolo específico. Un plugin de adquisición debe registrarse antes de restaurar un checkpoint que lo use, con la misma semántica y versión.
- `MixedSpace([Real(...), Integer(...), Categorical(...)])` valida entradas y produce características numéricas normalizadas y categorías one-hot. Las propuestas del optimizador conservan los valores de decisión originales. La búsqueda explora un conjunto finito de candidatos; su tamaño y el método de inicialización afectan al resultado. El contexto se aporta ya escalado por el investigador.
- Un surrogate intercambiable implementa `fit(X, y)` y `predict(X)`, con una fila por observación y dos vectores de salida (media y varianza). Los features internos son decisión codificada seguida de contexto fijo. Para una adquisición personalizada, `register_acquisition(name, function, direction)` registra una función vectorizada de media, varianza e incumbente en la convención interna de minimización. El sentido de su puntuación es `"min"` o `"max"`.

## Capacidades y alcance científico

| Área | Uso directo | Alcance de la evidencia |
|---|---|---|
| Posterior y kernels | `ExactGP`, RBF, Matérn 1/2, 3/2, 5/2, racional cuadrático, periódico, lineal, ARD y composición | Contrastes analíticos y scikit-learn; diagonal de ruido escalar o conocida por observación en fit directo. El optimizador con reservas requiere ruido escalar para construir observaciones ficticias. |
| Adquisiciones escalares | EI, PI, logEI, límites de confianza y Thompson sobre un posterior conjunto | El límite inferior para pérdidas proporciona el criterio de exploración superior tras invertir una utilidad. EI con el mejor valor observado es heurístico bajo ruido; `qlog_nei` modela explícitamente el incumbente incierto. |
| BoTorch | `TorchExperiment(X, utilities, variances=...).scores(batches, method=..., pending=..., reference=...)` | Entradas unitarias. **Todas** las salidas son utilidades a maximizar; transformar también la referencia de hipervolumen. Opciones `log_ei`, `qlog_ei`, `qlog_nei`, `qlog_ehvi`, `qlog_nehvi`. El resultado puntúa las alternativas suministradas en su orden original. `information_scores` ofrece MES/PES con conjunto de representaciones finito. |
| Modelos alternativos | `ResearchGP("variational")` o `ResearchGP("multitask")`; `SaasGP` | GP variacional con puntos inductores, GP multitarea con columna de identificador y SAAS con NUTS. El test SAAS comprueba la varianza de una mezcla; ocho extracciones no establecen mezcla de cadenas ni convergencia para un estudio. El multioutput de `TorchExperiment` modela las salidas de manera independiente. |
| DKL y memoria | `ContextMemory`, `DeepKernelGP`, `MultimodalDeepKernelGP`, `task_meta_gradient`, `meta_step` | Entradas externas como embeddings de 384 dimensiones, ramas entrenables/congeladas, GP exacto y adaptación por tarea. La prueba del hipergradiente reajusta el problema interno con cada perturbación. Las variantes lineal y MLP no lineal tienen ajuste GP por tarea y metagradientes probados mediante diferencias finitas; representan instancias específicas del procedimiento. [Ejemplo ejecutable](../../examples/multimodal_bo.py) |
| Restricciones | `probability_feasible`, `constrained_improvement`, `certified_candidates` | Las probabilidades se multiplican bajo independencia; las cotas del conjunto certificado requieren cotas simultáneas válidas y constantes Lipschitz suministradas. Una probabilidad pequeña de violación por sí sola no garantiza seguridad. |
| Riesgo y varias soluciones | `upper_cvar`, `unscented_improvement`, `diverse_solutions` | CVaR empírico con masa fraccional, EI de la respuesta media bajo perturbaciones con puntos sigma y archivo diverso por distancia. La métrica, el radio y las unidades los define el experimento. |
| Varios objetivos y fidelidades | Pareto, `hypervolume_2d`, métodos conjuntos de BoTorch, GP multifidelidad, `discrete_knowledge_gradient` | La KG usa covarianzas cruzadas y coste de observación, con recomendaciones de fidelidad alta finitas. La aproximación usa cuadratura de Gauss–Hermite; no integra sobre un dominio continuo de recomendaciones. |
| Alta dimensión | `LocalBayesianOptimizer`, `SaasGP` | Región local, Thompson correlacionado y reinicio; estudio corto en Sphere 10D. Es una variante local de investigación. TuRBO completo, REMBO y subespacios activos no tienen implementación propia validada aquí. |
| GP profundo y ruido heteroscedástico | `deep_gp.DeepVariationalGP`, `heteroscedastic.ReplicatedNoiseGP` | Dos capas estocásticas con ELBO variacional y ley de la varianza total; estimación en dos etapas de ruido desconocido con réplicas por entrada. El método con réplicas devuelve tres vectores: media latente, varianza latente y varianza de una observación individual. |
| Transferencia selectiva | `archive.TaskArchive`, `context.ContextMemory` | Registros por tarea, selección por distancia en embeddings, grupos permitidos y tiempos de disponibilidad estrictamente anteriores. El veto opcional usa pérdidas emparejadas de validación aportadas por el investigador. Guardar procedencia y asumir explícitamente cómo se obtuvieron esas etiquetas. |
| Extensiones pendientes | Reproducción histórica y validación en acelerador | El código y los índices originales del TFM son necesarios para distinguir varias interpretaciones. La evidencia ejecutada corresponde a CPU. |

## Integración con AxisForge3D

Compila el servicio WASM con la herramienta fijada por el repositorio e instala Node en el PATH. Desde la raíz:

```sh
rustup target add wasm32-unknown-unknown
cargo build --locked -p axisforge-wasm --target wasm32-unknown-unknown --profile wasm-release
python -m pytest tests/test_axisforge_service.py
```

`WasmService` recibe la ruta al binario y un timeout. El proceso Node instancia los bytes cuyo SHA-256 devuelve, inicia una sesión nueva y envía JSON al ABI existente. El adaptador admite descripción, listado de extensiones y `snapshot`. `SnapshotEvaluator.evaluate(candidate)` consume una configuración articular completa y devuelve manipulabilidad, estado, coste y procedencia. `evaluation.record` incorpora el resultado al experimento. La verificación real compara respuestas del servicio existente y examina errores; el evaluador simulado permite probar el protocolo por separado. Para una métrica o escena diferente hace falta un mapeo explícito con sus unidades y restricciones. [Diseño del adaptador](ARCHITECTURE.md)

## Repetir la validación

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m coverage run -m pytest
python -m coverage report
python tools/bo/benchmark.py /tmp/axisforge-scalar
python tools/bo/advanced_benchmark.py /tmp/axisforge-scalar/advanced.jsonl
python tools/bo/profile_gp.py --output /tmp/axisforge-scalar/scaling.json
```

La suite completa incluye pruebas opcionales de BoTorch, PyTorch, NumPyro y del binario WASM. Instala los extras y compila ese binario para ejecutarlas. [Protocolo congelado](BENCHMARK_PROTOCOL.md) · [Resultados y límites](BENCHMARK_REPORT.md) · [Trazabilidad del TFM](DEVELOPMENT_STATE.md) · [Auditoría independiente](AUDIT.md)

[Metagradientes y revisión de Chen et al.](META_RESEARCH.md) · [Evaluación final independiente](FINAL_AUDIT.md)

Las convenciones, hipótesis, complejidad y diferencias entre criterios se detallan en la [guía de adquisiciones](ACQUISITIONS.md).
