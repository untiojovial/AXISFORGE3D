# Reproducción de la fase IV

**Entrada rápida.** El notebook incluye 352 vistas selladas y el código que las procesa. El ZIP de embeddings se puede importar y analizar sin acceder a mallas ni volver a pagar consultas nativas. Para reproducir las figuras de visión usando el resultado que acompaña a la rama basta con Python, NumPy, scikit-learn y Matplotlib. [Notebook](../../notebooks/AxisForge_Phase4_Frozen_Foundation_Models.ipynb) · [Hashes y límites de la prueba local](automation/notebook_cpu.json)

```bash
python -m research_grasping.phase4.embedding_report --analysis research/phase4/embedding_experiment/openclip_analysis.json --out /tmp/axisforge-visual-report
PYTHONPATH=. python tools/research/phase4/import_embedding_bundle.py --job research/phase4/embedding_experiment/development-images.zip --bundle /ruta/bundle.zip --destination /ruta/modelo-importado --analysis-out /ruta/analisis.json
```

Para DINO, añadir `--expected-revision SHA_GIT --expected-checkpoint-sha256 SHA_PESOS --expected-processor-sha256 SHA_PROCESADOR` cotejados con la fuente oficial. La importación reconstruye las características por candidato y ejecuta las 360 trayectorias si están las 22 piezas. Una extracción incompleta se identifica como tal. OpenCLIP comprueba una revisión y digests fijos. El modo CPU/GPU se elige con `--device cpu` o `--device cuda` en el runner. El notebook detecta CUDA automáticamente. [Descarga y licencias de los modelos](vision/README.md) · [Contrato de trabajos, reanudación y bundles](automation/README.md)

## Entorno y archivos externos

En esta ejecución se usaron Python 3.12 sobre Linux x86-64, PyTorch 2.9.1+cpu, torchvision 0.24.1+cpu, OpenCLIP 4.2.2, Transformers 4.57.3, MuJoCo 3.3.7, NumPy 2.3.5, SciPy 1.17.0 y scikit-learn 1.8.0. Las versiones del entorno científico están en [constraints-cpu.txt](verification/constraints-cpu.txt). En otro equipo la versión CUDA de PyTorch debe corresponder a su driver. No se contrató infraestructura.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu' 'torchvision==0.24.1+cpu'
python -m pip install -c research/phase4/verification/constraints-cpu.txt -e '.[phase3,research,test]' 'transformers==4.57.3' 'open_clip_torch==4.2.2'
```

El catálogo original, las mallas, los proxies del proveedor y el oráculo fijado de fase III están fuera de Git. La procedencia de las 22 piezas y su atribución están en la ficha de datos. Para recuperar los activos de desarrollo, usar el convertidor separado trimesh 5.2.2 y `python -m research_grasping.phase3.dataset_reproduce --root "$D" --splits train validation`. Recrear catálogos y vistas mediante las recetas de fase III. Para reanudar exactamente una corrida se necesitan los binarios, los proyectos nativos originales y sus rutas, cuyos hashes están en los locks. La regeneración puede cambiar UUID, hashes y tiempos. Una ejecución nueva debe utilizar un directorio nuevo. [Datos y licencia CC BY 4.0](../phase3/dataset/README.md) · [Reconstrucción previa y proxies](../phase3/REPRODUCIBILITY.md)

Se usaron `D=/workspace/axisforge-phase3-data` y `W=/workspace/axisforge-phase4-data`. El binario original para medir F1/F2 en `$W/bin/axisforge-phase3` tiene SHA-256 `138173de8c197df70662ed15ae087e6f8e4c7f66f2b0cbe2d34d0de5241c8c5a`. Compilar el selector de la rama en otro directorio conserva el oráculo original.

```bash
cargo build --locked --release -p axisforge-cli --target-dir "$W/build"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m research_grasping.phase4.workflow feasibility_analysis --data "$D" --work "$W" --binary "$W/bin/axisforge-phase3" --selector "$W/build/release/axisforge" --jobs 2
```

## Grafo de experimentos

[workflow.json](automation/workflow.json) registra los nodos y sus dependencias. Los modos públicos son `development`, `foundation_model_comparison`, `feasibility_analysis`, `transfer_diagnostics`, `dkl_calibration`, `physical_validation`, `multi_fidelity_pilot`, `ablations` y `confirmatory`. Este último está bloqueado mientras M10 siga sin aprobación. Los límites de concurrencia, hilos y segundos por nodo son explícitos. El estado y los tiempos se escriben en `$W/workflow-state`. Para una prueba corta, usar `development --only reference`. Un segundo lanzamiento comprueba los hashes y devuelve `REUSED`. La planificación no autoriza por sí sola una evaluación confirmatoria.

El perfil de modelos genera por defecto la comparación observada de OpenCLIP. Para incluir DINO autorizado, pasar `--model-registry /ruta/oficial.json` y, en un equipo compatible, `--device cuda`. El JSON tiene una entrada por `dinov2` o `dinov3`, con revisión Git exacta (`revision`) y SHA-256 independientes (`weights`, `config`, `processor`). Cada descarga oficial, extracción congelada, importación y análisis se ejecuta en un directorio por modelo. La tabla verifica que todos usen el mismo trabajo de imágenes, panel y control geométrico. El redibujado del dataset y la preparación del ZIP tienen nodos individuales `development_assets` y `prepare_image_job`. El primero requiere red y el entorno de conversión fijado. El ZIP ya distribuido permite omitir ambos para los modelos.

El piloto físico completo fija MuJoCo y admite tres shards con comprobación posterior de los 84 pares. El analizador usa únicamente el archivo completo corregido. El módulo de transferencia usa semillas y catálogos fijados y reanuda a partir de la última observación. El resultado anidado V2b se consume solo si coincide con el hash registrado. Los archivos físicos y resultados de BO guardan costes por evaluación. La caché compartida afecta la asignación de tiempo fresco a cada método. [Protocolos físicos](robotics/PHYSICAL_PROTOCOL.md) · [Memoria y políticas](transfer/PILOT.md) · [Informe de calidad](verification/COVERAGE.md)

## Verificar el código y medir la cobertura

La suite usa activos locales reales cuando encuentra las variables siguientes. El entorno de esta corrida ejecutó los casos externos. Si faltan activos, pytest explicita las omisiones. Para computar la cobertura de los entrypoints lanzados como subprocesos se usa coverage.py 7.17.5 con su parche de subprocess. El directorio de cobertura debe estar vacío al comenzar cada medición.

```bash
export AXISFORGE_OPENCLIP_CHECKPOINT="$D/weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt"
export AXISFORGE_PHASE3_DATA="$D" AXISFORGE_PHASE3_GSO=/ruta/mujoco_scanned_objects AXISFORGE_PHASE4_DATA="$W"
export COVERAGE_RCFILE="$PWD/research/phase4/verification/coverage.ini"
covdir=$(mktemp -d)
export COVERAGE_FILE="$covdir/.coverage"
python -m coverage run -m pytest -q
python -m coverage combine "$covdir"
python -m coverage json -o "$covdir/raw.json"
python tools/research/phase4/summarize_coverage.py "$covdir/raw.json" --out "$covdir/coverage.json"
cargo test --workspace --locked
```
