# Reproducción de la fase III

Los resultados corresponden a CPU Linux, Python 3.12, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.9.1+cpu, torchvision 0.24.1+cpu, open_clip_torch 4.2.2, trimesh 4.8.3, MuJoCo 3.3.7 y scikit-learn 1.8.0. El exportador original de STL utilizó trimesh 5.2.2 en un entorno separado. Las comprobaciones distinguen el hash de los datos, la identidad del binario y la configuración del experimento.

Desde la raíz del repositorio, en un entorno Python para investigación y con Rust instalado:

```bash
cargo build --locked --release -p axisforge-cli
cargo test --workspace --locked
python -m pip install -e '.[phase3,research,test]'
python -m research_grasping.phase3.fetch_openclip /ruta/datos/weights
```

La preparación inicial descarga exclusivamente los objetos del manifiesto. Ejecútese desde el mismo checkout. Un virtualenv independiente evita que el exportador cambie la versión de trimesh usada por percepción.

```bash
python -m venv /tmp/axisforge-converter
/tmp/axisforge-converter/bin/python -m pip install trimesh==5.2.2 numpy==2.3.5
/tmp/axisforge-converter/bin/python -m research_grasping.phase3.dataset_reproduce --root /ruta/datos
bash research/phase3/pilot/reproduce.sh /ruta/datos
```

El script ejecuta 16 vistas por objeto, genera hasta 48 poses por catálogo de desarrollo, crea el archivo histórico de 160 evaluaciones y compara nueve políticas. Las ocho identidades test solo intervienen en la comprobación del manifiesto. El bloqueo rechaza cambios de malla, proyecto, binario, embeddings e historia al reanudar. AxisForge genera identificadores UUID al crear un nuevo proyecto. Las reconstrucciones independientes pueden tener hashes de proyecto y tiempos distintos. Una reanudación debe usar los mismos artefactos de la ejecución original.

Para reproducir los análisis adicionales, con `D` como directorio externo y `P` como directorio del protocolo:

```bash
D=/ruta/datos
P=research/phase3/pilot
common=(--data "$D" --catalogues "$D/pilot-expanded/catalogs" --protocol "$P" --binary target/release/axisforge --vision "$D/vision-development-v3" --run "$D/run-development")
python -m research_grasping.phase3.contact_similarity "${common[@]}" --out "$D/contact_similarity.json"
python -m research_grasping.phase3.representation_gp "${common[@]}" --out "$D/representation_gp.json"
python -m research_grasping.phase3.source_appearance --root "$D" --vision "$D/vision-development-v3" --checkpoint "$D/weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt" --out "$D/appearance-development" --summary "$D/appearance_summary.json"
git clone https://github.com/kevinzakka/mujoco_scanned_objects "$D/gso-source"
git -C "$D/gso-source" checkout 6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c
python -m research_grasping.phase3.gso_contact_probe --catalogues "$D/pilot-expanded/catalogs" --run "$D/run-development" --source "$D/gso-source" --binary target/release/axisforge --data "$D" --protocol "$P" --out "$D/contact_probe.json"
AXISFORGE_PHASE3_DATA="$D" AXISFORGE_PHASE3_GSO="$D/gso-source" python -m pytest -q
```

El visor se compila con `bash apps/viewer/build.sh /tmp/axisforge.html` y el target Rust `wasm32-unknown-unknown`. En **Experimentos de agarre**, cargar el JSON de ejemplo de esta rama o el informe completo generado en `run-development/viewer.json`. El CSV exporta las observaciones, predicciones anteriores a la consulta, costes e identidad del método.

Las trazas comprimidas guardan los registros de cada método y el historial compartido. Los tiempos de ejecución de una consulta nueva y los costes imputados a la repetición de esa consulta quedan identificados. Medir rendimiento del optimizador requiere además contabilizar preparación, representación y ajuste de modelos. Los pesos, mallas, paquetes y cachés completos se mantienen en almacenamiento externo. Sus datos de origen, versión y licencias están enlazados en el manifiesto y en las notas de visión.

### Paneles adicionales

Los resultados también se publican en los directorios `development-20`, `visual`, `geometry` y `functional`. Para reusar exclusivamente el coste físico determinista entre métodos, el runner admite `--cache-dir "$D/run-development/cache"`. Cada política sigue pagando nominalmente la evaluación elegida. Con el mismo listado de 22 catálogos del script principal, se reproduce el presupuesto 20 pasando `--budget 20 --history-budget 10`. Los paneles visual y geométrico usan respectivamente `--context visual --methods visual_fixed adaptive` y `--context geometry --methods adaptive`.

Para reconstruir las 33 características por candidato:

```bash
python -m research_grasping.phase3.functional_catalogue --data "$D" --catalogues "$D/pilot-expanded/catalogs" --protocol "$P" --binary target/release/axisforge --vision "$D/vision-development-v3" --out "$D/functional-catalogs"
```

El panel funcional emplea únicamente los 22 JSON de objeto de esa carpeta, con diez consultas y `--methods target_local equal no_evidence adaptive`. `pca.json` es el artefacto de transformación. El runner verifica el lote exacto usado en esta investigación. Una regeneración con UUID de proyectos o entorno de álgebra lineal distintos requiere una nueva revisión y un lock propio para esa ejecución. La verificación exacta y las diferencias por BLAS se registran en el informe de resultados.

## Verificación de entrega

Pasaron 261 pruebas Python con los datos de desarrollo y el checkout GSO, incluidas las pruebas de reanudación, integridad, aislamiento del test y marcos físicos. Pasaron 354 pruebas Rust en el workspace. También se construyó el wheel y se compiló el visor WASM; Chromium leyó una trayectoria real y exportó correctamente las diez observaciones a CSV.

| Área | Líneas ejecutadas por pruebas | Ramas ejecutadas por pruebas |
|---|---:|---:|
| Núcleo AxisForgeBO | 95.26 % | 88.66 % |
| Módulos nuevos de fase III | 62.74 % | 50.59 % |

El núcleo supera el objetivo de 90 % de líneas y 85 % de ramas. La cobertura de los módulos nuevos sigue por debajo de ese objetivo. La métrica corresponde a pytest con coverage y no suma la ejecución de los pilotos como si fueran pruebas adicionales.
