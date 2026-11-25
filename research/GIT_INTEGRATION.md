# Referencia de integración para la fase II

El 25 de noviembre de 2026 se consultó el remoto por Git y se descargó su punta. El ancestro común era `a422ea335d84c3256317f62827e9d71d49efa932`. Desde ese punto la biblioteca tenía tres commits y el simulador siete. La rama local `integration/axisforge-phase2` integra ambos historiales en el merge `2ab53e0e34ea8adc8f378ed9e8816bfb99a5c769`. Los padres, en orden, son `fa9314f` (simulador) y `eec3c21` (biblioteca). Las referencias locales de respaldo son `backup/phase2-simulator-fa9314f` y `backup/phase2-library-eec3c21`. La rama `work` también conserva el segundo padre.

Git fusionó el README raíz automáticamente. Revisamos su diferencia contra cada padre. Conserva la descripción de los ciclos del simulador y la entrada a AxisForgeBO. La comparación del árbol integrado con el padre de la biblioteca no muestra diferencias en `axisforge_bo`, `docs/axisforge-bo`, `tests`, `experiments` ni `pyproject.toml`. Las carpetas Rust, las aplicaciones, Cargo.toml y Cargo.lock coinciden con el padre del simulador. Los CSV históricos conservan sus finales de línea originales y hashes. El merge preserva sus contenidos byte por byte.

| Verificación en el merge | Resultado |
|---|---|
| `cargo test --workspace --locked` | 353 aprobadas, cero fallidas e ignoradas. La cifra previa de 337 correspondía al simulador anterior. |
| WASM para wasm32-unknown-unknown, perfil wasm-release | Compilado, SHA-256 `c514b643af908c11237ece5a4264147d42b7a191fa6a5ca5eea2868ae45ac110`. Las pruebas Python ejercitan su ABI mediante Node. |
| AxisForgeBO 0.1.0 | Wheel reconstruido con setuptools a partir del merge. SHA-256 del artefacto medido `243831bb1b192818dba8ed779799e9a2124e0d819a5065f31ed26c69d393384d`. Instalación forzada sin dependencias en el entorno aislado de la fase I, `pip check` satisfactorio. No se reinstalaron de cero todas las dependencias en esta ejecución. |
| `python -I -m coverage run --branch --source=<paquete instalado> -m pytest -q --import-mode=importlib tests` | 163 aprobadas en 17,89 s. Dos avisos conocidos sobre compilación opcional qLogEHVI y estandarización de un objetivo constante. |
| Cobertura de AxisForgeBO instalado | 1.948 de 2.045 sentencias y 687 de 776 ramas. |
| Dependencias | Núcleo NumPy/SciPy. Versiones CPU del entorno probadas con el fichero de restricciones de la biblioteca, extras científicos PyTorch, BoTorch, GPyTorch y NumPyro. La validación GPU requiere hardware. |
| Datos de la fase I | Evidencia anterior conservada en los árboles originales. La auditoría de la integración contrasta la estructura y los hashes. Los recuentos conservados no equivalen a otra ejecución de 1.290 trayectorias en esta fase. |

La etiqueta local de referencia es `baseline/axisforgebo-0.1.0-integrated`. La versión de referencia identifica una combinación de software trazable en CPU. La auditoría científica global previa continúa abierta. La publicación externa de la rama depende de autorización expresa y de que GitHub confirme el SHA. [Auditoría previa](../docs/axisforge-bo/FINAL_AUDIT.md) · [Revisión independiente de fase II](audit/PHASE_II_INITIAL_REVIEW.md)
