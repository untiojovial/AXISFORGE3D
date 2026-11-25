# Piloto sintético de transferencia

Estado: IMPLEMENTADO, PRUEBAS LOCALES SUPERADAS; pendiente de auditoría independiente y no aprobación de M5. La configuración `configs/transfer/synthetic_pilot.json` se escribió antes de ejecutar. No son objetos, representaciones fundacionales ni agarres físicos. Ocho semillas por escenario son insuficientes para inferencia confirmatoria; diseño y umbrales finales siguen abiertos. El experimento usa el valor sin ruido de cada punto evaluado para medir el best-so-far: es observable solo porque se conoce la función sintética y no se entrega al optimizador. La escala de calidad es [-1,1], así que el área es la media de best-so-far y no se normaliza entre 0 y 1. La inicialización Sobol, candidatos y ruido por índice de evaluación se emparejan; las coordenadas posteriores pueden diferir. Fuente y objetivo comparten fase en este problema deliberadamente simple, el adversario invierte signo. El análisis se limita a esta familia.

| Escenario | Método | Área media | Diferencia media frente a target | Semillas perjudicadas / 8 |
|---|---|---:|---:|---:|
| related | random | 0.7186 | -0.1428 | 8 |
| related | sobol | 0.7651 | -0.0962 | 8 |
| related | target | 0.8614 | 0 | 0 |
| related | context | 0.8627 | 0.0014 | 2 |
| related | uniform | 0.8637 | 0.0023 | 2 |
| related | adaptive | 0.8643 | 0.0029 | 1 |
| adversarial | random | 0.7186 | -0.1428 | 8 |
| adversarial | sobol | 0.7651 | -0.0962 | 8 |
| adversarial | target | 0.8614 | 0 | 0 |
| adversarial | context | 0.5999 | -0.2615 | 8 |
| adversarial | uniform | 0.5999 | -0.2615 | 8 |
| adversarial | adaptive | 0.8517 | -0.0097 | 1 |

El peso target final adaptativo promedia 0.260 en related y 1.000 en adversarial; ello muestra recuperación de pesos en este ejemplo, no seguridad general. Las diferencias pequeñas no justifican afirmar superioridad. Sobol es una secuencia independiente tras tres puntos iniciales compartidos. Context y uniform son mezclas fijas; adaptive añade evidencia predictiva. No se ha implementado aquí GP con historia agregada, RGPE, GP multitarea, DINO ni el conjunto de baselines obligatorios. Presupuesto 12 incluye inicialización; coste de historia no se carga al presupuesto objetivo y debe contabilizarse separadamente. Registros crudos incluyen historial, trayectorias, pesos, tiempos, hash de configuración y fuentes, entorno y HEAD; el HEAD por sí solo no identifica cambios sin commit. Los tiempos incluyen ajuste de modelos incluso en random/Sobol por simplicidad y no sirven para comparar eficiencia de esos baselines.

Reproducir desde raíz en entorno con NumPy, SciPy y AxisForgeBO: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research_grasping.contextual_transfer.pilot --output /tmp/transfer-pilot.json`. Los resultados numéricos son deterministas en el entorno salvo tiempos; comparar únicamente campos numéricos científicos, no tiempo. `python -m pytest -q tests/phase2/test_transfer_model.py` ejecutó cuatro pruebas: varianza total y PSD, EI frente a Monte Carlo, prefijos y ausencia de fuentes, y rechazo temporal. Falta revisión independiente de implementación y experimentos.

Integración: `from research_grasping.contextual_transfer import AdaptiveTransfer, Source`; crear fuentes en unidades comunes con instante de disponibilidad anterior, crear `AdaptiveTransfer(context, sources, target_time=..., dimension=...)`; llamar `predict(X)` para media y varianza latente, `predict_joint(X)` para covarianza conjunta, `acquisition(X)` para EI una vez observado un resultado, y `observe(x,y)` después de cada evaluación. El llamador debe controlar unidades, grupos autorizados, escala, procedencia, selección y evaluación física. No existe verificación de contacto físico, ni un protocolo completo de archivo ni integración con AxisForge3D. Usa GP exacto AxisForgeBO con hiperparámetros fijos, ruido común conocido y utilidad maximizada; la calibración y normalización deben ser fijadas con entrenamiento y validación autorizados.

Repetición local: una segunda ejecución produjo 96 registros idénticos en todos los campos después de excluir `seconds`; no es una repetición independiente ni una validación en otra máquina.
