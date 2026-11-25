# Calibración predictiva de desarrollo

**PARTIALLY_VALIDATED; auditoría independiente pendiente.** Diagnóstico leave-one-object-out de 16 objetos train y 160 observaciones históricas nativas F2, incluidas las rechazadas en F1. No se accedió a los seis objetos de validación ni a los ocho de test. Resultados condicionados a la selección previa de candidatos y a estos objetos; epsilon analítico no es éxito dinámico. Diseño y revisión previa en [CALIBRATION_DESIGN.md](CALIBRATION_DESIGN.md); predicciones, métricas por objeto y versiones en [calibration_results.json](calibration_results.json).

| Modelo | MAE macro | RMSE macro | NLPD macro | Cobertura 90 % macro | Ancho 90 % macro |
|---|---:|---:|---:|---:|---:|
| Constante | 0,0130 | 0,0154 | −2,5915 | 0,9125 | 0,0557 |
| RBF | 0,0119 | 0,0142 | −2,6650 | 0,9000 | 0,0503 |
| Matérn 5/2 | 0,0119 | 0,0142 | −2,6642 | 0,9062 | 0,0504 |
| Matérn ARD | 0,0101 | 0,0128 | −2,7637 | 0,8375 | 0,0365 |
| DKL semilla 11 | 0,0128 | 0,0170 | 3,1664 | 0,4812 | 0,0164 |
| DKL semilla 29 | 0,0128 | 0,0164 | 3,8285 | 0,4813 | 0,0160 |
| DKL semilla 47 | 0,0148 | 0,0197 | 7,0487 | 0,4625 | 0,0157 |

La configuración DKL explorada es demasiado confiada en este diagnóstico: menor amplitud y cobertura observada inferior a 50 % con peor NLPD. No generalizar este resultado a todo DKL, ni promover esta configuración a adquisición. El predictor constante, RBF y Matérn tienen cobertura macro próxima a 90 % pero el número de objetos es pequeño y los folds comparten entrenamiento; esto no prueba calibración para objetos nuevos. ARD obtuvo menor MAE y NLPD pero menor cobertura y 167 advertencias acumuladas, principalmente parámetros en límites: no confundir ajuste numérico con validación. No elegir una semilla favorable ni asumir que cobertura macro garantiza cobertura por categoría o tarea.

Reproducir desde raíz, con dependencias registradas en JSON:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/axisforge-bo-env/bin/python -m research_grasping.phase4.calibration --manifest research/phase3/pilot/manifest.prepilot.json --catalogs /workspace/axisforge-phase3-data/pilot-expanded/catalogs --history /workspace/axisforge-phase3-data/run-development --out research/phase4/methodology/calibration_results.json
/tmp/axisforge-bo-env/bin/python -m unittest discover -s tests/phase4 -p test_calibration.py
```

Las pruebas verificaron cálculo de métricas y aislamiento del ajuste de preprocesado frente a cambios del objeto retenido (2 superadas). La ruta del intérprete es del entorno actual, no requisito portable. Los hiperparámetros GP se ajustaron solo dentro de cada fold; el coste y todas las advertencias se conservan en JSON. Los intervalos gaussianos no respetan el rango no negativo de epsilon, y la muestra no es representativa de todos los candidatos. Para verificar corrección y reproducibilidad se ha solicitado auditoría independiente.

Auditoría preliminar independiente: verificados los 16 folds/160 etiquetas, promedios y pruebas; el ajuste de escala/PCA es por fold y el ruido DKL agregado corresponde a varianza observacional. La reproducción completa y el cotejo independiente de hashes siguen pendientes. El dictamen solo respalda la conclusión sobre la configuración estudiada y no aprueba una selección prospectiva de modelo.

## Recalibración exploratoria anidada

Una corrección posterior estimó, en cada fold exterior, un factor de varianza a partir de tres folds internos agrupados por objeto; [diseño revisado previamente](RECALIBRATION_DESIGN.md) y [resultados completos](recalibration_results.json). Para DKL semilla 11, cobertura macro 0,4813 → 0,9563, ancho 0,0164 → 0,0756, NLPD 3,1664 → −2,4948; MAE permanece 0,0128. Los factores van de 16,02 a 34,21. El aumento de cobertura se acompaña de intervalos aproximadamente 4,6 veces más anchos; la NLPD corregida sigue siendo peor que la del predictor constante y GP estacionarios de esta muestra. Se ajustaron 48 modelos internos adicionales; sus tiempos se registran por fold, excluyendo coste previo exterior. Esto no garantiza calibración en objetos nuevos ni justifica promover DKL a adquisición.

El auditor verificó fórmula, exclusión del objeto exterior, agregación macro y cifras; no repitió los 48 entrenamientos. Tres pruebas automatizadas verifican además que los grupos internos sean disjuntos y excluyan el objeto exterior. Reproducir tras el comando anterior:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/axisforge-bo-env/bin/python -m research_grasping.phase4.calibration_recalibrate --manifest research/phase3/pilot/manifest.prepilot.json --catalogs /workspace/axisforge-phase3-data/pilot-expanded/catalogs --history /workspace/axisforge-phase3-data/run-development --outer research/phase4/methodology/calibration_results.json --out research/phase4/methodology/recalibration_results.json
```
