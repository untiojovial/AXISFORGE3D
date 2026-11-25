# Piloto de potencia sintético

**IMPLEMENTED, revisión independiente posterior pendiente.** No hay datos robóticos ni acceso al test. Reproducir desde raíz con `python research_grasping/statistics/simulate_design.py --repetitions 10000 --seed 20261125 --output research/results/synthetic_power.json`; requiere NumPy y SciPy, versiones registradas en JSON. Ejecución CPU, 10 000 simulaciones por escenario; prueba básica de reproducibilidad y coste: `python -m unittest discover -s tests/phase2 -p test_statistics_simulation.py` (2 pruebas superadas). Se conservaron todos los escenarios en JSON. Los resultados dependen de los valores arbitrarios seleccionados antes de ejecutarlos, no son estimaciones de variabilidad real.

Para referencia, con σ_obj=0,20, σ_semilla=0,20, 10 semillas y diferencia media hipotética 0,05:

| Escenario | Potencia estimada (IC Wilson 95 %) | Coste nominal evaluaciones |
|---|---:|---:|
| Normal, 50 objetos, memoria fija | 0,502 (0,492–0,511) | 50 000 |
| Normal, 100 objetos, memoria fija | 0,775 (0,767–0,783) | 100 000 |
| Normal, 200 objetos, memoria fija | 0,957 (0,953–0,961) | 200 000 |
| Normal, 5 historias independientes × 50 objetos | 0,220 (0,212–0,228) | 250 000 |
| Normal, 20 historias independientes × 50 objetos | 0,665 (0,655–0,674) | 1 000 000 |

El coste incluye dos métodos por 50 evaluaciones, pero excluye entrenamiento, extracción, historia, piloto y reintentos. Los escenarios con múltiples historias incorporan desviación entre historias de 0,10 y son una población de historias distinta de la inferencia condicional de memoria fija; no comparar como si estimaran exactamente el mismo objetivo.

La forma de la distribución importa: con diferencia nula y lognormal centrada, cobertura bilateral nominal 95 % fue 91,9 % con 50 objetos y 93,5 % con 200, frente a rechazo unilateral 1,6 % y 2,5 % respectivamente. La simulación no satisface la tolerancia de cobertura especificada en esos escenarios. **No se recomienda adoptar aún el contraste t como procedimiento general**, ni elegir un tamaño final: estudiar métodos válidos bajo asimetría, simular de nuevo y obtener piloto independiente para efecto mínimo, varianzas, dependencia y pérdidas. Para un estudio futuro, un piloto de al menos decenas de objetos diversos y réplicas puede estimar estos parámetros; la cantidad final depende del objetivo y presupuesto, no se infiere del número 100–200 por sí solo. La sensibilidad MCAR no valida pérdidas dependientes de resultados. No existe validación de potencia para familia Holm ni garantía de ausencia de sesgo real.

Auditoría independiente preliminar: el auditor comprobó coherencia del modelo y replicó directamente 91,95 % de cobertura en el escenario asimétrico de N=50; reproducir la cuadrícula completa permanece pendiente. Con H=1 se fija el efecto de historia en cero: la inferencia es condicional, y no captura variabilidad entre historias. Esta revisión no aprueba potencia real ni un contraste universal.
