# Alternativa acotada: simulación sintética

**Pendiente de auditoría.** Diseño previo y alcance en [BOUNDED_DESIGN.md](../methodology/BOUNDED_DESIGN.md); sin datos de robot ni test. Reproducir `python research_grasping/statistics/simulate_bounded.py --output research/results/bounded_power.json`. Las tres pruebas estadísticas pasan con `python -m unittest discover -s tests/phase2 -p 'test_statistics*.py'`. El JSON incluye 10 000 repeticiones, IC Wilson, versiones y cuadrícula completa.

| Objetos | Efecto sintético | Rechazo Hoeffding | Rechazo t | Semiamplitud Hoeffding bilateral | Coste nominal |
|---:|---:|---:|---:|---:|---:|
| 200 | 0,05 | 0,0000 | 0,9999 | 0,192 | 200 000 |
| 200 | 0,15 | 0,0132 | 1,0000 | 0,192 | 200 000 |
| 1 000 | 0,05 | 0,0000 | 1,0000 | 0,086 | 1 000 000 |
| 1 000 | 0,15 | 1,0000 | 1,0000 | 0,086 | 1 000 000 |
| 5 000 | 0,05 | 1,0000 | 1,0000 | 0,038 | 5 000 000 |

En el nulo, N=50, tasa de rechazo Hoeffding 0/10 000 y t 0,0324; cobertura bilateral Hoeffding 1,0000 y t 0,9432. Un valor simulado de 0 o 1 no es garantía de probabilidad exacta; consultar intervalos Wilson en JSON. Los resultados indican que la garantía es muy conservadora para esta distribución hipotética; no comparar esta DGP de pequeña varianza con el modelo no acotado anterior como si fuesen la misma población. La cota es útil como análisis de respaldo si la escala y la independencia se validan, pero el coste y potencia simulados desaconsejan recomendarla como única prueba práctica sin un piloto. No usarla con datos no acotados, historias aleatorias compartidas, objetos dependientes o faltantes sin resolver. Ningún diseño ni potencia real queda congelado.
