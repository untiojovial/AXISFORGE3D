# Diagnóstico exploratorio F0/F1/F2 frente a F3

**PARTIALLY_VALIDATED, no aprueba M8.** Se analizaron 84 candidatos predeterminados en 14 objetos train; 71 nominales numéricamente válidos, 13 bloqueados por incompatibilidad local del proxy y conservados como desconocidos. Los seis candidatos por objeto no son objetos independientes. Se usa un efector aislado, sin ejecución dinámica del brazo; no hay test ni validación sim-to-real. El diseño fue revisado con información cualitativa parcial, por lo que no es ciego.

Hubo 82 rechazos y dos planes F1 viables. De los 71 nominales observables, cinco cumplen `sustained` bruto y cuatro el criterio con settling `eligible_sustained`; solo uno de esos cuatro tiene plan F1 viable. De los cuatro, solo uno satisface adicionalmente una penetración máxima de objeto ≤1 mm: esa condición es una **sensibilidad posterior**, no el criterio físico predefinido. Se observaron 25 asentamientos inestables y 57 nominales con penetración de objeto máxima >1 mm. De 71 candidatos con variantes solver/paso válidas, cuatro cambiaron de etiqueta eligible bajo al menos una variante; son sensibilidades del mismo candidato, no réplicas adicionales. La cifra de cero variantes numéricas inválidas en el JSON se refiere exclusivamente a la lista de solver/paso `numerics`: no incluye controles. En particular, el control de fricción cero tuvo 57 de 71 ensayos numéricamente inválidos, que no se consideran fallos físicos. Estos problemas impiden considerar los cuatro eventos éxitos físicos validados.

| Estrategia contrafactual | Seleccionados | F3 nominal conocido / desconocido | Eventos isolated eligible retenidos / descartados | Eventos con F1 viable retenidos | Tiempo F3 nominal medido (s) |
|---|---:|---:|---:|---:|---:|
| Siempre F3 | 84 | 71 / 13 | 4 / 0 | 1 | 51.989 |
| Filtro F1 | 2 | 2 / 0 | 1 / 3 | 1 | 0.833 |
| Filtro F1 y epsilon > 0 | 2 | 2 / 0 | 1 / 3 | 1 | 0.833 |

Sensibilidad extrema a los 13 F3 desconocidos: el número de eventos isolated eligible entre los 84 va de 4 a 17 si todos fueran respectivamente fallos o éxitos; el filtro F1 selecciona dos candidatos con resultado conocido, de modo que conserva exactamente un evento observado y podría descartar entre 3 y 16. Estos límites no corrigen el sesgo de selección del proxy ni establecen una tasa poblacional.

Los tiempos anteriores suman únicamente ensayos F3 nominales medidos, sin imputar coste a los 13 bloqueados. Las 84 llamadas nativas conjuntas F1+F2 consumieron 45,939 s medidos, incluidos otros trabajos del proceso; no se pueden separar fielmente F1 y F2, ni computar coste total sin generar candidatos, procesar mallas, extraer embeddings y planificación. Los filtros conservan uno de cuatro eventos del efector aislado, aunque conservan el único evento nominal que también pasó F1; esta distinción no constituye prueba de no inferioridad ni eficiencia prospectiva. No hay política adaptativa predefinida y validada: **no evaluada**.

En pares disponibles, matriz F1 frente a `eligible_sustained` aislado: TP=1, FP=1, FN=3, TN=66; precisión 1/2 y recall 1/4, sin extrapolación poblacional. F2 >0: TP=3, FP=37, FN=1, TN=30. Correlaciones F2 continuo frente a binario eligible: Pearson 0,201 y Spearman 0,146, descriptivas y condicionadas a proxy comparable; se omiten si variable constante o pares insuficientes por objeto. No implican que F2 sea fidelidad de F3. F0 usa ancho + 10 mm de holgura frente a apertura 80 mm, conforme código nativo y configuración del piloto; se reporta junto a la matriz completa en JSON.

| Objeto | F1 viable | Nominal válido / desconocido | Eligible aislado | Eligible con penetración ≤1mm | Cambio de etiqueta solver/paso |
|---|---:|---:|---:|---:|---:|
| 2_of_Jenga_Classic_Game | 0 | 5 / 1 | 0 | 0 | 0 |
| ACE_Coffee_Mug_Kristen_16_oz_cup | 0 | 6 / 0 | 1 | 0 | 1 |
| Air_Hogs_Wind_Flyers_Set_Airplane_Red | 0 | 3 / 3 | 0 | 0 | 0 |
| Android_Figure_Orange | 2 | 6 / 0 | 1 | 0 | 1 |
| Aroma_Stainless_Steel_Milk_Frother_2_Cup | 0 | 5 / 1 | 0 | 0 | 0 |
| BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup | 0 | 6 / 0 | 0 | 0 | 0 |
| BIRD_RATTLE | 0 | 6 / 0 | 0 | 0 | 0 |
| Big_Dot_Aqua_Pencil_Case | 0 | 4 / 2 | 0 | 0 | 0 |
| BlackBlack_Nintendo_3DSXL | 0 | 6 / 0 | 0 | 0 | 0 |
| Cole_Hardware_Hammer_Black | 0 | 4 / 2 | 0 | 0 | 0 |
| Craftsman_Grip_Screwdriver_Phillips_Cushion | 0 | 4 / 2 | 0 | 0 | 0 |
| Ecoforms_Cup_B4_SAN | 0 | 6 / 0 | 0 | 0 | 0 |
| Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle | 0 | 6 / 0 | 2 | 1 | 2 |
| Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister | 0 | 4 / 2 | 0 | 0 | 0 |

El auditor independiente verificó conteos, lock y criterios principales; no aprobó M8 como evidencia multifidelidad. El [JSON completo](multifidelity_results.json) contiene matrices, costes y correlaciones por objeto, además de IDs, estados y hashes. El primer intento se detuvo por QACC y no se utilizó. Reproducir con `python -m research_grasping.phase4.fidelity_analysis --result research/phase4/robotics/physical_pilot.json --protocol research/phase4/robotics/physical_protocol.json --catalogs /workspace/axisforge-phase3-data/pilot-expanded/catalogs --out research/phase4/methodology/multifidelity_results.json`.

Revisión final del auditor: acepta el informe como diagnóstico exploratorio parcial, no como aprobación de M8 ni afirmación de coste total o no inferioridad. No verificó individualmente todos los tiempos ni repitió las 84 simulaciones.
