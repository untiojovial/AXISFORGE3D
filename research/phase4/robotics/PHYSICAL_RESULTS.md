# Piloto físico de desarrollo: resultados y límites

El protocolo `physical_protocol.json` predefinió 14 objetos train, seis IDs por objeto (84 pares) y condiciones; `physical_pilot.json` registra la unión verificada de tres shards. Las etiquetas F1/F2 son nativas; F3 utiliza un efector aislado con geometría de proyecto y proxies de proveedor, no la trayectoria del brazo completo. Ningún objeto final fue abierto. La selección tiene seis candidatos por objeto, inferior al rango orientativo de 10–20 por el coste del piloto, y no constituye una muestra poblacional.

De 84 candidatos, F1 aceptó 2 y rechazó 82. Se ejecutó el escenario F3 nominal en 71 y 13 quedaron bloqueados por discrepancia local entre contactos visuales y proxies. Los 71 nominales tuvieron estado numérico válido; 5 cumplieron el criterio sostenido bruto, pero sus máximos de penetración objeto fueron 8.48, 1.73, 0.91, 12.59 y 1.64 mm. Este criterio no limita penetración durante todo el ciclo y no demuestra éxito físico validado. La correspondencia por rayos locales tampoco valida la geometría global.

| Intervención | Cumple criterio bruto | No cumple | Numéricamente inválida |
|---|---:|---:|---:|
| Nominal | 5 | 66 | 0 |
| Pinza abierta | 0 | 71 | 0 |
| Fricción cero | 0 | 14 | 57 |
| Masa 2 kg | 1 | 70 | 0 |
| Desvío +0.1 m | 0 | 71 | 0 |

La masa elevada no garantiza un fracaso: en un caso satisfizo el criterio bruto; no se eliminó. Con fricción cero, 57 casos son inestabilidades numéricas, no fallos físicos. En total, 57 filas presentaron al menos una variante numéricamente inválida; otras 14 completaron todas sus variantes. Los contadores, paso y estado se conservan por variante, con `sustained=null` si falla numéricamente. Hay que distinguir el número de filas del de ejecuciones de control.

Se repitió el cálculo con Newton/CG y pasos 0.0005/0.001/0.002 s: 5/5/6 criterios brutos positivos por solver respectivamente entre los 71 nominales, sin que esto demuestre convergencia. La máxima penetración nominal fue 17.60 mm. El diagnóstico post hoc de tres casos identificó contactos palma/mandíbula–objeto con penetraciones de 8.48–12.91 mm; cambiar a una envolvente convexa alteró resultados y también COM/inercia. Los controles adicionales de COM y fuerza están en `perturbations.json`, son tres casos y no validan robustez general.

La primera corrida parcial se interrumpió al detectar warnings de MuJoCo y se conserva en `physical_pilot_pre_warning_fix.json`; otra repetición parcial se conserva en `physical_pilot_v2_partial.json`. La corrida definitiva invalida warnings previos o posteriores, no finitos y reinicios. Para 65 nominales solapados, los valores y booleanos coinciden exactamente con la corrida anterior; esto no revalida sus controles afectados por warnings. El hash del constructor de escena de fase III se registró durante la ejecución, no en el preregistro, como se detalla en `PHYSICAL_PROTOCOL.md`. Se requiere auditoría independiente de las filas completas antes de aceptar el diagnóstico. M7/M8 siguen pendientes; no se ha comprobado correlación ni ahorro de multifidelidad.
