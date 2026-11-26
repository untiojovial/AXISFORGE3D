# Fase VI — decisiones de continuar o parar

Cada puerta dice qué evidencia la abre, qué se observó y qué se decidió. Las puertas de las etapas B–D se
fijaron en el [protocolo](PHASE6_PROTOCOL.md) (commit `f1f0891`) antes de los resultados de fusión.

| Puerta | Criterio | Observado | Decisión |
|---|---|---|---|
| **A. Auditoría** | Cifras de la Fase V verificables; rasgos sin fuga | Cifras verificadas. Tres errores de la Fase V documentados ([fe de erratas](ERRATUM_PHASE5_DYNAMICS.md)). Rasgos sin cinemática inversa, colisiones ni veredicto | **Pasa** |
| **B. Mecanismo** | Explicar la pérdida de robustez de la Fase V | Venía de la carga de 2 kg y del brazo de palanca (AUC 0,86). El prior anula el centrado del planificador. Con las masas correctas no hay pérdida (+6,6 puntos para el aprendido) | **Pasa**: explicada |
| **C. Fusión simple** | Regla fijada (robustez no inferior a M0, más rápida que M0, conserva la velocidad de M1) | La cumplen M3, M4, M6, M7 y M8, pero **sin mérito propio**: M1 ya la cumpliría. En la secundaria de cargas (prevista en el protocolo), M4 recupera +11,9 puntos de tolerancia frente a M1 [6,6; 17,9] con la misma velocidad | **Pasa como desarrollo**. M4 es el candidato para la confirmación de la tolerancia a la carga |
| **D1. Calibración** | Siempre | ECE de 0,010 (0,005 con Platt). Algo infraconfiado entre 0,1 y 0,6 | **Hecha**. Para productos de probabilidades, usar Platt |
| **D2. P(H\|F) aprendido** | Ninguna fusión cumple la regla, **o** el centrado explica poco la robustez (AUC < 0,70) | AUC del centrado para la robustez estándar: 0,57 → la puerta se abre | **Ejecutado**: P(F)·P(H\|F) con P(H\|F) aprendido da robustez 0,960 frente a 0,907 de M1 (+5,3 puntos [2,0; 9,3]) a la misma velocidad, pero no mejora con cargas grandes (0,464) |
| **E. Inviabilidad temprana** | Barato | Ninguna pieza se puede declarar inviable solo con el filtro. Un umbral conservador sobre el clasificador marca 13 de 49 no viables sin perder ninguna viable | **Hecha** (nivel «estimado») |
| **F. Rasgos locales de DINO, MLP, DKL, ADKF-IFT** | Que un modelo simple deje una mejora grande y demostrable por explicar | La AUC de viabilidad ya es 0,993. P(H\|F) llega a una AUC de 0,956 con 19 rasgos geométricos. En la Fase V, DINOv3 no aportó dentro del mejor sistema. ADKF-IFT necesita tareas con conjuntos de soporte, y aquí no hay evaluaciones del objeto nuevo antes de planificar | **No se ejecuta**. Reabrir solo si un test confirmatorio deja margen grande y hay datos de soporte por objeto |
| **G1. Coste online real** | Medir segundos, no candidatos | Hecho en 60 piezas con un hilo, sin discrepancias con el simulador | **Hecha** (parcial): el ahorro real es del 24 % por llamada, porque el coste fijo domina |
| **G2. Orden por coste P·P/C_eval y multifidelidad** | Necesita el coste por candidato y por etapa de rechazo | El planificador no expone tiempos por etapa | **Bloqueado**: haría falta instrumentar el planificador en Rust |
| **H. Simulador articulado (UR5 con palma y dedos)** | Que la conclusión dependa del brazo o de la palma | La conclusión depende sobre todo de la fricción torsional, no del brazo | **No se ejecuta** ahora. Está en las limitaciones |
| **I. Baselines externos (GPD, Dex-Net, Contact-GraspNet)** | GPU y pesos | Aquí solo hay CPU | **Bloqueado**. Requiere un entorno con GPU (Colab) |
| **J. Test confirmatorio y tercer dataset** | Objetos nuevos con licencia y acceso | GSO agotado. EGAD y Objaverse son posibles ([registro](dataset_access_register.md)) | **Diseñado, no ejecutado**. Tamaño en [statistical_power.json](results/statistical_power.json): ~105 piezas nuevas para la tolerancia de M4 frente a M1 (80 %) y ~71 para la no inferioridad de la robustez a ±5 puntos |

## Qué haría falta para confirmar

1. **Congelar un protocolo nuevo antes de abrir objetos**, con estas hipótesis y medidas:
   - **H1**: M4 (p̃^0,8·Q_cen^0,2) supera a M1 en tolerancia a la carga.
   - **H2**: P(F)·P(H|F) supera a M1 en robustez estándar.
   - **H3**: ambos son no inferiores a M1 en N_F (margen 1,25).
   - Pesos y modelos congelados tal como salen de este desarrollo, entrenados con todas las piezas de
     desarrollo.
2. **Reunir de 140 a 180 objetos nuevos** (EGAD y/o un subconjunto CC0/CC-BY de Objaverse) con la
   regla de tamaño, fijando la lista y sus hashes.
3. **Calibrar la fricción torsional** con una medida real, o informar la conclusión con al menos tres
   valores.
