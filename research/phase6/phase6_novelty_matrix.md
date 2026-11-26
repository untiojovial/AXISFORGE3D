# Fase VI — matriz de novedad

Regla: solo se compara con trabajos verificados ([revisión](phase6_literature_review.md)). La búsqueda no
fue sistemática: unas 20 consultas dirigidas. Por eso ninguna fila afirma originalidad mundial; como mucho,
«no se encontró antecedente directo en esta búsqueda».

| Elemento | Antecedente verificado más cercano | Qué hay aquí que no está en ese antecedente | Estado |
|---|---|---|---|
| Clasificador aprendido de viabilidad que ordena candidatos (M1) | Lou et al. 2020; Akinola et al. 2021; arXiv:2111.02571; Qin et al. 2025 | Nada esencial en el método. Se evalúa con el orden exacto de un planificador validado con 7 588 ejecuciones | **No es novedoso** |
| Producto utilidad × probabilidad de viabilidad (M4, P(F)·P(H\|F)) | Gardner et al. 2014 (restricciones en optimización bayesiana) | Aplicado a ordenar candidatos de un planificador con restricciones cinemáticas | **No es novedoso** como idea |
| Robustez frente a «mejor puntuación» | Weisz y Allen 2012 (épsilon frente a error de pose); Nogueira et al. 2016 | El conflicto se da entre **acelerar la viabilidad** y la **robustez dinámica**: el prior aprendido desplaza el término de centrado del planificador y elige brazos de palanca mayores | Posible contribución empírica; sin antecedente directo en esta búsqueda; **no verificada como nueva** |
| Medida N_FR (candidatos hasta el primer viable **y** robusto) | No encontrado | Une el coste de planificación y la validación dinámica en una sola medida por pieza | Propuesta metodológica; sin antecedente en esta búsqueda |
| Modelos fundacionales dentro del mejor sistema (DINOv3 no aporta, equivalencia ±25 %) | Caron et al. 2021; Siméoni et al. 2025 (los modelos) | Medida controlada del valor de DINOv3 en este problema (Fase V) | Resultado negativo acotado; no es novedad metodológica |
| Simulador exacto del orden del planificador | No buscado de forma específica | Permite evaluar cualquier prior sin replanificar | Herramienta; sin afirmación de novedad |
| ADKF-IFT para agarre (línea del TFM) | Chen et al. 2023 (ADKF-IFT, moléculas); Wilson et al. 2016 (DKL) | El TFM lo aplica a 8 objetos con una semilla | **No ejecutado en la Fase VI**; sin evidencia nueva |

**Cómo leerlo para un manuscrito.** La aportación defendible es empírica y metodológica:

1. Medición controlada del compromiso entre velocidad y robustez.
2. La fe de erratas y la reevaluación con masas correctas.
3. Si la etapa C lo sostiene en un test confirmatorio futuro, una regla simple para recuperar la robustez.

No lo es «un método nuevo de agarre».
