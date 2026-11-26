# Fase VI — revisión científica interna

Es una autocrítica del trabajo (fases V y VI) y del TFM de partida, escrita para un verificador externo.
No sustituye a una revisión independiente.

## 1. Errores encontrados en la Fase V (corregidos en la Fase VI, sin tocar sus ficheros)

| Gravedad | Error | Efecto | Detalle |
|---|---|---|---|
| Alta | «Masa ×2/×0,5» simuló 2 kg y 0,5 kg | La pérdida de robustez del aprendido que publicó la Fase V es una pérdida **con 2 kg**. La robustez 6/6 de la Fase V equivale a «sostiene 2 kg»: aprendido 0,490 y 0,490; defecto 0,642 y 0,649 | [fe de erratas](ERRATUM_PHASE5_DYNAMICS.md) §1 |
| Media | Centro de masas del simulador distinto del del planificador | El brazo de palanca simulado no era el que puntúa el planificador; 4 de 180 veredictos cambian | §2 |
| Media | Orden aleatorio fuera del filtro de carrera | «El defecto es 3,3 veces mejor que el azar» exagera: respetando el filtro, el defecto es solo un 18 % mejor | §3 |
| Baja | «Causas de inviabilidad» como el rechazo más frecuente | En las 49 piezas no viables a 80 mm, todas tienen entre 302 y más de 600 candidatos que pasan el filtro. Ninguna se puede declarar inviable solo con el filtro ([infeasibility_benchmark.json](results/infeasibility_benchmark.json)) | — |

Ninguno de estos errores afecta a los contrastes principales de la Fase V, que son de viabilidad (N_F con
peso 2) y no usan la dinámica. Sí afectan a la lectura de la RQ7 y a la «respuesta a la revisión» sobre
el orden aleatorio.

## 2. Lo que sostiene la evidencia de la Fase VI

- **El mecanismo (etapa B)**:
  - La pérdida venía casi entera de la carga de 2 kg (aprendido: 74 de 151 sostenidos; defecto: 98).
  - En nominal, el aprendido es igual o mejor (143 frente a 134).
  - El brazo de palanca predice sostener 2 kg (AUC con validación cruzada por pieza de 0,86).
  - El aprendido elige brazos 7,4 mm mayores [4,7; 10,3] porque su prior, con peso 2, anula el término de
    centrado del planificador. No los busca: su probabilidad no se relaciona con el brazo de palanca
    entre los viables (Spearman mediana 0,04).
  - El brazo de palanca explica unos 9 de los 15 puntos de diferencia; el resto queda sin explicar.
- **Lo que depende de las masas correctas** (etapa B5) y **la fusión** (etapa C): ver los resultados en
  el [README](README.md).

## 3. Debilidades que un revisor señalaría

1. **Desarrollo, no confirmación.** Todo se ajusta y se mide en las mismas 151 piezas, aunque con
   validación cruzada anidada. Sin un test nuevo no hay afirmaciones confirmatorias.
2. **El modelo físico manda.**
   - La fricción torsional no calibrada decide los casos con cargas grandes.
   - Un robot real con almohadillas de goma podría tener otra capacidad torsional.
   - El criterio de sostener no detecta giros.
3. **N_FR supone un sistema que valida cada viable en simulación.** Es una medida útil, pero no es el
   coste de un planificador online sin simulador.
4. **Una celda.** El clasificador aprende el alcance de este robot en esta mesa. Su transferencia a otra
   celda no está medida.
5. **El protocolo de la Fase VI se escribió conociendo ya la exploración de la etapa B.** La hipótesis
   del brazo de palanca se vio antes de fijar el protocolo, y así consta en él. Las fusiones y la regla
   de decisión sí se fijaron antes de calcular ningún resultado de fusión (commit `f1f0891`).
6. **Desviación de implementación del barrido** (parada en el primer fallo; cargas solo en los primeros
   agarres). Documentada; no cambia las medidas definidas.

## 4. El TFM de partida

*Robot grasping with Bayesian optimization informed with foundational models* (Universidad de Zaragoza,
2025). Se verificó contra las fuentes ([revisión bibliográfica](phase6_literature_review.md), §9).

| Aspecto | Valoración |
|---|---|
| Diseño | 8 objetos, una semilla y una sola secuencia por orden (directo e invertido). No permite separar el efecto del contexto del ruido. El propio texto lo admite para los primeros objetos |
| Medida | Calidad épsilon/volumen en Simox con la mano del iCub. No hay validación dinámica ni robot real |
| Posible fuga | El vector de contexto se calcula con la media de la métrica de agarre del objeto (§5.2.4). Si incluye agarres evaluados del propio objeto que se optimiza, la entrada contiene el resultado. Hay que comprobarlo en su código |
| Referencias | DINO citado con el arXiv de ADKF-IFT; Montana con el año equivocado (1995 por 1988); GP-UCB atribuido a «Proceedings of the IEEE» (es ICML 2010); ADKF-IFT descrito con siglas que no son las del artículo; Rasmussen y Williams duplicado |
| Relación con este repositorio | Este repositorio no usa ni reproduce los resultados del TFM. ADKF-IFT no se ha implementado (etapa F, «no-go» por ahora: [phase6_go_no_go.md](phase6_go_no_go.md)) |

**Recomendación** si el TFM se usa como antecedente en un artículo:

1. Corregir las referencias.
2. Repetir la comparación con y sin contexto con varias semillas, intervalos y más objetos.
3. Comprobar la posible fuga del vector de contexto.
