# Fe de erratas de la Fase V: dinámica y orden aleatorio

Detectada en la auditoría de la Fase VI (etapa A). Los ficheros de la Fase V **no se modifican**. Esta
nota los corrige, y la Fase VI repite el análisis con el modelo corregido.

## 1. Las perturbaciones de masa no eran ×2 ni ×0,5

[`research_grasping/phase5/dynamics.py`](../../research_grasping/phase5/dynamics.py) define:

```python
('mass x2', dict(mass=2.)),
('mass x0.5', dict(mass=.5)),
```

`scene()` toma `mass` como masa absoluta en kilogramos, no como factor. Se comprueba leyendo el MJCF
generado: «mass x2» da `mass="2.0"` y «mass x0.5» da `mass="0.5"`. La pieza nominal pesa 0,1 kg, igual que
en la celda del planificador. Por tanto:

| Etiqueta en la Fase V | Lo que dice el protocolo | Lo que se simuló |
|---|---|---|
| masa ×2 | 0,2 kg | **2,0 kg** (×20) |
| masa ×0,5 | 0,05 kg | **0,5 kg** (×5) |

**Qué cambia en la lectura de la Fase V** ([README](../phase5/README.md), [RESPONSE](../phase5/RESPONSE.md),
[MANUSCRIPT](../phase5/MANUSCRIPT.md)):

- «Masa ×2 es el fallo principal» y «los priors eligen agarres menos robustos, sobre todo con masa doble»
  hablan en realidad de una **carga de 2 kg** sostenida por 45 N por almohadilla.
- La robustez «6/6» (49–64 %) y el McNemar del aprendido frente al defecto (7 frente a 30, p = 0,0002)
  son resultados válidos para esas condiciones (0,1, 2 y 0,5 kg, μ 0,3, dos ruidos de pose), pero **no**
  para la incertidumbre de masa que describe el protocolo.
- Lo nominal (89–95 %) no cambia.

## 2. El centro de masas del simulador no era el del planificador

La Fase V ponía el centro de masas de la pieza en la media de los vértices de sus piezas de colisión. El
planificador mide el brazo de palanca de cada agarre (`center_offset`) desde el centroide de la malla.
Recuperado del catálogo con un residuo menor de 1 µm, la distancia entre los dos es:

| | Mediana | Percentil 90 | Máximo |
|---|---:|---:|---:|
| Distancia entre los dos centros de masas | 6,4 mm | 16,5 mm | 51 mm |

En 60 agarres reales y 3 condiciones, el veredicto cambia en 4 de 180 ensayos
([physics_validation.json](results/physics_validation.json)).

## 3. El orden aleatorio no estaba «dentro del filtro de carrera»

La Fase V pasó el orden aleatorio como prior con peso 10⁶. El planificador suma +10 a los candidatos que
pasan el filtro de carrera y fuerza (`planner.rs`), pero 10⁶ × (diferencia entre valores aleatorios)
anula ese +10. El orden aleatorio examinaba, pues, candidatos que el filtro rechaza.

| Medida en las 151 piezas viables (B, 80 mm) | Valor |
|---|---|
| Orden aleatorio de la Fase V ÷ defecto (media geométrica, 20 semillas por pieza) | 2,15 |
| Aleatorio que respeta el filtro ÷ defecto (mismas semillas) | **1,18** |
| Candidatos que pasan el filtro (mediana por pieza) | 49 % |

La frase «el orden aleatorio es 3,3 veces peor que el defecto, así que el defecto no es arbitrario» de la
[respuesta de la Fase V](../phase5/RESPONSE.md) exagera. Respetando el filtro, el orden por defecto solo
es un 18 % mejor que el azar. Las demás comparaciones de la Fase V usan el peso 2 y respetan el filtro;
no les afecta.

## 4. Qué hace la Fase VI

- **Modelo v2** ([physics.py](../../research_grasping/phase6/physics.py)): centro de masas del
  planificador y masas correctas (0,2 y 0,05 kg). Con el centro de masas de la Fase V y las mismas masas,
  reproduce la v1 en 12 de 12 ensayos.
- **Cargas de estrés**: las cargas que la Fase V aplicó de hecho (0,5 y 2 kg) y 1 kg se mantienen, con su
  nombre real.
- **Etapa B**: explica el resultado de la Fase V con las cargas reales y comprueba si se mantiene con las
  masas correctas
  ([robustness_mechanism.json](results/robustness_mechanism.json)).

El error es del autor de la Fase V (este asistente). No lo detectaron los tests porque
`tests/phase5/test_dynamics.py` comprueba el ensayo nominal y no la masa de cada perturbación. La Fase VI
añade un test que sí lo comprueba.
