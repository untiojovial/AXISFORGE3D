# Candidatos, memoria y alcance

El generador nativo emitió entre 766 y 1.410 poses por pieza. La opción `research-candidates --diverse-limit N` selecciona una secuencia reproducible de IDs originales mediante distancia en centro, anchura y orientaciones de aproximación y cierre. Cuatro de cada cinco posiciones prefieren anchuras compatibles. La quinta solicita un control incompatible cuando hay uno. El comando histórico conserva el orden previo. Selección y medición usan ejecutables independientes, y el medidor verifica identidad y descriptor nativos.

Los 22 escaneos mantienen las unidades de la adaptación MuJoCo. Los tamaños por eje van de 6,2–29,2 cm en X, 3,6–27,2 cm en Y y 1,5–29,2 cm en Z. La base del objeto está en (0,45, 0, 0,42) m y el robot tiene la base en el origen. La escena usa una mesa con superficie a 0,42 m y una pinza paralela simétrica de apertura total de 0 a 80 mm, recorrido de mandíbula de 40 mm, profundidad de dedo de 45 mm y fuerza máxima declarada de 90 N. El planificador suma 5 mm por mandíbula a la anchura requerida. Usa preagarre de 80 mm, elevación de 100 mm, margen de colisión de 1 mm, seis muestras del camino y seis reinicios IK. Los ejes articulados conservan sus límites nativos. La fuerza declarada del efector y la ε de wrench son parámetros de modelo, no una medida de fricción o masa real de los escaneos.

| N por pieza | Fracción compatible uniforme | Fracción compatible estratificada | Octantes medios uniforme / estratificado |
|---:|---:|---:|---:|
| 48 | 0,466 | 0,806 | 7,95 / 8 |
| 128 | 0,466 | 0,769 | 8 / 8 |
| 256 | 0,462 | 0,721 | 8 / 8 |
| 512 | 0,464 | 0,640 | 8 / 8 |

Los octantes miden una propiedad gruesa de la dirección de aproximación. La fracción de anchuras compatibles crece, pero el generador también puede agotar el estrato preferido. Se midieron desenlaces en el catálogo de 128. Las otras tres escalas describen geometría, sin extrapolación de la tasa de éxito.

| Conjunto y selección | Consultas por objeto | Intentos | F1 viables | Objetos con ≥1 viable |
|---|---:|---:|---:|---:|
| Train, uniforme | 10 / 25 / 50 | 160 / 400 / 800 | 2 / 5 / 16 | 2 / 4 / 6 |
| Train, estratificada | 10 / 25 / 50 | 160 / 400 / 800 | 3 / 9 / 17 | 3 / 5 / 7 |
| Validación, uniforme | 10 / 25 | 60 / 150 | 3 / 7 | 1 / 2 |
| Validación, estratificada | 10 / 25 | 60 / 150 | 3 / 4 | 2 / 2 |

Las 1.900 consultas de ambos diseños representan 1.841 pares objeto–ID distintos. El oráculo ejecutó 1.800 mediciones nuevas y acumuló 1.558,53 segundos en ellas. Un acierto de caché conserva el coste nominal del oráculo original. El recuento de tiempo fresco incluye cada ejecución nueva una sola vez. El archivo de transferencia usa solo el brazo estratificado de train. El JSON comprimido conserva IDs, rechazos, ε, estado de cierre y costes de ambos diseños, con los resultados de validación en claves separadas.

El selector sustituyó el cuello de botella. En train, los rechazos por anchura pasaron de 428 a 168, mientras los de preagarre inalcanzable subieron de 233 a 392 y las colisiones de aproximación de 41 a 102. Se registra la primera causa del planificador. Este diseño compara el paquete de selección completo y no identifica el efecto independiente de cada componente. No hay recuperación general demostrada. [Protocolo](protocol.json) · [IDs congelados](selection.json) · [Resultados por pieza](results.json) · [Archivo medido](native_observations.json.gz)

## Cambio de posición en la misma mesa

En el primer objeto train por cada una de cinco familias, desplazamos el objetivo 10 cm hacia la base del brazo y mantuvimos el apoyo completo sobre la mesa. De 50 poses, se emparejaron 49 por contactos y normales en el marco del objeto, orientación, ancho y centro. En las 49 coincidió ε dentro de 10⁻⁶. F1 dio 2 planes viables antes y 2 después. Una pieza ganó una pose y otra perdió una. Esto muestra sensibilidad al alcance y a colisiones incluso con idéntica geometría de contacto. El refinamiento continuo de poses SE(3) y una comparación entre efectores necesitarían un contrato equivalente para evaluar cada nueva pose. [Protocolo previo](placement_protocol.json) · [Mediciones pareadas](placement.json)

## Intento de recuperación mediante un predictor de alcance

Con los 800 resultados estratificados de train se ajustó un predictor geométrico con regularización. La información sobre el preagarre estaba disponible en 594 casos tras censurar causas anteriores, y solo 17 obtuvieron F1 viable. Se fijaron los coeficientes, el orden y los IDs antes de consultar el oráculo. En las seis piezas de validación se eligieron por pieza los doce primeros candidatos compatibles y doce aleatorios del remanente, todos nuevos. Ninguna de las 144 consultas produjo un plan viable. Entre los 72 mejores puntuados hubo 30 preagarres inalcanzables y 27 colisiones de aproximación. En los 72 controles, hubo 61 y 2, respectivamente. Los brazos son disjuntos y el control procede del remanente. Estas frecuencias describen un fallo de recuperación en este panel. [Protocolo](reach_protocol.json) · [Selección congelada](reach_selection.json) · [Causas y coste](reach_results.json) · [Observaciones](reach_observations.json.gz)
