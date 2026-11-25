# Auditoría independiente de fase III

Referencia inspeccionada: `e4e4f70795f0d880b817510e35180408de36e13a`, checkout inicialmente limpio en rama `integration/axisforge-phase2`. Se mantiene abierto el dictamen científico de fase II; un test de interfaz con pesos aleatorios no demuestra un modelo fundacional preentrenado, y el piloto de cubo no demuestra transferencia entre objetos.

M0 fase III **PENDING**: faltan revisión de capacidades nativas y nueva regresión de referencia. M1–M12 **PENDING** hasta evidencia específica. No se asigna puntuación 95/100 sin evaluación.

Diseño preliminar dataset: aceptable condicionado a licencia individual y fuente verificables, originales inmutables y hashes, unidad basada en evidencia, límites de convex hull/concavidades, separación por identidad y familia, y test sellado. No convertir falta de derechos o escala en valores supuestos presentados como reales.

Diseño preliminar renderer: aceptable para desarrollo diagnóstico con marcos y ejes declarados, z axial, z-buffer, planos y oclusiones analíticos, normales y background definidos, pruebas de proyección, y restricciones explícitas de neutralidad/texturas y coste de vistas. Solo pesos verificablemente preentrenados acreditan modelo; no utilizar test para ajuste.

BO: diseño `research/phase3/bo/DESIGN.md` aceptado para interfaces y pilotos condicionados a prueba de no-futuro y costes, distinguir features pre-evaluación de contactos posteriores y controlar target-only con las mismas features. B14/B15 no quedan aprobados. F2 y F3 no son por definición fidelidades de una misma función.

Robótica: catálogo nativo y aplicación de pose completa aceptados para desarrollo condicionado a validar marcos, resultados de IK/colisiones, modelos y fuerzas, fallos y costes; pinza simplificada no representa fielmente brazo o mano nativa. Se exige bloqueo de concavidades cuando la geometría de colisión afecte la región de agarre. No se aprueban M4/M5 por este diseño.

Integración PI: banco y callback admisibles condicionados a aislamiento efectivo de etiquetas futuras, prueba de reanudación/identidad, separación de costes y fallos, y viewer atado a hashes y observaciones reales. El posible filtrado del planificador limita la población de candidatos. Test permanece sellado.

Primera auditoría de `CataloguePolicy`: **REJECTED para integración hasta corrección**. El constructor admite directamente fuentes que omiten `approved_history`, por lo que no impone autorización de split/identidad; `ask` puede avanzar inicialización antes de completar predicción; el diagnóstico F2/F3 no categoriza fallos físicos. Hallazgos enviados para corrección y nueva auditoría. Renderer inicial: interpolación perspectiva y z-buffer razonables; faltan casos analíticos inclinados, transformaciones y límites de recorte. No equivale a observación realista.

Reauditoría BO: 15 pruebas pasan tras comprobación interna de historial, transacción en `ask` y conteo de fallos físicos. Se retira veto para fixtures. **Permanece veto para datos reales hasta comprobar runner y vinculación del mapa de splits al manifiesto íntegro y familias**: hash suministrado junto al mapa prueba consistencia, no autenticidad ni ausencia de fuga. Renderer: tres pruebas independientes superadas para plano inclinado, oclusión y transformación; aceptado para diagnóstico geométrico limitado, no M2.

Dataset externo: manifiesto SHA-256 `6c5291288adcd365b873a4c983b30b9f19fb2ed0110a0575112514b866cdf660`; 30 objetos, 16 train/6 validation/8 test. Se cotejaron directamente 60 hashes de originales OBJ y STL, todos correctos; ids y originales únicos por hash. M1 **PARTIALLY_VALIDATED**, no aprobado: hashes exactos no identifican near-duplicates o identidad/familias; escala inferida del MJCF no es medición independiente; licencias de activos, disponibilidad real de texturas, colisiones/concavidades y físicas aún deben verificarse. Los objetos test no se han evaluado. 

Se comprobó OpenCLIP local: 605159845 bytes, SHA-256 `d867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e`; una imagen negra CPU produjo embedding finito de 512 dimensiones y norma 1. Esto acredita únicamente ejecución local; falta cotejo primario de procedencia y licencia y evaluación de utilidad. No acredita DINOv2 ni M3.

El runner de desarrollo y pruebas adversariales fueron revisados: lock del manifiesto, acceso a train/validation y entrega posterior a `ask` ofrecen barreras para piloto F1/F2; 6 pruebas de experiment/prepare aprobadas. El desacuerdo temporal de esquema entre `metadata` y campos raíz fue corregido y comprobado en disco. Se retira el veto exclusivamente para piloto exploratorio F1/F2 con artefactos congelados; no autoriza test, F3 ni afirmación de éxito dinámico. `physical_failure` en este protocolo significa rechazo F1 de planificación. Catálogos dependen de marcos, identificadores y datos visuales cuya procedencia debe quedar verificada y trazable.
