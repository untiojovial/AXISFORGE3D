# Esquema del manuscrito

**Título propuesto**: *Do Vision Foundation Models Improve Robotic Grasp Transfer? A Controlled Study of
Metric Geometry, Learned Feasibility and Visual Priors under Kinematic Constraints*

**Revista objetivo**: Engineering Applications of Artificial Intelligence o Expert Systems with Applications.
Alternativa: Intelligent Service Robotics o Robotics and Autonomous Systems. Antes de enviar hay que
verificar cuartiles y alcance.

## Resumen (borrador)

Los modelos fundacionales de visión se proponen a menudo para transferir conocimiento de agarre entre
objetos. Estudiamos, con controles y protocolos fijados de antemano, si mejoran la búsqueda de agarres de
un planificador con restricciones cinemáticas (brazo de seis ejes, pinzas paralelas de 40, 80 y 120 mm)
frente a priors geométricos y aprendidos.

- **Herramienta**: un simulador exacto del orden de examen del planificador, validado con 7 588 ejecuciones
  reales, permite comparar 59 ordenaciones en 200 objetos nuevos de Google Scanned Objects con una
  biblioteca congelada de 116 objetos.
- **Resultado principal**: un clasificador de viabilidad de candidatos aprendido de la biblioteca reduce
  los candidatos examinados hasta el primer agarre viable a 0,21 (IC 95 % 0,15–0,28) del orden por defecto,
  y a 0,57 (0,46–0,69) de transferir desde objetos de tamaño parecido.
- **DINOv3 no aporta**: añadirlo a un modelo adaptativo no cambia el resultado más allá de ±7 % (prueba de
  equivalencia con margen de ±25 %).
- **Apariencia y preentrenamiento**: buscar vecinos por apariencia es peor que por tamaño, aunque el
  preentrenamiento supera claramente a un ViT sin entrenar.
- **Robustez**: los resultados se mantienen entre datasets, con categorías no vistas y con tres pinzas. Una
  validación dinámica en MuJoCo indica que acelerar la búsqueda selecciona agarres menos robustos a
  perturbaciones.

## Contribuciones

1. Un protocolo controlado y reproducible para evaluar priors de transferencia de agarres bajo
   restricciones cinemáticas: un simulador exacto del orden validado con el planificador real, tres
   experimentos de generalización (biblioteca congelada, entre datasets y categorías no vistas) y tres
   pinzas.
2. Evidencia de que, en este problema, la viabilidad aprendida a nivel de candidato domina a la
   transferencia a nivel de objeto. Los modelos fundacionales mejoran la búsqueda por apariencia frente a
   un ViT sin entrenar, pero no aportan dentro del mejor sistema (equivalencia demostrada).
3. Un hallazgo de compromiso: priorizar la viabilidad reduce la robustez dinámica del primer agarre.

## Preguntas de investigación

| RQ | Respuesta | Evidencia |
|---|---|---|
| RQ1. ¿Transferir conocimiento reduce el coste de planificación? | Sí; sobre todo, el aprendizaje de viabilidad por candidato | Aprendido 0,21 frente al defecto; 0,57 frente al tamaño (P1) |
| RQ2. ¿La apariencia mejora la transferencia respecto al tamaño? | No: por apariencia es peor (1,22) y la fusión es equivalente (0,97) | Secundarios B 80 mm |
| RQ3. ¿El preentrenamiento aporta frente a un ViT aleatorio? | Sí para buscar vecinos (0,32, P4); no dentro de AFAT (equivalencia, P3) | P3, P4 |
| RQ4. ¿Las representaciones que conservan la escala mejoran la transferencia? | No demostrado en dos pruebas | Fase IV (escala) y secundarios de la Fase V |
| RQ5. ¿Se detecta y evita la transferencia negativa? | El aprendido y AFAT la reducen al 6–7 %, frente al 24 % de DINOv3; AFAT no mejora al aprendido | P2 y tasas de transferencia negativa |
| RQ6. ¿Se mantiene con otras pinzas? | Sí, con 40 y 120 mm | Tabla por pinza |
| RQ7. ¿Se traduce en agarres físicamente estables? | Nominalmente sí (89–95 %); bajo perturbaciones, el 49–64 %, y los priors eligen agarres menos robustos | MuJoCo (exploratorio) |

## Figuras y tablas

1. Esquema del planificador, el simulador exacto y el flujo de priors.
2. Éxito frente a presupuesto (200 piezas) y contrastes principales: [results/phase5.png](results/phase5.png).
3. Tabla de experimentos A/B/C × pinzas ([README](README.md)).
4. Tabla de validación dinámica y de robustez.
5. Desarrollo de la Fase IV (render normalizado frente a con escala), en material suplementario.

## Trabajo relacionado que hay que situar

- Transferencia por similitud geométrica: The Columbia Grasp Database (Goldfeder et al., 2009); DGCM-Net
  (Patten et al., 2020); transferencia con representaciones implícitas autoalineadas (2023); ZeroGrasp (2025).
- Modelos fundacionales de visión: DINOv2 (Oquab et al., 2023), DINOv3 (Siméoni et al., 2025), OpenCLIP.
- Descriptores de forma: D2 (Osada et al., 2002).
- Datasets: YCB (Calli et al., 2015) y Google Scanned Objects (Downs et al., 2022).
- Equivalencia: TOST (Schuirmann, 1987).
- Agarre aprendido: Contact-GraspNet (Sundermeyer et al., 2021), como comparación pendiente.

Antes de citar hay que comprobar cada referencia completa en su fuente.

## Limitaciones que el artículo debe declarar

- Sin robot real; la dinámica es un modelo de almohadillas aisladas con masa supuesta.
- Una sola celda y un solo brazo; pinzas paralelas, no manos.
- Margen de equivalencia de ±25 %, limitado por el tamaño de GSO.
- Sin comparación directa con métodos publicados de aprendizaje profundo.
- GSO tiene muchas variantes de producto. Las conclusiones se mantienen al excluirlas, pero la población es
  de productos de consumo.
