# Fase VI — preguntas de investigación

**Pregunta central.** ¿Pueden los priors de viabilidad aprendidos acelerar la planificación de agarres
sin perder robustez física cuando las propiedades del objeto son inciertas y el presupuesto de cálculo
es limitado?

| Id | Pregunta | Etapa | Medida | Dónde se responde |
|---|---|---|---|---|
| Q1 | ¿Son correctas y reproducibles las cifras y los artefactos de la Fase V? | A | Auditoría | [PHASE6_REPOSITORY_AUDIT.md](PHASE6_REPOSITORY_AUDIT.md) |
| Q2 | ¿Usa el clasificador información que no existe antes de planificar? | A | Auditoría de rasgos | [feature_audit.json](results/feature_audit.json) |
| Q3 | ¿Es fiable el modelo físico para este uso? | A/B | Controles con respuesta conocida y sensibilidad | [physics_validation.json](results/physics_validation.json) |
| Q4 | ¿En qué condición y por qué mecanismo perdía robustez el orden aprendido? | B | Modos de fallo, AUC por rasgo, estratificación | [robustness_mechanism.json](results/robustness_mechanism.json) |
| Q5 | ¿Se mantiene la pérdida con las masas que el protocolo pretendía? | B | Primeros agarres con v2 y masas correctas | ídem, B5 |
| Q6 | ¿Recupera la robustez una fusión simple de viabilidad y calidad, sin perder la aceleración? | C | N_F, R y N_FR con validación cruzada anidada; regla fijada | [fusion_benchmark.json](results/fusion_benchmark.json) |
| Q7 | ¿Son las probabilidades del clasificador probabilidades? | D | Brier, NLL y ECE; Platt e isotónica | [calibration_results.json](results/calibration_results.json) |
| Q8 | ¿Se puede saber pronto que una pieza no tiene agarre viable? | E | Demostrable, estimada o ninguna | [infeasibility_benchmark.json](results/infeasibility_benchmark.json) |
| Q9 | ¿Ahorrar candidatos ahorra segundos? | G | Tiempo real del planificador con un hilo | [cost_profile_online.json](results/cost_profile_online.json) |
| Q10 | ¿Cuántas piezas nuevas haría falta para confirmar Q6? | G | Potencia con la variabilidad de desarrollo | [statistical_power.json](results/statistical_power.json) |

Las respuestas, con sus límites, están en el [README](README.md).
