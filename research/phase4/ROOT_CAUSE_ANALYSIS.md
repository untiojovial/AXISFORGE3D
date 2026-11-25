# Hipótesis de causa y experimentos discriminantes

| Observación anterior | Explicación comprobable | Intervención y control |
|---|---|---|
| 245 rechazos por apertura | El muestreo uniforme del orden nativo dedica consultas a anchuras incompatibles con la pinza. La escala física del escaneo añade incertidumbre. | Secuencia nativa de máxima separación en centro, apertura y orientación. Cuatro de cada cinco posiciones priorizan la anchura compatible y la quinta reserva un control incompatible si existe. Se conserva la holgura de 5 mm por mordaza y la pinza original de 80 mm. |
| 94 rechazos por preagarre inalcanzable | Una anchura válida no garantiza alcance, colisión libre ni cierre de fuerza. | Medir la primera causa nativa por pieza y por selección. Conservar las mismas poses de escena y registrar propuestas y F1 por separado. |
| ACT-BO y control alcanzaron la misma utilidad | Tres positivos históricos y una mezcla escasamente informativa pueden modificar pesos sin cambiar la decisión. | Reproducir los rankings de adquisiciones en prefijos idénticos, y comparar por separado pesos, predicciones, elecciones y utilidad. Ampliar memoria reteniendo los rechazos. |
| DKL con intervalos demasiado estrechos | La varianza condicional y la pequeña población de entrenamiento infrarrepresentan el error entre objetos. | Validación externa al objeto dentro de train y recalibración en grupos internos. Medir simultáneamente anchura, cobertura y densidad predictiva contra GP y constante. |
| Éxitos dinámicos concentrados | Selección condicionada por F1, proxies de contacto, desplazamiento durante asentamiento y brazo simplificado. | Muestreo previo en más piezas, controles abiertos, perturbaciones y sensibilidad a integrador/contacto. Contrastar cinemática y seguimiento de un brazo articulado con inercias identificadas. |

Las frecuencias después de seleccionar por anchura describen el muestreo diseñado. Estimar la prevalencia en todas las poses exigiría un muestreo representativo independiente. Un contacto del proxy en una pose cinemáticamente rechazada carece de una trayectoria ejecutable del brazo.
