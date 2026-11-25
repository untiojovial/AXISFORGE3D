# Diagnóstico post hoc de penetración

Tras observar penetraciones centimétricas en el piloto congelado, se fijan tres casos nominales ya observados: ACE Coffee Mug ID 0 y Android Figure Orange IDs 0 y 220. Se reejecuta exactamente el mismo escenario y trayectoria sin modificar criterios ni seleccionar casos por éxito. Se registra el contacto objeto–otra geometría con mayor penetración por etapa, su nombre de geometría, paso y fuerza normal. El máximo en todo el ciclo debe coincidir con el almacenado en el ensayo original. El análisis es mecanístico descriptivo, no réplica independiente ni regla retroactiva de exclusión; el resto de penetraciones permanece sin descomposición por pareja. Proxies y escena conservan los hashes del protocolo original.

## Sensibilidad post hoc a proxy

Para los mismos tres casos, se comparará el proxy original de piezas convexas con una única envolvente convexa del conjunto de vértices, manteniendo masa 0.1 kg, pose y comandos. Se registran hash, volumen de la envolvente, distancias de contactos, COM e inercias en ambos modelos. La envolvente rellena concavidades y no es ground truth; la inercia y COM pueden cambiar, de modo que no se atribuirán las diferencias de comportamiento exclusivamente a fidelidad superficial. No modifica el piloto congelado ni sus etiquetas.

## Controles adicionales de masa y perturbación

Extensión preregistrada después del estudio de penetración y antes de estas pruebas: se mantienen exactamente los tres casos anteriores. Para cada uno comparar nominal con desplazamiento +0.01 m del COM local del objeto a masa y tensor de inercia fijos, y con fuerza externa de +0.2 N en eje X mundial aplicada solo al objeto durante t∈[1.45,1.55) s de sostén. Verificar positividad y desigualdades triangulares de inercia; actualizar constantes y conservar estado inicial. Mantener controles numéricos y criterio originales. El cambio de COM con tensor fijo es perturbación paramétrica, no una nueva distribución de masa real validada. La fuerza es una perturbación explícita, no un mecanismo de sujeción; estas pruebas no demuestran robustez general.
