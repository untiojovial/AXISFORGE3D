# Resultados de desarrollo

## Qué mide el piloto

Se congelaron 48 poses por objeto mediante espaciado uniforme en el orden de generación de AxisForge. Las poses contienen orientación, apertura, posición y contactos propuestos. La política selecciona un ID antes de recibir su resultado. F1 exige que el planificador nativo encuentre un plan; F2 mide el espacio de llaves. La utilidad del piloto toma el `score` nativo acotado si existe plan y cero ante un rechazo F1. El `score` incluye heurísticas de la pinza. La regresión de representaciones predice por separado el valor continuo de ε de F2.

El historial contiene 160 consultas de 16 objetos y solo tres planes viables. Se ejecutaron nueve políticas × tres semillas × seis objetos de validación × diez consultas, con una inicialización compartida de dos candidatos por semilla. De las 1.620 consultas registradas, 87 produjeron un plan y 1.533 fueron rechazadas. Hubo cero errores de infraestructura. Se evaluaron 256 pares objeto–candidato distintos en validación, de los que ocho resultaron viables. Las 1.364 repeticiones reutilizaron la medición determinista. El archivo costó 65,05 s de evaluación nativa. Los costes de las consultas en línea fueron 628,73 s imputados y 100,03 s de mediciones nuevas. Estos tiempos excluyen, entre otros, preparación, cómputo del optimizador y extracción de embeddings. [Trazas](bo/results/development/observations.json.gz) · [Desglose por objeto](bo/results/development/summary.csv)

| Política | Media del mejor valor acumulado a lo largo de diez consultas |
| --- | ---: |
| Aleatoria | 0,0522 |
| GP objetivo con los mismos rasgos locales | 0,0529 |
| RGPE | 0,0526 |
| GP contextual agrupado | 0,0898 |
| Transferencia equiponderada | 0,0575 |
| Transferencia sin actualización por evidencia | 0,0575 |
| Transferencia adaptativa | 0,0575 |

La media pondera por igual las seis identidades y después las tres semillas. El valor medio del GP contextual fue mayor en esta partición exploratoria. El gate adaptativo produjo el mismo valor que sus controles. El número de objetos y la escasez de planes viables impiden establecer superioridad. [Trayectorias por objeto](bo/results/development/best_so_far.png)

## Qué aportaron las representaciones

El checkpoint real de OpenCLIP produjo representaciones globales en 352 vistas de los 22 objetos de desarrollo. La similitud coseno media entre la representación de 16 vistas neutrales y la de 16 vistas con las texturas del proveedor fue 0,753. Sobre 342.005 muestras de superficie visibles, 341.794 admitieron interpolación UV con la tolerancia de 2 µm. Las cachés externas incluyen nubes en metros, colores por punto, índices de vista y calibración. Los puntos repetidos entre vistas se conservan. La correlación exploratoria entre similitud visual neutral y similitud de dimensiones exteriores fue 0,114 en 231 pares dependientes. [Apariencia](vision/appearance_summary.json) · [Matriz visual](vision/similarity_16.png)

Se compararon propuestas de regiones de contacto proyectándolas en las cámaras y aceptando tokens visibles con tolerancia axial de 3 mm. Tres vecinos históricos (incluidos empates) predijeron el ε de diez consultas aleatorias por objeto de validación, 60 en total. El error absoluto medio por objeto fue 0,01530 con geometría, 0,01483 con visión global, 0,01668 con tokens locales y 0,01422 con la fusión equiponderada. Los descriptores locales por sí solos rindieron peor en esta muestra. Son predicciones de una medida estática en contactos propuestos. [Datos y predicciones](vision/contact_similarity.json)

La comparación de regresión multimodal usó PCA ajustada sobre entrenamiento, un GP RBF, un GP Matérn-5/2 y un DKL neuronal con tres inicializaciones. La cobertura empírica de intervalos nominales del 90 % fue 80 % para ambos GP estacionarios y entre 45 % y 55 % para DKL. La incertidumbre del DKL quedó mal calibrada. Estos resultados desaconsejan promoverlo a política de adquisición en este piloto. [Resultados DKL/GP](bo/results/representation_gp.json)

## Qué se levantó en MuJoCo

Se eligieron, por ID ascendente, hasta cuatro candidatos que habían pasado F1 por objeto de validación. Esa selección condicionada dejó seis poses en tres objetos. Para cada pose se conservó el objeto libre, se cargaron las piezas de colisión oficiales GSO y se reprodujeron el marco, geometría de palma y dedos, apertura, ejes y límites de fuerza de la pinza nativa. Dos etapas actuadas sustituyen al brazo para el acercamiento y la elevación. Se supusieron masa de 0,1 kg y fricción de 0,5. La escala se interpreta a partir del modelo de origen. [Geometrías y resultados](robot/contact_probe_development.json)

Tres poses del tricerátops cumplieron simultáneamente elevación ≥40 mm, desplazamiento relativo ≤20 mm, contacto bilateral durante al menos el 90 % del mantenimiento y ausencia de contacto con la mesa en ese intervalo. Elevaron el centro de masas 54–59 mm. Las otras tres poses fallaron. Ninguno de los seis controles con la pinza abierta cumplió el criterio. En el caso de la botella, el contacto bilateral persistió pero la altura y el desplazamiento relativo incumplieron el umbral. Al reducir la fricción a 0,2, dos poses superaron el criterio. El registro incluye fuerzas normales, penetración, desplazamiento durante asentamiento y barridos de masa.

Los seis candidatos quedaron dentro de 3 mm de la frontera de la unión de proxies a lo largo de las normales propuestas. Esta medición local deja abiertas discrepancias en otras zonas. La masa, inercia y fricción no se midieron físicamente. Las tres sujeciones del mismo objeto tampoco permiten estimar una tasa poblacional o la correlación F2–F3 en piezas nuevas. El test final permanece sellado. [Revisión robótica](reviews/robotics.md)
