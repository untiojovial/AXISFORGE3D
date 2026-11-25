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

La media pondera por igual las seis identidades y después las tres semillas. El valor medio del GP contextual fue mayor en esta partición exploratoria. Entre los 416 pares distintos de entrenamiento y validación, 245 se rechazaron por la apertura requerida y 94 porque el brazo no alcanzó la pose previa. Tres de las seis piezas de validación no produjeron ningún plan viable. [Diagnóstico F1](methodology/F1_DIAGNOSTIC.md) El gate adaptativo produjo el mismo valor que sus controles. El número de objetos y la escasez de planes viables impiden establecer superioridad. [Trayectorias por objeto](bo/results/development/best_so_far.png)

El ensayo multiefector generó seis catálogos adicionales sobre tres piezas de entrenamiento. Se evaluaron tres IDs prefijados por catálogo para la pinza de tres dedos y la mano articulada. El motor calculó contactos y rechazó los 18 candidatos por cinemática, colisión o cierre de fuerza. El ensayo dinámico se limita a la pinza paralela. [Resultados por efector](robot/native_multieffector_results.json)

## Qué aportaron las representaciones

El checkpoint real de OpenCLIP produjo representaciones globales en 352 vistas de los 22 objetos de desarrollo. La similitud coseno media entre la representación de 16 vistas neutrales y la de 16 vistas con las texturas del proveedor fue 0,753. Sobre 342.005 muestras de superficie visibles, 341.794 admitieron interpolación UV con la tolerancia de 2 µm. Las cachés externas incluyen nubes en metros, colores por punto, índices de vista y calibración. Los puntos repetidos entre vistas se conservan. La correlación exploratoria entre similitud visual neutral y similitud de dimensiones exteriores fue 0,114 en 231 pares dependientes. [Ejemplo RGB-D y nube de la taza GSO, Google Research, CC BY 4.0](vision/observations_mug.png) · [Apariencia](vision/appearance_summary.json) · [Matriz visual](vision/similarity_16.png)

Se compararon propuestas de regiones de contacto proyectándolas en las cámaras y aceptando tokens visibles con tolerancia axial de 3 mm. Tres vecinos históricos (incluidos empates) predijeron el ε de diez consultas aleatorias por objeto de validación, 60 en total. El error absoluto medio por objeto fue 0,01530 con geometría, 0,01483 con visión global, 0,01668 con tokens locales y 0,01422 con la fusión equiponderada. Los descriptores locales por sí solos rindieron peor en esta muestra. Son predicciones de una medida estática en contactos propuestos. La referencia constante aprendida del historial obtuvo MAE 0,01541 y Brier F1 0,03243. Esa referencia superó a todas las variantes de vecinos en Brier. [Datos y predicciones](vision/contact_similarity_v2.json)

La comparación de regresión multimodal usó PCA ajustada sobre entrenamiento, un GP RBF, un GP Matérn-5/2 y un DKL neuronal con tres inicializaciones. La cobertura empírica de intervalos nominales del 90 % fue 80 % para ambos GP estacionarios y entre 45 % y 55 % para DKL. La incertidumbre del DKL quedó mal calibrada. Estos resultados desaconsejan promoverlo a política de adquisición en este piloto. La referencia constante alcanzó cobertura del 85 %. [Resultados DKL/GP](bo/results/representation_gp_v2.json)

## Qué se levantó en MuJoCo

Se eligieron, por ID ascendente, hasta cuatro candidatos que habían pasado F1 por objeto de validación. Esa selección condicionada dejó seis poses en tres objetos. Para cada pose se conservó el objeto libre, se cargaron las piezas de colisión oficiales GSO y se reprodujeron el marco, geometría de palma y dedos, apertura, ejes y límites de fuerza de la pinza nativa. Dos etapas actuadas sustituyen al brazo para el acercamiento y la elevación. Se supusieron masa de 0,1 kg y fricción de 0,5. La escala se interpreta a partir del modelo de origen. [Geometrías y resultados](robot/contact_probe_development.json)

Tres poses del tricerátops cumplieron el criterio sostenido comprobado en los 400 pasos del mantenimiento. La altura mínima del centro de masas fue 51–55 mm sobre el nivel tras el asentamiento, el desplazamiento relativo máximo fue 4–10 mm y ambas mordazas registraron fuerza normal superior a 0,01 N durante todo el intervalo. Hubo cero contactos con la mesa y cero penetración inicial de la pinza. Los umbrales del diagnóstico fueron elevación mínima de 40 mm, desplazamiento máximo de 20 mm y fuerza bilateral durante al menos el 90 % del intervalo. La altura final de esas tres poses fue de 54–59 mm. Las otras tres poses fallaron. Ninguno de los seis controles con la pinza abierta cumplió el criterio. En el caso de la botella, el contacto bilateral persistió pero la altura y el desplazamiento relativo incumplieron el umbral. Al reducir la fricción a 0,2, dos poses superaron el criterio. El registro incluye fuerzas normales, penetración, desplazamiento durante asentamiento y barridos de masa.

Los seis candidatos quedaron dentro de 3 mm de la frontera de la unión de proxies a lo largo de las normales propuestas. Esta medición local deja abiertas discrepancias en otras zonas. La masa, inercia y fricción no se midieron físicamente. Las tres sujeciones del mismo objeto tampoco permiten estimar una tasa poblacional o la correlación F2–F3 en piezas nuevas. El test final permanece sellado. [Revisión robótica](reviews/robotics.md)

## Presupuesto y transferencia local

Al ampliar a veinte consultas, se conservaron los mismos 160 candidatos históricos, 48 candidatos por pieza, seis objetos de validación, nueve métodos y tres semillas. En estas 162 trayectorias, la media del mejor valor acumulado fue 0,09833 con el GP contextual, 0,08328 con búsqueda aleatoria, 0,07031 con RGPE y 0,06872 con la política adaptativa. La variante sin actualización por evidencia también obtuvo 0,06872. Los ensayos de 10 y 20 consultas comparten candidatos y semillas. Los presupuestos 50 y 100 exceden el catálogo congelado de 48 poses sin reemplazo. [Resultados a 20 consultas](bo/results/development-20/summary.csv)

Los contextos únicamente visual y únicamente geométrico obtuvieron 0,05750 con la política adaptativa a diez consultas. El control visual fijo también obtuvo 0,05750. En un panel adicional se incorporaron estadísticas de los contactos propuestos en el marco de agarre y cuatro componentes de los tokens locales visibles. La PCA se ajustó sin etiquetas, sobre 768 candidatos de 16 piezas históricas, ponderando por igual cada objeto. Se verificaron las 22 matrices y la PCA antes de ejecutar las políticas. El bloqueo exacto corresponde a la configuración original; con BLAS de un solo hilo la discrepancia numérica máxima en las características fue 1,61 × 10⁻¹⁵. [Protocolo del panel](pilot/functional_protocol.json)

| Política con las mismas 33 características | Calidad acumulada media a 10 consultas |
|---|---:|
| GP ajustado solo a la pieza objetivo | 0,08507 |
| Transferencia equiponderada | 0,07589 |
| Transferencia adaptativa | 0,06232 |
| Transferencia sin actualización por evidencia | 0,06232 |

En este panel de 72 trayectorias el control sin historia superó a las tres variantes de transferencia en la media descriptiva. Por tanto, los datos de desarrollo no justifican promover la correspondencia visual local o el gate como mejora general de ACT-BO. [Resultados funcionales](bo/results/functional/paired.csv) · [Control solo visual](bo/results/visual/paired.csv) · [Control solo geométrico](bo/results/geometry/paired.csv)
