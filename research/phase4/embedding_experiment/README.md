# ¿Añaden información los descriptores OpenCLIP?

Se renderizaron 16 vistas RGB por cada uno de 22 escaneos de desarrollo. Dos extracciones congeladas en CPU devolvieron los mismos bytes en los 22 NPZ. El archivo de observaciones aporta diez configuraciones por objeto. Los 16 objetos de entrenamiento aportan 160 respuestas históricas. Las seis piezas de validación aportan un pequeño dominio de búsqueda de diez configuraciones ya medidas por pieza. Cada trayectoria selecciona seis consultas de ese dominio, con dos iniciales compartidas y tres semillas. El resultado es la utilidad analítica condicionada a que exista el plan nativo. Se conservan los tiempos del oráculo original. La reproducción desde caché no ejecuta el robot.

El panel contiene cinco representaciones y cuatro políticas. Los componentes principales globales y locales se ajustan exclusivamente con train. El descriptor global se agrega por objeto. Los locales se proyectan mediante cámara, profundidad y recorte, se deduplican por token dentro de la vista y se promedian sobre las vistas visibles. El control aleatorio conserva las dimensiones y usa una semilla estable por objeto. OpenCLIP usa los mismos pesos para todos los métodos.

| Características | AUC aleatorio | AUC GP objetivo | AUC contextual | AUC adaptativo | MAE ε | Brier F1 |
|---|---:|---:|---:|---:|---:|---:|
| Geometría | .04860 | .03824 | .04170 | .04170 | .01378 | .03176 |
| Control aleatorio | .04860 | .05206 | .04515 | .04503 | .01402 | .03247 |
| Geometría + global | .04860 | .03824 | .04170 | .04170 | .01350 | .03231 |
| Geometría + local | .04860 | .03824 | .05217 | .05217 | .01335 | .03204 |
| Geometría + ambos | .04860 | .03824 | .05217 | .05217 | .01352 | .03246 |

Las señales locales cambian la AUC media exploratoria en +.01047 respecto a geometría en el método adaptativo. El Brier de factibilidad empeora ligeramente. La constante estimada en train tiene MAE .01541 y Brier .03243. Las 360 trayectorias comparten seis objetos, los primeros candidatos y los desenlaces almacenados. Estos contrastes descriptivos no establecen superioridad de OpenCLIP ni informan sobre DINO o éxito dinámico. Los diez candidatos por objeto limitan la extrapolación a los catálogos más grandes.

[Trayectorias y predicciones](openclip_analysis.json) · [Resumen verificable](summary.json) · [Figura](openclip_panel.png) · [Dos ejecuciones y sus hashes](../automation/openclip_reproduction.json)

La reconstrucción de las entradas de geometría, visibilidad y etiquetas se puede ejecutar mediante `python -m research_grasping.phase4.portable_panel --data DATA --binary ORACULO_FASE_III --out DIRECTORIO`. El importador contrasta el ZIP original, el sello independiente de las imágenes y las identidades del modelo. Los resultados de un tercero requieren replicación independiente de la inferencia. El notebook incluye el trabajo de imágenes de desarrollo, atribuido a Google Scanned Objects (Downs et al., 2022) y a la adaptación MuJoCo de Kevin Zakka, CC BY 4.0. [Procedencia del conjunto](../../phase3/dataset/README.md)
