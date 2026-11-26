# Fase VI — registro de acceso a datasets

Comprobado el 2026-11-26 solo con peticiones HEAD a las páginas públicas. **No se descargó ni se evaluó
ningún objeto.** No se aceptó ninguna licencia en nombre del usuario y no se esquivó ningún control de
acceso.

| Dataset | Licencia (según la fuente) | Estado del acceso hoy | Uso en el proyecto | Qué haría falta para un test confirmatorio |
|---|---|---|---|---|
| Google Scanned Objects (Downs et al. 2022) | CC-BY 4.0 | Clon git sin blobs, revisión `6ff8d275…` | Biblioteca, Fase IV y prueba de la Fase V (ahora desarrollo) | **Agotado** con la regla de tamaño: la Fase V examinó 858 modelos para reunir 200 nuevos |
| YCB | Según su web; no se volvió a verificar en esta fase | Ya descargado (16 piezas) | Biblioteca | Pocas piezas; ya usadas |
| EGAD (Morrison et al. 2020) | BSD-3 | La página de datos de QUT responde 200 (en la Fase V respondió 403); GitHub responde 403 a través del proxy de esta sesión | Ninguno | Comprobar la descarga de las mallas desde QUT. Las 49 piezas de evaluación, imprimibles en 3D, encajan bien en un test confirmatorio |
| Objaverse (Deitke et al.) | ODC-By para el conjunto; cada objeto con la suya (CC-BY-NC, CC-BY-NC-SA, CC-BY-SA, CC0…) | Hugging Face responde 200 | Ninguno | Filtrar por licencia compatible (p. ej., CC0 o CC-BY), con malla cerrada y tamaño de agarre. Fijar la lista y su hash **antes** de evaluar |
| GraspNet-1Billion | No verificada | La página responde 200; la descarga suele requerir registro | Ninguno | Registro del usuario |
| BigBIRD | No verificada | Redirección 301 | Ninguno | Comprobar |
| ShapeNet | Requiere cuenta (Fase V) | Redirección 301 | Ninguno | Cuenta del usuario y licencia |
| KIT object models | No verificada | El intento por archive.org dio 404 | Ninguno | Buscar la fuente oficial |
| 8 objetos sellados de la Fase III | — | Presentes en el manifiesto, **no abiertos** | Reservados | Siguen reservados |

**Recomendación** para un test confirmatorio (diseño en [statistical_power.json](results/statistical_power.json)):

1. EGAD (49 objetos de evaluación más el conjunto de entrenamiento) si la descarga de QUT funciona.
2. Un subconjunto CC0/CC-BY de Objaverse elegido por reglas fijadas antes de mirarlo.
3. En ambos casos: congelar el protocolo, la lista de objetos y sus hashes antes de resolver ninguno.
