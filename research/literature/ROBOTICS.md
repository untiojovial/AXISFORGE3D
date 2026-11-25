# Fuentes primarias y permisos

| Recurso | Fuente | Alcance y límites |
|---|---|---|
| MuJoCo | https://github.com/google-deepmind/mujoco/blob/main/LICENSE | Apache-2.0; conservar avisos. |
| Bullet | https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt | zlib en la base; exclusiones para Extras y librerías de terceros. |
| iCub models | https://github.com/robotology/icub-models | modelos CC BY-SA 4.0; atribución y condiciones de compartir igual; variantes Gazebo alteran inercias y amortiguación. La variante original del TFM no se conoce. |
| Objetos del TFM | Tesis secciones 4.1, 5.1 | Ocho objetos nombrados, origen, archivos, licencia y parámetros físicos no identificados; no redistribuir como originales. |
| YCB | https://www.ycbbenchmarks.com/object-models/ | Página inaccesible al consultar; licencia de datos no verificada. |

El TFM describe epsilon del espacio de wrenches en §2.3 y también un proxy de volumen/tasa de éxito en §5.1.2; no se conoce implementación exacta. Sus datos originales usan Simox e iCub, no esta pinza. AxisForge3D implementa cierre geométrico con objeto fijo, fricción idealizada y epsilon estático; no demuestra un agarre dinámico.

## Datos y modelos candidatos (revisión 2026-11-25)

La licencia del software no autoriza automáticamente activos, vídeos, anotaciones ni checkpoints; «no verificada» significa que no se incorporarán hasta resolverla. Ningún dataset se ha descargado en este trabajo.

| Fuente primaria | Datos, efector y anotaciones | Acceso y derechos |
|---|---|---|
| [GraspNet-1B / API](https://github.com/graspnet/graspnetAPI) | Escenas RGB-D y etiquetas de agarre para pinza paralela; pose, apertura y calidad de candidatos. | Acceso por sitio del proyecto; API [MIT](https://github.com/graspnet/graspnetAPI/blob/master/LICENSE), términos de imágenes, mallas y etiquetas no verificados. |
| [ACRONYM](https://github.com/NVlabs/acronym) | 17,7 millones de agarres simulados, 8.872 objetos y pinza paralela; éxito/fracaso FleX. | Código MIT y dataset CC BY-NC 4.0 según README. Mallas ShapeNet y condiciones de procedencia requieren revisión individual; uso comercial excluido por NC. |
| [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) | 1,32 millones de agarres, 5.355 objetos, ShadowHand, validación descrita en Isaac Gym; variantes Allegro y MANO. | Repositorio y descargas descritos por autores; licencia de código, etiquetas y activos no verificada; algunos datos derivan de objetos externos. No implica compatibilidad con iCub. |
| [DexGraspNet 2.0](https://github.com/PKU-EPIC/DexGraspNet2) | Escenas sintéticas con clutter y grasps diestros, código también contempla experimentos con pinza. | Dataset enlazado desde README a Hugging Face; términos de código, datos, mallas y componentes Isaac Gym no verificados. |
| [YCB-Video](https://github.com/yuxng/YCB_Video_toolbox) | 21 objetos YCB, 92 vídeos, 133.827 fotogramas y pose 6D; no son etiquetas de agarre ni datos iCub. | README declara dataset MIT; verificar alcance exacto por componente y licencias de mallas YCB originales antes de redistribuirlas. |
| [ShapeNet](https://shapenet.org/) | Colecciones CAD 3D y categorías; sin etiquetas físicas de agarre. | Acceso y licencia de cada versión/modelo no verificados en fuente accesible; no asumir licencia general ni redistribuir mallas. |
| [Objaverse-XL](https://github.com/allenai/objaverse-xl) | Más de 10 millones de objetos 3D heterogéneos; no son por sí mismos datos de agarre. | Herramientas Apache-2.0; base como conjunto ODC-By 1.0, pero cada activo tiene licencia propia que debe verificarse. Sin licencia verificable, excluir. |
| [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) | Modelo de propuestas 6D a partir de nubes de puntos; pinza paralela, no mano iCub. | README enlaza checkpoints y datos externos; licencia de código y condiciones concretas de pesos/datos no verificadas; no redistribuir ni usar hasta resolver. |

### Plan de manifiesto y particiones

Cada activo requiere identificador y URL de origen, versión o revisión inmutable, checksum, autor, licencia exacta y condiciones de atribución, permiso de uso/redistribución, fecha de acceso, modelo y unidad de escala, transformaciones, familia de procedencia y equivalencia geométrica. Cada etiqueta debe registrar simulador/hardware, efector, marco, convención, método y disponibilidad temporal. Crear particiones por identidad y familia geométrica antes de ajustes, junto a una lista explícita de exclusiones; mantener duplicados, variantes del mismo CAD y vídeos del mismo objeto en un único split; registrar semilla y hash del manifiesto. Reservar categorías y objetos fuera de entrenamiento para evaluación y detectar solapamientos entre fuentes; si no puede determinarse la procedencia o la licencia de un activo, excluirlo. No tratar imágenes de YCB-Video ni mallas de ShapeNet/Objaverse como etiquetas de éxito. Las etiquetas de pinza y mano deben conservar sus efectores y no son intercambiables.
