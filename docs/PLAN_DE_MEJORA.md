# Plan de mejora: de demostrador a producto vendible

Este documento dice dónde está AXISFORGE hoy, qué le falta para venderse y en
qué orden conviene construirlo. Cada fase tiene un criterio de aceptación
comprobable; una fase no está hecha hasta que lo cumple.

No hay fechas: el tamaño de cada fase se da como S (días), M (semanas) o
L (meses) de una persona, y es una estimación, no un compromiso.

## 1. Dónde estamos

Hecho y probado (353 pruebas automáticas, sin avisos del analizador):

- Modelo de robot, cinemática directa e inversa, jacobiano y condicionamiento.
- Colisiones por GJK entre primitivas y envolventes convexas.
- Pinza paralela como articulaciones normales del árbol cinemático.
- Síntesis de agarres para caja, cilindro, cápsula y esfera.
- Calidad por **espacio de llaves**: cierre de fuerza y épsilon sobre los
  parches de contacto de las almohadillas.
- Planificador de recogida: pre-agarre, agarre, cierre y elevación, con el
  motivo de cada descarte.
- Parámetros con el vocabulario de GraspStudio (sección 3).
- Registro de extensiones para conectar bibliotecas (sección 4).
- Formato de proyecto versionado, CLI, motor en WebAssembly y visor web con
  render de estudio.
- Importación de robots URDF y XML de Simox, objetos de Simox y mallas STL,
  OBJ y COLLADA, en la CLI y en el visor (fase 1).
- Agarre de mallas por normal de superficie con cierre hasta contacto,
  descomposición convexa, y **manos**: pinza de tres dedos y mano de tres
  dedos articulados con actuadores y preformas (fase 2, en parte).
- **Movimiento completo** (fase 3): comprobación de colisión continua, RRT
  bidireccional con atajos, ciclo recoger–transportar–depositar con perfiles
  dentro de los límites de cada eje, y exportación a URScript, CSV y JSON.

Lo que hoy impide venderlo:

| Carencia | Por qué bloquea la venta |
| --- | --- |
| Mallas Open Inventor (.iv) sin leer | Casi todos los modelos de Simox las usan; esos eslabones quedan sin colisión |
| Visor de demostración, no editor | No se puede crear ni guardar una celda propia |
| Sin ventosa | Muchas celdas de manipulación usan vacío |
| Movimientos sin enlazar (parada en cada punto) | Ciclos más lentos que los de un programa afinado a mano |
| Sin instalador, licencias ni soporte | No hay nada que entregar ni cobrar |
| Sin validación contra hardware | Ningún número está contrastado con una celda real |

## 2. Fases

### Fase 1. Importar el mundo real (M)

- Importador URDF: enlaces, articulaciones, límites, mallas STL y OBJ.
- Importador del XML de robots y objetos de Simox (sección 4.3).
- Mallas en el visor y en colisión (envolvente convexa por pieza; descomposición
  convexa para piezas cóncavas).

**Aceptación:** tres robots públicos distintos se importan, se dibujan y su
cinemática directa coincide con la de referencia a menos de 0,1 mm en diez
posturas cada uno.

**Estado: criterio de aceptación cumplido. Queda pendiente la descomposición
convexa de piezas cóncavas de una sola cáscara y la lectura de Open
Inventor.**

- Hecho: importador URDF (enlaces, articulaciones de revolución, prismáticas,
  continuas y fijas, límites, orígenes, mallas STL y OBJ con escala,
  materiales); importador del XML de robots de Simox (unidades, `Transform`
  con todos sus hijos, DH, desfases, primitivas, efector final) y de objetos
  `ManipulationObject`; lector de STL binario y ASCII, de OBJ y de COLLADA
  (triángulos, polilistas, polígonos, tiras y abanicos; nodos con matriz,
  traslación, giro y escala; `instance_node`; unidad; color difuso de cada
  material). Las mallas se dibujan con sus colores y chocan como la
  envolvente convexa de cada cáscara cerrada que contienen, o de la malla
  entera si alguna está abierta. El visor importa archivos y
  trae un UR5 real de ejemplo; la CLI tiene `import-urdf`, `import-simox`,
  `add-object` y `fk`, con `--gripper` y `--demo-cell` para planificar en
  seguida.
- Verificación (detalle en `tools/verify/README.md`): UR5, Kinova Jaco2 y
  Unitree Z1, de `example-robot-data`, comparados con una cinemática directa de
  referencia independiente del producto: peor error 4,0e-13 mm y 7,2e-14°
  en 340 poses de eslabón. El UR5 se compara además con los parámetros DH que
  publica Universal Robots, que no comparten nada con el URDF: 4,7e-9 mm, y
  esa comprobación corre en `cargo test`.
- Dibujo: el UR5 y el Kinova con sus mallas COLLADA reales y sus colores.
  Las mallas visuales del UR5 coinciden con sus mallas de colisión STL,
  modeladas aparte, a menos de 0,53 mm en los siete eslabones.
- Fallos encontrados por el camino, los dos en el cálculo de distancias
  (GJK): no convergía cerca del contacto entre caras casi paralelas (464 de
  2000 casos de prueba entre 0,1 µm y 10 cm), y con una malla apoyada en el
  suelo de 100 m podía tomar un tetraedro casi plano que contenía el origen
  por uno que no, y acabar leyendo fuera de un vector: el motor se caía al
  importar un robot con mallas reales. Corregidos los dos; pruebas de 600 y
  2000 casos los vigilan.
- Pendiente: descomposición convexa de una cáscara cóncava (hoy choca como
  su envolvente: puede ver choques que no hay, nunca pasar por alto uno);
  lectura de Open Inventor (los eslabones afectados no tienen colisión, y la
  etiqueta de contacto y el veredicto del plan lo dicen);
  robots de siete ejes como el Panda se rechazan, porque el producto es de uno
  a seis; las articulaciones `mimic` se importan como independientes, con
  aviso; las flotantes y planas se rechazan.

### Fase 2. Agarre genérico, paridad con GraspStudio (L)

- Generador por **normal de superficie**: puntos muestreados en la superficie
  de la pieza y aproximación a lo largo de la normal.
- **Cierre hasta contacto**: las mordazas avanzan hasta tocar, y los contactos
  salen de la geometría real, no de una fórmula por forma. Esto abre el agarre
  de mallas.
- Efectores con varios actuadores y preformas (pinza de tres dedos, mano).
- Conjuntos de agarres guardados por objeto y pinza en el proyecto.
- Ventosa como segundo tipo de efector, con su propio modelo de sujeción.

**Aceptación:** en veinte piezas de malla de un banco público, el planificador
encuentra agarres con cierre de fuerza en al menos las mismas piezas que
GraspStudio con parámetros equivalentes, y se documenta cada diferencia.

**Estado: hechos el agarre de mallas y las manos de varios dedos; la
comparación con GraspStudio, los conjuntos de agarres guardados y la
ventosa siguen pendientes.**

- Hecho: generador `surface_normal` con las dos familias (cerrar a lo largo
  de la normal, llegar por ella), cada una a la profundidad que permite la
  palma y en variante corta que no toca el lado lejano; cierre hasta
  contacto de almohadillas planas sobre la geometría real, con tolerancia de
  0,3 mm y la normal de la superficie donde aprieta el borde de la
  almohadilla; calidad tomada en el centro de masas real de la malla; `auto`
  elige el generador según la pieza. Una pieza se añade desde un STL, OBJ o
  COLLADA suelto, en el visor, la CLI y el servicio.
- Comprobado: cerrar las almohadillas en cada pose del generador de
  primitivas reproduce exactamente su anchura, sus contactos y sus normales
  (caja y cilindro); el cubo planificado como malla da épsilon 0,382 frente
  a 0,386 exacto; una pieza de dos cajas, antes rechazada, se planifica.
- Banco YCB (los 16 objetos de GRASPA, detalle en `tools/verify/README.md`):
  la síntesis encuentra agarres con cierre de fuerza que caben en la pinza
  en 15 de 16 (la que falta es una lata de 102 mm que la pinza de 80 mm no
  puede coger); el brazo de demostración recoge 12. Las cuatro que no
  recoge fallan por su alcance (caja de 21 cm de alto, martillo tumbado) o
  por ser latas más anchas que la carrera.
- Descomposición convexa de las mallas de colisión cóncavas, conservadora
  (los cascos de las piezas contienen todo el sólido, no solo su
  superficie): el hueco entre el modelo de colisión y el sólido baja, por
  ejemplo, de 574 a 34 cm³ en el martillo y de 94 a 17 cm³ en el plátano, y
  el brazo pasa de recoger 9 objetos YCB a 12.
- Manos de varios dedos, hechas: la pinza describe **actuadores** (dedos:
  articulaciones que cierran juntas, cada una con su sentido y razón, como
  los `direction` de los actores de Simox) y **preformas** con nombre. Dos
  plantillas: pinza centrada de tres mordazas radiales y mano de tres dedos
  de dos falanges. Cada dedo cierra por avance conservador con una cota del
  recorrido de sus eslabones, así que nunca atraviesa una pieza fina; una
  falange que toca detiene su articulación y la punta sigue curvándose. Los
  agarres se proponen como GraspStudio con una mano (punto de la superficie,
  aproximación por la normal, retirada hasta despejar, cierre) más tres
  variantes: hasta el fondo, centrada en la pieza e inclinada 25°. Los
  contactos son los de GraspStudio (un punto por par de piezas convexas que
  se tocan) y se juzgan en el espacio de llaves. Un efector de Simox con
  actores se importa como mano, con sus preformas.
- Comprobado en la celda de demostración: la pinza de tres dedos y la mano
  recogen el cubo y la clavija con cierre de fuerza; el cierre se detiene en
  las caras del cubo (20 a 28 mm del eje) y nunca deja un dedo dentro de la
  pieza.
- Pendiente: ejecutar GraspStudio con parámetros equivalentes y comparar en
  veinte piezas (exige compilar Simox con Qt y Coin3D); conjuntos de
  agarres guardados y ventosa.

### Fase 3. Movimiento completo (L)

- Planificador de trayectorias sin colisión (RRT bidireccional con suavizado).
- Comprobación de colisión continua en lugar de por muestras.
- Ciclo recoger, transportar y depositar, con perfiles de velocidad y
  aceleración respetando los límites de cada eje.
- Exportación del programa a formatos de controlador.

**Aceptación:** en la celda de demostración y en dos celdas con obstáculos, el
ciclo completo se planifica sin contacto verificado de forma continua, y el
tiempo de ciclo calculado coincide con el del perfil a menos del 2 %.

**Estado: hecha; la aceptación se cumple** (`crates/motion/tests/acceptance.rs`).

- Comprobación continua: cada tramo recto en el espacio articular se
  demuestra libre par a par con la prueba de Schwarzer, Saha y Latombe
  (distancias en los extremos frente a una cota de lo que puede moverse
  cualquier punto entre ellos; se subdivide solo donde un par se acerca).
  Da una cota inferior certificada de la holgura. Un pasador de 3 mm que la
  comprobación por seis muestras no ve se detecta.
- RRT-Connect sobre los ejes del brazo, con atajos aleatorios; cada arista y
  cada atajo se demuestran libres. Determinista por semilla, cancelable.
- Ciclo: abrir, aproximación (RRT), descenso, cierre, elevación,
  transporte (RRT, con la pieza), bajada, soltar, retirada, regreso (RRT),
  reposo. En cada tramo solo se permite el contacto que corresponde (dedos y
  pieza al cerrar y abrir; pieza y la superficie que deja o en la que se
  posa al elevar y bajar). Se prueba el ciclo entero con cada agarre viable
  y se queda el más rápido (agarres de calidad parecida pueden diferir en
  segundos: con la UR5 importada, el primero exigía media vuelta de muñeca
  y su ciclo duraba 21,5 s; el elegido, 15,5 s); de los que no dan ciclo se
  dice en qué tramo y por qué.
- Perfiles trapezoidales sincronizados: todos los ejes recorren la misma
  recta, así que se ejecuta exactamente lo que se comprobó; velocidad y
  aceleración de cada eje dentro de su límite (escalado), con parada en
  cada punto.
- Exportación: URScript (`movej` con `a` y `v` del eje que más recorre, sin
  enlazar), y la trayectoria muestreada en CSV y JSON. Una simulación del
  interpolador de un controlador, a partir solo de los números del
  programa, vuelve a cronometrarlo.
- Resultado de la aceptación (pinza paralela, cubo llevado a 330, 210 mm):

  | Celda | Ciclo | Programa re-cronometrado (8 ms) | Holgura demostrada | Desvío |
  | --- | --- | --- | --- | --- |
  | Demostración | 9,632 s | 9,632 s (0,002 %) | ≥ 1,0 mm | — |
  | Pared | 11,240 s | 11,240 s (0,002 %) | ≥ 1,0 mm | el transporte rodea la pared |
  | Dos postes | 11,544 s | 11,544 s (0,002 %) | ≥ 1,0 mm | el transporte rodea los postes |
  | Pared, con la mano | 12,220 s | 12,220 s (0,002 %) | ≥ 1,0 mm | — |

  Cada movimiento se vuelve a demostrar con un comprobador nuevo y coincide
  con 41 muestras densas por movimiento; en las celdas con obstáculos la
  recta directa del transporte choca, así que el desvío es necesario.
- Límites: sin enlace entre movimientos (más lento que un programa
  afinado); la pieza se deposita con su misma orientación; ejes sin límites
  declarados reciben unos supuestos, que se avisan. El tiempo es el del
  perfil y del programa, no medido en un robot real (fase 5).

### Fase 4. Editor de escritorio (L)

- Aplicación de escritorio con árbol de escena, inspector, deshacer y rehacer,
  guardado atómico y recuperación ante fallo.
- Edición directa en la vista 3D: mover, girar, medir.
- Mapas de alcanzabilidad y manipulabilidad del espacio de trabajo.
- Informes en PDF del plan, la calidad y las holguras.

**Aceptación:** una persona que no conoce el producto modela una celda nueva,
planifica una recogida y exporta el informe en menos de treinta minutos, sin
ayuda, en una prueba con cinco usuarios.

### Fase 5. Preparación comercial (M, en paralelo desde la fase 3)

- Instaladores firmados para Windows, macOS y Linux; actualizaciones.
- Licenciamiento: clave por puesto y modo sin conexión.
- Integración continua con pruebas, análisis estático y auditoría de
  dependencias en cada cambio; compilaciones reproducibles.
- Manual de usuario, referencia de la API y ejemplos ejecutables.
- **Validación con hardware**: al menos un robot y una pinza reales, con
  agarres ejecutados y medidos, y el margen entre simulación y realidad
  publicado.
- Revisión legal de licencias de terceros y del aviso de seguridad.

**Aceptación:** una instalación limpia en cada sistema arranca, activa una
licencia y abre el proyecto de ejemplo; el informe de validación existe y
cuantifica el error.

### Orden recomendado

1 → 2 → 3 son el producto. La 4 es lo que el cliente ve; puede empezar en
cuanto la 1 esté hecha, porque el motor ya tiene una interfaz única. La 5 no
se deja para el final: la validación con hardware es la que más tarda y la
que más cambia el resto.

## 3. Parámetros: correspondencia con Simox

El planificador acepta los mismos conceptos que GraspStudio, con nombres
reconocibles. No se ha usado código de Simox: es una implementación propia, y
los valores numéricos de calidad **no son comparables uno a uno** con los de
Simox mientras no se replique su normalización (fase 2).

| Concepto en Simox / GraspStudio | Parámetro en AXISFORGE | Estado |
| --- | --- | --- |
| Número de agarres a planificar | `nrGrasps` | Hecho |
| Tiempo máximo de planificación | `timeoutSeconds` | Hecho en nativo; en WebAssembly no hay reloj y se rechaza, con `maxCandidates` como alternativa |
| Calidad mínima | `minQuality` | Hecho |
| Exigir cierre de fuerza | `forceClosure` | Hecho |
| Medida de calidad por espacio de llaves | `qualityMeasure: "wrench_space"` | Hecho |
| Fuerza unitaria de contacto | `quality.unitForce` | Hecho |
| Coeficiente de fricción | `quality.frictionCoefficient` (o el de la pieza) | Hecho |
| Muestras del cono de fricción | `quality.frictionConeSamples` | Hecho |
| Calidad del agarre y prueba de cierre de fuerza | `quality.quality` (épsilon), `quality.forceClosure` | Hecho |
| Punto central de agarre (GCP) y TCP | marco de agarre de la pinza | Hecho |
| Aproximación por normal de superficie | `candidateGenerator: "surface_normal"` (y `auto`, por defecto) | Hecho: se llega por la normal o se cierra a lo largo de ella |
| Cerrar actuadores hasta contacto | almohadillas que avanzan hasta tocar (pinza paralela); dedos que cierran por avance conservador (manos) | Hecho |
| Actores del efector (`Actor`, `direction`) | `Gripper.actors` (`ActorJoint.direction`) | Hecho; se importan de Simox |
| Preformas del efector | `Gripper.preshapes`; opción `preshape` del planificador | Hecho; se importan de Simox |
| Objeto manipulable con conjunto de agarres guardado | — | Fase 2 |
| Formato XML de robot y objeto | `import_simox_robot`, `import_simox_object`; `axisforge import-simox`, `add-object` | Hecho, solo lectura; las mallas .iv no se leen |
| Planificación de movimiento (RRT) | `plan_path` (RRT-Connect con atajos), `plan_cycle` | Hecho, con comprobación continua |
| Mapas de alcanzabilidad y manipulabilidad | condicionamiento en una postura | Fase 4 |

Parámetros propios, sin equivalente directo: `standoff` (retirada del
pre-agarre), `liftHeight`, `jawClearance`, `collisionMargin`, `pathSamples` y
`quality.torsionalFriction` (almohadilla blanda).

## 4. Cómo se conectan bibliotecas

### 4.1 Hoy: extensiones en Rust

El planificador no sabe qué medida de calidad ni qué generador de agarres
usa: los pide por nombre a un registro. Una biblioteca nueva implementa una de
dos interfaces y se registra:

```rust
struct MiMedida;
impl QualityMeasure for MiMedida {
    fn id(&self) -> &'static str { "mi_medida" }
    fn summary(&self) -> &'static str { "Qué mide." }
    fn measure(&self, input: &QualityInput<'_>) -> Measurement { /* ... */ }
}

session.registry_mut().register_measure(Arc::new(MiMedida))?;
// A partir de aquí: {"op":"plan_pick", "options": {"qualityMeasure": "mi_medida"}}
```

`CandidateGenerator` funciona igual para proponer agarres. Los nombres son
únicos; registrar dos veces el mismo se rechaza. Hay pruebas que registran una
extensión y comprueban que el planificador la usa.

### 4.2 Siguiente: bibliotecas que no están en Rust

- **Nativas (C o C++)**: un adaptador en Rust que implemente la interfaz y
  llame a la biblioteca por su API de C. Sirve para la aplicación de escritorio
  y la línea de órdenes. Tamaño S por biblioteca.
- **En el navegador**: módulos WebAssembly cargados en tiempo de ejecución con
  una interfaz binaria fija. Tamaño M, una sola vez.

### 4.3 Simox en concreto

Enlazar Simox directamente es posible en la aplicación nativa mediante un
adaptador como el de 4.2, pero **no en el navegador**: depende de Eigen, Boost,
Qt y Coin3D, que no se compilan a WebAssembly de forma razonable.

Lo que sí da valor pronto y sin enlazar nada es la **interoperabilidad de
datos**: leer sus XML de robot y de objeto (fase 1) y leer y escribir
conjuntos de agarres (fase 2). Con eso, un usuario de Simox trae sus modelos y
compara resultados.

Simox se distribuye bajo licencia LGPL según su repositorio. Antes de enlazarlo
en un producto comercial hay que confirmar la versión de la licencia y cumplir
sus condiciones (enlace dinámico, avisos, posibilidad de sustituir la
biblioteca). Leer sus formatos de archivo no tiene esa carga.

## 5. Riesgos

| Riesgo | Efecto | Mitigación |
| --- | --- | --- |
| El modelo de contacto no predice la realidad | Agarres "buenos" que fallan | Validación con hardware desde la fase 3; publicar el margen |
| Envolvente convexa 6D lenta con muchos contactos | Planificación lenta con mallas | Ya se mide una vez por conjunto de contactos; reducir contactos por agrupación; medir antes de optimizar |
| Piezas cóncavas | Colisión y contactos incorrectos | Descomposición convexa conservadora de las mallas cerradas; una malla abierta (no estanca) sigue chocando como una sola envolvente |
| Mallas que no se leen (.iv) | Eslabones sin colisión | Aviso en la etiqueta de contacto y en el plan; leer Open Inventor |
| Responsabilidad legal por uso en maquinaria | Reclamaciones | Aviso de seguridad en toda salida; revisión legal en la fase 5; no certificar lo que no se ha certificado |
| Dependencia de una sola persona | Parón | Documentación, pruebas y CI como parte de cada fase |

## 6. Qué significa "listo para vender"

1. El ciclo completo de recoger y depositar funciona con robots y piezas
   importados, no solo con los de ejemplo.
2. Hay un instalador, una licencia y un manual.
3. Existe un informe que dice cuánto se desvía la simulación de un robot real.
4. Cada número que muestra el producto tiene una prueba que lo respalda o una
   nota que dice que es una estimación.
