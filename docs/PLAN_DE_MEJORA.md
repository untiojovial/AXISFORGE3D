# Plan de mejora: de demostrador a producto vendible

Este documento dice dónde está AXISFORGE hoy, qué le falta para venderse y en
qué orden conviene construirlo. Cada fase tiene un criterio de aceptación
comprobable; una fase no está hecha hasta que lo cumple.

No hay fechas: el tamaño de cada fase se da como S (días), M (semanas) o
L (meses) de una persona, y es una estimación, no un compromiso.

## 1. Dónde estamos

Hecho y probado (291 pruebas automáticas, sin avisos del analizador):

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

Lo que hoy impide venderlo:

| Carencia | Por qué bloquea la venta |
| --- | --- |
| Solo recogida, sin depósito ni trayectorias | Un cliente quiere el ciclo completo |
| Piezas solo primitivas | Las piezas reales son mallas CAD |
| Mallas Open Inventor (.iv) sin leer | Casi todos los modelos de Simox las usan; esos eslabones quedan sin colisión |
| Visor de demostración, no editor | No se puede crear ni guardar una celda propia |
| Una sola pinza (paralela de dos dedos) | Ventosas y manos multidedo quedan fuera |
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

### Fase 3. Movimiento completo (L)

- Planificador de trayectorias sin colisión (RRT bidireccional con suavizado).
- Comprobación de colisión continua en lugar de por muestras.
- Ciclo recoger, transportar y depositar, con perfiles de velocidad y
  aceleración respetando los límites de cada eje.
- Exportación del programa a formatos de controlador.

**Aceptación:** en la celda de demostración y en dos celdas con obstáculos, el
ciclo completo se planifica sin contacto verificado de forma continua, y el
tiempo de ciclo calculado coincide con el del perfil a menos del 2 %.

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
| Aproximación por normal de superficie | `candidateGenerator` | Parcial: hoy solo `primitive_antipodal`; el muestreo por normal es la fase 2 |
| Cerrar actuadores hasta contacto | — | Fase 2 |
| Preformas del efector | — | Fase 2 |
| Objeto manipulable con conjunto de agarres guardado | — | Fase 2 |
| Formato XML de robot y objeto | `import_simox_robot`, `import_simox_object`; `axisforge import-simox`, `add-object` | Hecho, solo lectura; las mallas .iv no se leen |
| Planificación de movimiento (RRT) | — | Fase 3 |
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
| Piezas cóncavas | Colisión y contactos incorrectos | Ya se separan las cáscaras cerradas; falta descomponer una cáscara cóncava. Hasta entonces la envolvente da choques de más, nunca de menos |
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
