# Piloto dinámico exploratorio: especificación congelada antes de ejecutar

Diseño revisado independientemente; no es reproducción del TFM ni validación sim-to-real. MuJoCo 3.3.7 CPU, gravedad 9.81 m/s²; cubo libre de 0.04 m, 0.05 kg; mesa plana, coeficiente de fricción 1. Dos dedos planos verticales de 0.01 × 0.04 × 0.06 m y 0.1 kg, centros iniciales (±0.035, 0, 0.04) m; cubo inicialmente (0,0,0.020) m. Aproximación centrada lateralmente con offset candidato en metros. Dedos cierran a ±0.024 m en 1 s (0.011 m/s), actuadores con fuerza máxima 10 N por dedo; elevación guiada 0.1 m en 2 s y sostén 1 s; fuerza de elevación limitada a 50 N. Objeto libre, sin fuerza aplicada ni restricción artificial. Fricción torsional y rotacional cero, contacto `condim=3`, margen 0; solref=(0.02,1), solimp=(0.9,0.95,0.001); paso base 0.001 s, integrador Euler, solver Newton, 100 iteraciones.

Éxito exige altura sobre inicial ≥0.08 m durante todo el sostén, ninguna pareja de contacto mesa-objeto y deriva relativa respecto al centro de dedos ≤0.01 m durante sostén. Calidad exploratoria continua `clip(lift_m / 0.1, 0, 1)` sin confundir con epsilon, estabilidad universal o éxito. Fallo por contacto inicial dedo/objeto o penetración inicial superior a 0.001 m; se permite contacto mesa-objeto de reposo. Controles: desplazamiento 0.15 m; fricción 0 para dedos verticales sin soporte; masa 5 kg; caída libre 0.1 s error analítico <0.003 m; reposo velocidad <0.01 m/s. Sensibilidad: pasos 0.0005, 0.001, 0.002 s y solvers Newton/CG; guardar trayectorias, fuerzas y penetraciones además de éxito. Sin afirmar equivalencia con iCub. Coordenadas SI, z vertical y x eje de cierre.

## Resultados exploratorios y límites

MuJoCo 3.3.7 CPU: el agarre centrado alcanza altura mínima 0.093881 m en sostén, deriva máxima 0.001973 m desde el comienzo del sostén y no toca la mesa; desplazamiento 0.15 m, fricción cero en esta geometría y masa 5 kg no cumplen éxito. Caída libre, reposo y repetibilidad pasan en tests. Con Newton/CG y tres pasos predefinidos, todos dan éxito, con alturas 0.092552–0.093922 m; diferencia máxima de altura de trayectoria respecto a Newton/0.001 s de 0.001794 m, y picos de fuerza de dedos 1.8545–2.0057 N. Esto es una comprobación de sensibilidad, no una prueba de convergencia general. Las seis trazas completas se conservan en [el registro versionado](../results/mujoco_sensitivity.json). MuJoCo genera toda la evidencia dinámica; el piloto aislado no consume todavía candidatos de AxisForge3D.

Detalles de control registrados tras la primera ejecución: servos de posición con `kp=3000` N/m por dedo y `kp=1000` N/m para elevación; no quedaron especificados numéricamente en el documento antes de la primera ejecución, por lo que este piloto no constituye un protocolo completamente preregistrado. La deriva mide el cambio de posición relativa objeto–soporte de pinza desde el inicio del sostén; no mide deslizamiento acumulado durante el ascenso. Los controles de fallo inicial no generan trayectoria simulada y devuelven calidad cero por convención de interfaz.

## Capacidades de AxisForge3D inspeccionadas

| Capacidad | Evidencia y alcance |
|---|---|
| URDF/Simox | `crates/project-io/src/urdf.rs:418`, `crates/project-io/src/simox.rs`: importadores; revisar avisos y mallas no compatibles antes de usar. |
| FK, IK, jacobiano | `crates/kinematics/src/fk.rs:47`, `crates/kinematics/src/ik.rs`, `crates/kinematics/src/jacobian.rs:72`; cálculo cinemático, no prueba de agarre físico. |
| Colisión | `crates/collision/src/lib.rs:1`: GJK y adjuntos rígidos; contactos sin profundidad de penetración, no dinámica de contacto. |
| Manos y agarre | `crates/grasping/src/hand.rs:1`, `crates/grasping/src/quality.rs:132`, `crates/grasping/src/wrench.rs:137`: cierre geométrico y epsilon estático; objeto fijo durante cierre. La documentación antigua que limita a pinza no refleja este código. |
| Fricción y masa | `crates/domain/src/environment.rs:40`, `crates/grasping/src/wrench.rs`: parámetros y conos de fricción para cálculo estático; no una integración temporal de fuerzas. |
| Fuerza/torque y dinámica | `crates/domain/src/joint.rs:271` almacena límites; no se ha identificado un integrador temporal o solver físico de contacto en AxisForge3D. La ausencia de validación de hardware es adicional. |
| Visualización y cámaras | `crates/service/src/lib.rs:366` expone geometría para visualizar y el viewer renderiza; no se identificó interfaz validada de cámara calibrada RGB-D ni salida de mapa de profundidad/nube de puntos. No se concluye imposibilidad técnica de desarrollarla. |
| Planificación/objeto transportado | `crates/service/src/lib.rs:1290`, `crates/collision/src/lib.rs:86`: objeto unido geométricamente durante comprobaciones; no demuestra sujeción por fricción. |

Regeneración de las seis trazas incluidas en `research/results/mujoco_sensitivity.json`: `python -m research_grasping.simulation.regenerate_sensitivity research/results/mujoco_sensitivity.json`, con MuJoCo 3.3.7 instalado. Las trazas incluyen versión, parámetros, posición relativa y fuerza de actuadores; comparar diferencias cuantitativas no demuestra convergencia general ni validez física.
