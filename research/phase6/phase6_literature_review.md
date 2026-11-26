# Fase VI — revisión bibliográfica (referencias verificadas)

Solo aparecen referencias comprobadas en su fuente durante esta fase (arXiv, actas o la página del
editor), con el enlace usado. Lo que no se pudo verificar se marca así y no se usa como argumento. No se
declara originalidad: la comparación está en [phase6_novelty_matrix.md](phase6_novelty_matrix.md).

## 1. Viabilidad o alcance aprendidos para no comprobar todos los candidatos

Es la familia a la que pertenece el clasificador aprendido de la Fase V (M1).

- **X. Lou, Y. Yang, C. Choi.** «Learning to Generate 6-DoF Grasp Poses with Reachability Awareness».
  ICRA 2020, pp. 1532–1538. [arXiv:1910.06404](https://arxiv.org/abs/1910.06404v3). Aprende la
  alcanzabilidad como etiqueta binaria a partir de la pose del agarre.
- **I. Akinola et al.** «Dynamic Grasping with Reachability and Motion Awareness». IROS 2021.
  [arXiv:2103.10562](https://arxiv.org/pdf/2103.10562). Liga la alcanzabilidad a la existencia de una
  solución de cinemática inversa y la aprende.
- **Graspabilidad de succión con alcanzabilidad.** [arXiv:2111.02571](https://arxiv.org/pdf/2111.02571).
  Predice a la vez calidad y alcanzabilidad para evitar la comprobación online de cinemática inversa y
  colisiones. Los autores no se verificaron; se cita por su identificador.
- **L. Qin, W. Wan, J. Takahashi, R. Negishi, M. Matsushita, K. Harada.** «Learning from Planned Data to
  Improve Robotic Pick-and-Place Planning Efficiency». [arXiv:2506.15920](https://arxiv.org/abs/2506.15920v1),
  2025. Aprende de datos del propio planificador para acelerar la elección de agarres.

**Consecuencia.** Usar un clasificador aprendido para podar u ordenar candidatos antes de la cinemática
inversa **no es nuevo**. Lo que la Fase V mide (el orden exacto de un planificador real y el coste en
candidatos) es una evaluación controlada de una idea conocida.

## 2. Calidad del agarre y robustez ante la incertidumbre

- **D. J. Montana.** «The Kinematics of Contact and Grasp». *IJRR* 7(3):17–32, **1988**.
  [DOI 10.1177/027836498800700302](https://pages.github.berkeley.edu/EECS-106/sp22-site/assets/misc/David%20Montana.1988.pdf).
- **J. Weisz, P. K. Allen.** «Pose Error Robust Grasping from Contact Wrench Space Metrics». ICRA 2012,
  pp. 557–562. [PDF](https://www.cs.columbia.edu/~allen/PAPERS/weisz_icra12.pdf). Los agarres mejor
  puntuados por la métrica épsilon no son siempre los más robustos al error de pose. Es el antecedente
  directo de la tensión entre «mejor según una puntuación» y «robusto» que estudia la Fase VI.
- **J. Nogueira, R. Martinez-Cantin, A. Bernardino, L. Jamone.** «Unscented Bayesian Optimization for
  Safe Robot Grasping». IROS 2016, pp. 1967–1972. [arXiv:1603.02038](https://arxiv.org/abs/1603.02038v1).
  Favorece regiones de agarre seguras bajo incertidumbre de entrada.
- **J. Mahler et al.** «Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and
  Analytic Grasp Metrics». RSS 2017. [arXiv:1703.09312](https://arxiv.org/abs/1703.09312v2). Usa métricas
  de robustez como etiqueta.

## 3. Optimización bayesiana con restricciones

- **N. Srinivas, A. Krause, S. Kakade, M. Seeger.** «Gaussian Process Optimization in the Bandit Setting:
  No Regret and Experimental Design». **ICML 2010**, pp. 1015–1022.
  [DBLP](https://dblp.uni-trier.de/rec/conf/icml/SrinivasKKS10.html).
- **J. R. Gardner, M. J. Kusner, Z. Xu, K. Q. Weinberger, J. P. Cunningham.** «Bayesian Optimization with
  Inequality Constraints». ICML 2014, PMLR 32:937–945.
  [PMLR](https://proceedings.mlr.press/v32/gardner14.html). Pone priors sobre el objetivo y las
  restricciones. Multiplicar una utilidad por la probabilidad de cumplir la restricción es la idea que
  subyace a la fusión por producto (M4) y a P(F)·P(H|F).

## 4. Núcleos profundos y meta-aprendizaje

Es la base del TFM.

- **A. G. Wilson, Z. Hu, R. Salakhutdinov, E. P. Xing.** «Deep Kernel Learning». AISTATS 2016, PMLR
  51:370–378. [PMLR](https://proceedings.mlr.press/v51/wilson16.html).
- **W. Chen, A. Tripp, J. M. Hernández-Lobato.** «Meta-learning Adaptive Deep Kernel Gaussian Processes
  for Molecular Property Prediction». ICLR 2023. [arXiv:2205.02708](https://arxiv.org/abs/2205.02708).
  ADKF-IFT significa *Adaptive Deep Kernel Fitting with Implicit Function Theorem*:
  - aprende unos parámetros del núcleo compartidos entre tareas (meta-aprendizaje) y otros por tarea;
  - resuelve la optimización anidada con el teorema de la función implícita;
  - incluye DKL y DKT como casos particulares.

  Necesita **tareas con conjuntos de soporte y de consulta**.

## 5. Modelos fundacionales de visión

- **M. Caron et al.** «Emerging Properties in Self-Supervised Vision Transformers» (DINO). ICCV 2021,
  pp. 9650–9660. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294v1).
- **O. Siméoni et al.** «DINOv3». [arXiv:2508.10104](https://arxiv.org/abs/2508.10104), 2025.
- DINOv2 (Oquab et al.) se citó en fases anteriores; en esta fase no se ha vuelto a verificar.

## 6. Métodos aprendidos de agarre (baselines externos pendientes)

- **M. Sundermeyer, A. Mousavian, R. Triebel, D. Fox.** «Contact-GraspNet: Efficient 6-DoF Grasp
  Generation in Cluttered Scenes». ICRA 2021. [arXiv:2103.14127](https://arxiv.org/abs/2103.14127v1).
- Dex-Net 2.0: ver §2.

## 7. Calibración

- **C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger.** «On Calibration of Modern Neural Networks». ICML 2017,
  PMLR 70:1321–1330. [PMLR](https://proceedings.mlr.press/v70/guo17a.html). Define el ECE por intervalos
  iguales y el escalado de temperatura, una variante de Platt.

## 8. Datasets

- **L. Downs et al.** «Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items».
  ICRA 2022. [arXiv:2204.11918](https://arxiv.org/abs/2204.11918v1). CC-BY 4.0.
- **D. Morrison, P. Corke, J. Leitner.** «EGAD! an Evolved Grasping Analysis Dataset for diversity and
  reproducibility in robotic manipulation». RA-L 2020. [arXiv:2003.01314](https://arxiv.org/abs/2003.01314v3).
  Licencia BSD-3.
- **M. Deitke et al.** «Objaverse: A Universe of Annotated 3D Objects».
  [arXiv:2212.08051](https://huggingface.co/datasets/allenai/objaverse). Conjunto bajo ODC-By, pero cada
  objeto tiene su propia licencia (CC-BY-NC, -SA, CC0…). No se verificó la sede de publicación.

## 9. El TFM de partida y lo que hay que corregir antes de citarlo

*Robot grasping with Bayesian optimization informed with foundational models* (Universidad de Zaragoza,
2025): 8 objetos en Simox con la mano del iCub, BayesOpt con EI y un GP con núcleo profundo en JAX sobre
DINO (384 dimensiones) más la posición (3). Contrastado con las fuentes de arriba:

| Punto del TFM | Verificación |
|---|---|
| Ref. [2] (DINO, Caron et al.) con «arXiv:2205.02708, 2022» | Ese identificador es el de ADKF-IFT. DINO es arXiv:2104.14294, ICCV 2021 |
| Ref. [3] Montana, «1995» | Es 1988, *IJRR* 7(3):17–32 |
| Ref. [24] Srinivas et al., «Proceedings of the IEEE» | Es ICML 2010, pp. 1015–1022 |
| §2.7.2–2.7.3: «Approximate Deep Feature Kernels» e «Independent Feature Transformations» como lectura de ADKF-IFT | El artículo define *Adaptive Deep Kernel Fitting with Implicit Function Theorem*. La figura 2.5 del TFM lo dice bien, pero el texto no |
| Refs. [20] y [37] | Son la misma obra (Rasmussen y Williams) con dos años distintos (2004 y 2006) |
| §5.2.4: el vector de contexto sale de la media de la métrica de agarre del propio objeto | Si esa media incluye agarres evaluados del objeto que se optimiza, es información del resultado en la entrada. Habría que comprobarlo en el código del TFM, que no está en este repositorio |
| Resultados del cap. 6 | Una semilla, curvas sin intervalos y 8 objetos. Son indicios, no evidencia; no se reutilizan aquí como evidencia nueva (principio 15) |

**Sin verificar en esta fase** (no se usan como argumento): Castanheira et al., IROS 2018; Simox
(Vahrenkamp et al.); BayesOpt (Martinez-Cantin); MuJoCo (Todorov et al.); Osada et al. 2002; YCB (Calli
et al.); TOST (Schuirmann 1987).
