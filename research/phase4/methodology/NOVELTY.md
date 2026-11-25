# Novedad y antecedentes: evaluación provisional

**PENDING.** Revisión dirigida con fuentes primarias accesibles al 2026-11-25; no sistemática ni completa. No se ha demostrado prioridad ni superioridad. La combinación de factibilidad, GP y transferencia no constituye por sí sola novedad.

- [Gardner et al., *Bayesian Optimization with Inequality Constraints*, ICML 2014](https://github.com/mlresearch/v32/blob/gh-pages/gardner14.pdf) estudia BO con restricciones desconocidas: modelar factibilidad por separado es un antecedente, no contribución automática.
- [Sui et al., *Safe Exploration for Optimization with Gaussian Processes*, ICML 2015](https://github.com/mlresearch/v37/blob/gh-pages/sui15.pdf) estudia exploración segura con GP. No llamar «seguro» un filtro de factibilidad sin validar sus supuestos y restricciones.
- [Kandasamy et al., BOCA, ICML 2017](https://github.com/mlresearch/v70/blob/gh-pages/kandasamy17a/kandasamy17a.pdf): optimización multifidelidad ya existe; métricas geométricas o de wrench no son automáticamente fidelidades de éxito dinámico.
- [Simeonov et al., Neural Descriptor Fields](https://github.com/anthonysimeonov/ndf_robot/tree/59eb1fb152092347e995eafaaf332f13840c5002) transfiere correspondencias locales para manipulación, de modo que la transferencia de regiones funcionales no es nueva; revisar texto completo y tareas antes de afirmar diferencia.
- [GraspTwin](https://github.com/VT-Collab/GraspTwin/tree/9490e75518e519a89fecfcf76f755b2602ed0892) combina selección semántica y optimización de pose con simulador; preprint según autores, no ejecutado aquí.
- [Revisión fase III](../../phase3/NOVELTY_ASSESSMENT.md) incluye RGPE, ADKF-IFT y transferencia de GP, además de límites de revisión.

Posible contribución investigable: cuantificar bajo qué condiciones evidencia predictiva secuencial y factibilidad ayudan o perjudican al transferir experiencias entre objetos, con archivo común, controles target-only emparejados y coste completo. Para sostenerla hacen falta baselines fieles, calibración, poblaciones independientes, física validada, análisis de fallos y contraste detallado con trabajos próximos. Si resultados no apoyan mejora, un análisis negativo robusto puede ser informativo; aún no hay justificación para manuscrito.
