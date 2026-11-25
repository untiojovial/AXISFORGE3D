# Hipótesis y decisiones previas

**Estado: PENDING — borrador de diseño, no preregistro congelado ni aprobado.** Fecha del entorno: 2026-11-25. Ningún resultado final ha sido observado o declarado aquí. Antes de acceder a test se exige versión inmutable, aprobación independiente, manifest de objetos, método y configuración realizables, tamaño y comparaciones definitivas; las modificaciones posteriores se fecharán y se distinguirán como exploratorias.

## Hipótesis principal

En la población definida por el manifest de objetos simulados admisibles, con memoria histórica congelada y el mismo presupuesto, ACT-BO (B11) aumenta la media sobre objetos de la AUC discreta best-so-far frente a BO sin contexto ni transferencia (B2). Estimando: Δ = E_objeto[E_aleatoriedad(AUC_B11 − AUC_B2 | objeto)]; mayor es mejor. H0: Δ ≤ 0; H1: Δ > 0. La prueba primaria será unilateral α=0,05, mientras que el intervalo bilateral de 95 % y el efecto absoluto se informarán siempre. Un resultado no concluyente no demuestra equivalencia. El presupuesto primario propuesto es 50 evaluaciones totales, incluida inicialización; debe confirmarse antes del congelado mediante coste y piloto independientes de test.

## Secundarias, con contraste bilateral

1. B11 frente a B8 cuantifica la diferencia respecto a transferencia robusta consolidada; B11 frente a B9, respecto a transferencia indiscriminada. Estas dos comparaciones constituyen una familia confirmatoria secundaria, Holm α=0,05; no se infiere éxito de la hipótesis principal de ellas.
2. El gate, la evidencia predictiva y fallback contribuyen a reducir degradación frente a B2 en tareas desfavorables; comparar variantes emparejadas y comunicar frecuencia y magnitud de daño. Sin umbral clínico o práctico fijado, no afirmar seguridad ni no inferioridad.
3. El contexto visual y geométrico aporta información útil: comparaciones B4/B5/B6 y ablaciones; son exploratorias salvo que se definan familia y tamaños antes del congelado. La superioridad de DINOv3 sobre DINOv2 no se presupone.
4. Perturbaciones y categorías no vistas afectan rendimiento y transferencia; analizar interacciones y distribuciones, exploratorio salvo registro específico.

Desenlaces, restricciones, pérdida de datos y multiplicidad se definen en [STATISTICS.md](methodology/STATISTICS.md). No atribuir causalidad a un componente sin ablación controlada ni extrapolar de simulación a robot real. La hipótesis y los baselines no afirman novedad; requieren contrastar literatura y auditoría.
