# Fase VI — limitaciones

1. **Sin test confirmatorio.** Las 151 piezas de desarrollo fueron la prueba de la Fase V, así que todo lo
   de la Fase VI es desarrollo o piloto. GSO está agotado con la regla de tamaño. El diseño de un test
   nuevo está en [statistical_power.json](results/statistical_power.json) y en el
   [registro de datasets](dataset_access_register.md).
2. **Simulación, no robot.**
   - El modelo dinámico solo tiene almohadillas sobre una palma cinemática; no hay brazo ni palma con
     colisión.
   - Inercia de caja.
   - Densidad uniforme.
   - La fricción torsional no está calibrada, y es el parámetro que más cambia los veredictos: 17–20 de
     180 en el control ([physics_validation.json](results/physics_validation.json)).

   «Robusto» significa robusto **en este modelo**.
3. **Viable no es agarrar.** «Viable» es el veredicto del planificador (alcance, cinemática inversa,
   colisiones, cierre de fuerzas). La robustez dinámica es otra cosa y se mide aparte.
4. **El criterio de sostener viene de la Fase V.** Sube al menos 80 mm y se desliza menos de 20 mm. No
   detecta giros grandes de la pieza en la pinza si el origen del cuerpo apenas se mueve: un caso de
   desarrollo giró 66° y contó como sostenido. Se informa el giro, pero no forma parte de la definición.
5. **Candidatos, no segundos.** N_F y N_FR cuentan candidatos examinados. Un candidato rechazado por el
   filtro cuesta mucho menos que uno que llega a la cinemática inversa y a las colisiones. No hay banco de
   coste online.
6. **Una celda y una pinza.**
   - Mesa, robot (seis ejes) y pose de la pieza fijos; pinza de 80 mm en las etapas B y C.
   - La altura del agarre, el rasgo que más pesa en el clasificador, lleva información de alcance
     específica de esta celda.
7. **Una sola población.** Objetos de consumo de GSO, con muchas variantes de producto (auditadas en la
   Fase V).
8. **Desviación de implementación del barrido.**
   - La pasada 1 se detiene en el primer fallo de cada candidato.
   - Las cargas de estrés solo se simulan en los primeros agarres de cada orden.
   - El protocolo decía «las 9 condiciones para cada viable». No cambia ninguna medida definida: R y N_FR
     solo usan las 6 condiciones estándar, y la tolerancia a la carga solo se mide en el primer viable.
9. **Desviación de implementación del prior.**
   - El protocolo decía «prior con peso 10⁶», pero el planificador nativo solo admite pesos entre 0 y 5.
   - El orden evaluado se pasa con un prior que lo codifica, con peso 5 (`fusion.encoded_prior`).
   - El orden es idéntico: un test lo comprueba, y el banco de coste lo confirma con el planificador real.
     En las 60 piezas con M4, el número de candidatos examinados coincidió siempre con el del simulador.
10. **P(H|F) aprende la robustez estándar, no la tolerancia a la carga.** Con 2 kg, el orden con P(H|F)
    es peor que M4. Combinar los dos sería ajustar a posteriori y queda como hipótesis.
11. **Errores heredados de la Fase V** ([fe de erratas](ERRATUM_PHASE5_DYNAMICS.md)):
   - las masas de la dinámica;
   - el centro de masas del simulador;
   - el orden aleatorio fuera del filtro.

   Los ficheros de la Fase V no se corrigen; la Fase VI los reanaliza.
12. **Búsqueda bibliográfica dirigida, no sistemática.** Unas 20 consultas. Ninguna afirmación de novedad
    va más allá de «no encontrado en esta búsqueda».
13. **No ejecutado:**
    - ADKF-IFT, DKL, MLP y los rasgos locales de DINO;
    - el simulador articulado de la mano;
    - los baselines externos (GPD, Dex-Net, Contact-GraspNet);
    - el tercer dataset;
    - la generalización a otra configuración del robot.

    Las razones están en [phase6_go_no_go.md](phase6_go_no_go.md).
