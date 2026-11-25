# Diagnóstico exploratorio F1 — piloto de desarrollo

**PARTIALLY_VALIDATED; revisión descriptiva interna, no evidencia confirmatoria.** Cifras derivadas de archivos únicos `run-development/cache/<signature>/native-f12/<id>.json` asociados a los 22 catálogos `pilot-expanded/catalogs` mediante el hash de proyecto, binario, malla y cuerpo. Se cuentan IDs únicos por objeto, no observaciones de trayectorias repetidas. Las causas son la primera razón de rechazo devuelta por el planificador; no implican que no existan otras causas tras ese rechazo. No se accedió al conjunto test.

| Split | Objeto | IDs evaluados únicos | Resultado F1 / primera causa |
|---|---|---:|---|
| train | 2_of_Jenga_Classic_Game | 10 | too_wide: 10 |
| train | ACE_Coffee_Mug_Kristen_16_oz_cup | 10 | too_wide: 8, pregrasp_unreachable: 1, approach_collision: 1 |
| train | Air_Hogs_Wind_Flyers_Set_Airplane_Red | 10 | too_wide: 3, grasp_unreachable: 3, pregrasp_collision: 2, pregrasp_unreachable: 2 |
| train | Android_Figure_Orange | 10 | pregrasp_unreachable: 4, pregrasp_collision: 3, approach_collision: 1, no_force_closure: 1, too_wide: 1 |
| validation | Animal_Crossing_New_Leaf_Nintendo_3DS_Game | 38 | too_wide: 23, pregrasp_unreachable: 11, approach_collision: 2, pregrasp_collision: 2 |
| train | Aroma_Stainless_Steel_Milk_Frother_2_Cup | 10 | too_wide: 6, pregrasp_unreachable: 3, pregrasp_collision: 1 |
| train | BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup | 10 | too_wide: 9, pregrasp_collision: 1 |
| train | BIRD_RATTLE | 10 | too_wide: 4, pregrasp_unreachable: 3, no_force_closure: 1, approach_collision: 1, ok: 1 |
| train | BUNNY_RATTLE | 10 | too_wide: 4, no_force_closure: 3, pregrasp_unreachable: 3 |
| train | Big_Dot_Aqua_Pencil_Case | 10 | too_wide: 9, pregrasp_unreachable: 1 |
| train | BlackBlack_Nintendo_3DSXL | 10 | too_wide: 4, ok: 2, pregrasp_unreachable: 2, pregrasp_collision: 2 |
| train | Cole_Hardware_Hammer_Black | 10 | pregrasp_unreachable: 6, too_wide: 4 |
| train | Craftsman_Grip_Screwdriver_Phillips_Cushion | 10 | pregrasp_unreachable: 6, too_wide: 3, approach_collision: 1 |
| train | Ecoforms_Cup_B4_SAN | 10 | too_wide: 10 |
| validation | Great_Dinos_Triceratops_Toy | 45 | too_wide: 19, pregrasp_unreachable: 9, ok: 6, approach_collision: 5, pregrasp_collision: 5, grasp_unreachable: 1 |
| train | Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle | 10 | too_wide: 9, pregrasp_unreachable: 1 |
| train | Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister | 10 | too_wide: 6, lift_unreachable: 2, pregrasp_collision: 1, pregrasp_unreachable: 1 |
| validation | OXO_Soft_Works_Can_Opener_SnapLock | 43 | too_wide: 18, pregrasp_unreachable: 18, approach_collision: 5, pregrasp_collision: 2 |
| validation | Orbit_Bubblemint_Mini_Bottle_6_ct | 44 | too_wide: 36, pregrasp_unreachable: 5, pregrasp_collision: 1, ok: 1, lift_unreachable: 1 |
| validation | PINEAPPLE_MARACA_6_PCSSET | 43 | too_wide: 17, pregrasp_unreachable: 11, pregrasp_collision: 7, approach_collision: 5, no_force_closure: 2, ok: 1 |
| train | Razer_Abyssus_Ambidextrous_Gaming_Mouse | 10 | too_wide: 4, pregrasp_unreachable: 4, pregrasp_collision: 1, approach_collision: 1 |
| validation | Threshold_Porcelain_Teapot_White | 43 | too_wide: 38, pregrasp_unreachable: 3, approach_collision: 2 |

Total por primera causa en 416 evaluaciones únicas (160 train, 256 validation): too_wide: 245, pregrasp_unreachable: 94, pregrasp_collision: 28, approach_collision: 24, ok: 11, no_force_closure: 7, grasp_unreachable: 4, lift_unreachable: 3. La caché repite resultados entre métodos y semillas, por lo que 1.620 registros de trayectorias de validación no son 1.620 evaluaciones únicas. En validación hay ocho IDs F1 viables en tres objetos (dinosaurio 6, botella 1, maraca 1), sin F3 implícito. Los 48 candidatos de cada objeto ya eran una submuestra determinista del generador; estas frecuencias no representan todas las posibles poses ni objetos.

En `summary.json` son **tres de seis** objetos con AUC media igual a cero para los nueve métodos: Animal Crossing, OXO y tetera. La maraca tiene un único ID F1 viable en caché y al menos un método registra AUC media positiva (máximo 0,03553); por tanto una afirmación anterior de cuatro objetos con AUC cero era incorrecta. Los tres objetos con al menos un plan F1 son dinosaurio, botella y maraca. Estos conteos usan IDs únicos en caché; no extrapolar a candidatos no consultados.

| Objeto validación | AUC adaptive | AUC target_local | Diferencia | AUC contextual | Diferencia adaptive − contextual |
|---|---:|---:|---:|---:|---:|
| Animal_Crossing_New_Leaf_Nintendo_3DS_Game | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Great_Dinos_Triceratops_Toy | 0.34498 | 0.25284 | 0.09214 | 0.34498 | 0.00000 |
| OXO_Soft_Works_Can_Opener_SnapLock | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Orbit_Bubblemint_Mini_Bottle_6_ct | 0.00000 | 0.06459 | -0.06459 | 0.19377 | -0.19377 |
| PINEAPPLE_MARACA_6_PCSSET | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Threshold_Porcelain_Teapot_White | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

Las cifras son medias de tres semillas por objeto y diferencias descriptivas; no son contrastes, no prueban superioridad ni aislación causal y la etiqueta contextual no se equipara automáticamente a B10. La selección de pose, la viabilidad cinemática, contactos y escala/cuerpo requieren auditoría independiente. Documentar unidades, geometría y causas de colisión antes de modificar condiciones, y repetir piloto afectado si cambian. F1 `ok` no equivale a agarre dinámico exitoso.
