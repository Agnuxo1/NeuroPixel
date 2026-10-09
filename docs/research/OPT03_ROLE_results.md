# OPT03-ROLE: consulta y asignación de roles

Diagnóstico de desarrollo terminado y auditado el8deoctubrede2026. Se usaron los doce pesos existentes, sin optimizador, gradientes ni entrenamientos nuevos. Las predicciones originales de MODE se reutilizaron. H1, el gate original95%/90% y el estado pendiente de la tarea3 amplia no se modifican.

El generador contiene cuatro pares horizontales[ROL][VALOR] en filas aleatorias. Cambiar solo consulta AGENTE↔PACIENTE debe cambiar el noun correcto, manteniendo todo el contexto. Intercambiar **pares completos** conserva las asignaciones y la tupla semántica; no es un intercambio de valores bajo claves fijas. Así las tres vistas congeladas(queryflip,pairrelocate,both) preservan la composición y partición, sin introducir tripletas de test. Los seis contratos construidos verificaron estas propiedades antes de las evaluaciones.

## Resultado principal

La exactitud conjunta exige acertar el original **y** el caso de consulta opuesta sobre el mismo contexto y máscara. Una respuesta que ignore la consulta no puede acertar ambos nouns distintos. La media de exactitudes sobre ocho máscaras en modelos0,5 no es un ensemble ni ocho nuevas inicializaciones. Las unidades de entrenamiento son tres,seeds100/101/102.

Valores del subconjunto de validación512,128por rol; porcentajes.

| PAD/firingtrain, inferencia correspondiente | Exactitud conjunta consulta media | Seeds100/101/102 | IC95t,df2 | Cambia predicción al cambiar consulta | Predice uno de los dos noun candidatos |
|---|---:|---|---|---:|---:|
| Legado/0,5 | 7,503 | 0,195 /21,143 /1,172 | [−21,864;36,871] | 18,506 | 63,574 |
| Legado/1,0 | 0,521 | 0,781 /0,000 /0,781 | [−0,600;1,641] | 63,672 | 29,557 |
| Cero/0,5 | 2,572 | 1,709 /3,223 /2,783 | [0,637;4,506] | 13,216 | 80,501 |
| Cero/1,0 | 2,344 | 0,000 /2,734 /4,297 | [−3,059;7,747] | 60,286 | 45,443 |

En los modelos0,5, pertenecer a la categoría correcta o copiar un noun visible resulta mucho más frecuente que resolver correctamente ambas consultas. El cambio de predicción bajo consulta es reducido. En los modelos1,0 la predicción cambia más, pero la pertenencia a los candidatos es menor y el éxito conjunto sigue muy bajo. **No se atribuye a todos los modelos una única causa.** Sensibilidad a consulta por sí sola no equivale a una asignación correcta.

## Sensibilidad a posición

Al reubicar pares completos, la composición y todos los targets permanecen iguales. La proporción de predicciones nominales que conserva el mismo token es60,205%,82,292%,49,886% y68,099% para las cuatro condiciones de la tabla. La exactitud conjunta original+reubicación es22,493%,11,328%,20,980% y13,542%, respectivamente. Estas medidas describen dependencia de geometría en estos pesos/subconjuntos; no separan automáticamente lectura de claves, propagación, ruido o la causa de fallos de optimización.

La intervención combinada se conserva con todos sus resultados y controles ACCION/LUGAR en los arrays y el informe de auditoría. No se escogen ejemplos, masks o seeds favorables. Los intervalos amplios se conservan sin recortarlos; no constituyen prueba de ausencia ni estimación poblacional precisa.

## Completitud, recuperación y auditoría

El diagnóstico se detuvo por RAM después de11checkpoints. Se preservaron los11resultados y el incidente. Un auditor independiente recontó1050checks/162816predicciones como **parcial**, sin declarar cohorte completa ni seleccionar resultados. La reanudación reutilizó esos archivos; solo el checkpoint pendiente seed102/PADcero/firing1 se evaluó.

El margen operativo de arranque se ajustó8,75→8,5GiB, manteniendo sin cambios el mínimo científico8GiB del worker antes del runtime y entre checkpoints. La admisión real del reanudador registró8,6749GiB; el worker pasó sus propios controles y terminó. El wrapper previo fue detenido únicamente mientras esperaba y no tenía hijo. No se cambiaron fuentes experimentales, pesos, datos, thresholds ni procesos ajenos. Se conservan el incidente, el primer rechazo de admisión, el ajuste operativo y el transcript de reanudación.

La auditoría de la cohorte completa efectuó**1384checks, cero incidencias y165888predicciones nuevas recontadas**. Verificó esquema por un decoder independiente, conservación de todos los valores/tuplas, objetivos/roles, hashes, decisiones originales cacheadas, masks/seeds y controles de filas no modificadas, finitud, métricas, valores por seed e intervalos con n=3. Los estados y archivos de los checkpoints no cambiaron.

## Conclusión y trabajo que sigue

La baja competencia persiste incluso al medir el modo de actualización correspondiente al entrenamiento. La accuracy nominal de un solo caso puede ocultar el fallo de resolver las dos consultas contrapuestas. Los resultados muestran perfiles condicionados de selección de candidatos, dependencia de consulta y posición; no demuestran incapacidad arquitectónica universal ni una explicación única. No son una replicación externa.

La tarea3 necesita una intervención de entrenamiento nueva con controles adecuados. Los próximos candidatos deberán especificarse prospectivamente sobre desarrollo, manteniendo comparabilidad de capacidad, número de ejemplos/updates, diversidad contextual y supervisión. La conclusión de ROLE no autoriza escoger un ganador de la faseD negativa ni abrir test final. El trabajo sobre precisión y beneficios de componentes a competencia elevada sigue sin completarse, y etapas4–8 permanecen pendientes.

Artefactos: `docs/research/OPT03_role_plan.json`(SHA25606de45dfd8d861dfd6bfb49ef05c217dfb8b36a744fec7740fbd25ab84cd470d), `results/research/OPT03_roles/`, `results/research/OPT03_roles_review/audit.json`, `partial_audit.json`, `resource_incident.json`, `admission_amendment.json` y `resume_transcript.log`. Originales y derivados conservados separados en D:.
