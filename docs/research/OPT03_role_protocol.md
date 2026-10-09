# OPT03-ROLE: consulta y vinculación rol–valor

Registro prospectivo local NP-OPT03-ROLE-20261008-D2. Fase de desarrollo posterior a D/MODE, congelada antes de nuevos resultados. Pregunta: ¿hasta qué punto las respuestas dependen de la consulta y preservan las asignaciones de rol cuando cambia la colocación espacial?

Se usan los doce checkpoints ya auditados y los subconjuntos512por panel(128por rol) ya utilizados en MODE. Las predicciones originales se reutilizan, sin repetir esas evaluaciones cerradas. Se mantiene el modo de inferencia correspondiente al entrenamiento: ocho máscaras78000–78007 para firing0,5; una evaluación densa para firing1,0. Las mismas seeds, orden de ejemplos y lotes256 se usan para cada contrafactual, de modo que las máscaras no se conviertan en una segunda intervención.

El esquema real consiste en cuatro pares horizontales[ROL][VALOR] situados en cuatro filas distintas0–6; consulta(7,6), salida(7,7). El parser verifica35IDs,nouns distintos para agente/paciente, claves únicas, categorías, labels y roles. Se ejecutan tres vistas nuevas:

1. Cambiar solo consulta AGENTE↔PACIENTE: los ejemplos de ambos roles deben responder el otro noun; ACCION/LUGAR permanecen idénticos.
2. Intercambiar posiciones de los **pares completos** AGENTE/valor y PACIENTE/valor: claves viajan con sus valores, todos los targets se conservan.
3. Ambas transformaciones: la composición se conserva y las consultas nominales deben responder el otro noun.

Las tres vistas conservan exactamente la tupla semántica(agente,verbo,paciente) y el valor de lugar; no introducen composiciones del pool test ni cambian partición. Intercambiar solo los valores habría cambiado la tupla y podía introducir ejemplos de test; esa opción fue descartada durante diseño, antes de congelación/evaluación. La selección de ejemplos no usa scores. Se verificó involución de las tres vistas, preservación de originales y rechazo de claves/labels corruptos con seis contratos construidos.

Métrica primaria diagnóstica: exactitud conjunta original+consulta cambiada, equiponderada entre los roles AGENTE/PACIENTE y promediada sobre rollouts individuales. Una predicción que ignore la consulta y conserve el mismo token no puede acertar ambos targets distintos con máscaras idénticas. Esto es un control analítico del contrato, no un resultado aprendido.

Métricas secundarias: binding de cada vista, tasa de cambio de predicción, pertenencia de la predicción a los dos noun candidatos, conservación bajo reubicación y conjunción original+ambas. Se guardan también controles ACCION/LUGAR. Ocho máscaras son variación de inferencia condicional; se resumen primero por checkpoint, luego los valores y CI t95df2 entre tres inicializaciones, separando PAD y firing. No se seleccionan máscaras ni modelos.

Las etiquetas y roles se utilizan para auditoría y evaluación, nunca como entradas de la red. El parser exacto sirve para certificar transformaciones, sin repetir el benchmark cerrado de controles elementales. Los diagnósticos pueden revelar insensibilidad a la consulta, selección deficiente de candidatos o dependencia de posición, pero no identifican por sí solos una causa única de optimización ni prueban inteligencia relacional general.

No se entrena, calcula gradiente, crea optimizador ni abre test. Los pesos/archivos se verifican antes/después; fuentes y entradas quedan por hash. CPU2threads, RAM≥8GiB al admitir y entre checkpoints; originales y derivados separados en D:. Resultados completos con el mismo plan/entorno/hash se reutilizan. La tarea3 amplia,H1 y gate original95%/90% se mantienen sin cambio.
