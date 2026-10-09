# OPT03-QTRAIN: consultas emparejadas y supervisión intermedia

Registro prospectivo local NP-OPT03-QTRAIN-20261008-D3. Es un nuevo estudio de desarrollo, motivado por D/MODE/ROLE ya observados. No es una repetición de sus modelos ni confirmación sobre test. H1 permanece cerrada y no soportada. La implementación y los criterios se congelan antes de los nuevos entrenamientos.

**Pregunta:** ¿presentar las cuatro consultas sobre cada contexto, frente a repetir una sola consulta de ese contexto, mejora la exactitud conjunta de consultas contrapuestas, y cómo interactúa con la escuela intermedia?

## Diseño y controles

Factorial 2×2: política paired/repeated × escuela 0/0,3; inicializaciones 200/201/202. Doce entrenamientos nuevos de 8.192 updates, batch de 64, 16 contextos muestreados por update y 16 ejemplos por rol. Se muestrea con reemplazo: pueden coincidir contextos, por lo que no se declaran 16 únicos. Ambas políticas y escuelas usan exactamente el mismo stream de contextos por seed.

En paired cada contexto se consulta en los cuatro roles. En repeated cada contexto recibe una consulta asignada balanceadamente (cuatro contextos por rol), repetida cuatro veces. Las máscaras de firing por ejemplar son independientes; la repetición no equivale a una selección de una máscara favorable. La diversidad de consultas y etiquetas por contexto es la intervención; no se afirma igualdad de información supervisora efectiva. La asignación usa un RNG propio que se consume también en paired, manteniendo el stream semántico y el de firing separados.

Backbone histórico fijo: NP 8×8, vocabulario 35, identidad 16, estado 48, hidden 128, T16, tying/reinjection activos, PAD legado y firing train 0,5; 29.824 parámetros nominales. No se escoge el mejor brazo fallido de D. AdamW LR 0,001 común, betas (0,9;0,999), eps 1e-8, decay 0,0001, clip global 1,0. LR es un paso nuevo fijado menor después de observar gradientes grandes; no se demuestra optimalidad ni se incluye un contraste causal de LR.

Init seeds 200/201/202; sampler 92002, query assignment 92003 y firing 92001. Igualar parámetros, ejemplos y updates no iguala FLOPs/coste: escuela 0,3 añade el lens y objetivo intermedio. Se registran pérdidas separadas, tiempos y recursos. Todas las condiciones y fallos se conservan.

## Datos y evaluación

Solo pools train/validation de split 0 históricamente expuestos. Probe 2.048/seed77001 y validation 4.096/seed77002, balanceados por los cuatro roles. No se generan ejemplos de test. Los dos datasets se guardan por arrays/hash, y la estructura clave–valor se verifica al construir las consultas contrapuestas, conservando las tuplas semánticas.

Endpoints 1.024/4.096/8.192. En todos se guardan decisiones por ejemplo, labels y NLL de inferencia densa para original y consulta AGENTE↔PACIENTE. En 8.192 se añade modo firing 0,5 con ocho seeds 79000–79007, mismas máscaras y lotes entre originales y contrafactuales. El valor es la media de rollouts individuales, no ensemble ni mejor máscara. Las máscaras no aumentan el número de inicializaciones independientes.

**Métrica primaria:** exactitud conjunta original+consulta opuesta en ejemplos AGENTE/PACIENTE, media equiponderada y sobre máscaras del modo correspondiente al entrenamiento. Se reportan binding nominal, accuracy global y por rol, NLL, modo denso y controles ACCION/LUGAR por separado. Se preserva el RNG de firing al evaluar; el flag training solo controla la máscara en el backbone sin BatchNorm/Dropout.

**Gate de competencia de esta fase:** train-probe joint ≥95% y validation joint ≥90%, en modo correspondiente al entrenamiento, en las tres inicializaciones de la condición al endpoint 8.192; además fuentes/datos/params/opt/gradientes/pesos/recursos íntegros. Es más estricto que binding nominal. No cambia gates históricos ni rescata H1. Si ninguna condición pasa, no se abre test final. Si pasa alguna, solo autoriza diseñar la siguiente fase prospectiva; no establece precisión poblacional o utilidad externa.

Selección de desarrollo, únicamente después de todas las condiciones y auditorías: máximo del peor validation joint entre las tres seeds; empate por menor NLL validation media, luego menor tiempo de entrenamiento medido y orden lexical. Solo condiciones que pasen el gate son elegibles. No seleccionar sobre scores de test, máscaras o endpoints intermedios.

## Análisis y reproducibilidad

Efecto paired−repeated e interacción con escuela dentro de cada seed, y efecto de escuela; conservar tres diferencias, media, SD e IC t95 df2 sin truncar. El contraste primario es paired−repeated promediado sobre escuela. Los intervalos sin ajuste son exploratorios. Endpoints, roles, ejemplos y máscaras no son nuevas réplicas. Una fase mayor de precisión necesita otro protocolo, controles y datos realmente nuevos antes de obtener resultados.

Se guardan primero los checkpoints antes de evaluar endpoints. Las evaluaciones parciales tienen identidad de pesos/dataset/seeds y se reutilizan sin recalcular si el hash coincide; se rechaza una identidad distinta. Modelo, optimizador y los tres RNG streams se preservan cada 128 updates; checkpoints de cada endpoint se guardan de forma independiente. Un digest de contextos por ventana y counts por rol permiten verificar comparabilidad. Se conserva un witness del primer batch.

Fuentes y plan usan UTF-8 con LF explícitos; el hash esperado del plan procede de una congelación independiente y se comprueba al ejecutar. No se normaliza un plan después de resultados para esconder una discrepancia. El incidente CRLF/LF anterior y su fallo literal permanecen archivados.

Recursos: CPU dos threads en runner estándar de repositorio público, sin gasto, cache/artifacts de Actions ni test. Mínimo RAM disponible 8 GiB antes de admitir y a lo largo de los controles de entrenamiento. Se comprueban recursos cada ventana; no se baja el mínimo. Archivos en D: local o en workspace del runner externo. Cada caso completo se archiva por ZIP, manifest y SHA en su rama Git aislada antes de pasar al siguiente; los parciales/errores se preservan con estado explícito en la limpieza operativa.

## Validación previa

La suite local pasó seis contratos, pero el de fitting/reanudación Tiny fue rechazado por RAM <8 GiB. Ese fallo operativo se conserva. En el runner estándar la suite completa pasó siete contratos: inventario/hash exacto, 16 contextos/64 ejemplos/roles balanceados y estructura paired/repeated, políticas inválidas, aislamiento del RNG de firing, evaluación que preserva RNG/modo y cache inmutable, rechazo de RAM baja y paridad de interrupción/reanudación de modelo/opt/sampler/query/firing en fixture. Run 37727219968/source aa0f77feed5ed4cb0855a8c2227e2655bc8ae8e2; no entrenamiento científico allí.

Estas pruebas verifican implementación, no rendimiento. La tarea 3 amplia seguirá abierta hasta tener evidencia de competencia, controles de capacidad/supervisión y precisión suficiente; las etapas 4–8 no se adelantan por un workflow verde.
