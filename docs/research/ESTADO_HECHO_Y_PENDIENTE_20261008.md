# NeuroPixel: trabajo realizado y trabajo pendiente

Estado comprobado el 8 de octubre de 2026, 06:48 UTC. Este inventario distingue trabajos terminados dentro de su alcance de afirmaciones científicas todavía no demostradas. Se conserva el orden del programa; la tarea 3 está abierta y las tareas 4–8 permanecen pendientes.

## Trabajo realizado

- **Antecedentes, hipótesis y controles iniciales:** los hitos históricos 1–5 están cerrados en su alcance. Incluyen revisión de antecedentes, protocolo prospectivo, controles elementales, baselines comparables y replicación con incertidumbre. El informe del punto 5 se recuperó por continuidad; su identidad byte a byte con el informe Linux original no está verificada (`05_results.md`).
- **Resultado principal conservado:** NeuroPixel alcanzó 12,393% de binding y el Transformer relativo 49,395%; diferencia −37,002 puntos porcentuales, IC95 pareado [−42,030; −31,974]. H1 no está soportada bajo esa receta y presupuesto. Los probes fueron limitados por presupuesto; esto no demuestra incapacidad arquitectónica universal.
- **Receta exacta A06.1 recuperada y auditada:** 26 entrenamientos y 34 evaluaciones del núcleo, seis trayectorias de crecimiento, 18 etapas y ocho ajustes de gate. Se recontaron contrastes e incertidumbre. Las decisiones de las 34 evaluaciones y ocho bancos se reproducen; se conservan cinco discrepancias continuas frente a la tolerancia original, sin ampliarla retrospectivamente. Los efectos generales de los componentes siguen sin establecerse (`06_A06_1_results.md`).
- **Aportaciones exploratorias cuantificadas:** diccionario/readout compartido +2,014 pp; escuela +0,439 pp; reinyección −1,135 pp. Sus intervalos incluyen cero. Recurrencia, daño y crecimiento/routing tienen contrastes y límites documentados; no se acredita un beneficio general ni una partición única del rendimiento que sume 100%.
- **Continuaciones conciliadas:** las dos recetas se conservan separadas. No se promedian sus resultados ni se cuentan dos veces las mismas semillas. Los informes cloud sobre hitos posteriores están recuperados, pero sus declaraciones de cierre no se han incorporado como conclusiones locales (`CONTINUATIONS_RECONCILED_20261008.md`).
- **Desarrollo OPT03-D terminado:** doce entrenamientos nuevos y auditados; ninguna condición pasó el gate congelado. No se eligió un ganador ni se abrió test final (`OPT03_D_results.md`).
- **Diagnósticos MODE y ROLE terminados:** reutilizaron pesos existentes, sin nuevos entrenamientos. Se midieron sensibilidad a la dinámica de evaluación, lectura de consulta y asignación de roles. Los resultados no resuelven la baja competencia ni identifican una causa única (`OPT03_MODE_results.md`, `OPT03_ROLE_results.md`).
- **Nueva intervención QTRAIN congelada y ejecutándose:** supervisión de cuatro consultas por contexto frente a una consulta repetida, cruzada con escuela 0/0,3; tres semillas y doce entrenamientos. A este corte hay cinco casos recuperados y auditados. Los checkpoints reproducen exactamente 675.840 decisiones, sin discrepancias. Es verificación interna de inferencia, no replicación científica externa. Las fuentes, datos y receta se conservan por hashes; no se repiten casos cerrados.

## Trabajo pendiente, en orden

1. **Terminar la tarea 3: optimización, competencia y precisión.** Completar los siete casos QTRAIN restantes, reunir archivos y recibos de toda la cohorte, auditar controles y checkpoints y calcular los contrastes e intervalos previstos. No seleccionar con una cohorte incompleta. El gate exige joint-query probe ≥95% y validación ≥90% en las tres semillas de una condición. Si ninguna pasa, conservar el resultado negativo y diseñar prospectivamente la siguiente intervención; completar QTRAIN no implica cerrar la tarea 3.
2. **Establecer efectos reproducibles con competencia suficiente.** Separar optimización, capacidad efectiva y supervisión mediante controles comparables. Alcanzar precisión útil con inicializaciones y particiones independientes y objetivos de precisión fijados antes de nuevos resultados. Tres semillas exploratorias, máscaras de inferencia o miles de ejemplos no sustituyen replicación entre entrenamientos. Los gates históricos y H1 no se reescriben.
3. **Tarea 4: verificar trabajos existentes sobre splits y capacidades posteriores.** Auditar fuentes, datasets, predicciones, checkpoints, controles y alcance de los informes sobre separación definitiva desarrollo/validación/test, generalización, memoria, aprendizaje continuo, reparación y estabilidad. Incorporar únicamente afirmaciones respaldadas por sus artefactos; distinguir exposición previa de evaluación nueva.
4. **Tarea 5: completar scanner causal y generalización.** Ejecutar intervenciones que permitan atribuir mecanismos y probar generalización visual, percepción, símbolos y tiempo con hipótesis y controles prospectivos. No inferir estas capacidades desde el binding sintético original.
5. **Tarea 6: medir coste completo y escalabilidad.** Incluir entrenamiento, supervisión, routing, crecimiento, evaluaciones, almacenamiento y cómputo omitido; medir energía con una fuente de medición válida y comparar trabajo equivalente. Evaluar fotónica solo si una hipótesis y un modelo de coste completo la justifican. No presentar estimaciones de software como mediciones físicas.
6. **Tarea 7: reproducibilidad externa y manuscrito.** Preparar código, datos, pesos, entorno, instrucciones, licencias, hashes, incidencias y resultados negativos; obtener replicación por una parte independiente y redactar un manuscrito acorde con la evidencia. Un segundo ejecutor del mismo proyecto no demuestra esa independencia.
7. **Tarea 8: originalidad, predicciones nuevas y utilidad importante.** Delimitar la contribución frente a bibliografía primaria, formular predicciones que distingan el mecanismo y demostrar utilidad relevante confirmada por otros. Actualmente no hay evidencia suficiente de ese resultado excepcional.

Completar el inventario no garantiza un premio Nobel. La distancia científica principal sigue siendo demostrar una contribución original, explicativa y útil que otras personas puedan confirmar.

## Evidencia operativa actual

- Entrenamiento original QTRAIN: run `37728384380`, fuente `6a5af5b68aaa9173edf46658dd8f9d16845fbad6`.
- Plan QTRAIN SHA-256: `20610e354288b48eeb7316367f61da5173fb2a124a01effa93e7d40e61bf3fb5`.
- Verificación interna de checkpoints: run `37732670363`; recibos anclados a blobs y commits Git en `results/research/OPT03_QTRAIN_review/37728384380/pinned_replay_receipts/`.
- Auditoría parcial de cinco casos: `partial_audit_five_cases.json`, 1.847 comprobaciones, cero incidencias. Replay: 805 comprobaciones, cero incidencias y 675.840 decisiones exactas.
- Estado del programa: `programme_20261008.json`; checkpoint: `coord/recovery/status-audit-20261007/checkpoint.md`.
- Restricciones vigentes: RAM disponible ≥8 GiB antes de entrenamiento, conservación de históricos/main, separación de recetas, cero repetición de entrenamientos cerrados y cero selección sobre test.

Este documento es una fotografía fechada. El progreso posterior se verifica en el checkpoint y en los recibos, sin reinterpretar esta fotografía como resultado de una cohorte completa.
