# OPT03-QTRAIN: diseño pendiente de implementación y congelación

**Borrador; no autoriza ejecución ni afirma resultados.** Se formula tras los resultados negativos deD y los diagnósticos auditadosMODE/ROLE. El siguiente ciclo debe modificar el entrenamiento de forma controlada, sin repetir los doce modelos cerrados ni rescatar H1.

Pregunta: ¿presentar las cuatro consultas sobre el mismo contexto, frente a repetir una sola consulta de ese contexto, facilita aprender la asignación de roles, y cómo interactúa con la escuela intermedia?

Diseño previsto: factorial2×2 (consultas emparejadas/repetidas × escuela0/0,3), tres seeds nuevas200/201/202; doce entrenamientos de8192updates y64ejemplos por batch. Ambas políticas partirán de los mismos16contextos muestreados por batch; al muestrear con reemplazo puede haber coincidencias, por lo que no se afirma que sean16contextos únicos. Ambos batches tendrán16ejemplos por rol. En política emparejada cada contexto se consulta en los cuatro roles; en política repetida cada contexto tiene un rol asignado y su consulta se repite cuatro veces. La asignación de los16contextos a roles es balanceada y usa un RNG separado.

Así se controlan número de contextos muestreados, ejemplos, distribución de roles, número nominal de parámetros y updates. La diversidad de consultas/labels por contexto es la intervención; no se afirmará igualdad de información supervisora efectiva. La presencia de escuela se controla factorialmente. La repetición incluye máscaras independientes de firing por ejemplar; no se elige una máscara favorable.

Backbone previsto: receta histórica NP con PADlegado, tying/reinjection activos, T16 y firingtrain0,5. No se escoge el mejor brazo fallido deD. LR0,001 común, como paso fijado menor tras observar gradientes grandes; no se afirma optimalidad. AdamW/decay/clip, semilla de muestreo92002, firing92001 y asignación de query92003 se fijarán en el manifiesto definitivo antes de entrenar.

Solo train/validation históricamente expuestos. Se registrarán pérdidas separadas, recursos, checkpoints completos reanudables y predicciones por ejemplo en todos los endpoints que se vayan a analizar; no se repetirán los fallos de archivo de endpoints deD. Los artefactos de código y plan tendrán finalesLF explícitos y se comprobará la identidad exacta del blob de ejecución antes de entrenar, conservando el precedenteCRLF/LF.

La evaluación de desarrollo registrará modo denso por separado y modo0,5 correspondiente al entrenamiento, con todas las máscaras prescritas. Además del binding nominal se medirá la exactitud conjunta de consultas contrapuestas con las mismas máscaras. La fase no abre test ni cambia los gates o conclusiones históricas. Los thresholds y selección de una fase posterior se fijarán en el protocolo final, sin usar resultados nuevos para escogerlos.

Antes de ejecutar: implementar políticas de muestreo y persistencia; verificar que comparten contextos,16ejemplos por rol,64inputs, tuples/split y originales, que el optimizador recibe todos los parámetros y que la reanudación preserva modelo/opt/RNG; fijar evaluación/gates/contrastes/n=3 y todos los hashes; reservar recursos legales y sin gasto, con RAM8GiB real mantenida. Un workflow verde o una mejora de desarrollo no completará por sí sola competencia, precisión, reproducibilidad externa o utilidad.
