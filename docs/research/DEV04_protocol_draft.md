# DEV04: reproducción interna y precisión del candidato competente

**Borrador prospectivo; receta no congelada y ningún entrenamiento iniciado.** Se prepara después del cierre completo de READ03. No sustituye H1 original ni constituye una replicación externa o un test final.

## Pregunta y alcance

¿La receta de cuatro consultas por contexto, escuela 0,3 y lectura de atención mantiene alta competencia conjunta y una ventaja frente al control local activo cuando se entrenan cuerpos nuevos sobre distintas particiones de desarrollo?

Se selecciona esta condición porque fue la única que pasó el gate READ03 en las tres semillas. La selección pertenece al desarrollo ya expuesto. El nuevo contraste será específico de esta condición; **no se denominará réplica exacta del efecto READ03 de +54,225 pp**, que promediaba cuatro condiciones originales distintas.

## Separación y ausencia de reutilización de pesos

- Conservar las 264 composiciones del test original de split 0 fuera de todos los entrenamientos y evaluaciones DEV04. No abrir ni volver a puntuar ese test. Su exposición histórica tampoco permite llamarlo un test final nuevo.
- Reparticionar exclusivamente las 1.056 composiciones del pool original de desarrollo, con políticas/semillas de partición 101, 102 y 103. Cada política tendrá 924 composiciones de entrenamiento y 132 de validación, disjuntas dentro de ella; test original fijo y excluido.
- Las composiciones se pueden repetir entre políticas. Son variaciones del mismo universo finito, no tres dominios nuevos. Registrar intersecciones y hashes de todos los conjuntos.
- Entrenar desde cero cuatro inicializaciones por política: semillas 400–403, 404–407 y 408–411 respectivamente. Son **12 entrenamientos de cuerpo nuevos**, no continuaciones de pesos U16/READ03. Prohibir cargar cualquier checkpoint anterior en estos cuerpos.
- La nueva validación puede contener composiciones usadas en el entrenamiento histórico de otro modelo. Por eso no se utilizarán cuerpos preentrenados: cada pareja de lecturas comparte únicamente su propio cuerpo nuevo y sus datos de entrenamiento.

## Receta que deberá implementarse y congelarse

Cuerpo: misma arquitectura C48/Cid16/hidden128/T16, diccionario/readout compartido, reinyección, PAD legado y firing 0,5. Cuatro consultas por contexto, batch64 formado por 16 contextos × cuatro consultas; escuela 0,3 y CE de respuesta según el código QTRAIN/U16 conservado. AdamW LR0,001, betas0,9/0,999, eps1e-8, decay0,0001, clip1; 16.384 updates desde una inicialización nueva. Fijar antes de ejecutar los generadores privados de muestreo, consulta y dinámicas.

Después, congelar cuerpo/diccionario/decoder y ajustar dos lecturas por cuerpo: atención por consulta y control local activo con acceso directo a consulta. **24 ajustes de cabeza**, 3.168 parámetros nominales activos en cada lectura, mismo cache, labels, batches, firing y presupuesto de 8.192 updates por pareja. No afirmar igualdad funcional o de FLOPs entre las arquitecturas. La uniforme se conserva como evidencia READ03, sin repetir sus ajustes cerrados ni incorporarla retrospectivamente al contraste nuevo.

Fijar conjuntos de entrenamiento de cabeza, probes y validación antes del ajuste. Evaluar ocho realizaciones individuales del cuerpo con máscaras nuevas y deterministas, sin ensemble ni selección de máscara. Mantener el cambio AGENTE↔PACIENTE sobre el mismo contexto y la igualdad de controles ACTO/LUGAR. Guardar checkpoints de cuerpo, cabeza, optimizadores, tres streams RNG, pérdidas, witnesses y predicciones.

La RAM disponible mínima seguirá siendo 8 GiB antes de entrenamiento y carga numérica; no reducirla. Fijar software y recursos por ejecutor, preservar interrupciones y reanudar estados exactos sin repetir prefijos cerrados. El presupuesto completo y la energía no se deducirán del contador de updates.

## Contraste y precisión previstos

Primario: atención menos control local en exactitud conjunta de consulta original y AGENTE↔PACIENTE en validación. Calcular la diferencia pareada por cuerpo; promediar sus cuatro inicializaciones dentro de cada política de partición. El estimador principal promediará las tres políticas con igual peso. Mostrar los doce pares, las tres medias por política y un ICt95 df2 exploratorio sin truncar. No tratar 24 cabezas, máscaras o ejemplos como réplicas independientes.

Objetivo prospectivo de precisión: semianchura del intervalo principal ≤5 pp y límite inferior >0. Competencia: atención con probe conjunto ≥95% y validación ≥90% **en los doce cuerpos**, sin exigir que el control local pase el gate. Mostrar también todos sus resultados. Estos criterios evaluarán esta reproducción interna del candidato; no bastan por sí solos para cerrar el programa o demostrar convergencia universal.

Secundarios descriptivos: competencia de cada lectura y del padre, cada rol, dispersión entre inicializaciones, sensibilidad al firing, curvas y finitud. Comparar cabeza con padre mezcla lectura adicional y exposición a labels. Las comparaciones de firing deberán concretarse y congelarse antes de evaluar.

La cohorte será fija: 12 cuerpos y 24 lecturas. No parar cuando el resultado sea favorable ni ampliar semillas retrospectivamente para estrechar el intervalo. Si no alcanza los criterios, conservar el resultado negativo y definir cualquier estudio siguiente como una receta nueva. Mantener las limitaciones del intervalo de tres políticas sobre un universo semántico compartido; no llamarlo replicación independiente externa.

## Antes de ejecutar

Implementar y auditar el reparticionado que conserva el test original, el rechazo de checkpoints históricos, el controlador y los streams compartidos. Verificar recuentos 12/24, separación, preflight y recuperación exacta. Concretar todos los seeds, paneles, endpoints y contrastes pendientes; guardar sus hashes en un manifiesto y congelarlo antes de ajustar. Hasta entonces este borrador no autoriza a afirmar que DEV04 está ejecutándose.
