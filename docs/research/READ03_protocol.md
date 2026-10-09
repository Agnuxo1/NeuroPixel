# DiagnÃ³stico causal de lectura sobre modelos congelados

**Borrador de desarrollo, antes de cualquier ajuste cientÃ­fico de las nuevas lecturas.** No es todavÃ­a una receta congelada. U16 estÃ¡ cerrado; la tarea 3 amplia permanece abierta y H1 no soportada. Este diagnÃ³stico complementa los requisitos de optimizaciÃ³n, capacidad, supervisiÃ³n y precisiÃ³n; no los sustituye.

## Pregunta y padres

Â¿Los estados ya aprendidos permiten mejorar la asignaciÃ³n de roles mediante una lectura global condicionada por consulta, frente a una lectura local con capacidad nominal y acceso directo a consulta comparables?

Usar los doce checkpoints U16 finales a 16.384 actualizaciones: inicializaciones 200/201/202 Ã— polÃ­tica original de cuatro consultas/una repetida Ã— escuela original 0/0,3. NingÃºn padre se selecciona por su score. Inventario `READ03_parent_inventory.json`, SHA256 `f2574cabde30ff6a613ddbae591646efb4dc4aba6124e99761ade7c140637414`. Los doce modelos, su diccionario, decoder y buffers permanecerÃ¡n congelados; no se repite su entrenamiento ni se actualizan sus optimizadores originales.

Tres lecturas por padre: atenciÃ³n condicionada por consulta; agregaciÃ³n global uniforme; MLP local con compuerta de identidad condicionada por consulta. Son 36 ajustes de cabezas nuevas sobre las mismas tres inicializaciones de cuerpo por condiciÃ³n. No son 36 nuevas rÃ©plicas independientes de NeuroPixel.

## Capacidades y mecanismo

Las tres cabezas tienen 3.168 parÃ¡metros nominales; cuerpo mÃ¡s cabeza: 32.992. La uniforme no utiliza Q/K para ponderar celdas: sus 1.312 parÃ¡metros Q/K no reciben gradiente de respuesta, aunque se calculen sus proyecciones. Su capacidad efectiva difiere. El control local tiene parÃ¡metros activos y acceso a la identidad visible de consulta, pero no agrega celdas lejanas en la cabeza. Igual conteo nominal no iguala funciÃ³n, inicializaciÃ³n o cÃ³mputo.

La atenciÃ³n recibe claves/valores de estados e identidades visibles, consulta en (7,6), y excluye celdas vacÃ­as y la celda de consulta como claves. La contribuciÃ³n se suma al estado de salida (7,7), leÃ­do con el decoder/diccionario originales. El control local usa ese estado y una compuerta de identidad. Ninguna lectura recibe asignaciones del generador ni respuestas como caracterÃ­sticas.

## Datos, cachÃ© y supervisiÃ³n propuesta

- Nuevo conjunto de entrenamiento: 4.096 ejemplos, obtenidos de 1.024 extracciones de contexto con reemplazo y las cuatro consultas de cada contexto. Generador de contextos 89002, asignaciÃ³n consumida 89003; particiÃ³n train del mismo split 0. No denominar Ãºnicas a esas 1.024 extracciones.
- Todas las cabezas reciben exactamente el mismo conjunto, etiquetas y cachÃ© para un padre. La nueva supervisiÃ³n es solo CE de respuesta; no se aÃ±ade escuela ni se modifica la supervisiÃ³n original del cuerpo.
- Cachear ocho realizaciones del cuerpo con firing 0,5: entrenamiento 88000â€“88007. Probe y validaciÃ³n conservan los datos 77001/77002 y las mÃ¡scaras 79000â€“79007 del endpoint U16 cerrado. Son pools de desarrollo previamente expuestos; no final test.
- La extracciÃ³n de estados es un observable nuevo. Las decisiones originales/contrafactuales de referencia se reutilizan desde las evaluaciones cerradas. Si se recalculan al extraer estados, exigir concordancia exacta con esas decisiones; no contar ese control como otro experimento independiente.
- CachÃ© compacta: ocho estados de celdas de hechos visibles y el estado de salida, posiciones derivadas exclusivamente del canvas, mÃ¡s canvas/targets/roles por separado. Reconstruir el mismo layout de memoria; no confundir exactitud de valores con exactitud de la ruta numÃ©rica. Los cinco contratos sintÃ©ticos incluyen ambos layouts y las tres cabezas, igualdad exacta sin ampliar tolerancia.

## Ajuste propuesto

Por cabeza: 8.192 actualizaciones, AdamW LR0,001, betas0,9/0,999, eps1eâˆ’8, decay0,0001, clip1. Batch64, seleccionando16 grupos de contexto y sus cuatro consultas; selecciÃ³n de grupos89004 y de realizaciÃ³n89005, privados y comunes entre cabezas. Semillas de cabeza500/501/502 segÃºn la inicializaciÃ³n del cuerpo200/201/202; Q/uniforme tienen pesos iniciales idÃ©nticos. Local usa el mismo identificador de semilla y su arquitectura diferente, sin afirmar identidad funcional de inicializaciÃ³n.

Evaluaciones previstas a1.024/4.096/8.192; decisiÃ³n de competencia solo en8.192. Guardar estados completos de cabeza/optimizador/RNG, pÃ©rdidas y witnesses de batches; recuperaciÃ³n no debe repetir ajustes cerrados. Registrar finitud y digest de cuerpo/diccionario antes/despuÃ©s y entre hitos. No permitir gradientes o cambios de los parÃ¡metros originales. RAM disponible â‰¥8GiB antes de cargar o ajustar; CPU con dos threads y software cientÃ­fico fijado al preflight original. Registrar tiempos completos de cachÃ©, ajuste, evaluaciÃ³n y archivo por separado; energÃ­a fÃ­sica no estÃ¡ establecida.

## Endpoints y anÃ¡lisis que deberÃ¡ congelarse

Primario: atenciÃ³n âˆ’ control local en exactitud conjunta de consulta original y AGENTEâ†”PACIENTE sobre el mismo contexto/mÃ¡scara, promediado entre las cuatro condiciones originales del cuerpo dentro de cada inicializaciÃ³n. Secundarios: atenciÃ³n âˆ’ uniforme; atenciÃ³n âˆ’ padre original; local âˆ’ padre; uniforme âˆ’ padre. Las comparaciones con el padre combinan nueva lectura y exposiciÃ³n a etiquetas adicionales; no son un efecto puro de atenciÃ³n. Comparar nominal binding y cada rol como descriptivos.

Mantener n=3, contrastes pareados, ICt95df2 exploratorios sin ajuste ni truncamiento, todos los valores por inicializaciÃ³n. Sin inflaciÃ³n por mÃ¡scaras, ejemplos, doce cuerpos o etapas. Exigir cohorte completa y auditorÃ­as antes de efectos/selecciÃ³n. El criterio descriptivo de competencia permanece probe conjuntoâ‰¥95% y validaciÃ³nâ‰¥90%, en las tres inicializaciones de cada condiciÃ³n original/cabeza. NingÃºn vencedor de una fase negativa ni apertura de test.

Si atenciÃ³n mejora y local no, el resultado apoyarÃ­a utilidad de la nueva ruta global/selecciÃ³n en esos estados, con sus lÃ­mites de funciÃ³n y cÃ³mputo. Si uniforme mejora de forma similar, la selecciÃ³n por Q/K no serÃ­a necesaria en ese diagnÃ³stico. Si local mejora, acceso directo a consulta/capacidad no lineal serÃ­a una alternativa. Si todas fallan, no se demuestra ausencia de informaciÃ³n ni incapacidad del cuerpo: las cabezas y su optimizaciÃ³n tambiÃ©n pueden limitar el resultado. NingÃºn caso prueba interpretaciÃ³n causal de los pesos de atenciÃ³n, originalidad o utilidad externa.

## Estado de preparaciÃ³n

Primer preflight fallÃ³ la igualdad exacta de logits por cambio de layout: diferencia mÃ¡xima8,88eâˆ’16; fuente y fallo conservados. CorrecciÃ³n conserva layout global y vista local; la ejecuciÃ³n37824591943 pasÃ³ cinco contratos y admitiÃ³/restaurÃ³/congelÃ³ los doce padres reales con cero inferencia cientÃ­fica y cero fitting. Source0f64dd755c557c889b38878b619d2a900fc0a267; evidencias en `results/research/READ03_preflight_review/`.

Falta implementar/verificar el controlador y las auditorÃ­as, fijar inventarios y contrastes en JSON, congelar todos los hashes y ejecutar. No hay resultados cientÃ­ficos de las nuevas lecturas.
