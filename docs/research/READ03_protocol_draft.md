# Diagnóstico causal de lectura sobre modelos congelados

**Borrador de desarrollo, antes de cualquier ajuste científico de las nuevas lecturas.** No es todavía una receta congelada. U16 está cerrado; la tarea 3 amplia permanece abierta y H1 no soportada. Este diagnóstico complementa los requisitos de optimización, capacidad, supervisión y precisión; no los sustituye.

## Pregunta y padres

¿Los estados ya aprendidos permiten mejorar la asignación de roles mediante una lectura global condicionada por consulta, frente a una lectura local con capacidad nominal y acceso directo a consulta comparables?

Usar los doce checkpoints U16 finales a 16.384 actualizaciones: inicializaciones 200/201/202 × política original de cuatro consultas/una repetida × escuela original 0/0,3. Ningún padre se selecciona por su score. Inventario `READ03_parent_inventory.json`, SHA256 `f2574cabde30ff6a613ddbae591646efb4dc4aba6124e99761ade7c140637414`. Los doce modelos, su diccionario, decoder y buffers permanecerán congelados; no se repite su entrenamiento ni se actualizan sus optimizadores originales.

Tres lecturas por padre: atención condicionada por consulta; agregación global uniforme; MLP local con compuerta de identidad condicionada por consulta. Son 36 ajustes de cabezas nuevas sobre las mismas tres inicializaciones de cuerpo por condición. No son 36 nuevas réplicas independientes de NeuroPixel.

## Capacidades y mecanismo

Las tres cabezas tienen 3.168 parámetros nominales; cuerpo más cabeza: 32.992. La uniforme no utiliza Q/K para ponderar celdas: sus 1.312 parámetros Q/K no reciben gradiente de respuesta, aunque se calculen sus proyecciones. Su capacidad efectiva difiere. El control local tiene parámetros activos y acceso a la identidad visible de consulta, pero no agrega celdas lejanas en la cabeza. Igual conteo nominal no iguala función, inicialización o cómputo.

La atención recibe claves/valores de estados e identidades visibles, consulta en (7,6), y excluye celdas vacías y la celda de consulta como claves. La contribución se suma al estado de salida (7,7), leído con el decoder/diccionario originales. El control local usa ese estado y una compuerta de identidad. Ninguna lectura recibe asignaciones del generador ni respuestas como características.

## Datos, caché y supervisión propuesta

- Nuevo conjunto de entrenamiento: 4.096 ejemplos, obtenidos de 1.024 extracciones de contexto con reemplazo y las cuatro consultas de cada contexto. Generador de contextos 89002, asignación consumida 89003; partición train del mismo split 0. No denominar únicas a esas 1.024 extracciones.
- Todas las cabezas reciben exactamente el mismo conjunto, etiquetas y caché para un padre. La nueva supervisión es solo CE de respuesta; no se añade escuela ni se modifica la supervisión original del cuerpo.
- Cachear ocho realizaciones del cuerpo con firing 0,5: entrenamiento 88000–88007. Probe y validación conservan los datos 77001/77002 y las máscaras 79000–79007 del endpoint U16 cerrado. Son pools de desarrollo previamente expuestos; no final test.
- La extracción de estados es un observable nuevo. Las decisiones originales/contrafactuales de referencia se reutilizan desde las evaluaciones cerradas. Si se recalculan al extraer estados, exigir concordancia exacta con esas decisiones; no contar ese control como otro experimento independiente.
- Caché compacta: ocho estados de celdas de hechos visibles y el estado de salida, posiciones derivadas exclusivamente del canvas, más canvas/targets/roles por separado. Reconstruir el mismo layout de memoria; no confundir exactitud de valores con exactitud de la ruta numérica. Los cinco contratos sintéticos incluyen ambos layouts y las tres cabezas, igualdad exacta sin ampliar tolerancia.

## Ajuste propuesto

Por cabeza: 8.192 actualizaciones, AdamW LR0,001, betas0,9/0,999, eps1e−8, decay0,0001, clip1. Batch64, seleccionando16 grupos de contexto y sus cuatro consultas; selección de grupos89004 y de realización89005, privados y comunes entre cabezas. Semillas de cabeza500/501/502 según la inicialización del cuerpo200/201/202; Q/uniforme tienen pesos iniciales idénticos. Local usa el mismo identificador de semilla y su arquitectura diferente, sin afirmar identidad funcional de inicialización.

Evaluaciones previstas a1.024/4.096/8.192; decisión de competencia solo en8.192. Guardar estados completos de cabeza/optimizador/RNG, pérdidas y witnesses de batches; recuperación no debe repetir ajustes cerrados. Registrar finitud y digest de cuerpo/diccionario antes/después y entre hitos. No permitir gradientes o cambios de los parámetros originales. RAM disponible ≥8GiB antes de cargar o ajustar; CPU con dos threads y software científico fijado al preflight original. Registrar tiempos completos de caché, ajuste, evaluación y archivo por separado; energía física no está establecida.

## Endpoints y análisis que deberá congelarse

Primario: atención − control local en exactitud conjunta de consulta original y AGENTE↔PACIENTE sobre el mismo contexto/máscara, promediado entre las cuatro condiciones originales del cuerpo dentro de cada inicialización. Secundarios: atención − uniforme; atención − padre original; local − padre; uniforme − padre. Las comparaciones con el padre combinan nueva lectura y exposición a etiquetas adicionales; no son un efecto puro de atención. Comparar nominal binding y cada rol como descriptivos.

Mantener n=3, contrastes pareados, ICt95df2 exploratorios sin ajuste ni truncamiento, todos los valores por inicialización. Sin inflación por máscaras, ejemplos, doce cuerpos o etapas. Exigir cohorte completa y auditorías antes de efectos/selección. El criterio descriptivo de competencia permanece probe conjunto≥95% y validación≥90%, en las tres inicializaciones de cada condición original/cabeza. Ningún vencedor de una fase negativa ni apertura de test.

Si atención mejora y local no, el resultado apoyaría utilidad de la nueva ruta global/selección en esos estados, con sus límites de función y cómputo. Si uniforme mejora de forma similar, la selección por Q/K no sería necesaria en ese diagnóstico. Si local mejora, acceso directo a consulta/capacidad no lineal sería una alternativa. Si todas fallan, no se demuestra ausencia de información ni incapacidad del cuerpo: las cabezas y su optimización también pueden limitar el resultado. Ningún caso prueba interpretación causal de los pesos de atención, originalidad o utilidad externa.

## Estado de preparación

Primer preflight falló la igualdad exacta de logits por cambio de layout: diferencia máxima8,88e−16; fuente y fallo conservados. Corrección conserva layout global y vista local; la ejecución37824591943 pasó cinco contratos y admitió/restauró/congeló los doce padres reales con cero inferencia científica y cero fitting. Source0f64dd755c557c889b38878b619d2a900fc0a267; evidencias en `results/research/READ03_preflight_review/`.

Falta implementar/verificar el controlador y las auditorías, fijar inventarios y contrastes en JSON, congelar todos los hashes y ejecutar. No hay resultados científicos de las nuevas lecturas.
