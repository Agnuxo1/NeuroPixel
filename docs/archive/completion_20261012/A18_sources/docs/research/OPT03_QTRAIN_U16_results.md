# OPT03-QTRAIN-U16: presupuesto adicional con estados conservados

La receta de desarrollo terminó y quedó auditada el 8 de octubre de 2026. **El presupuesto adicional mejoró el endpoint primario en 19,554 pp, IC 95 % exploratorio[14,862;24,247], pero ninguna condición alcanzó el criterio completo de competencia.** La tarea3 amplia permanece abierta, no se selecciona ganador ni se abre test final. H1 sigue no soportada.

## Qué se intervino

Se continuaron los doce estados originales de QTRAIN de 8.192 a 16.384 actualizaciones: semillas 200/201/202 × cuatro consultas por contexto frente a una repetida × escuela 0/0,3. Arquitectura y 29.824 parámetros, LR 0,001, AdamW y clipping 1, 16 pasos recurrentes y firing 0,5, PAD legado, tying y reinyección se conservaron. Modelo, momentos/contador AdamW y tres streams de RNG se restauraron exactamente. Primer update nuevo 8.193; el prefijo de 8.192 no se repitió. No son nuevas inicializaciones: n=3 por condición, no n=6 por etapas.

El presupuesto cambia conjuntamente actualizaciones, exposición adicional a ejemplos y cómputo; no identifica una mejora pura del algoritmo de optimización. Se añadieron 98.304 actualizaciones y 6.291.456 ejemplos procesados en total. Cada batch mantiene 16 extracciones de contexto con reemplazo, 64 ejemplos y 16 consultas por rol. Las dos políticas difieren en diversidad de etiquetas dentro del mismo contexto, controlada como intervención.

Se reutilizó el endpoint de 8.192 cerrado, sin reevaluarlo. Nuevos endpoints de 12.288 y 16.384: original y cambio AGENTE↔PACIENTE en modo denso y en ocho rollouts individuales de firing 0,5. Se promedian rollouts, sin ensemble ni selección de máscaras. El primario es acertar ambas consultas sobre el mismo contexto. Gate fijo: probe conjunto ≥95 % y validación ≥90 % a 16.384, en las tres inicializaciones de una condición. Probe de 2.048 y validación de 4.096 ejemplos, con semillas 77001/77002, siguen siendo pools expuestos de desarrollo, no un test final nuevo.

## Competencia absoluta

| Condición a 16.384 | Semillas 200 / 201 / 202, % | Media, % | IC 95 %, % | Gate en las tres |
|---|---:|---:|---:|---|
| Cuatro consultas / escuela 0 | 46.973 / 4.303 / 3.809 | 18.361 | [-43.194; 79.917] | No |
| Cuatro consultas / escuela 0.3 | 88.593 / 80.621 / 84.808 | 84.674 | [74.769; 94.579] | No |
| Una consulta repetida / escuela 0 | 1.685 / 0.409 / 0.616 | 0.903 | [-0.797; 2.604] | No |
| Una consulta repetida / escuela 0.3 | 8.405 / 1.740 / 2.722 | 4.289 | [-4.649; 13.227] | No |

La celda cuatro consultas/escuela 0,3 pasó descriptivamente de 28,225 % conjunto a 84,674 %. Sus tres valores nuevos 88,593/80,621/84,808 % indican menor dispersión observada que los 74,677/2,466/7,532 % anteriores; no se preregistró un test de igualdad de varianzas. Su IC 95 %nuevo[74,769;94,579]% continúa siendo amplio con tres inicializaciones.

Los probes de esa celda son88,086/80,469/84,607%, todos inferiores a 95 %; sus validaciones son todas inferiores a 90 %. Los 12 casos fallan su gate. La exactitud nominal de una consulta en la celda favorable es94,287/89,960/91,962%, media 92,069 %, y no sustituye el endpoint conjunto. No se compara este 92,069 % como victoria sobre el Transformer histórico: cambian receta, presupuesto, supervisión, endpoint y uso del pool de evaluación.

![Cada inicialización y los tres endpoints; la línea roja es90% de validación](../../results/research/OPT03_QTRAIN_U16_review/37771588722/analysis/joint_competence_continuation.png)

## Siete contrastes prospectivos

Diferencias dentro de cada inicialización, n=3. Presupuesto promedia las cuatro condiciones; consultas/escuela promedian los otros factores. Interacciones respetan las diferencias de diferencias congeladas. IC t del 95 %, con 2 grados de libertad, exploratorios, sin ajuste de multiplicidad ni truncamiento.

| Contraste | Diferencias 200 / 201 / 202, pp | Media, pp | IC 95 % pareado, pp |
|---|---:|---:|---:|
| Presupuesto adicional (primario) | 17.374 / 20.596 / 20.692 | 19.554 | [14.862; 24.247] |
| Cuatro consultas − una repetida | 50.302 / 21.539 / 23.512 | 31.785 | [-8.128; 71.697] |
| Escuela 0,3 − 0 | 30.383 / 19.751 / 22.298 | 24.144 | [10.353; 37.934] |
| Consultas × escuela | 54.260 / 37.714 / 42.490 | 44.821 | [23.666; 65.976] |
| Presupuesto × consultas | 24.872 / 39.697 / 38.254 | 34.274 | [13.967; 54.581] |
| Presupuesto × escuela | -12.427 / 38.147 / 38.510 | 21.410 | [-51.385; 94.206] |
| Interacción triple | -38.721 / 74.548 / 72.809 | 36.212 | [-125.007; 197.432] |

El incremento de presupuesto es positivo en las tres inicializaciones. Los intervalos de escuela, consultas×escuela y presupuesto×consultas también excluyen cero bajo el análisis exploratorio previsto. Los de consultas, presupuesto×escuela e interacción triple incluyen cero. No se presentan los siete intervalos sin ajuste como siete confirmaciones independientes ni se deduce ausencia de efecto cuando incluyen cero.

Los intervalos dependen del supuesto del modelo t entre tres inicializaciones, difícil de comprobar con n=3. Un split y streams comunes no prueban generalidad entre datasets/entornos. Las máscaras, ejemplos y etapas no incrementan n. Los extremos formalmente fuera de límites físicos se conservan. Esta fase no interviene tying/reinyección, recurrencia, daño o crecimiento: sus contribuciones originales A06.1 no se recalculan ni se mezclan aquí.

![Todos los contrastes previstos, medias e intervalos sin recortar](../../results/research/OPT03_QTRAIN_U16_review/37771588722/analysis/prespecified_effects.png)

## Curvas, recursos y límites

La pérdida de respuesta desciende entre las dos últimas ventanas agregadas de 1.024 actualizaciones en los 12 casos. En cuatro consultas/escuela 0,3: semilla 200: 0,151→0,142; 201: 0,236→0,200; 202: 0,179→0,164. No demuestra convergencia ni una incapacidad arquitectónica general. Los resultados permiten afirmar mejora de aprendizaje con presupuesto/exposición adicionales en estas condiciones; no localizan una causa neuronal única.

El modo de evaluación correspondiente al entrenamiento sigue siendo distinto del denso. En esa celda, conjunto denso a 16.384: 80,176/79,785/63,184%, frente a88,593/80,621/84,808% en el modo correspondiente. Se conservan ambos, sin escoger masks o modo por score.

RAM disponible mínima registrada: 14,172 GiB, superior al mínimo obligatorio de 8 GiB. Las ventanas temporizadas suman 16.464,282 segundos: incluyen muestreo, update, hashing de witnesses y parte del bookkeeping. Excluyen gran parte de checkpoints, evaluaciones, archivos y preparación; no son coste computacional completo ni energía física. La energía no está medida. Tensores de entrenamiento/verificaciónCPU y dos threads configurados, Python 3.12.14, Torch 2.6.0+cpu, NumPy 2.2.6 y SciPy 1.15.1.

![Curvas completas; línea vertical: comienzo de la extensión](../../results/research/OPT03_QTRAIN_U16_review/37771588722/analysis/answer_loss_continuation.png)

## Integridad y reproducibilidad

- Plan LF SHA256 a4093bb8d559404b9f68587b753f8cea180a1e75d33363a04c1f2c7ebcb22dc0 y 19 fuentes gobernantes sin cambios después del inicio.
- Entrenamiento[37771588722](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37771588722), source adab1f3dc0ded9da3c42a3e7e5adcb96cdaba67e, completed/success. Recibo final fijado a Git cde5ff9b10c77fc4dcf901a873e3e6c7c570fd7e.
- Verificación[37774614332](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37774614332), source aa50cb08766fa0661719672b4dc364b31147da6f, completed/success. Recibo final fijado a Git a36f61347a3076ee16c872b7d5a2f3dff8188bb5.
- Auditoría: 4.105 comprobaciones, cero incidencias, doce casos y 2.654.208 decisiones. Replay: 2.196 comprobaciones, cero incidencias, las 2.654.208 decisiones exactas y máximo error de NLLmedia0. Checkpoints reales, optimizador y tres RNG validados; no nuevas actualizaciones ni test en replay.
- Recuento estadístico independiente: 60 comprobaciones, tolerancia1e−12; diferencias anidadas frente a pesos factoriales ortogonales. CDF con 2 grados de libertad contrastada analíticamente, residual 5,74e−13. Derivados locales: Python 3.13.7 y SciPy 1.15.1; no confundir con software de entrenamiento.
- Se conservaron todos los negativos y parciales, primer preflight fallido por Python 3.12.15 frente a 3.12.14, rechazo local por RAM y observaciones que negaban un cierre prematuro. No se flexibilizaron gates/tolerancias ni se reinició por un timeout de observación.

Tablas CSV, resumen JSON y figuras PNG/SVG con hashes en results/research/OPT03_QTRAIN_U16_review/37771588722/analysis/. Regeneración de derivados sin entrenamiento/inferencia: scripts/summarize_OPT03_QTRAIN_U16.py y scripts/supplement_OPT03_QTRAIN_U16.py. Auditoría y recibos en el mismo directorio de revisión; originales en la recuperación separada. Esta verificación es interna al proyecto, no replicación científica externa.

**Cierre limitado:** la receta U16 está terminada y auditada, con un resultado positivo de presupuesto/exposición y un resultado negativo de competencia. Quedan optimización suficiente, controles a competencia elevada y precisión/generalidad adecuadas. La siguiente intervención deberá conservar los doce estados y todos los negativos, registrar otra receta y distinguir beneficio de atención, acceso a consulta, capacidad y supervisión. Los módulos preparados de atención todavía no se han entrenado científicamente. No avanzar a tareas 4–8 ni reinterpretar H1 desde este cierre.
