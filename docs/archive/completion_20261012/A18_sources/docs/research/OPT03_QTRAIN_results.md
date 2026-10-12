# OPT03-QTRAIN: supervisión de consulta y escuela

La fase de desarrollo terminó y se auditó el 8 de octubre de 2026. **Ninguna condición pasó el criterio congelado de competencia.** No se selecciona un ganador ni se abre test final. La tarea 3 amplia sigue abierta: optimización suficiente, competencia reproducible, controles de capacidad/supervisión a competencia elevada y precisión útil no están resueltos. H1 permanece no soportada.

## Receta exacta y ejecución

Doce entrenamientos nuevos: semillas 200/201/202 × cuatro consultas por contexto frente a una consulta repetida cuatro veces × escuela 0/0,3. Ambas políticas reciben los mismos 16 draws de contexto por minibatch, con reemplazo, 64 ejemplos y 16 consultas de cada rol. No se garantiza que los 16 draws sean contextos únicos. Las cuatro consultas exponen más diversidad de etiquetas dentro de cada contexto; esa diferencia es la intervención. Los metadatos de fillers/roles no se añaden a la entrada de la red.

Backbone histórico fijo: 29.824 parámetros, T16, firing 0,5, PAD legado, tying y reinyección activos. AdamW LR0,001, clipping1,0 y 8.192 actualizaciones por caso; parámetros, contextos, ejemplos, roles y actualizaciones controlados. Escuela añade supervisión y cómputo; igualdad de actualizaciones no demuestra igualdad de FLOPs o coste completo. El LR común es una nueva elección de desarrollo, no una mejora causal de LR demostrada.

El endpoint primario exige acertar **la consulta original y la consulta AGENTE↔PACIENTE cambiada sobre el mismo contexto**. Se promedian ocho rollouts individuales de firing0,5, seeds79000–79007, sin seleccionar máscaras ni hacer ensemble. El criterio requiere joint-query probe≥95% y validación≥90%, en las tres inicializaciones de una condición. No cambia los gates históricos. Probe2048/validación4096, seeds77001/77002, son pools de desarrollo ya expuestos históricamente; no hay nuevo test final.

- [Entrenamiento original](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37728384380): completed/success; source `6a5af5b68aaa9173edf46658dd8f9d16845fbad6`.
- [Verificación de inferencia](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37732670363): completed/success; source `6d8341a59dbe99475e0b747d8340de493273c6d6`. No nuevos entrenamientos ni test.
- Plan LF SHA-256: `20610e354288b48eeb7316367f61da5173fb2a124a01effa93e7d40e61bf3fb5`; once fuentes congeladas verificadas. Recibos finales anclados a Git en commits `fd800145917408d19b01e15b1ed8edc2f81ad433` y `9762b99a82e3738cb0c88cffc5c3c3716c0e9bda`.
- RAM disponible mínima registrada: 14,156 GiB; mínimo obligatorio8GiB cumplido. No se rebajó el límite ni se repitieron entrenamientos cerrados.

## Competencia absoluta

Porcentajes de exactitud conjunta en validación; valores por semilla 200/201/202. IC95 t entre tres inicializaciones, df2, sin truncar.

| Condición | Semillas 200 / 201 / 202 | Media | IC95 | Criterio en las tres |
|---|---:|---:|---:|---|
| Cuatro consultas / escuela 0 | 1,270 / 1,569 / 1,447 | 1,428 | [1,055; 1,802] | No |
| Cuatro consultas / escuela 0,3 | 74,677 / 2,466 / 7,532 | 28,225 | [-71,906; 128,356] | No |
| Una consulta repetida / escuela 0 | 0,214 / 0,098 / 0,104 | 0,138 | [-0,024; 0,300] | No |
| Una consulta repetida / escuela 0,3 | 0,000 / 0,555 / 0,104 | 0,220 | [-0,514; 0,953] | No |

El caso paired/escuela0,3/seed200 alcanza 76,599% conjunto en probe y 74,677% en validación; el binding nominal de una consulta es 86,865%. Los otros dos casos de esa condición tienen 2,466% y 7,532% conjunto en validación. Es aprendizaje parcial con fuerte dependencia de inicialización. El score nominal alto de una semilla no satisface el criterio conjunto ni permite compararlo como superioridad al resultado histórico del Transformer sobre otra receta/protocolo.

Las doce ejecuciones fallan el gate por caso; ninguna de las cuatro condiciones es elegible. La media 28,225% de paired/escuela0,3 no describe una solución estable. No se elige solo seed200 ni se descartan seeds201/202.

## Contrastes prospectivamente definidos

Puntos porcentuales; diferencias calculadas dentro de cada inicialización. El primario promedia paired−repeated sobre escuela. Los intervalos no tienen ajuste de multiplicidad y son exploratorios.

| Contraste | Diferencias 200 / 201 / 202 | Media, pp | IC95 pareado, pp |
|---|---:|---:|---:|
| Cuatro consultas − una repetida, promedio sobre escuela (primario) | 37,866 / 1,691 / 4,385 | 14,647 | [-35,416; 64,711] |
| Escuela 0,3 − 0, promedio sobre política | 36,597 / 0,677 / 3,043 | 13,439 | [-36,467; 63,345] |
| Interacción consultas × escuela | 73,621 / 0,439 / 6,085 | 26,715 | [-74,437; 127,868] |

Los tres contrastes tienen signos positivos en las tres inicializaciones, pero sus intervalos incluyen cero y están dominados por seed200. No confirman beneficio poblacional reproducible ni equivalencia. La interacción tiene especial incertidumbre; no se presenta como un mecanismo causal único del aprendizaje insuficiente.

La unidad de replicación es la inicialización: **n=3**, un split y streams de muestreo controlados. Las ocho máscaras, los tres endpoints, los roles o los 6.291.456 ejemplos procesados durante los doce entrenamientos no aumentan n. No se puede comprobar adecuadamente el supuesto distributivo del intervalo t con tres muestras. Los extremos que exceden los límites físicos se conservan como resultado formal del método congelado.

## Curvas y límites de optimización

La pérdida de respuesta media desciende entre las dos últimas ventanas agregadas de 1.024 updates en los doce casos. En paired/escuela0,3/seed200 baja de0,376 a0,290; en seeds201/202 baja de0,896 a0,818 y de0,813 a0,737. Los endpoints densos de seed200 suben de1,904% conjunto a4.096updates a60,010% a8.192; el endpoint correspondiente al entrenamiento da74,677% a8.192. Son métricas de modos diferentes y se conservan por separado.

Estas curvas no demuestran convergencia, una causa única ni incapacidad arquitectónica. Justifican estudiar prospectivamente el presupuesto y la estabilidad de aprendizaje, preservando controles y todas las semillas. La extensión tendría que ser una nueva receta registrada, reutilizar los estados completos y no repetir los primeros8.192updates. No constituye selección de un ganador de esta fase negativa.

## Auditoría, precisión numérica y reproducibilidad

- Auditoría completa: 4.338 comprobaciones, cero incidencias, doce casos y 1.622.016 decisiones recontadas. Verifica fuentes, manifiestos, datasets, roles/targets, streams de contextos, ventanas, recursos, métricas y gates.
- Replay de checkpoints en1024/4096/8192: 1.932 comprobaciones, cero incidencias y las 1.622.016 decisiones reproducidas exactamente; máximo error de NLL media0. Modelos y optimizadores finitos, estados y datasets identificados por hashes. Recibos por caso anclados a commits y blobs Git.
- Recuento estadístico suplementario: 28 comprobaciones de vectores, medias, SD e intervalos a tolerancia1e−12, usando aritmética independiente y el valor crítico de la biblioteca congelada; CDF df2 contrastada analíticamente, residual5,74e−13.
- Se conserva un **primer recuento analítico fallido**: cuatro intervalos difieren de los de SciPy1.15.1 por el valor crítico numérico, máximo1,254e−11 en proporción (1,254e−9pp). No se elevó su tolerancia ni se borró el fallo. El recuento suplementario mantiene el cuantil del análisis congelado y valida su CDF de manera independiente; este detalle no cambia ninguna decisión científica. Evidencia en `statistics_initial_failure.json` y su generador original archivado.
- Esta verificación es interna al proyecto. Un segundo ejecutor y un segundo algoritmo de recuento no son replicación científica externa independiente.

Tablas CSV, resumen JSON y figuras PNG/SVG, con hashes de entradas y salidas, en `results/research/OPT03_QTRAIN_review/37728384380/analysis/`. Regeneración de derivados, sin entrenamiento ni nuevas evaluaciones de modelos:

```text
python scripts/summarize_OPT03_QTRAIN.py
```

![Valores por semilla, medias e intervalos; puntos azules/naranjas/verdes:200/201/202](../../results/research/OPT03_QTRAIN_review/37728384380/analysis/joint_and_effects.png)

![Pérdidas de respuesta archivadas, sin modificar pesos](../../results/research/OPT03_QTRAIN_review/37728384380/analysis/answer_loss_curves.png)

**Cierre limitado:** OPT03-QTRAIN está terminado y auditado; no se alcanzó competencia suficiente ni precisión útil y no se abre test final. La tarea3 y el programa completo siguen abiertos. Las tareas4–8 permanecen pendientes.
