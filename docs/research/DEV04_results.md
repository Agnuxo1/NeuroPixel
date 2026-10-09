# DEV04: reproducción interna del candidato y precisión registrada

Cohorte completa: 12 cuerpos nuevos y 24 lecturas. Atención menos control local: **+16.244 pp**, ICt95 exploratorio **[10.586; 21.902] pp**, semianchura **5.658 pp**.

Gate de competencia: **superado**, 12/12 candidatos cumplen probe ≥95% y validación ≥90%. Objetivo de precisión (semianchura ≤5 pp y límite inferior positivo): **no superado**.

## Los doce pares registrados

| Política | Inicialización | Atención validación % | Local validación % | Diferencia pp | Atención probe % | Gate |
|---|---:|---:|---:|---:|---:|---|
| 101 | 400 | 99.780 | 75.256 | +24.524 | 99.817 | Pasa |
| 101 | 401 | 99.792 | 81.488 | +18.304 | 99.866 | Pasa |
| 101 | 402 | 99.811 | 89.911 | +9.900 | 99.707 | Pasa |
| 101 | 403 | 99.762 | 88.538 | +11.224 | 99.805 | Pasa |
| 102 | 404 | 99.835 | 77.313 | +22.522 | 99.744 | Pasa |
| 102 | 405 | 99.884 | 70.117 | +29.767 | 99.792 | Pasa |
| 102 | 406 | 99.707 | 88.458 | +11.249 | 99.451 | Pasa |
| 102 | 407 | 99.683 | 88.666 | +11.017 | 99.792 | Pasa |
| 103 | 408 | 99.518 | 67.261 | +32.257 | 99.561 | Pasa |
| 103 | 409 | 99.231 | 92.139 | +7.092 | 99.316 | Pasa |
| 103 | 410 | 99.579 | 89.795 | +9.784 | 99.402 | Pasa |
| 103 | 411 | 99.927 | 92.639 | +7.288 | 99.951 | Pasa |

## Promedios por política

| Política | Media pareada pp | Inicializaciones |
|---|---:|---:|
| 101 | +15.988 | 4 |
| 102 | +18.639 | 4 |
| 103 | +14.105 | 4 |

El estimador promedia cuatro diferencias pareadas dentro de cada política y luego las tres medias con igual peso. El intervalo df2 no se trunca, no se ajusta por multiplicidad y es exploratorio. Las políticas reutilizan un universo semántico de desarrollo; sus composiciones se solapan. Ni 24 cabezas ni máscaras, ejemplos, endpoints o reintentos se consideran réplicas independientes.

## Procedencia y auditoría

Plan SHA-256: `2484f3c5917e474ca3755bcdbbaca7d322077615ba47d51653116042d980f20a`. Se preservan los 43 archivos gobernantes. Ejecución original `37845945011`: nueve éxitos y tres fallos de transporte antes del entrenamiento. Recuperación operativa `37871777801`: exclusivamente los tres fallos bajo una nueva ejecución; cero casos exitosos reentrenados. Verificación interna de endpoints `37876687599`, fuente `0ef7446ff8a0b872b3d2f85d5deb11d387ac4375`.

Se recuperaron ZIP originales, manifiestos, checkpoints de todos los endpoints, optimizadores y tres streams RNG. Replay independiente de decisiones y recuentos, datasets regenerados, controles ACTO/LUGAR inalterados, mismos witnesses de batches/máscaras para cada pareja, finitud y ventanas de RAM ≥8 GiB. La tolerancia conserva decisiones exactas y NLL media ≤1e-4. La prueba completa contiene 9.732.096 decisiones individuales; no equivale a otros tantos sujetos experimentales.

Recuento estadístico ortogonal con denominadores enteros: 41 comprobaciones sin discrepancias. Los archivos derivados están vinculados por hashes al original, al replay y al análisis.

## Alcance y conclusiones

La evidencia se refiere únicamente a cuatro consultas, escuela 0,3 y nuevas inicializaciones de cuerpos bajo la receta seleccionada después de READ03. Es desarrollo interno y no una comparación nueva justa contra Transformer. Las 264 composiciones del test original se conservaron excluidas y no se puntuaron. H1 original permanece no soportada bajo su receta.

Las lecturas comparten cuerpo, decoder, datos, labels, batches, máscaras y presupuesto de updates. Ambas tienen 3.168 parámetros nominales activos; sus espacios funcionales y FLOPs difieren. El resultado no atribuye toda la capacidad a dinámica celular ni excluye atajos. Atención sobre estado visible tiene antecedentes; una ganancia no demuestra originalidad excepcional.

El replay y la reproducción de cuerpos pertenecen al mismo proyecto. No constituyen replicación externa, generalización visual, memoria continua, estabilidad prolongada, reparación, medición física de energía ni utilidad independiente. El coste completo y la energía siguen pendientes; el tiempo de updates excluye extracción de caches, evaluación, transporte y otras operaciones.

DEV04 puede cerrarse como receta completa auditada incluso si sus criterios no se superan. El cierre de la tarea 3 amplia exige una decisión separada sobre controles, límites de optimización y alcance promovible; este informe no la cierra automáticamente ni adelanta tareas 4–8. No se añaden semillas ni presupuesto para estrechar el intervalo retrospectivamente.

## Reproducir el análisis

`collect_DEV04_replay_receipts.py` requiere snapshots Git y de ejecuciones completas; `summarize_DEV04.py` verifica los doce ZIP y proofs antes de cualquier efecto; `render_DEV04_report.py` recontabiliza el estimador con denominadores enteros; `close_DEV04.py` verifica hashes, el informe y el alcance. Cada comando conserva los negativos y rechaza entradas incompletas.

Contexto primario y límites de comparación: [DEV04_context_literature_20261009.md](DEV04_context_literature_20261009.md).
