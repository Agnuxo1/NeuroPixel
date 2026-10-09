# Estado de READ03: ejecución y verificación

**Actualización tras el cierre:** ambos ejecutores terminaron correctamente y los 36 ajustes están auditados y reproducidos. El informe vigente es `READ03_results.md`; el estado general y lo pendiente están en `ESTADO_HECHO_Y_PENDIENTE_20261008_READ03.md`. El texto siguiente conserva la fotografía operativa anterior al cierre.

Actualizado el 8 de octubre de 2026. La receta está congelada en `READ03_execution_plan.json`, SHA-256 `d273b8284d9a387b49266dc4ec62cfad76aefaa11c5f82c7ab6a97ba0e431ea0`. Sus 32 archivos gobernantes permanecen intactos.

El archivo congelado `READ03_protocol.md` conserva las expresiones históricas «borrador» y «falta implementar». Describen su preparación anterior. El manifiesto JSON fechado antes del ajuste y los recibos de ejecución establecen el estado operativo actual. Esta nota corrige la descripción del estado; no modifica datos, hipótesis, receta, endpoints, contrastes, tolerancias ni gates.

Trabajo realizado:

- A06.1 y la conciliación de las continuaciones están cerradas dentro del alcance documentado.
- QTRAIN y su extensión U16 terminaron y fueron auditadas. U16 reutilizó los doce estados completos; ninguno de sus prefijos se repitió. La media conjunta de cuatro consultas con escuela 0,3 llegó al 84,674%, pero las tres semillas siguen por debajo de los gates registrados. H1 original permanece no soportada.
- READ03 compara atención condicionada por consulta, agregación global uniforme y control local con consulta. Son 36 ajustes nuevos de cabeza sobre doce cuerpos y diccionarios congelados; cero actualizaciones del cuerpo. El conteo nominal de las cabezas coincide, pero la uniforme tiene menos parámetros con gradiente, lo cual está declarado.
- El preflight de la receta pasó 14 contratos y las admisiones de los doce padres reales. La ejecución científica es `37831051538`, fuente `a19c0000b4175984c14cd4eaf08392ceffd9e917`.
- Se implementaron una auditoría de archivos, un replay de checkpoints y un recuento independiente de métricas. El análisis final contempla los cinco contrastes registrados y exige los 36 ajustes verificados y el éxito terminal de ambos ejecutores. Doce contratos analíticos de recuento, contrastes y cierre pasaron.
- El primer verificador `37833686776` falló por una dependencia de recuperación ausente, antes de verificar ningún ajuste. Su recibo y análisis se conservaron. Se corrigió únicamente el empaquetado del verificador; la receta científica no cambió. El nuevo verificador es `37834470656`, fuente `ce0bdc7ab3080fe851cfe0134b7a68ffc4ec7b98`.

Trabajo pendiente inmediato:

1. Recuperar todos los ajustes y checkpoints; verificar sus hashes, streams comunes y congelación efectiva.
2. Reproducir los tres endpoints de cada cabeza y recontar las métricas. Conservar cualquier discrepancia antes de concluir.
3. Reunir los recibos de éxito de ambos ejecutores; entonces calcular los cinco contrastes pareados, intervalos y gates sin selección retrospectiva.
4. Concluir qué demuestra este diagnóstico y qué límites de optimización, competencia y precisión siguen abiertos. Un replay del mismo proyecto no es replicación externa.

Las cifras dinámicas de recuperación/verificación se guardan en `programme_20261008.json`, `READ03_live_observation.json` y los recibos bajo `results/research/READ03_review/37831051538/`. Recuperado, íntegro y reproducido son estados diferentes. Todavía no se afirma una mejora por atención ni se da por terminada la tarea 3. Las tareas 4–8 siguen pendientes.
