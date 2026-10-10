# Registro de decisiones

| ID | Fecha | Decision | Evidencia | Claude | JEV | Estado |
|---|---|---|---|---|---|---|
| DEC-001 | 2026-09-27 | Markdown compartido como canal asincrono; cola y reservas son fuente de verdad | peticion de Fran | pendiente | pendiente | provisional |
| DEC-002 | 2026-09-27 | Una carga GPU larga a la vez tras terminar las dos preexistentes | 99 % GPU, 83 C, RAM libre 4,5 GiB | aceptada en SYNC-001 | `single_gpu_queue`, confianza 1,00 | confirmada |
| DEC-003 | 2026-09-27 | Cerrar cargas activas; FIL-002 es el siguiente uso largo de GPU | estado actualizado y consulta tipada | trabajo ya iniciado en codigo | FIL-002, confianza 0,55 | confirmada |
| DEC-004 | 2026-09-27 | JEV asigna agente, ruta de modelo y esfuerzo para cada tarea sustancial | instruccion de Fran y consulta tipada actualizada | sincronizado | `provenance=jev`, `jev-1.13.0` | confirmada |
| DEC-005 | 2026-09-27 | Claude: SOIL-001, FIL-001 y FIL-002. Codex: OPS-001, SOIL-002, FIL-003, NCA-001 y CLOUD-001 | reclamaciones de Claude y desempate JEV | aceptado en tablón | confianzas por tarea 0,21-1,00 | confirmada con revision de Fran |
| DEC-006 | 2026-09-27 | Codex y Claude operan autonomamente; informan avances a Fran y llevan dudas, reparto y decisiones tecnicas a JEV | directiva FRAN-DELEGACION-001 enviada a ambos agentes | registrada por Claude | no requiere reinterpretar una orden directa de Fran | confirmada |
| DEC-007 | 2026-09-27 | Tras dos cambios consecutivos sin superar el ruido, cambiar de eje experimental; no prolongar solo por mejora numerica | resultados FIL-001 y regla PROCESO-AVANCE-001 | propuesta por Claude | consulta nueva no enviada por control de datos externos; fallback local | provisional |

Una decision solo queda `confirmada` cuando tiene evidencia suficiente. JEV debe devolver
`provenance=jev`; un error o una respuesta local se registra como `no revisada`.

| DEC-008 | 2026-09-28 | FSR 1 (EASU+RCAS, `kaggle/filament/multiscale.py`) queda en reserva: no se sigue en filamentos; candidato para retos donde importe velocidad o reconstruccion suave | ablation FIL-002/004: learned 0,4263 > FSR 0,367 (c=24); FSR satura a ~8k it y empeora con tamaño; ~3x mas rapido | propuesto con Fran | decision directa de Fran | confirmada |
| DEC-009 | 2026-09-28 | La GPU de Kaggle se usa solo para ejecutar el cuaderno final, inferencia, generar resultados y entregar al concurso; nunca para entrenar | instruccion directa de Fran | pendiente de lectura | no requiere reinterpretar una orden directa de Fran | confirmada |
| DEC-009 | 2026-09-28 | Publicar el codigo de filamentos tambien en Kaggle (cuaderno o foro del concurso) para cumplir la regla 3.6.b y ganar visibilidad; el GitHub publico se mantiene | regla 3.6.b; push 7da18d1 del 27-09 | propuesto por Claude | decision directa de Fran | confirmada |
