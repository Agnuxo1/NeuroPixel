# Registro de decisiones

| ID | Fecha | Decision | Evidencia | Claude | JEV | Estado |
|---|---|---|---|---|---|---|
| DEC-001 | 2026-09-27 | Markdown compartido como canal asincrono; cola y reservas son fuente de verdad | peticion de Fran | pendiente | pendiente | provisional |
| DEC-002 | 2026-09-27 | Una carga GPU larga a la vez tras terminar las dos preexistentes | 99 % GPU, 83 C, RAM libre 4,5 GiB | aceptada en SYNC-001 | `single_gpu_queue`, confianza 1,00 | confirmada |
| DEC-003 | 2026-09-27 | Cerrar cargas activas; FIL-002 es el siguiente uso largo de GPU | estado actualizado y consulta tipada | trabajo ya iniciado en codigo | FIL-002, confianza 0,55 | confirmada |
| DEC-004 | 2026-09-27 | JEV asigna agente, ruta de modelo y esfuerzo para cada tarea sustancial | instruccion de Fran y consulta tipada actualizada | sincronizado | `provenance=jev`, `jev-1.13.0` | confirmada |
| DEC-005 | 2026-09-27 | Claude: SOIL-001, FIL-001 y FIL-002. Codex: OPS-001, SOIL-002, FIL-003, NCA-001 y CLOUD-001 | reclamaciones de Claude y desempate JEV | aceptado en tablón | confianzas por tarea 0,21-1,00 | confirmada con revision de Fran |
| DEC-006 | 2026-09-27 | Codex y Claude operan autonomamente; informan avances a Fran y llevan dudas, reparto y decisiones tecnicas a JEV | directiva FRAN-DELEGACION-001 enviada a ambos agentes | registrada por Claude | no requiere reinterpretar una orden directa de Fran | confirmada |

Una decision solo queda `confirmada` cuando tiene evidencia suficiente. JEV debe devolver
`provenance=jev`; un error o una respuesta local se registra como `no revisada`.



## SCI-005-RECOVERY completed - 2026-10-07T00:44:04.8593721Z
 Tool owner: ChatGPT Recovery Tool / architecture_audit. New isolated local clone at D:\PROJECTS\NeuroPixel-Validation-20261007-continuation on research/scientific-validation-2026-10-07-continuation. Frozen fbf576f0 source ZIP restored with 77 safe entries and eight scientific hashes unchanged. Exact recovered report and receipt preserved. Item 5 closed by continuity reconstruction: 28 completed, zero failed, H1 not_supported, two verified audits with zero issues, 28 budget_limited probes. Linux raw artifacts and later Git history remain inaccessible, without demonstrated deletion. Item 6 remains pending. No experiments, GPU, original-repository edits, resource-limit changes, or interference with other processes. JEV unavailable fallback inherited from the explicitly authorized session. See coord/recovery/migration_20261007.json. Status: completed; no resources held.
