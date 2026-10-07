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


## SCI-006-CLOUD — 2026-10-07

Following the already documented JEV recovery failure, continue the existing conservative fallback for reversible scientific work within item 6. This is a local operational decision, not a JEV decision. Recover historical scientific sources by exact recorded hashes, record the missing GPU worker, and validate a separately versioned standard public GitHub CPU environment before freezing and executing the declared ablations. Keep paired comparisons in the same software environment, preserve negative findings and failed attempts, and retain the no-paid-compute constraint. No main-branch merge or external scientific submission is part of this action.

## SCI-006-FREEZE — 2026-10-07T03:22:22Z

Root scientific/operational review approves the complete item-6 execution plan after the passing cloud suite and independent supervisor review. This remains the documented local conservative fallback following JEV recovery failure; it is not represented as a JEV decision. Both panels are frozen together and cannot be changed between panels in response to results. Preserve failures, optimization limits and negative findings. The worker may write only its owned synthetic-study evidence to the separate results branch, using bounded waits and an unchanged scientific execution HEAD. Item 5's H1 remains not supported; the budget ablation is exploratory. Main is unchanged, and no external scientific submission or paid compute is authorized by this decision.

## 2026-10-07T05:32:16.665091+00:00 — SCI-006 scientific closure

Root closes only item 6 after complete frozen execution and verified evidence. Larger budget improved each declared paired binding score but all 26 probes remained below 0.95; all 38 core contrast intervals included zero. On a fixed retained expert bank, scanner routing improved binding over random slots, but absolute binding stayed about 22–24% and the learned gate added unmatched supervised fitting. Novelty reproduced the fixed bank; resonance retained K=1/2 by seed. These conditional findings do not demonstrate generally effective growth, autonomous repair, an architecture advantage at matched cost, or a rescue of H1. All negative comparisons and scope limits remain in the report. Routine continuation follows the user's standing point-by-point instruction and the previously recorded conservative fallback; no new expenditure, external scientific submission or main merge is authorized by this record.
