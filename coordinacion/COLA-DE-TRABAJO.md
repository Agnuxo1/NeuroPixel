# Cola de trabajo compartida

Actualizada: 2026-09-27 19:34 Europe/Madrid.

| ID | Prioridad | Estado | Responsable | Modelo/ruta | Esfuerzo | Reto | Recursos | Entregable / criterio de aceptacion |
|---|---:|---|---|---|---|---|---|---|
| OPS-001 | 0 | REVISION | Codex | determinista | — | global | ninguno | Mantener protocolo y resolver conflictos de cola |
| SOIL-001 | 1 | HECHA (20:13, EMD CV 39,195 vs media 85,393) | Claude | modelo general | medio | soil | GPU, hasta 6 GiB | Completar 6 folds; `result.json`; comparar EMD con curva media |
| FIL-001 | 2 | ACTIVA | Claude | modelo rapido | bajo | filament | GPU, hasta 14 GiB | Terminar 6000 it o documentar corte; mejor PQ y checkpoint |
| SOIL-002 | 3 | LISTA | Codex | modelo general | medio | soil | GPU 6 GiB | Auditar split y crear submission solo si CV supera referencia; validar esquema y monotonia |
| FIL-002 | 4 | ACTIVA (codigo) | Claude | modelo general | alto | filament | GPU exclusiva despues de FIL-001 | Lienzo multiescala a 1/4 + reconstruccion; CV comparable a 0,370 |
| FIL-003 | 5 | LISTA | Codex | modelo general | medio | filament | CPU/RAM | CV por pliegues para posprocesado sin ajustar sobre un unico split |
| NCA-001 | 6 | PROPUESTA | Codex | modelo general | alto | bateria | GPU exclusiva | Destilar 24 pasos a 6 y medir exactitud, latencia y robustez |
| NCA-002 | 7 | PROPUESTA | JEV por asignar | pendiente | pendiente | bateria | GPU exclusiva | Prototipo reversible y memoria maxima medida |
| CLOUD-001 | 8 | REVISION | Codex | determinista | — | global | red | Verificar cuentas/cuotas disponibles antes de crear trabajos externos |

Regla: una tarea `ACTIVA` debe tener reserva en `RECURSOS.md`. Si no actualiza evidencia durante
90 minutos, pasa a `BLOQUEADA/POR_CONFIRMAR`; nunca se mata automaticamente el proceso.

Toda fila nueva incorpora, tras la decision de JEV: `responsable`, `modelo`, `esfuerzo` y motivo
breve. Hasta entonces su responsable es `JEV por asignar` y no se inicia.

Decision JEV vigente: 2026-09-27 19:34, `provenance=jev`, modelo `jev-1.13.0`.
Politica: `single_gpu_queue`. Siguiente tras cerrar las cargas activas: `FIL-002`.



## SCI-005-RECOVERY - 2026-10-07T00:44:04.2842718Z
 Owner: ChatGPT Recovery Tool / architecture_audit. Status: in_progress. Scope: isolated restoration and continuity reconstruction of item 5 closure, explicitly authorized by parent. Item 6 remains pending. No experiments, GPU, or original-repository changes.


## SCI-005-RECOVERY completed - 2026-10-07T00:44:04.8593721Z
 Tool owner: ChatGPT Recovery Tool / architecture_audit. New isolated local clone at D:\PROJECTS\NeuroPixel-Validation-20261007-continuation on research/scientific-validation-2026-10-07-continuation. Frozen fbf576f0 source ZIP restored with 77 safe entries and eight scientific hashes unchanged. Exact recovered report and receipt preserved. Item 5 closed by continuity reconstruction: 28 completed, zero failed, H1 not_supported, two verified audits with zero issues, 28 budget_limited probes. Linux raw artifacts and later Git history remain inaccessible, without demonstrated deletion. Item 6 remains pending. No experiments, GPU, original-repository edits, resource-limit changes, or interference with other processes. JEV unavailable fallback inherited from the explicitly authorized session. See coord/recovery/migration_20261007.json. Status: completed; no resources held.
