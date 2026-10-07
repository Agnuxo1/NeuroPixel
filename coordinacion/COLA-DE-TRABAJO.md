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


## 2026-10-07 — Scientific validation continuation

SCI-006 is active in the isolated research/scientific-validation-2026-10-07-cloud branch. Root is responsible for integration and review; focused collaborators cover source recovery, implementation and independent scientific checks within item 6. Items 1–5 are completed investigations, with H1 not supported in item 5. Items 7–30 remain pending. The immediate action is a source/environment preflight with zero optimizer updates and no final-test access. See docs/research/progress.json and 06_recovery_manifest.json. This dated section does not assert that the historical September reservations above are still active.

## 2026-10-07T03:22:22Z — SCI-006 frozen study

Root has closed implementation validation and approved the complete item-6 execution plan. The final cloud suite passed 107 tests with no failures and one unrelated optional CIFAR skip. SCI-006 now proceeds sequentially through 26 core trainings, the core final gate, six growth trajectories (18 stages), eight train-only gate fits, the growth final gate and both independent recounts. Both panels are frozen before the first training. Items 7–30 remain pending. The live owned attempt is recorded in worker_status.json on the separate research/scientific-validation-2026-10-07-cloud-results branch. Resource reservation and cut conditions are recorded below in RECURSOS.md.
