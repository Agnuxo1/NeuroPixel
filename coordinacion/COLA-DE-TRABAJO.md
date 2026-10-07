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

## 2026-10-07T05:32:16.665091+00:00 — SCI-006 closed

SCI-006 is completed: all frozen execution, independent saved-array recounts, final archive verification and report review passed. Results are retained without configuration selection. Deliverable: docs/research/06_results.md, full tables and hash-anchored receipts. The actual completed source is 08d0d52edd05da6835e71479f3ba4399fcbeabae; raw archive 15e76456bc2b4cce5faec0b08fb5288fe7844547. Items 1–6 are completed investigations, not validated Nobel-level claims. Items 7–30 remain pending at this closure; the next action is to claim SCI-007 sequentially.

## 2026-10-07T05:39:32.627738+00:00 — SCI-007 opened

SCI-007 is ACTIVE, owned by root under the standing point-by-point authorization and recorded conservative fallback after JEV recovery failure. Separate code/exposure/method reviews are justified by the risk of misclassifying historical test access. Acceptance: a sourced exposure inventory, fail-closed prospective split/evaluation rules, meaningful tests or simulations, preserved failures and a reviewed limitations report. No model training is planned in this item.

## 2026-10-07T05:57:21.510280+00:00 — SCI-007 recipes frozen

Root approves the two item-7 plans after source/method and implementation review. Local governance tests pass 17/17 after one preserved denominator failure and correction; four Torch adapter checks await the declared CPU environment. Exact memberships will be exported without model execution, and the null selection simulation will run once from its committed source/plan with all predeclared conditions. Both plans precede those outputs. H1 stays closed and item 8 remains pending. Evidence stays on the isolated research branches; no response is requested.

## 2026-10-07T06:22:34.038049+00:00 — SCI-007 closed

SCI-007 is completed after frozen execution, independent saved-artifact audits and final report review. Source 0ed43bd5bb1d8bcb166b8e63e909195cbe69f769; cloud raw archive f0aa1f17e08a495e92e7e8084003f357abfe9a8d; run 37578991388. Cloud tests passed 21/21, exact split audit has zero issues, and all four null-simulation conditions passed their declared analytical diagnostics and independent recount. The initial denominator failure and all subsequent corrections are preserved. Deliverable: docs/research/07_results.md, SHA256 895d261bf2c11896d8b011cf196fdc3004a5e5b154c668144214b4c2e4853cca; report review covers 42 numeric table rows with zero discrepancies.

The audit distinguishes internally disjoint configurations from cross-configuration eligibility: TRAIN unions cover 1,319/1,320 triples, but actual historical minibatches were not audited. The new contract is opt-in, and no historically fresh holdout or actual NeuroPixel bias estimate was created. H1 remains not supported. Items 1–7 are completed investigations; items 8–30 remain pending at this closure. The next action is to claim SCI-008. No response is requested.

## 2026-10-07T06:27:00.782435+00:00 — SCI-008 opened

SCI-008 is ACTIVE after the complete item-7 closure was archived and fetched as ef25ed6497cd3554d8eb78a0ef278f7b589e1e1f. Root owns integration and coordination under the standing user authorization and previously documented conservative fallback after JEV recovery failure. Investigate metric and invariant defects with source-level hypotheses, minimal reproductions, focused fixes and regression checks. Padding, routing summaries and Soil CDF/fold calculations are in scope; no external submission or main-branch change is requested. Focused reviewers may work on disjoint paths within this item. Items 9–30 remain pending. No response is requested.


## 2026-10-07T06:52:42.874308+00:00 — SCI-008 integration recipe frozen

Root freezes docs/research/08_cloud_validation_plan.json (SHA256 46b7f19063325b78e593ea53fbbc9bf4372a3f81e7d503a8ac58ee301c15455b) after four focused source audits, preserved regression failures/corrections, historical-fixture checks and independent worker review. The plan binds 95 files and 187 pytest cases across 13 files, including 58 new methods. Only the declared optional CIFAR absence may skip. The two-route legacy PAD witness and entire integration suite are admitted under the existing short CPU reservation. Ordinary failures remain archived; any post-freeze correction requires an explicit new source/plan revision. Historical controller/hash policies remain unchanged and reject modern code as a historical replay. H1 remains not supported; items 9–30 remain pending. No response is requested.


## 2026-10-07T07:17:56.375298+00:00 — SCI-008 closed

SCI-008 is completed after frozen source execution, source/archive identity checks, analytical PAD recount, local-evidence audit and report review. Deliverable: docs/research/08_results.md (SHA256 c9ae8a5dab162e7b7ca2b82255c2dfb2105dce7d01f6228b37a0e081a31070ec); review has zero unresolved issues. Source d01e20625068a2ec19bf25554e106c1fc2cc4420; raw archive 505a7098a077f90421e4a5ed0cd4c06882e98211; run37584119610. The suite has187 collected cases:186 passed, one optional CIFAR skip, zero failures/errors. All58 new methods passed. The341 JUnit header counts187 parents plus154 reconstructed nested subtests; it is not341 independent cases. Items1–8 are completed investigations; items9–30 remain pending at this closure. H1 remains not supported. Next action: claim SCI-009 sequentially.


## 2026-10-07T07:19:16.323499+00:00 — SCI-009 opened

SCI-009 is ACTIVE after verified item-8 closure6b2f2554b56c3ac2d7811ea4f6732befd7a93817. Root owns sequential integration under standing authorization and the documented conservative fallback after JEV recovery failure. Focused source/method/implementation reviews within item9 are justified by the risk of shortcut tasks and changed labels. Acceptance: explicit complex-binding task/estimands, primary-source comparison, positive and shortcut controls, reviewed frozen bounded evaluation, saved predictions/source/configuration, independent recount and a limitations report. Items10–30 remain pending.


## 2026-10-07T07:45:18.180685+00:00 — SCI-009 Stage-A reservation and implementation review

SCI-009 implementation reviews are complete in the isolated worktree. The finite two-event grammar and six paired transformations are specified;20 local grammar tests and12 current local protocol/metric tests passed, while two Torch integration cases await the pinned environment. The complete cloud inventory will contain34 cases with no allowed skip. Preserve the earlier12-case protocol receipt (11pass,1skip) and its exact recovered source preimages separately from the current14-case source (12pass,2skip). Source reviews found and corrected pilot-selection transfer and fixture-manifest identity checks before any performance execution. Root will freeze Stage A after checking final review receipts. Items10–30 remain pending.


## 2026-10-07T07:45:55.891523+00:00 — SCI-009 Stage A frozen

Root freezes09_preflight_plan.json, SHA256 7802ee68ae8ce19bf17e6135815e0c796216696e05c8d322347a314a2c88df04, binding49 source/recipe/review/evidence files and34 collected test cases. Reviews have no unresolved blocker in the frozen scientific or operational source. The new35-to37-token task uses the corrected core directly. The declared contract tests precede two bounded memorization diagnostics and all four development pilots; zero skipped tests are permitted in the pinned CPU runtime. No Stage-B training or final performance evaluation is launched by this plan. All prior evidence and failure boundaries remain preserved. H1 remains not supported; items10–30 remain pending.


## 2026-10-07T08:22:31.852462+00:00 — SCI-009 Stage B admitted for prospective freeze

The reviewed Stage-A evidence is complete: two successful 32-row memorization diagnostics and four finite development pilots, 34 collected cloud tests passed without skips, 138,074 independent scientific checks and 861 archive checks with zero issues. Admit the unchanged ten-run, 4,096-update Stage-B inventory under the separate bounded resource reservation. The frozen validation-binding priority selects NeuroPixel LR 0.001 and RelativeTransformer LR 0.003; neither lower cross-entropy at the other rate nor runtime changes this rule. The complete source-linked handoff receipt, analysis source, 19 passing analysis tests, preserved earlier analysis failure and exact offline Python/NumPy/SciPy runtime will be bound before final data exist. Items 10–30 remain pending; H1 remains closed and not supported. No response is requested.


## 2026-10-07T08:28:04.762952+00:00 — SCI-009 frozen Stage B launched

Source commit 83fe135e301f76bc0c74e30c66bb18e067ca5959 (tree a2b8319335bd6fb05de8b1c9749d537d10150218) contains the reviewed plan, SHA256 7e9089ae013abbc4fa042e30606d5d1329dc8b7e89281fcab080addc832c7792, and all 83 bound files. Root verified the clean frozen checkout, all 41 changed/new Git blobs and every binding. Workflow run 37593731891, job 112701154832, was observed in progress after the source-only trigger. This records launch, not completion or scientific outcomes. The independent final analyzer and all scientific criteria remain fixed. Work stays within item 9 until complete raw archival, recount and report review; items 10–30 remain pending.


## 2026-10-07T09:50:38+00:00 — SCI-009 completed Stage B and frozen offline continuation

GitHub run 37593731891 / job 112701154832 completed successfully from scientific source 83fe135e301f76bc0c74e30c66bb18e067ca5959. Its stopped final archive is b64d0e2ba222df76caf72bc9870c7602873e7236, also the root-observed results head; the pre-final gate archive is 2e58f5deb47f5676e8663d666d9db7ea98e3e080. The worker recorded 3,061.636433142 seconds before final archival. Its final manifest declares 134 files and 56,105,745 bytes excluding the manifest. These observations establish completion and inventory, not an independent scientific recount.

The isolated local executor stopped returning commands after 09:02:53 UTC. Continue the already frozen offline analysis on one bounded standard public CPU runner. No model training, checkpoint replacement, final-data regeneration, reselection, criterion change or historical H1 replacement is authorized by this operational continuation. Root verified that all 83 scientific bindings retain their original Git blobs, modes and types, and that all 16 operational bindings match candidate d58aeebd9877326002eac9be5edf8799728bd9db. The new plan SHA256 is 7cc8b995ebc388089d1668820a60174fa228de4e5bba1285b14e3fb35d4b6eb3; relative to the saved candidate, only its status and freeze timestamp change.

Four configuration failures (37599727642, 37600603127, 37601029823, 37601447941) had zero jobs and occurred before any offline audit began. Their provider receipts are retained separately. Two workflow defects were corrected: a colon-space command now uses a YAML literal block, and runner.temp is now in the supported step environment scope. Root completed a source/context review; the two requested focused agent follow-ups were still pending at freeze. Absence of a new rejected non-triggering candidate at the last observation is not claimed as a completed platform validation.

Execution order remains full archive/source/runtime audit, actual Git gate audit, unchanged independent scientific recount, then figures. Preserve first failures and all full outputs in a new 09_offline_runs run/attempt directory; the raw input checkout and frozen source are separate and unchanged. Item 9 stays ACTIVE through complete archival, recount, interpretation and report review; items 10–30 remain pending. This is a same-custody host continuation, not external replication. H1 remains closed and not supported.
