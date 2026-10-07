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


## 2026-10-07T10:09:16+00:00 — SCI-009 investigation closed

SCI-009 is complete as a feasible investigation; its target capability was not demonstrated. Report: docs/research/09_results.md, SHA256 f1dec259b08fd4c0ec6593c1eae1f64bed5ab606d58a62d6678bfc3158a65ee2. Scientific source 83fe135e301f76bc0c74e30c66bb18e067ca5959, raw final archive b64d0e2ba222df76caf72bc9870c7602873e7236, pre-final gate 2e58f5deb47f5676e8663d666d9db7ea98e3e080. All ten 4,096-update trainings and their final predictions completed. Final evidence contains 256 bags, 12,288 nominal rows and 10,240 unique canvases per checkpoint, with disclosed within-bag dependence.

The unchanged independent analysis ran once on offline source 87e1f1c0b388190474e8190952c148c0501c8abf, run37603398040/job112732965039, and archived at 2b8203f15ec5f6fe190876c80bf29232034603ce. Full archive:1,100 checks/0issues; actual Git gate:690/0; scientific recount verified/0issues. Root verified the complete 7,142,199-byte analysis, its projection and stored bootstrap-index digest, completed 462 saved-value checks and 393 report-table checks, and viewed both hash-verified PNGs. Agent receipt reviews and their corrected documentary preimages are retained. Check counts are not experimental sample sizes or external replications.

Mean base binding was 0.0962890625 for NeuroPixel and 0.4935546875 for the Transformer. The mean NP-minus-TF paired difference was -0.397265625, t95[-0.4125410251515038,-0.3819902248484962], n=5/df=4 conditional training realizations. All three NeuroPixel competence criteria failed. The reference also lacked reliable agent/patient discrimination; analytical 50% references and counterfactual role-specific results prevent treating its high global score as solved binding. Both networks could memorize the small Stage-A fixture, but the causes of weak broader performance remain unresolved. Historical H1 stays closed and not supported.

Keep all operational failures, original frozen sources, data, checkpoint records, analyses and report revisions. The local-executor interruption did not cause retraining or data regeneration. No GPU, paid service, external outreach, main-branch merge or independent-laboratory claim is part of this closure. Items1–9 are completed investigations; items10–30 remain pending at this instant. The next sequential item is SCI-010, generalization outside the generator, to be opened separately.


## 2026-10-07T10:11:14+00:00 — SCI-010 opened after verified SCI-009 closure

SCI-010 is ACTIVE after closure e6a3a5dffa602cd9731076390e9b20b6c738958e and independent publication read-back of the complete item-9 report. Root retains sequential ownership under the user's standing scientific-work authorization. Review externally authored benchmark sources and the model's actual input/output contracts before defining any transfer score. Distinguish same-generator perturbation, a separately implemented renderer, an external synthetic benchmark and real-world generalization. A semantic adapter must not copy the target into the input or use gold labels to construct the answer-bearing representation.

The weak item-9 competence and unresolved optimization/representation causes remain explicit. Do useful source, analytic and controlled-fixture work, and admit a bounded scientific evaluation only after its exact source, data identity, selection and estimand are reviewable and frozen. Do not add labels or change task semantics to make a claimed external benchmark fit. Physical or genuinely external validation may remain a dependency. H1 stays closed; items11–30 remain pending until this feasible investigation is completed.


## 2026-10-07T10:44:50+00:00 — SCI-010 next concrete action

Run24 frozen data contracts, then the bounded TRAIN-only provenance, duplicate and shortcut audit if all pass. Review raw bytes, actual geometry/vocabulary, grouping, controls and failures before admitting neural preflight. No item11 or later work is opened. A neural recipe is being prepared but is not part of this acquisition plan.


## 2026-10-07T11:01:48+00:00 — SCI-010 acquisition completed and independently recounted

Public CPU run37609533224/job112753126084 completed successfully; final raw commit19c8c40ab50141ba128623e7bbbc04cfb2460f31 contains34 files/23,778,454 bytes excluding its manifest. All24 contract methods and58 subtests passed. Root independently rehashed17 fetched text/JSON files and completed3,176 raw-TRAIN/parser/grouping/encoding/control checks with0 issues. Original-author endpoint404 is retained; the documented HTTPS mirror supplies the exact TRAIN member. No original-host byte comparison or external custody is claimed. TRAIN1000 records/919 unique inputs/752 connected groups; split895/105; geometry4x8/vocabulary18. No TEST parsing or neural execution occurred. The acquisition runner is released; local executor remains unavailable. Item10 continues with a prospective six-run neural preflight and exact fixture/source/runtime review; items11–30 remain pending. H1 is unchanged.


## 2026-10-07T11:12:59+00:00 — SCI-010 neural preflight frozen

Freeze21 bound files,32 contract methods and six TRAIN/DEV-only runs: two seed58 small-set memorization diagnostics with at most4096 updates, and four seed59 pilots at two declared learning rates with1024 updates each. RecipeSHA256 8492892f58a4065a92df99dbbbf30f0bd7340111c853cf5533a7102109fe8d19; planSHA256 f01846c1c6685328d49a5cdcb6d5c399222f99b8f160e37ccff62ee5e0418993. The149-context input-only fixture ledger is bound and checked against test definitions. All prospective source corrections and draft identities are preserved in preflight_preparation_receipt.json. Actual contract/model execution remains pending at this freeze. No official TEST member access belongs to this phase. Main seeds60–64 remain conditional on all six completions, both memorization criteria, independent audit and a separately fixed finite timing budget. Negative complete preflight evidence can be verified with admissionfalse. Item10 remains active; items11–30 unopened; H1 unchanged.


## 2026-10-07T11:31:58+00:00 — SCI-010 completed neural preflight; independent audit frozen

Public CPU run 37612623870/job 112763295080 completed successfully and released its runner. Archive 707be7d4c1df58bde0706d965e6678c0be6a2cfa retains all 90 files/14,645,751 bytes excluding manifest, all six runs and all 19 memorization snapshots. All 32 parent test methods and 87 subtests passed. Both small-set criteria passed under the frozen consecutive-check rule; selected DEV rates are 0.003 for both families. DEV correct counts are 36/105 (NeuroPixel) and 79/105 (Transformer). These are development scores, not final evidence. All full-TRAIN probes remain below 95%.

The separate JSON review passed 125 checks with zero issues and 25 text files rehashed. Full saved-array and archive verification now has a frozen operational plan (SHA-256 8272b44b2cfec01659121584e5c4a888ea520d693593d35ba366a12abe963061), wrapping the unchanged scientific auditor d16aeb308029df8ac80716efbe84f524e6aa7680b259746c9102bd82293ce949. No neural execution, TEST extraction or reselection belongs to the audit. Main admission is pending its actual success. Item 10 remains active; items 11–30 unopened; historical H1 unchanged.


## 2026-10-07T11:36:23+00:00 — SCI-010 independent preflight verified; primary study frozen

The single offline audit run 37614761119/job 112770312614 completed successfully and released its runner. Immutable archive 83a4f36f07ee05d8fb34fabe37b61b41fbdbe0f8 contains 19 files/10,025,025 bytes excluding manifest. Archive audit: 5,819 checks, zero failures/issues; complete preflight/acquisition files, actual Git ancestry, 432/403 source inventories, full ZIP comparison and TRAIN-only re-extraction verified. Fresh scientific audit: verified, zero issues, 105 input files, all six runs and all 19 memorization snapshots recounted. Root's 84 result-field comparisons agree exactly; no model inference or TEST extraction occurred in the audit.

Admit the same recipe with fresh paired seeds 60–64, two model families, 4,096 updates each, DEV-selected learning rate 0.003 each. Main plan SHA-256 571dd747cb9649bc435ffb006afc6558132b82e0d1c40e7b251827c51821b6f0, 26 implementation bindings. All ten checkpoints/configurations must be archived before consumed final access. Preserve failed attempts and all seeds; no optional tuning, dataset adaptation or outcome-dependent budget extension. The independent final analysis remains pending actual study completion. This is training from scratch on an externally authored synthetic task, not zero-shot or real-world transfer. Item 10 stays active; items 11–30 unopened; H1 unchanged.


## 2026-10-07T12:02:19+00:00 — SCI-010 primary execution completed; saved-artifact audit frozen

Run 37615277149/job 112771993909 completed successfully, all 32 parent methods/87 subtests passed, all ten main runs completed, and the single final evaluation completed. Source f619ac2150960ff658a797e3ac8594f49c8d96c1; frozen scientific plan 571dd747cb9649bc435ffb006afc6558132b82e0d1c40e7b251827c51821b6f0. Actual training archive G=e8f5cb4913574fd0e27ca52f986da9153bd6996d was confirmed at 11:59:30.212256 UTC before consumed final access. Final raw F=0e135ad10518e1ccd9b7adc562265aa4e2d94d2b retains 141 files/30,293,136 bytes excluding manifest. The standard public CPU runner is released.

Root rehashed ten fetched final artifacts, retained the complete Actions log, and recorded all execution/resource facts in study_completion_receipt.json. A read-only metadata lookup initially requested the final-manifest name from an interim snapshot; the actual interim manifest was recovered from its observed Git tree and the contract files rehashed. This was not a numerical failure. The observed final populations are 1,000 official rows, 886 novel relative to optimization TRAIN and 866 novel relative to the full declared TRAIN/fixture exposure ledger; no neural input overflow or unsupported gold is reported. These figures and every scientific metric remain pending independent recount.

Freeze operational audit plan 670c1999a60a4a0d213ca430a633613d43b10f4aa36db7192760ba578e5244bc: unchanged auditor/wrapper, full target F, acquisition, preflight and actual gate G inventories including their own final/interim manifests. Verify all Git/source ZIP identities, strict G-to-F ancestry and all ten checkpoint subtrees; independently reconstruct predictions, controls, masks and five-seed/cluster-bootstrap statistics without model inference. Item 10 remains active; items 11–30 unopened; H1 unchanged.


## 2026-10-07T12:25:26+00:00 — Item 10 closed after final independent verification

The feasible item-10 investigation is complete. Report: docs/research/10_results.md, SHA-256 ac14d3a4c5125e0e420f441f2c27ac373e10379a6ca90eaaf2a76ea1cbb8d5c8. Scientific source f619ac2150960ff658a797e3ac8594f49c8d96c1; main F=0e135ad10518e1ccd9b7adc562265aa4e2d94d2b; actual before-final G=e8f5cb4913574fd0e27ca52f986da9153bd6996d; audit O=bb2323ac0e0ba028975a4a4c89d2a97a20c7ff6d. Audit run 37618268225/job 112781813916 completed successfully and is released. All five item-10 jobs are complete; no active numerical child remains.

Final archive audit: 10,988 checks, zero issues. Scientific saved-array audit: 306 checked input files, 35 grouped checks, zero issues; no Torch or model execution. Actual Git G strictly precedes F and all ten training subtrees are identical. Root matched 2,166 numerical values without disagreement and independently reconstructed the paired t4 interval. Additional JSON review: 259 checks, zero issues, 21 text files rehashed. Final report tables and inferential limits were separately reviewed. Completion receipt, decoded provider log, numerical projection, independent review and preparation/exposure context are retained in results/research/10_validation/.

All-official mean accuracies are NeuroPixel 50.40% and Transformer 88.66%; NP minus TF is -38.26 percentage points, nominal paired t95% [-52.3857,-24.1343] over five fresh paired seeds. Every NeuroPixel seed also falls below the prespecified fact-frequency-excluding-query control (65.70% globally) in all three populations. Neither family passes an all-five competence screen; the NP screen was declared prospectively, the TF application is descriptive. All NP full-TRAIN probes remain below 95%, leaving optimization versus representation unresolved. This is training from scratch on externally authored synthetic bAbI, with exact input novelty only, not transfer, complete blinding, natural-language competence or external laboratory replication. Historical H1 remains closed/not supported.

Close item 10 and release its analysis/resource reservation. Items 11–30 remain unopened at this closure; item 11 (durable memory and interference) is next and requires its own scoped opening. Closure records completion of the feasible investigation, not achievement of its scientific capability. No main merge, paid/GPU compute, outreach, deployment or access/billing changes occurred.

One final editorial observation was resolved: the report distinguishes the audit-stage monitor minimum RAM (14.464985 GiB) from a later pre-archive worker boundary (14.460808 GiB). Original report, exact correction receipt and final independent review are preserved; results and admission are unchanged.


## 2026-10-07T12:26:57+00:00 — Item 11 opened after verified item-10 closure

Claim item 11 (durable memory and interference) on isolated source branch at parent 4411de991098ac592a42b408e266061786396395. Item 10 report/ledger and branch identity have been read back and verified. All previous runners are released. No contest-specific Kaggle work is affected.

Scope: distinguish parameter/expert storage, recurrent state within one call, activation carried between calls, and information reintroduced with current input. Read the actual model/phase3 memory paths and historical evidence, derive counterfactual requirements, then freeze a bounded diagnostic protocol before execution. Default-call reset, returned state, weights, RNG, hooks and deliberate optimizer updates must be distinguished. A finite-horizon activity trace or a saved parameter checkpoint is not automatically durable episodic memory. No item-10 retuning or historical-H1 replacement.

Reserve source/report preparation for root with independent architecture, evidence and primary-literature review subtasks, justified by avoiding conflation of memory mechanisms. Reservation renews within 90 minutes. No numerical runner is admitted yet; any execution requires its frozen scope, available RAM at least 8 GiB, aggregate threads at most 4, explicit deadlines and retained failures. Local executor remains unavailable. No JEV callable capability appears in the current tool inventory; continue the previously established conservative fallback and record decisions. Items 12–30 remain unopened.


## 2026-10-07T12:55:25+00:00 — SCI-011 memory-path diagnostic frozen

Historical and current-source review distinguishes default-call reset, native within-sequence state, explicit caller continuation/hooks and parameter/ground-buffer persistence. The historical memory table records 96.5% at eight blank frames; the README shorthand is corrected with provenance, without replacing historical results. No learned-memory result has been produced in item 11.

Freeze probe plan SHA-256 e93c42d0a6c62f4870f26d360dbdc9d877fe0d5196e5d207db86ce1b65db8bc5 with 13 source bindings, eight collected contract methods and all declared finite untrained counterfactuals, including actual zero/swap boundary tensors and returned-state ownership. Source and peer protocol review found no remaining material blocker. The prior probe/helper drafts are retained. Scientific interpretation requires a saved-array and actual Git/source-archive recount after execution.

Reserve one standard public ubuntu-24.04 CPU runner: numerical threads1/inter-op1/Git1, aggregate active CPU cap4, available RAM at least8 GiB. Worker600 seconds TOTAL includes180-second final archive reserve; contract120/probe120 caps share its execution remainder. Workflow20 minutes, first-step deadline1170 seconds; earlier limit wins. Supervision1 second, heartbeat60 and archive300 seconds. No optimizer, learned final scoring or later-item work is admitted. Local shell remains unavailable; source/report reservation renewed for90 minutes. Preserve failures and release only owned resources. No GPU, paid compute, main merge, outreach or access/billing change.


## 2026-10-07T13:11:49+00:00 — SCI-011 diagnostic completed; learned-memory preflight frozen

Diagnostic run37624600542/job112803038520 completed successfully and released its standard public CPU runner. Source bf8e4829ef609df3d7487bfe18ceaf6be24f949f; final raw ae82b5154c8d3d7da132c4b95ff05009be253437 retains36 files/10,315,839 bytes excluding manifest. Eight parent methods plus31 subtests and47 finite untrained checks passed. Worker14.016487243 seconds; six monitor samples had minimum available RAM14.475296020507812 GiB. Root rehashed eight fetched final texts and checked13 JSON/runtime/summary assertions. These are current evidence checks, not a learned memory score. Complete independent array/Git recount remains pending.

Freeze learned-memory preflight plan SHA-256 dbdd2e02da175522cff93323b32a89538b842423fe15eaef2c53d11a35aa25d5; recipe b3c397b106c9c40776f13db7c5f45f1455d821be27474aa845f772c8a5c7e82f; 23 source bindings. Four seed69 pilots, two families by two rates, each1024 updates on the known48-case six-symbol census with training delays0–2. Choose rate by mean DEV accuracy atd3/4, then CE, then lower rate. Main requires BOTH selected pilots >=46/48 on delay2, complete independent probe/preflight/source/operational audits and prospective timing admission. A completed negative preflight remains completed and blocks main; no rescue or outcome-driven budget extension. Primary future endpoint is normal-delay8, five fresh paired seeds70–74; only it receives a nominal t95/df4 interval. Cumulative state payload, local updates and parameter counts are separately labeled.

Root/peer review corrected two pre-execution defects: negative scientific admission was formerly raised as an operational error, and final gate formerly accepted an empty development mapping. New eight-method contracts test both and all three development outputs are now mandatory. Independent auditor mirrors completed/admission requirements. Original code candidates, correction receipts, source/metric/protocol reviews and internal model-service capacity interruptions are retained; no C training outcome preceded these corrections.

Reserve one standard public ubuntu-24.04 CPU preflight job, numericalthreads2/inter-op1/Git1, aggregate CPU cap4, available RAM>=8 GiB. Worker1800 seconds TOTAL including180 reserve; contracts180/pilots1440 within shared remainder; workflow40 minutes and first-step deadline2370. Supervision1 second, heartbeat60/archive300. Source/report preparation renews for90 minutes. Local executor remains unavailable. No main-study runner, GPU, paid compute, main merge, outreach or access/billing change is admitted. Item11 remains active and items12–30 unopened.
