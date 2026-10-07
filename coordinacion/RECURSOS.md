# Recursos y reservas

Actualizada: 2026-09-27 19:20 Europe/Madrid.

## Inventario local

| Recurso | Capacidad | Margen operativo acordado |
|---|---:|---|
| GPU | RTX 3090, 24.576 MiB | reservar 3,5 GiB para pantalla/sistema; no superar 83 C sostenidos |
| RAM | 23,7 GiB | conservar al menos 6 GiB disponibles; parar lanzamientos nuevos por debajo de 5 GiB |
| CPU | 20 hilos logicos | maximo 8 hilos por agente; total habitual <=16; bajar a 4 con GPU saturada |
| Disco de trabajo | D: y E: | datos, caches, entornos, modelos, temporales y resultados permanecen en D: o E:; evitar C: |

## Reservas activas

| Recurso | Tarea | Responsable | Inicio conocido | Limite | Liberacion esperada | Estado |
|---|---|---|---|---|---|---|
| GPU | SOIL-001 | Claude | antes de 19:04 | 6 GiB | al completar folds 4-5 | liberada 20:13 (6/6 folds; result.json) |
| GPU | FIL-001 | Claude | antes de 17:42 | 14 GiB | al completar 6000 it | activa, 3000/6000 registrados |
| GPU | FIL-002 | Claude | en cola gpuq desde 19:36 | 10 GiB, RAM 4 GiB | 3 corridas x ~2 h (learned, fsr, bilinear), secuenciales | pendiente: arranca cuando gpuq vea la GPU libre (tras FIL-001) |
| RAM/CPU | ambas | PREEXISTENTE | — | compartido | junto con GPU | RAM ya bajo el margen objetivo |

Las dos reservas actuales superponen 20 GiB nominales y dejan poco margen. Se toleran solo porque
ya estaban ejecutandose. A partir de su cierre: una sola carga GPU larga a la vez. Excepcion: una
prueba corta de menos de 10 minutos si la suma medida queda por debajo de 20 GiB y la GPU por debajo
de 80 C antes de empezar.

## Protocolo de reserva

Antes de ejecutar, añadir una fila con tarea, agente, VRAM/RAM/hilos, hora, duracion estimada y
condicion de corte. Renovar cada 90 minutos con un hito. Al acabar, marcar `liberada` y enlazar el
resultado. No finalizar procesos de otro agente sin autorizacion de Fran.

## Computo externo gratuito o promocional

| Plataforma | Oferta oficial observada | Uso recomendado | Limitacion clave |
|---|---|---|---|
| Kaggle Notebooks | P100 gratis; cuota semanal normalmente 30 h o variable | entrenamientos reproducibles ligados al concurso | cuota compartida y disponibilidad no garantizada |
| Google Colab gratuito | acceso sin coste a GPU/TPU | prototipos y replicas pequenas | recursos y limites fluctuan; sesiones pueden terminar |
| Lightning AI | creditos gratuitos para cuentas elegibles y Studio CPU gratuito | entrenamientos medianos persistentes | elegibilidad y creditos pueden cambiar |
| Hugging Face ZeroGPU | 5 min/dia para cuenta gratuita; hasta 2 Spaces ZeroGPU | demos e inferencias cortas | Gradio, cuotas cortas; no sirve para entrenamientos largos |
| Google Cloud Trial | 300 USD durante 90 dias para nuevos clientes | corrida puntual grande con presupuesto cerrado | GPU no es always-free; requiere cuenta de facturacion |

Fuentes oficiales verificadas el 2026-09-27:

- https://www.kaggle.com/docs/efficient-gpu-usage
- https://www.kaggle.com/docs/notebooks
- https://research.google.com/colaboratory/faq.html
- https://lightning.ai/docs/overview/ai-studio/
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://docs.cloud.google.com/free/docs/free-cloud-features

No crear cuentas, activar facturacion, subir datasets privados ni consumir creditos sin una tarea
aprobada y registro de coste/cuota. Preferencia: RTX 3090 local, luego Kaggle; nube adicional solo
cuando permita paralelizar una prueba independiente y reproducible.

Variables de cache y temporales de herramientas nuevas deben apuntar a una carpeta identificada en
D: o E:. Antes de una descarga pesada se registra origen, tamaño aproximado y destino. No se limpia
C: ni se mueve contenido existente sin autorizacion expresa de Fran.


## 2026-10-07 — SCI-006 isolated CPU preflight reservation

A standard GitHub-hosted ubuntu-24.04 runner in this public repository is reserved for one sequential preflight, at most 30 minutes. Two CPU threads, one inter-op thread and at least 8 GiB available RAM are required. No GPU job, paid runner, billing change, artifact storage upload or dependency cache is used. Work stays in the isolated Linux runner workspace; the Windows D:/E: storage rule concerns the disconnected Windows device. Standard public runner CPU time is free under the official documentation checked for this action. Source/environment validation precedes every affected scientific run. This preflight does not authorize the pending item-6 training freeze. The runner releases its resources automatically on completion; operational status is visible in this branch's Actions run.

Sources: https://docs.github.com/en/actions/reference/runners/github-hosted-runners and https://docs.github.com/en/billing/concepts/product-billing/github-actions .

## 2026-10-07T03:22:22Z — SCI-006 full sequential CPU reservation

The preceding implementation/preflight jobs have completed and released their runners. Reserve one standard public ubuntu-24.04 GitHub runner for this frozen study only. The observed admission runner has four logical CPUs and 14.313186645507812 GiB available RAM (2026-10-07T03:20:03.150279+00:00); every actual stage requires a fresh measurement. Scientific numerical work uses two threads and one inter-op thread; Git packing uses one thread, within the four-active-CPU ceiling. Preserve at least 8 GiB available RAM, with independent child supervision every second, heartbeats every 60 seconds and evidence archives every 300 seconds. No GPU is requested.

The worker budget is 18,000 seconds from its start, including source preservation and validation. Every new stage is admitted immediately before launch. Git waits are bounded by min(120 seconds, remaining budget); a stopped child may be followed by at most 180 seconds of final archival. The workflow timeout is 330 minutes to leave shutdown headroom. Only owned process sessions can be interrupted. Do not terminate or displace other tasks.

No paid compute, billing change, dependency cache or Actions artifact upload is used. Evidence is committed with normal fast-forward pushes to the dedicated results branch; the scientific execution HEAD stays fixed. The worker heartbeat and archive commits renew this reservation more frequently than the 90-minute coordination rule. The runner releases resources on completion or failure; root will record the outcome and release at closure. Earlier Windows reservations are neither interpreted as current nor modified. The local reconstructed checkout is used only for brief operational verification and report preparation, not a concurrent scientific training job.

## 2026-10-07T05:32:16.665091+00:00 — SCI-006 runner released

GitHub job 112615307349 / run 37566497890 completed successfully. Worker finished at 2026-10-07T05:16:05.490541Z after 6,736.55504302 seconds before final archival. All six stages returned 0. The 117 persisted worker resource samples have minimum available RAM 14.253311157226562 GiB; core trainer records include 14.249969482421875 GiB. These are sampled minima; continuous CPU utilization and peak RSS were not measured. The two-thread numerical/one-interop/four-active-CPU/8-GiB policies remained configured and no resource stop was recorded. The standard runner has released its resources; no GPU, paid runner or new Windows reservation was used. Historical unrelated reservations are unchanged.

## 2026-10-07T05:39:32.627738+00:00 — SCI-007 opened

Reserve the isolated Linux task workspace for SCI-007 source inspection and short deterministic audits, tests and statistical simulations, starting now, up to 90 minutes before renewal. At most four aggregate active CPU threads, one numerical thread per concurrent audit, and at least 8 GiB available RAM. Measure admission before numerical work; stop owned numerical work if the margin falls below 8 GiB. No GPU, paid compute, external dataset transfer or changes to the disconnected Windows device. No long scientific training is admitted by this reservation; earlier runner and Windows histories remain unchanged.

## 2026-10-07T05:49:00Z — SCI-007 bounded membership export reservation

Reserve one standard public ubuntu-24.04 runner for the item-7 contract tests and exact CPU Torch membership export, at most 35 minutes including dependency setup and evidence archival. No model training, final task-example sampling or model predictions are requested. Adapter tests use small training/validation sampler fixtures; the membership exporter samples no examples. Use pinned Torch 2.6.0+cpu and Python 3.12.8, one numerical thread and one inter-op thread; aggregate active CPU use remains below four. Every stage requires at least 8 GiB available RAM and is supervised once per second, with a 900-second stage limit and termination only of owned process groups. The source plan must be committed before execution. Preserve failures and export evidence on the existing isolated results branch; source/main are not modified by the worker. No paid runner, dependency cache or Actions artifact-storage upload is used. Standard public-runner terms were checked for the preceding item and remain the same service used here. Local work remains source inspection and small independent audits/simulations within its earlier reservation.

## 2026-10-07T06:22:34.038049+00:00 — SCI-007 resources released

GitHub job 112654156161 / run 37578991388 completed successfully. Both scientific stages returned 0; worker elapsed 4.227750006 seconds before archival. Seven worker samples had minimum 14.275299072265625 GiB available RAM; eleven exporter samples had minimum 14.275325775146484 GiB. The one local simulation finished in 0.6465705449991219 seconds with minimum 9.109718322753906 GiB across five samples; its independent recount and the split audit also completed. All owned numerical tasks and figure renders have finished. These are sampled available-memory minima, not continuous minima, process peak RSS or energy measurements. The standard runner released its resources; no GPU, paid service or historical Windows process was used or changed. This item-7 reservation is released before opening item 8.

## 2026-10-07T06:27:00.782435+00:00 — SCI-008 local reservation

Reserve short source audits, deterministic fixtures and regression development in the isolated Linux workspace for up to 90 minutes before renewal. At most four aggregate active CPU threads, one numerical thread per concurrent local audit, at least 8 GiB available RAM with admission measurement. No GPU, paid compute, competition dataset download or scientific training programme is admitted by this reservation. A separately declared standard public CPU validation job may be reserved after its concrete test recipe is reviewed. Historical Windows/competition processes remain untouched.


## 2026-10-07T06:48:26.339269+00:00 — SCI-008 bounded CPU integration reservation

Reserve one standard public ubuntu-24.04 runner for the reviewed item-8 regression recipe only, up to 35 minutes including setup and evidence archival. The source/plan must be frozen and committed before execution. Use Python 3.12.8, Torch 2.6.0+cpu and the existing pinned dependencies; no GPU, paid runner, cache, artifact-storage upload or competition-data download. One numerical and one inter-op thread are configured. The existing safety compatibility test explicitly requests two numerical threads, permitted only as a recorded bounded fixture with restoration between tests; aggregate active CPU remains at most four. Require at least 8 GiB available RAM at admission and sampled supervision. Each owned stage is limited to at most 900 seconds, further shortened if needed to reserve up to 180 seconds for archival before the 35-minute job deadline. No study training or final performance scoring is requested; tiny deterministic optimizer, sampler and gate-fit regression fixtures are included. Preserve ordinary failures from both declared stages, exact sources and runtime/test inventory in 08_cloud_runs on the isolated results branch. This is a short validation reservation after completed local source review, not an authorization to replay historical experiments with changed source. The current local reservation remains for lightweight receipt/source checks.


## 2026-10-07T07:17:56.375298+00:00 — SCI-008 closed

Release all SCI-008 local and standard public CPU reservations. GitHub run37584119610/job112670244618 completed successfully. The two stages returned0; worker elapsed11.713040590000006 seconds before archival, with15 saved resource samples and minimum available RAM14.328044891357422 GiB. All187 case boundaries start at1 numerical/1 inter-op thread; the declared two-intra-thread safety fixture restored1 afterward. Legacy import metadata precedes its CLI initialization, so no claim of universal1/1 before initialization is made. Local recount admission recorded9.12637710571289 GiB. These are sampled values, not continuous memory minima, measured CPU utilization, peak RSS or energy. No GPU, paid compute or unrelated process was used or changed.


## 2026-10-07T07:19:16.323499+00:00 — SCI-009 opened

Reserve the isolated Linux workspace for SCI-009 short source audits, deterministic task checks and bounded simulation development for90 minutes before renewal. One numerical thread per local audit, at most four aggregate active CPU threads, at least8 GiB available RAM checked before numerical work. No GPU, paid service or external dataset transfer. A separate standard public CPU reservation and frozen recipe are required before a training experiment. No unrelated or historical Windows process may be stopped.


## 2026-10-07T07:45:18.180685+00:00 — SCI-009 Stage-A reservation and implementation review

Reserve one standard public ubuntu-24.04 CPU runner for SCI-009 Stage A only: the reviewed contract suite, two bounded 32-row memorization diagnostics and four 1024-update development pilots. Workflow limit120 minutes; worker limit6600 seconds within the workflow remainder, with180 seconds reserved for final stopped archival. Two numerical threads, one inter-op thread, one Git pack thread and at most four aggregate active CPU threads; require at least8 GiB available RAM at every admission and sampled supervision. A separate supervisor checks owned children every second during Git/copy operations; heartbeats every60 seconds and partial archives every300 seconds renew the reservation. Contract tests have a900-second cap; scientific stages share the finite remaining budget. No final performance collection is generated in Stage A; registered grammar/driver fixtures are excluded from the performance population. No GPU, paid compute, cache, external private dataset or Actions artifact-storage upload is used. The Stage-B workflow file is an unlaunched implementation, not a Stage-B compute reservation. Its inventory must be admitted prospectively from observed Stage-A costs and reviewed evidence. Historical Windows and unrelated tasks remain untouched.


## 2026-10-07T08:06:41.893378+00:00 — SCI-009 Stage A completed; archive audit in progress

The standard public CPU preflight job112688648396/run37589908205 completed successfully. Its scientific source is e63764ccb924ab23d65c2deb94b477387d102c82 and final stopped archive is8311c052796aeef5a52484b597cf5d9675ca315e. All34 collected contract tests passed without skips, failures or errors; the JUnit suite counter137 includes nested subtest accounting and is not137 collected methods. Both memorization diagnostics and all four development pilots completed. Their scientific stage ended at2026-10-07T07:57:15.754888Z; the worker finished at07:57:18.234401Z after327.752899868 seconds before final archival. Release the Stage-A runner reservation. The local SCI-009 audit reservation continues; no Stage-B training reservation or final-data access has yet been opened.

Root's current archive audit verifies861 checks with zero issues, including all71 raw files plus their manifest,312 source files,49 bound files, exact tests/runtime and source guards. It strengthens two earlier preserved audit versions; repeated file checks are not independent experiments. The saved worker monitors contain5 test-stage and312 preflight-stage samples; their minimum available RAM is14.200397491455078 GiB. This is sampled availability, not energy, peak RSS or continuous CPU utilization. Independent scientific data/prediction recount remains pending before Stage-B admission.


## 2026-10-07T08:22:31.852462+00:00 — SCI-009 Stage-B reservation and local renewal

Renew the isolated local SCI-009 reservation for 90 minutes for source, provenance and independent analysis only, one numerical thread and at least 8 GiB available RAM. Reserve one standard public ubuntu-24.04 runner for the unchanged Stage-B contract suite, ten primary runs (five paired training realizations per family, 4,096 updates each) and the fixed six-condition final panel. Workflow limit 240 minutes; worker limit 13,800 seconds including a 180-second stopped archival reserve, additionally bounded by the workflow deadline. Two numerical threads, one inter-op thread and one Git packing thread, at most four aggregate active CPU threads, with independent one-second supervision and at least 8 GiB available RAM. Heartbeats every 60 seconds and partial archival every 300 seconds renew the cloud reservation. Every stage checks admission immediately before launch.

All Stage-A scientific and archive audits have completed without discrepancies. The prospectively calculated operational allowance is 6,148.67933326 seconds, including margins and archival reserve; it is not a confidence bound or measured future cost. Freeze the complete reviewed Stage-B plan and independent analysis before publishing the triggering source commit. All ten completed checkpoint records must be archived and remotely confirmed before final access. Use the existing pinned free public CPU runner only; no GPU, paid compute, billing change, cache or Actions artifact-storage upload. Preserve failed attempts and release only owned resources. Historical Windows and unrelated jobs remain untouched.


## 2026-10-07T09:50:38+00:00 — SCI-009 Stage-B resources released; offline audit reservation

Release the completed Stage-B standard public CPU runner reservation: run 37593731891 / job 112701154832 ended successfully, worker completion 2026-10-07T09:17:53.632580Z and elapsed 3,061.636433142 seconds before final archival. The final scientific/runtime/resource recount remains pending; no new minimum-memory or energy claim is inferred here.

Reserve one standard public ubuntu-24.04 runner for the unchanged offline audits and reviewed figure helper only, at most 30 workflow minutes. The worker allows at most 1,380 seconds, additionally bounded by the workflow deadline, including a 180-second stopped archival reserve. Configure one numerical thread and one Git packing thread, at most four aggregate active CPU threads, and at least 8 GiB available RAM checked before stages and once per second by the supervisor. Heartbeats are every 60 seconds; partial archives are attempted every 300 seconds and after each stage. Stage caps are 600 seconds for full archive audit, 600 for gate audit, 900 for scientific recount and 300 for figures, each further bounded by the shared remaining allowance. Bootstrap caps are 1/3/4/4 minutes for deadline/checkout/Python/dependencies. Preserve failures through complete Actions logs/provider status and the always-step receipt where possible; worker archives are not promised before worker startup.

Use the frozen offline Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0, with Matplotlib 3.10.8 and psutil 7.2.2. No GPU, paid runner, cache, artifact-storage upload or account/billing change. Results use ordinary fast-forward pushes to the dedicated results branch. Renew the local source/report reservation for 90 minutes for read-only provenance, text work and any brief numerical checks only if the local executor returns and the same one-thread/8-GiB admission policy passes. No local scientific process was launched during the executor interruption; do not stop unrelated or historical Windows processes.


## 2026-10-07T10:09:16+00:00 — SCI-009 resources released

Release all owned SCI-009 local and standard public CPU reservations after report review. Scientific run37593731891/job112701154832 completed successfully; worker duration before final archival was3,061.636433142 seconds. Its3,044 saved supervision samples have minimum available RAM14.214973449707031 GiB. Offline run37603398040/job112732965039 completed successfully; its worker duration before final archival was27.802332284000016 seconds, with15 supervision samples and minimum14.436992645263672 GiB. The scientific auditor additionally recorded86 samples with minimum14.435539245605469 GiB. No supervision failure or remaining owned child is recorded. These are sampled available-memory values, not continuous minima, peak RSS, power or energy. No GPU, paid compute or unrelated/historical Windows process was used or changed. The unavailable local executor is not assumed recovered.


## 2026-10-07T10:11:14+00:00 — SCI-010 source and bounded-analysis reservation

Reserve one-thread source/provenance work and brief deterministic checks for90 minutes before renewal, aggregate active CPU at most4 and available RAM at least8 GiB for numerical execution. The local shell executor remains unavailable; do not assume it has recovered or launch repeated blind probes. No scientific job is active. A separately frozen bounded standard public CPU continuation can be reserved when its concrete protocol is reviewed. No GPU, paid service, new billing/access changes, outreach or changes to historical Windows/unrelated processes. Public benchmark metadata may be researched; any actual dataset and scoring plan must be identified and preserved prospectively.


## 2026-10-07T10:44:50+00:00 — SCI-010 bounded acquisition reservation

Reserve one standard public ubuntu-24.04 CPU job, Python3.12.14, psutil7.2.2 and pytest9.1.1. Numerical thread environment1, Git1, aggregate active CPU at most4, available RAM at least8 GiB. Worker1200 seconds TOTAL including180 final-archive reserve; workflow25 minutes; tests300 and acquisition720 within the shared deadline. No GPU, paid service, billing/access change, outreach or unrelated-process control. Local shell remains unavailable. No neural training or TEST payload parsing/scoring belongs to this reservation.


## 2026-10-07T11:01:48+00:00 — SCI-010 acquisition completed and independently recounted

Public CPU run37609533224/job112753126084 completed successfully; final raw commit19c8c40ab50141ba128623e7bbbc04cfb2460f31 contains34 files/23,778,454 bytes excluding its manifest. All24 contract methods and58 subtests passed. Root independently rehashed17 fetched text/JSON files and completed3,176 raw-TRAIN/parser/grouping/encoding/control checks with0 issues. Original-author endpoint404 is retained; the documented HTTPS mirror supplies the exact TRAIN member. No original-host byte comparison or external custody is claimed. TRAIN1000 records/919 unique inputs/752 connected groups; split895/105; geometry4x8/vocabulary18. No TEST parsing or neural execution occurred. The acquisition runner is released; local executor remains unavailable. Item10 continues with a prospective six-run neural preflight and exact fixture/source/runtime review; items11–30 remain pending. H1 is unchanged.


## 2026-10-07T11:12:59+00:00 — SCI-010 bounded neural CPU reservation

Reserve one standard public ubuntu-24.04 CPU job, Python3.12.14/Torch2.6.0+cpu/NumPy2.2.6/SciPy1.15.1, numericalthreads2/inter-op1/Git1, aggregate activeCPU at most4, availableRAM at least8GiB. Worker2400seconds TOTAL including180 final-archive reserve; sharedexecution2220s, contracts300s and neuralpreflight1800s within it. Workflow50minutes and2970s fromfirststep; earlierdeadline alwayswins. RAMsupervision1s/archive300s. No GPU, paid compute, billing/access change, main merge, outreach or unrelated-process control. Acquisition runner released; local executor unavailable. Preserve failures before stopping any blocked phase.


## 2026-10-07T11:31:58+00:00 — SCI-010 completed neural preflight; independent audit frozen

Public CPU run 37612623870/job 112763295080 completed successfully and released its runner. Archive 707be7d4c1df58bde0706d965e6678c0be6a2cfa retains all 90 files/14,645,751 bytes excluding manifest, all six runs and all 19 memorization snapshots. All 32 parent test methods and 87 subtests passed. Both small-set criteria passed under the frozen consecutive-check rule; selected DEV rates are 0.003 for both families. DEV correct counts are 36/105 (NeuroPixel) and 79/105 (Transformer). These are development scores, not final evidence. All full-TRAIN probes remain below 95%.

The separate JSON review passed 125 checks with zero issues and 25 text files rehashed. Full saved-array and archive verification now has a frozen operational plan (SHA-256 8272b44b2cfec01659121584e5c4a888ea520d693593d35ba366a12abe963061), wrapping the unchanged scientific auditor d16aeb308029df8ac80716efbe84f524e6aa7680b259746c9102bd82293ce949. No neural execution, TEST extraction or reselection belongs to the audit. Main admission is pending its actual success. Item 10 remains active; items 11–30 unopened; historical H1 unchanged.


## 2026-10-07T11:31:58+00:00 — SCI-010 bounded offline audit reservation

The completed neural runner is released. Reserve one standard public ubuntu-24.04 CPU audit job, Python 3.12.14/NumPy 2.3.5/SciPy 1.17.0/psutil 7.2.2, numerical threads 1/Git 1 and aggregate CPU cap 4, available RAM at least 8 GiB. Worker 1,500 seconds TOTAL including 180 seconds final-archive reserve; audit stage at most 1,200 seconds; workflow 30 minutes and first-step deadline 1,770 seconds. Parent RAM supervision every second and archive at 300-second intervals. Earlier shared deadlines always win. Renew the read-only/source coordination reservation for 90 minutes from this timestamp; no local numerical process is assumed available. No GPU, paid compute, access/billing change, main merge, outreach or unrelated-process control.


## 2026-10-07T11:36:23+00:00 — SCI-010 independent preflight verified; primary study frozen

The single offline audit run 37614761119/job 112770312614 completed successfully and released its runner. Immutable archive 83a4f36f07ee05d8fb34fabe37b61b41fbdbe0f8 contains 19 files/10,025,025 bytes excluding manifest. Archive audit: 5,819 checks, zero failures/issues; complete preflight/acquisition files, actual Git ancestry, 432/403 source inventories, full ZIP comparison and TRAIN-only re-extraction verified. Fresh scientific audit: verified, zero issues, 105 input files, all six runs and all 19 memorization snapshots recounted. Root's 84 result-field comparisons agree exactly; no model inference or TEST extraction occurred in the audit.

Admit the same recipe with fresh paired seeds 60–64, two model families, 4,096 updates each, DEV-selected learning rate 0.003 each. Main plan SHA-256 571dd747cb9649bc435ffb006afc6558132b82e0d1c40e7b251827c51821b6f0, 26 implementation bindings. All ten checkpoints/configurations must be archived before consumed final access. Preserve failed attempts and all seeds; no optional tuning, dataset adaptation or outcome-dependent budget extension. The independent final analysis remains pending actual study completion. This is training from scratch on an externally authored synthetic task, not zero-shot or real-world transfer. Item 10 stays active; items 11–30 unopened; H1 unchanged.

Reserve one standard public ubuntu-24.04 CPU study job, numerical threads 2/inter-op 1/Git 1, aggregate CPU cap 4, available RAM at least 8 GiB. Worker 3,600 seconds TOTAL including 180-second final-archive reserve; contracts 300/train 2,700/final 300-second caps within shared execution. Workflow 70 minutes and first-step deadline 4,170 seconds; earlier deadline wins. Train admission projection 2,640.22012661 seconds includes twofold worst-pilot timing plus 60 seconds. Supervision 1 second, archive 300 seconds. Local executor remains unavailable. No GPU, paid compute, access/billing change, main merge, outreach or unrelated-process control.


## 2026-10-07T12:02:19+00:00 — SCI-010 primary execution completed; saved-artifact audit frozen

Run 37615277149/job 112771993909 completed successfully, all 32 parent methods/87 subtests passed, all ten main runs completed, and the single final evaluation completed. Source f619ac2150960ff658a797e3ac8594f49c8d96c1; frozen scientific plan 571dd747cb9649bc435ffb006afc6558132b82e0d1c40e7b251827c51821b6f0. Actual training archive G=e8f5cb4913574fd0e27ca52f986da9153bd6996d was confirmed at 11:59:30.212256 UTC before consumed final access. Final raw F=0e135ad10518e1ccd9b7adc562265aa4e2d94d2b retains 141 files/30,293,136 bytes excluding manifest. The standard public CPU runner is released.

Root rehashed ten fetched final artifacts, retained the complete Actions log, and recorded all execution/resource facts in study_completion_receipt.json. A read-only metadata lookup initially requested the final-manifest name from an interim snapshot; the actual interim manifest was recovered from its observed Git tree and the contract files rehashed. This was not a numerical failure. The observed final populations are 1,000 official rows, 886 novel relative to optimization TRAIN and 866 novel relative to the full declared TRAIN/fixture exposure ledger; no neural input overflow or unsupported gold is reported. These figures and every scientific metric remain pending independent recount.

Freeze operational audit plan 670c1999a60a4a0d213ca430a633613d43b10f4aa36db7192760ba578e5244bc: unchanged auditor/wrapper, full target F, acquisition, preflight and actual gate G inventories including their own final/interim manifests. Verify all Git/source ZIP identities, strict G-to-F ancestry and all ten checkpoint subtrees; independently reconstruct predictions, controls, masks and five-seed/cluster-bootstrap statistics without model inference. Item 10 remains active; items 11–30 unopened; H1 unchanged.

Reserve one standard public CPU audit job, Python 3.12.14/NumPy 2.3.5/SciPy 1.17.0/psutil 7.2.2, numerical threads 1/Git 1, aggregate cap 4 and available RAM at least 8 GiB. Worker 1,500 seconds TOTAL including 180-second archive reserve; stage 1,200 seconds; workflow 30 minutes/first-step deadline 1,770 seconds. Supervision 1 second and archival interval 300 seconds. Main runner released; local executor remains unavailable. No neural execution, GPU, paid compute, main merge, outreach or access/billing changes.


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


## 2026-10-07T13:18:16+00:00 — SCI-011 learned preflight completed negatively; independent audit frozen

Run37626660573/job112810057008 completed successfully and released its standard public CPU runner. Scientific source cef4e67358ca5c344cd4154a7f641feefeb406ae; final raw1b1759775adba1ab0d289ba34b6b306f9ad547c2 contains72 files/12,153,630 bytes excluding manifest. All eight contract methods passed, with no skips, failures or subtests. All four1024-update pilots completed; worker92.8944738 seconds,85 monitor samples, minimum sampled available RAM14.306175231933594 GiB. Root rehashed nine top-level texts plus all four run JSON records and verified34 summary/count/selection/resource assertions. Full array recount remains pending.

The declared selection chooses rate0.003 in both families. Near-delay2 counts are NeuroPixel38/48 and GRU48/48; the other NP rate has40/48. The required minimum is46/48 for BOTH selected models, so admission is FALSE and no main training, delay8 primary comparison or learned interference/final panel is admitted. All metrics, high finite NP development losses and both rates are preserved. This is a completed negative scientific preflight; there is no outcome-driven tuning, budget extension or substitution of the pilot for the planned five pairs. Historical H1 remains unchanged.

Freeze the operational saved-artifact audit plan SHA-256 2d34eb9492f9a8eccef24e575bf1cca4e10fc695b02de2b09712c807e9516742 with complete preflight and probe inputs, including their own manifests. Unchanged independent NumPy auditors will reconstruct the literal fixtures, sampling streams, labels, logits/NLL, scores, selection and intervention arrays; the wrapper also verifies every Git/ZIP/file, completed worker/source/runtime/test/monitor contract and input stability. No neural execution belongs to this audit.

Reserve one standard public CPU audit runner: Python3.12.14/NumPy2.3.5/SciPy1.17.0/psutil7.2.2, numericalthread1/Git1, aggregate CPU cap4, available RAM>=8 GiB; worker1500 seconds TOTAL including180 archival reserve; auditstage1200; workflow30 minutes/first-step1770. Supervision1 second, heartbeat60/archive300. Renew source/report preparation90 minutes. All numerical training runners are released and local executor remains unavailable. Item11 stays active until audit/report closure; items12–30 remain unopened. No GPU, paid compute, main merge, outreach or access/billing change.


## 2026-10-07T13:32:18+00:00 — SCI-011 closed after independent audit and report review

The source/API and historical-memory audit, finite untrained mechanism diagnostic, four frozen learned pilots and independent saved-artifact audit are complete. Report docs/research/11_results.md SHA-256 ff44578bbcb0faf3762718eca2d5bc8cfbb3ccf05a773891444392b6115889f5 records all outcomes and distinctions. Explicit state continuation is implemented and tested; automatic independent-call activity persistence is absent in the tested default path. Finite untrained history dependence is not learned retrieval or stability.

Independent audit run37627489876/job112812914654 completed and released its runner. Source f50ef0e55397ce4fb8e85360f5be8c12003bd6ee; final audit archive 0a8403b1001881acc8626369a1e4f9dde83cc8e2:24files/10,623,568bytes excluding manifest; worker10.522987138seconds. Archive/source/operational6409checks, Barray504checks and learned-preflight13517checks all passed,0issues. Root rehashed six final audit texts and compared924 producer/auditor metric/cost numbers,0issues (largest absolute difference1.2434497875801753e-13). Architecture reviewer independently confirmed12 population counts/subgroup sums/selection; scientific reviewer read the full report and corrected state-tensor-size language to avoid calling it measured information capacity.

Selected near-delay2 scores remain NeuroPixel38/48 and GRU48/48, with46/48 required for both. The alternate NP rate40/48 also fails. Main training, the delay8 primary comparison and learned interference/final panel were not admitted or executed; no outcome-driven rescue, interval or causal forgetting claim is made. All four pilots completed operationally and their high finite CE is retained. Historical H1 remains closed and not supported.

Close item11 as a feasible investigation, not as a passed aspirational capability. All owned item11 numerical runners and reservations are released; local executor remains unavailable. Items12–30 remain unopened at this closure. Item12 can now be opened in a separate entry. No GPU, paid compute, main merge, outreach or billing/access changes occurred.


## 2026-10-07T13:33:44+00:00 — SCI-012 opened: continual learning and retention evidence

Item11 was closed in 3ad305390bf9f9d07d18120f5d641ceb38edd28e. Open only item12. Audit current continual/expert/routing paths, historic matrices and already audited item6 growth outcomes; distinguish acquired competence, frozen-parameter invariance, task-ID/oracle routing and end-to-end retention. Consult primary continual-learning definitions and controlled baselines. No new training, scientific scoring or execution plan is frozen at opening. Preserve failed prerequisites and formulate any bounded check before execution. Do not open item13 or later until closure.

Reserve source/review work90 minutes under the established conservative routing fallback (no JEV callable capability found). No numerical runner is active or reserved at opening. Later public CPU work, if scientifically justified, requires an explicit finite budget and RAM>=8GiB/aggregateCPU<=4, including archival reserve and monitoring. Local executor remains unavailable. No GPU, paid compute, main merge, outreach or access/billing change.


## 2026-10-07T14:04:24+00:00 — SCI-012 frozen bounded evidence execution

Freeze docs/research/12_probe_plan.json SHA-256 c5ee67317b336d786f44f66572d274e7e95099a18ed87f6e486133512a302ffa after architecture, evidence and primary-literature static reviews. Only item12 is open. Eight contract methods precede a retrospective recount of the15 selected item6 artifacts (seeds20/21, three topics), a native deepcopy isolation diagnostic on8 copies of the same12-example fixture, a literal routing counterexample and two independent saved-array audits in fresh serial interpreters. The27 diagnostic SGD updates (plus2 separate unit-control steps) establish actual mutation for isolation, not8 learned tasks. Existing low binding acquisition is retained; no new six-task preflight or learned continual benchmark is admitted. No FWT baseline, new interval or external replication is asserted.

Reserve one standard public ubuntu-24.04 CPU runner, maximum20 minutes with earlier1170-second workflow deadline. Worker900 seconds includes180 seconds for archival; tests<=120 seconds, shared probe<=540 seconds including input recovery, each scientific child<=120 seconds. Numerical/inter-op/Git threads1, aggregate activeCPU<=4, availableRAM>=8GiB; owned-process supervision every1 second, heartbeats60 seconds and archive snapshots300 seconds. Preserve all failures, logs, original inputs, source ZIP and result manifests. Source/review reservation remains within its current90-minute interval; no other numerical work is active. Source guard and runtime/package checks are mandatory before execution. Local executor remains unavailable. All resulting changes stay on the two isolated research branches.


## 2026-10-07T14:20:46+00:00 — SCI-012 closed after continual-evidence execution and reviewed report

Complete the feasible investigation of item12. Report docs/research/12_results.md SHA-256 8d8e53792bcd5ffc9a5bbebe5cdf88e471b266c540f98382250be20a691598f2 preserves historical and new results with distinct source identities. One frozen public CPU job37633696345/job112834249348 succeeded and released its runner. Source aada1037271a621ac651c94c68feaba51143be05; final archive5381f6775ae4be64210b99b14360fe06faea28ec,51files/19,417,647bytes excluding manifest. Eight contract parents plus7 subtest calls passed,0failures/skips. Native producer10checks and independent matrix6330/isolation5013checks passed,0issues. Root rehashed all27 nonbinary result files and checked3508 statements including1716 numeric comparisons,0issues/maxdifference0. Peers added88 numerical and82 documentary checks without discrepancies; scientific review clarified one sentence to say prior weak acquisition, not a newly failed preflight.

Secondary matrices recount the old item6 source08d0d52edd05da6835e71479f3ba4399fcbeabae without model replay: seeds20/21 globalBWT +13.9160/+4.9316pp but bindingBWT -24.9023/-26.1719pp, with oldA/B binding0/512 and initial diagonals only19.7–29.1%. Improvements in ACCION/LUGAR explain the global aggregate arithmetically; no neural cause or external replication is established. New source verifies native deepcopy isolation through8 copies on one12-example fixture, with27SGDsteps plus2 separate contract steps and detecting alias/RNG controls. Eight copies use958,008 named-tensor bytes, not total process RAM or8learnedtasks. Literal routing outputs remain labelled constructed.

Worker23.008122629seconds;11 supervisor samples minimum availableRAM14.272777557373047GiB. All phases stayed within the frozen resource budget; no current item12 numerical resource remains. The three-topic retrospective analysis and isolated-copy diagnostic are completed; robust larger-scope continual learning, reliable acquisition and end-to-end routing remain unestablished. No new larger learned benchmark was admitted or executed. HistoricalH1 stays closed/not supported. Release item12 source/review reservation. Items13–30 remain unopened at this closure; item13 can now be opened separately.


## 2026-10-07T14:21:44+00:00 — SCI-013 opened: new tokens and distractors

Item12 closed in a1436a203f90a61428d2beb4277d2bddcb6dfc7d. Open only item13. Audit the model's vocabulary expansion and new-token/distractor claims; distinguish allocating new embedding rows from acquiring meanings, old-logit preservation from probability normalization after enlarging the answer space, and robustness from weak initial competence. Inspect old parameter, buffer, optimizer and configuration preservation; derive necessary contracts and controlled distractor cases before any new execution. Preserve prior evidence and failures. HistoricalH1 remains closed/not supported. Do not open14 or later before this point's closure.

Reserve source/review work90 minutes with the established conservative routing fallback. No numerical runner is active or reserved at opening. Later authorized standard public CPU work must be prospectively bounded, maintain availableRAM>=8GiB and aggregate activeCPU<=4, and include supervision, failure preservation and archival time. Local executor remains unavailable. No GPU, paid runner, main merge, outreach, billing or access changes.


## 2026-10-07T14:55:04+00:00 — SCI-013 frozen: bounded CPU vocabulary/distractor probe

Only item13 remains active. Freeze twelve contract methods, three serial components (saved item9 secondary recount, constructed native vocabulary/label witnesses, independent NumPy recount audit), and 29 implementation/source bindings. Preserve legacy phase3/battery functions and all pre-execution corrections. Archive input F=b64d0e2ba222df76caf72bc9870c7602873e7236 has12 selected files totaling23,928,770 bytes. No new checkpoint load, learning study, confidence interval or hypothesis revision. H1 remains closed/not supported; items14–30 unopened.

Reserve one explicitly authorized standard public CPU workflow, at most20 minutes. Worker900s includes180s archive reserve; contracts120s; shared probe540s including read-only recovery and three fresh serial child processes of at most120s. Numerical/interop/Git threads1; aggregate activeCPU<=4; availableRAM>=8GiB. Owned process-group supervision samples each1s, heartbeat60s, archive300s. Preserve partial/failure evidence and provider bootstrap logs. Public-repository guard and isolated source/results branches remain. No paid runner, GPU, billing/access changes, main merge or outreach. Root integrates; architecture/source, evidence and scientific reviews are justified subwork under the documented conservative routing fallback. Release numerical reservation after final archive/status confirmation.


## 2026-10-07T15:17:10+00:00 — SCI-013 closed: verified fixes and limited scientific claims

Closed only item13 after source S=9618ba8aae3b83d95a2813f99dd72c9d2e1d148a, successful public CPU run 37640756510/job112858763837, and final raw archive F=e16d0ab07a5aff93c66c4057a6509cc31b744160. Report docs/research/13_results.md SHA256 7459bdec8ddc3bc1853b4f28f1d7b2508f0d24dd3d7a595db5526ddd0616272a. Twelve parent regression methods and twenty additional subcalls passed; independent saved-array auditor32954 checks, root15870 checks and final peer reviews found no discrepancy. Preserved legacy failures and every pre-execution correction. README now states historical new-token conditions and limitations.

Expansion allocation/grounding/dtype and battery label faults are corrected on the isolated source branch. Constructed witnesses distinguish vocabulary capacity, softmax competition, projected-split overlap and semantic learning. Historical positive-only few-shot summaries and low-base-competence distractor retention do not establish semantic acquisition or robust high-competence binding. No historical H1 revision, external replication or Nobel-level result.

All numerical stages completed and zero active Actions runs were verified. Release numerical and source/review reservations for SCI-013. Items14–30 remain unopened in this closure; item14 may open in a separate subsequent commit. Existing RAM>=8GiB, aggregateCPU<=4, bounded public CPU-only, no main merge/paid/GPU/outreach/billing/access-change constraints remain.


## 2026-10-07T15:18:42+00:00 — SCI-014 opened after SCI-013 closure

Item13 closure dd9930694e5679e0a514525840a83b4c760f2730 was published and report/progress/README hashes read back. Only item14 is now active: assisted state repair versus stored memory. Inspect input reinjection, damage hooks, state continuation and historical evidence before freezing any numerical comparison. Distinguish competence, survival and actual recovery. Preserve source-version and old-result limits; H1 remains closed/not supported. Items15–30 remain unopened.

Reserve source/review work90 minutes. No numerical runner active or reserved at opening. Any later public CPU probe must have finite deadlines, RAM>=8GiB, aggregateCPU<=4, process supervision and archival time. Existing isolated-branch and no paid/GPU/main-merge/outreach/access/billing-change constraints remain. Source, evidence and primary-literature reviewers are justified parallel subtasks within item14; root integrates under the established JEV-unavailable conservative fallback.


## 2026-10-07T15:34:14+00:00 — SCI-014 frozen: constructed repair versus input census

Freeze only item14 with5 parent contract methods,216 constructed trajectories,1944 repeated endpoints,324 summary rows and88 saved arrays. Native equations distinguish input copying, passive retention and categorical reconstruction from surviving spatial redundancy. Capture pre/post damage, all8 later updates, native-held forward witnesses and before/after weights. Independent NumPy reconstruction and bounded V8 summary recount planned. No trained checkpoint, learning study, new confidence interval or H1 revision; items15–30 remain unopened.

Reserve one standard public CPU workflow, max20 minutes. Worker900s includes180s archival reserve; tests120s; shared probe540s; two serial fresh components max120s each. Numerical/interop/Git threads1, aggregate activeCPU<=4, availableRAM>=8GiB, owned-process supervision1s, heartbeat60s, archive300s. Source/review checks complete, three reviewers found no blocker after a preserved count-unit correction. Preserve raw results before interpreting expected controls and retain completed-case partial evidence on failure. No paid/GPU/main-merge/outreach/access/billing changes. Release after final archive and provider completion are verified.


## 2026-10-07T15:42:51+00:00 — SCI-014 closed: source assistance and state recovery distinguished

Closed item14 after successful CPU run37644986366/job112873311188, sourceS=844f4dfae1cb8909d4f87c0c8cfd96b965590adf, final raw archiveF=10c694cd299a4114bd447528d22b204d89b00b9e. Report docs/research/14_results.md SHA2563b9681492f10829cda3d2e6f4b0211b987d1ee85d8a35ffe94a0ca4ab3d5be6e. Five parent methods+15subcases passed; native8checks, independent6176checks, root10086checks and all final peer reviews found no unresolved discrepancy. Full88arrays/216constructedtrajectories/1944repeatedendpoints/324rows archived. Exact held-native equivalence and before/after weights preserved.

The controls separate source-dependent rebuilding, passive retention and categorical repair from surviving spatial redundancy. No learned autonomous repair or historical mechanism attribution is established. Historical99.35% remains an input-held endpoint without immediate lesion measurement; README conditions corrected. H1 remains closed/not supported. Source ZIP, logs, histories, pre-execution corrections and independent reviews are retained.

Provider completion and zero active Actions runs verified. Release all SCI-014 numerical/source-review reservations. Lowest retained availableRAM14.09469985961914GiB; numerical/interop/Git1, aggregateceiling4. Items15–30 remain unopened in this closure; item15 may open only in a separate next commit. No paid/GPU/main-merge/outreach/access/billing changes.


## 2026-10-07T15:43:57+00:00 — SCI-015 opened after SCI-014 closure

Item14 closured7c2e07473856ccd38591b2389babb235e8e29b5 report/README/progress hashes verified; numerical resources released. Only item15 now active: dynamical stability. Distinguish finite-horizon answer accuracy, bounded states, perturbation sensitivity, equilibria and attraction. Inspect actual update constraints and historical evidence before any new frozen numerical plan. Finite-horizon constructed witnesses are not a proof for all learned weights or all times. HistoricalH1 stays closed/not supported; items16–30 unopened.

Reserve source/review work90 minutes with conservative root integration and independently justified architecture/evidence/primary-source tasks. No active numerical runner. Any later standard public CPU diagnostic requires finite time/state-horizon limits, RAM>=8GiB, aggregateCPU<=4, process supervision and archival reserve. No paid/GPU/main-merge/outreach/access/billing changes. The completed item14 independent operational receipt is appended as provenance without changing its scientific result.


## 2026-10-07T16:07:56+00:00 — SCI-015 seven-case census frozen

Only item15 active. Freeze the complete native constructed dynamics census before execution:7cases,14paired trajectories,256continuation updates after an explicit post-update1 injection,3,598aligned states,214raw arrays,154summary rows. Original model core unchanged. Native forward FD and autograd Jacobians, weights/RNG witnesses and full-array NumPy reconstruction; one existing hook/snapshot contract. Source and report-schema reviews found no remaining blocker. A missing promised autograd check was corrected before freeze and its exact preimage retained. Metrics and NLL terminology clarified before outcomes. Plan SHA256 a39782409d87a172ca516a461e4c31a92ebd2ed2828477c02dfb9e38b060460d.

Reserve one standard public CPU run: numerical1/interop1/Git1, aggregateCPUceiling4, availableRAM>=8GiB, worker900s including180s archive, tests120s, diagnostic540s with two120s components, workflow20min and bootstrapdeadline1170s. Native absolute-state stop1e12 and256-step horizon are execution limits, not stability guarantees. Owned-process-group supervision/RAM samples every1s, heartbeat60s, interim archives300s. Pre-execution provider query showed no active runs. Retain every failure, sourceZIP, raw arrays, logs, manifests and independently reviewed counters.

This census contains no training and loads no trained checkpoint. Separate read-only feasibility work within item15 has located the5archived item9 NP weights and saved validation inputs/logits; any retrospective learned-state inference requires a different frozen plan after this census is verified. Low earlier competence and H1not-supported remain unchanged. Items16–30 unopened. No paid/GPU/main-merge/external-contact/nomination/access/billing action.


## 2026-10-07T16:23:07+00:00 — SCI-015 constructed census verified; learned phase source preparation

Census sourceff66dc4bacd19219bf4f754e40944396e7ce8c5b, run37649654847/job112889349696 completed successfully; finalarchiveff34cc4f44c535e5f4d00276e1e3be194ec70f21. Onecontract/12nativechecks/3,532independentNPchecks pass,214arrays/14trajectories/3,598aligned states/154rows. FrozenrootJS1,667checks(826numeric) and1,668producer/auditorJSONcomparisons have zero issues.19textchecksums/Gittreemodesandsizes verified.21archivefiles9,839,284B plusmanifest retained. Minimum15retainedRAMobservations14.32938003540039GiB;7supervisorsamples. Providercompleted and zero active runsverified. Release ONLY this numerical runner; item15 source/review reservation remains active.

Census confirms distinctions between output, state bounds, attraction, cycles and transient amplification for deliberately assigned native weights; no learned-model/global-ReLU certificate. Report docs/research/15_census_results.md SHA256bbcb531e01dddd9fa2b710a3b8805b8c065f9e07de0453c9a0e485120076cb10. Historical H1not-supported unchanged.

Prepare a separate retrospective learned-dynamics phase within item15:5archived item9NPweights, savedvalidationdata and64-rowT16compatibility gate beforeanyextendedtrajectory;4fixedqueriesfromoneknownscene,explicitRMScheckerboardperturbation,256continuations,allrawstates andguards retained.22originalinputfiles1,777,007B are identified by archiveandSHA. This is not newly trained competence, a held-out accuracy benchmark or an external replication. A different frozen plan is required before execution. Items16–30 remain unopened. No paid/GPU/main-merge/outreach/access/billing change.
