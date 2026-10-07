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
