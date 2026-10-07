# Item 12 results: continual learning, stored experts and routing

## Decision

**Close item 12 as a completed feasible investigation. Robust continual learning at larger scope remains unestablished.**

The new work provides three concrete results:

1. Retrospective matrices reconstructed from item-6 saved predictions show that the newest sequential descendant loses all observed AGENTE/PACIENTE successes on the two older topics, despite positive global backward transfer. Acquisition on those roles was already weak.
2. Current-source native deep copies are independent under the frozen diagnostic: updating each new expert preserves all older named parameters, buffers and fixture outputs through eight copies. A deliberately shared-reference control is detected.
3. A literal routing counterexample confirms that unchanged experts can still produce a deteriorating aggregate answer when expert selection or mixing changes.

The second result is an implementation property with a growing storage cost. It does not establish eight-task learning, useful acquisition, automatic task discovery or accurate end-to-end routing. No new six-task preflight or larger learned benchmark ran. Historical H1 remains closed and not supported.

## 1. What was executed and what was already known

The source, historical-evidence and primary-literature audit began only after item 11 closed. The [protocol](12_probe_protocol.md) and [execution plan](12_probe_plan.json) were frozen at **2026-10-07T14:04:24+00:00**, before the new programs ran. Architecture, evidence and scientific reviewers found no concrete static blocker. The preserved [static review](../../results/research/12_validation/probe_static_review.json) explicitly distinguishes source review from execution.

| Identity | Value |
|---|---|
| Frozen execution source | aada1037271a621ac651c94c68feaba51143be05 |
| Plan SHA-256 | c5ee67317b336d786f44f66572d274e7e95099a18ed87f6e486133512a302ffa |
| Public CPU workflow | [run 37633696345](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37633696345), job 112834249348 |
| Provider outcome | Completed successfully; runner released |
| Final result commit | 5381f6775ae4be64210b99b14360fe06faea28ec |
| Result directory | results/research/12_cloud_runs/37633696345-1-probe |
| Final manifest | [archive_manifest.json](https://github.com/Agnuxo1/NeuroPixel/blob/5381f6775ae4be64210b99b14360fe06faea28ec/results/research/12_cloud_runs/37633696345-1-probe/archive_manifest.json) |
| Archived inventory | 51 files and 19,417,647 bytes, excluding the manifest |
| Manifest itself | 7,950 bytes; SHA-256 8d36e0b3ff383227fdc66af73a4501462c52a9f76b06b6a5278effb5255d522b |

There was one frozen workflow execution for this point. It ran eight focused parent contract tests and four serial components in fresh interpreters: a saved-prediction producer, native isolation diagnostic, independent matrix auditor and independent isolation auditor.

The task-performance data were already generated and partially examined in item 6. This is a **secondary analysis of previously observed outcomes**, not a new confirmatory experiment. The selected old archive is commit 15e76456bc2b4cce5faec0b08fb5288fe7844547, directory results/research/06_cloud_runs/37566497890-1, originating from source 08d0d52edd05da6835e71479f3ba4399fcbeabae.

Exactly 15 old artifacts were bound by SHA-256: the final manifest, worker status, growth records, partitions, three datasets, two expert-output archives and six checkpoint files. They occupy 3,766,650 bytes including the old manifest. Checkpoints were hashed without deserialization. The original full archive audit remains in [item 6](06_results.md); this new selection does not repeat that whole audit.

The old predictions use the pre-item-8 PAD implementation. The fresh isolation diagnostic uses current source after the PAD correction. The analysis does not silently substitute current weights, replay the old model or combine the two source versions into a single experiment.

## 2. Source and historical claims

The audited paths include [scripts/phase3.py at the opening source](https://github.com/Agnuxo1/NeuroPixel/blob/bb9ac26a926a9ca2023145f0166e82a4f5e174de/scripts/phase3.py), [neuropixel/phase3.py](https://github.com/Agnuxo1/NeuroPixel/blob/bb9ac26a926a9ca2023145f0166e82a4f5e174de/neuropixel/phase3.py), and the item-6 controlled growth implementation.

Native growth has two materially different actions. Creating a new expert deep-copies an existing complete model; its previous copy remains stored. Revisiting an existing expert updates that expert itself, so its old behavior is not protected merely by having a bank. Optimizers are recreated by the training path. In the historical driver, creation receives the full iteration budget while a revisit receives one fifth at a different learning rate. The recorded policy comparisons consequently conflate capacity, updates, routing and which model is modified.

The old summary files contain these global accuracies:

| Historical recorded policy | Stored experts at end | Final A/B/C | Scope |
|---|---:|---|---|
| Sequential, after A/B/C | 1 predictive path | 49.1% / 49.35% / 99.3% | First three stages of the historical A/B/C/A sequence |
| Sequential, after revisiting A | 1 predictive path | 98.5% / 49.55% / 49.35% | Fourth stage, different endpoint |
| Resonance growth | 2 | 93.25% / 47.75% / 90.45% | Recorded routed result |
| Novelty growth | 3 | 82.55% / 96.4% / 99.0% | Recorded routed result |

Sources: [grow_seq.json](https://github.com/Agnuxo1/NeuroPixel/blob/bb9ac26a926a9ca2023145f0166e82a4f5e174de/results/phase3/grow_seq.json), [grow_grow.json](https://github.com/Agnuxo1/NeuroPixel/blob/bb9ac26a926a9ca2023145f0166e82a4f5e174de/results/phase3/grow_grow.json), [grow_novedad.json](https://github.com/Agnuxo1/NeuroPixel/blob/bb9ac26a926a9ca2023145f0166e82a4f5e174de/results/phase3/grow_novedad.json).

The first sequential acquisition scores were A 96.9%, B 98.45% and C 99.3%. After C, losses on A/B were 47.8 and 49.1 percentage points. The descriptive global BWT is therefore **−48.45 pp** and final average accuracy **65.9167%**. After revisiting A, average accuracy is **65.8%**; resonance and novelty end at **77.15%** and **92.65%**, respectively.

These are arithmetic summaries of one recorded trajectory per policy with rounded scores. The inspected historical artifacts do not provide per-example predictions, original checkpoints, per-role matrices or matched independent seeds. Their stronger reported accuracies do not override the differently configured, more fully archived item-6 results.

The historical fields called temaX_elige_lienzo are means of selected slot indices. A noninteger mean such as 1.0215 is neither routing accuracy nor a recoverable routing-frequency distribution. The source correction from item 8 is retained; historical artifacts are preserved.

The scanner evaluates readability of visible token identities under the expert. It is not intrinsically an oracle for novel tasks or for preserved task competence. Item 6 also cached expert results by a state_dict-based identity; that identity omits the current nonpersistent grounding buffers. Its common ungrounded configuration limits the immediate consequence there, but it is not a general full-model identity contract. The new diagnostic explicitly includes named buffers.

## 3. Reconstructed performance matrices

### Population and predictive path

Both seeds, 20 and 21, use split 0, three noun-disjoint topics and the ordered A/B/C curriculum. Each topic contributes 1,024 final examples, with 256 queries for each of AGENTE, ACCION, PACIENTE and LUGAR. The producer and auditor independently verify visible role/filler pairs, query and output positions, token categories, TEST triple membership and saved label order.

Each row below uses the newest descendant after training the indicated topic for 512 updates with a reset optimizer. Its checkpoint bytes and recorded parent chain match the fixed sequential trajectory. **These matrices describe that newest-descendant predictive path, not the end-to-end router of the stored three-expert bank.** The earlier experts remain available as separate snapshots.

All three fixed evaluation sets are scored at every checkpoint from saved 35-class log probabilities. No task-specific output mask is applied. The complete [JSON report](https://github.com/Agnuxo1/NeuroPixel/blob/5381f6775ae4be64210b99b14360fe06faea28ec/results/research/12_cloud_runs/37633696345-1-probe/secondary/report.json) retains six matrices per seed, correct counts, per-role scores and cross entropy. The [CSV](https://github.com/Agnuxo1/NeuroPixel/blob/5381f6775ae4be64210b99b14360fe06faea28ec/results/research/12_cloud_runs/37633696345-1-probe/secondary/retention_rows.csv) contains all 18 checkpoint/topic rows.

### AGENTE/PACIENTE binding

With equal role counts, the mean of AGENTE and PACIENTE accuracies equals their combined correct count divided by 512.

| Seed | Newest checkpoint after | A | B | C |
|---:|---|---:|---:|---:|
| 20 | A | 149/512 = 29.1016% | 0/512 | 0/512 |
| 20 | B | 0/512 | 106/512 = 20.7031% | 0/512 |
| 20 | C | 0/512 | 0/512 | 115/512 = 22.4609% |
| 21 | A | 142/512 = 27.7344% | 0/512 | 0/512 |
| 21 | B | 0/512 | 126/512 = 24.6094% | 0/512 |
| 21 | C | 0/512 | 0/512 | 101/512 = 19.7266% |

Two conclusions hold together. The initial scores are not strong task acquisition, and the limited observed successes on old-topic binding disappear after moving to a different noun topic. Zero means zero successes in these saved finite evaluations; it is not a proof of a zero population success probability.

The previous label-aware hard-selection bounds equal the binding acquisition diagonals because every off-diagonal binding score here is zero. Even a selector that knew which individual snapshot was correct could not recover high binding accuracy from these stored experts. This bound concerns selecting an individual argmax answer and does not cover probability mixtures.

### Global accuracy

| Seed | Newest checkpoint after | A | B | C |
|---:|---|---:|---:|---:|
| 20 | A | 204/1024 = 19.9219% | 84/1024 = 8.2031% | 62/1024 = 6.0547% |
| 20 | B | 143/1024 = 13.9648% | 228/1024 = 22.2656% | 145/1024 = 14.1602% |
| 20 | C | 354/1024 = 34.5703% | 363/1024 = 35.4492% | 480/1024 = 46.8750% |
| 21 | A | 231/1024 = 22.5586% | 62/1024 = 6.0547% | 92/1024 = 8.9844% |
| 21 | B | 223/1024 = 21.7773% | 376/1024 = 36.7188% | 264/1024 = 25.7813% |
| 21 | C | 346/1024 = 33.7891% | 362/1024 = 35.3516% | 477/1024 = 46.5820% |

The oldest topic's global score increases after C in both seeds. That does not conflict with zero binding: the additional correct answers come from ACCION and LUGAR.

### Backward transfer and forgetting

BWT compares performance after C with each old task's own acquisition diagonal, averaging over A and B. Max-past forgetting instead compares with the maximum over all earlier checkpoints. The separately named post-acquisition variant excludes observations before the task's acquisition. Both forgetting quantities remain signed; improvement can produce a negative value. The definitions follow the distinctions in GEM and RWalk, with the additional variant explicitly labelled [1,2].

| Seed | Measure | Average A/B/C after C | BWT on old A/B | Max-past forgetting | Post-acquisition forgetting |
|---:|---|---:|---:|---:|---:|
| 20 | Global | 38.9648% | +13.9160 pp | −13.9160 pp | −13.9160 pp |
| 21 | Global | 38.5742% | +4.9316 pp | −4.9316 pp | −4.9316 pp |
| 20 | Binding | 7.4870% | −24.9023 pp | +24.9023 pp | +24.9023 pp |
| 21 | Binding | 6.5755% | −26.1719 pp | +26.1719 pp | +26.1719 pp |

The role decomposition explains the sign change exactly:

| Role | Seed-20 BWT | Seed-21 BWT |
|---|---:|---:|
| AGENTE | −29.1016 pp | −27.3438 pp |
| PACIENTE | −20.7031 pp | −25.0000 pp |
| ACCION | +59.7656 pp | +47.4609 pp |
| LUGAR | +45.7031 pp | +24.6094 pp |

Because every role has equal weight, global BWT is the mean of these four quantities. This is an arithmetic explanation of the observed aggregate, not an experimental identification of the neural mechanism causing each change. The source's disjoint noun sets and shared verb/place sets make the task structure relevant, but do not isolate effects of representation, optimization, exposure or routing.

The two forgetting maxima are also observably different in a real row: seed-20 ACCION has final max-past forgetting −46.8750 pp and post-acquisition forgetting −48.0469 pp. Its BWT is +59.7656 pp because an intermediate improvement on A had already occurred. The nonmonotone unit example independently tests this distinction.

There is no initial untrained prediction baseline, so FWT stays null. No new confidence interval, p-value or bootstrap is attached to these two already observed seeds. Eighteen rows, correlated query views and three topics are not eighteen independent replications.

## 4. New native-copy isolation result

The [native diagnostic report](https://github.com/Agnuxo1/NeuroPixel/blob/5381f6775ae4be64210b99b14360fe06faea28ec/results/research/12_cloud_runs/37633696345-1-probe/isolation/isolation_report.json) records the current model with vocabulary 35, 3×3 canvas, four recurrent updates, c_id 16, c 48 and hidden width 128. Seed 120001 initializes the model.

The one fixed twelve-example fixture displays noun IDs 5–16 at (0,0), query token 1 at (2,1), and reads at (2,2). Evaluation mode disables stochastic firing while gradients remain enabled. SGD uses learning rate 0.01 with no momentum or weight decay.

At each of eight stages, the newest complete expert is copied and receives three fixture updates. The first stage initializes the first expert. All named parameters, nonpersistent buffers and logits are saved before and after every stage.

**All eight isolation stages passed.** Every older expert retained exact tensor values and fixture logits, while the new expert changed both tensors and logits. Every normal bank had disjoint reported storage classes across experts. This also verifies that old-output equality was not merely the consequence of a no-op optimizer update.

| Stored experts | Parameter elements in bank | Named tensor payload | Unique reported storage | Reported storage classes |
|---:|---:|---:|---:|---:|
| 1 | 29,824 | 119,751 bytes | 119,751 bytes | 12 |
| 2 | 59,648 | 239,502 bytes | 239,502 bytes | 24 |
| 4 | 119,296 | 479,004 bytes | 479,004 bytes | 48 |
| 8 | 238,592 | 958,008 bytes | 958,008 bytes | 96 |

Per expert, 119,296 parameter bytes plus 455 buffer bytes give 119,751 bytes. The values confirm the predicted linear tensor-storage cost for this native copy strategy. They exclude optimizer/autograd state, gradients, allocator overhead, Python objects and the runtime. They are not an estimate of general information capacity, physical energy or total application RAM.

The shared-reference negative control places the same object in two list slots. Updating the second slot also changes the first slot's tensors and both outputs, exactly as the detection criterion requires. The two slots remain mutually identical. Its unique storage is one expert even though the logical two-slot tensor references sum to two experts.

The diagnostic performs 24 positive updates plus three alias-control updates. The focused contract suite separately uses two one-step update witnesses. These 29 total update calls are integrity checks on fixed fixtures, not 29 trained tasks or independent trials.

All ten producer checks passed: eight stages, alias detection and unchanged saved RNG after initialization. Eighteen full before/after snapshot NPZs, the fixture and RNG NPZ are preserved. The independent auditor reconstructs full fingerprints and comparisons from those saved arrays.

This result applies to the tested native deepcopy/create path and fixed configuration. It does not claim that modifying an existing expert is harmless, that a changed shared component is isolated, or that an untested configuration is protected by a state_dict-only hash.

## 5. Routing can still deteriorate with unchanged experts

The frozen two-example counterexample uses eight literal two-class probability tables. Expert 0 predicts both labels correctly; each other expert predicts both oppositely. These tables never change.

| Expert count | Select expert 0 | Select last expert | Uniform probability mixture |
|---:|---:|---:|---:|
| 1 | 2/2 | 2/2 | 2/2 |
| 2 | 2/2 | 0/2 | 1/2 |
| 4 | 2/2 | 0/2 | 0/2 |
| 8 | 2/2 | 0/2 | 0/2 |

At K=2 the mixture ties, and the frozen lowest-class tie rule gives one success. Both producer and independent auditor recover the exact counts.

These are constructed outputs. They demonstrate a logical limitation of the inference "old experts are unchanged, therefore system answers are unchanged." They are not measured accuracies of trained NeuroPixel routers.

For context, item 6's actual saved snapshot-bank scanner binding was 23.3073%/22.4609% for seeds 20/21; its learned router gave 22.9167%/23.8281%. Those already reported routed scores are distinct from the newest-descendant matrices above, and they also fail to establish high binding competence.

Progressive Neural Networks are a relevant earlier architectural comparison because they freeze old columns and add new capacity, while also using lateral connections and discussing growth and task identity [3]. Native copying here does not reproduce that transfer mechanism. Task-IL, Domain-IL and Class-IL distinguish what information is available at inference and what must be inferred [4]. The current noun-disjoint, partially shared-label task structure should be reported explicitly.

## 6. Validation and resources

| Check layer | Observed result |
|---|---|
| Focused tests | 8 parent methods passed, plus 7 successful subtest call events; 15 call events total; zero failures, errors or skips |
| Native diagnostic | 10/10 producer checks passed |
| Independent saved-output audit | 6,330 checks, zero issues |
| Independent isolation-array audit | 5,013 checks, zero issues |
| Root JSON/CSV, provenance and inventory review | 3,508 checks, including 1,716 numerical comparisons; zero issues; maximum numerical difference 0 |
| Archived text integrity | All 27 nonbinary result files rehashed after retrieval; SHA-256 and byte lengths match the final manifest |
| Git inventory | Complete, nontruncated tree; exactly 52 run files including the manifest; regular file modes and sizes consistent |
| Source and inputs | All 21 bound source files and the frozen plan verified before/after; selected old inputs unchanged |
| Worker and provider | Completed; runner released |

The matrix auditor imports neither the producer nor its metric helper. It validates visible gold, partitions, saved labels, normalized log probabilities, argmax predictions, all matrices and the literal routing fixture. Maximum log-normalization discrepancies are approximately 2.17×10^-7 and 2.32×10^-7, within the frozen 10^-5 tolerance.

The isolation auditor imports no Torch or model. It independently hashes saved named tensors, checks old/new equality and mutation, counts payload/storage arithmetic, and compares saved RNG arrays. Live pointer identities are reported observations from the producer: the auditor cannot re-observe that process's pointers and does not independently replay SGD.

Root additionally checks numerical CSV rows against JSON, recalculates BWT and both maxima, verifies role decomposition, matches every binary payload descriptor against the final manifest and verifies Git sizes. The worker's archived source ZIP is preserved, but root did not independently unpack it in this point. The binary arrays were read by the independently authored auditors on the same runner; root did not deserialize them in a separate environment.

The workflow used Python 3.12.14, NumPy 2.2.6, CPU Torch 2.6.0+cpu, SciPy 1.15.1, Pillow 12.3.0, psutil 7.2.2 and pytest 9.1.1. One numerical/inter-op thread was configured, and there was no GPU execution.

Worker elapsed time before its final archive was **23.0081 seconds**. The serial scientific controller took **4.6934 seconds**, including input recovery; the four subprocess wall times were 0.2152, 3.1215, 0.3654 and 0.4186 seconds. These overlap with enclosing worker measurements and must not be summed as independent total costs. Dependency installation and provider setup lie outside worker time.

Eleven supervisor samples across the two worker stages gave a minimum observed available RAM of **14.2728 GiB**, above the 8 GiB floor. This is a sampled minimum, not a proof of the minimum between observations. All phases completed within their frozen deadlines, with no resource-abort or diagnostic failure. Pre-execution fixture clarification and the addition of saved RNG witnesses remain documented with the preimage preserved.

The independent code and peer reviews improve auditability. They share this project's execution and review context and do **not** constitute external laboratory replication.

## 7. Achieved and still missing

### Achieved in this investigation

- Audited the actual create/revisit behavior, optimizer resets, historical summaries and task/routing/capacity confounds.
- Added a reusable descriptive matrix implementation with hand-counted tests.
- Reconstructed every checkpoint/topic/role score from the selected preserved outputs without retraining.
- Exposed an aggregation failure: positive global BWT conceals complete loss of observed old-topic binding successes.
- Verified full parameter-and-buffer copy isolation through eight stored experts, with actual optimization and a detecting negative control.
- Measured the stated tensor-storage growth and preserved full array evidence.
- Tested the literal routing counterexample and completed independent saved-array, source and manifest checks.
- Retained the prior evidence of weak acquisition, did not admit a larger learned study, and kept the claim narrower than the implementation tests.

### Still missing for a substantive continual-learning result

1. Reliable initial acquisition on every proposed task, verified under a prospective finite pilot before retention is interpreted.
2. A larger learned sequence with independent seeds and task constructions, varied orders, recurrence and controlled domain/label overlap.
3. Comparisons with a shared sequential model, a shared model with a fixed rehearsal budget and an isolated expert bank; acquisition and retention must both be measured.
4. Explicit supplied-task-ID and inferred-routing settings, with full end-to-end accuracy and routing failures.
5. Matched or transparently accounted cumulative capacity, optimizer state, replay memory, data exposure and computation.
6. An initial baseline if FWT is claimed; an appropriate jointly trained or other specified reference if a formal intransigence gap is claimed.
7. Held-out content and untouched evaluation tasks suitable for confirmatory uncertainty and model selection.
8. Replication by an independent group, external practical utility and evidence of a scientifically new result.

GEM, EWC and A-GEM motivate different memory, regularization and efficiency controls [1,5,6]. This point does not implement those baselines or demonstrate superiority to them. A longer sequence of weakly acquired tasks would not resolve the acquisition problem. The chosen bounded work closes the feasible evidence questions while leaving the larger capability explicitly pending.

## Primary references

1. Lopez-Paz and Ranzato, *Gradient Episodic Memory for Continual Learning* (2017), task-performance and transfer definitions. https://arxiv.org/pdf/1706.08840
2. Chaudhry et al., *Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence* (2018), section 3. https://arxiv.org/pdf/1801.10112
3. Rusu et al., *Progressive Neural Networks* (2016), frozen columns, lateral connections and limitations. https://arxiv.org/pdf/1606.04671
4. van de Ven and Tolias, *Three Scenarios for Continual Learning* (2019), task identity and inference scenarios. https://arxiv.org/pdf/1904.07734
5. Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (2017). https://arxiv.org/pdf/1612.00796
6. Chaudhry et al., *Efficient Lifelong Learning with A-GEM* (2019), evaluation and computational/memory considerations. https://arxiv.org/pdf/1812.00420

All numerical conclusions are restricted to the identified artifacts and recipes. This point establishes neither a general solution to continual learning nor a Nobel-level discovery. Item 13 remains unopened until this report and its closure are recorded.
