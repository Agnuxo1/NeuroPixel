# Item 11 — Memory transport, learned recall and interference

## Decision

**The feasible item-11 investigation is complete. Durable, useful memory and resistance to interference are not established.** The audited implementation carries recurrent activity inside a stream and can continue it through an explicit caller-owned state. Ordinary independent forward calls reset that activity. A finite untrained diagnostic verifies these mechanisms, but it is not a learned recall result.

All four prespecified learned pilots completed. The selected NeuroPixel configuration recovered **38 of 48** known cue cases at the near delay; the selected GRU recovered **48 of 48**. Admission required at least **46 of 48 in both selected configurations**. Neither NeuroPixel candidate met that prerequisite. Accordingly, **the five-pair main experiment, delay-eight primary contrast and learned interference panel were not admitted or executed**. No interval, significance claim or interference effect is inferred from an unexecuted panel.

The independent saved-artifact audit verified 6409 archive/source/operational checks, 504 diagnostic-array checks and 13517 learned-preflight checks, with no reported issues. These are consistency and execution-contract checks, not independent scientific replications. Root additionally compared 924 numeric metric/cost values between the producer records and independent recount; the largest absolute difference was 1.2434497875801753e-13, with no discrepancy beyond the declared comparison tolerance.

Sources: [learned raw archive](https://github.com/Agnuxo1/NeuroPixel/tree/1b1759775adba1ab0d289ba34b6b306f9ad547c2/results/research/11_cloud_runs/37626660573-1-preflight), [independent scientific recount](https://github.com/Agnuxo1/NeuroPixel/blob/0a8403b1001881acc8626369a1e4f9dde83cc8e2/results/research/11_cloud_runs/37627489876-1-audit/analysis/scientific_audit.json), [diagnostic recount](https://github.com/Agnuxo1/NeuroPixel/blob/0a8403b1001881acc8626369a1e4f9dde83cc8e2/results/research/11_cloud_runs/37627489876-1-audit/probe_analysis/probe_audit.json), and [archive/source audit](https://github.com/Agnuxo1/NeuroPixel/blob/0a8403b1001881acc8626369a1e4f9dde83cc8e2/results/research/11_cloud_runs/37627489876-1-audit/offline_archive_audit.json).

## 1. Questions and definitions

This item asks what object stores information, how long that object persists, whether the information supports a trained retrieval behavior after the cue is removed, and whether a later distractor disrupts that behavior. Its conclusions distinguish the following.

| Property | Object or operation | Item-11 conclusion |
|---|---|---|
| Learned parameter storage | Model weights and constructor/buffer metadata | Parameters can be saved; a parameter-only checkpoint is incomplete for the tested grounded configuration. |
| Recurrent activation within a stream | The current canvas state | Implemented and verified in a finite untrained diagnostic. |
| Explicit continuation across chunks | Caller passes the returned state back to the helper | Implemented and verified; ownership remains explicit. |
| Automatic persistence between ordinary independent calls | Model remembers a prior episode without an external state argument or hook | Absent in the tested default forward path. |
| Learned cue recall after withdrawal | Correct original-symbol output from the intact carried state | Prespecified near-delay competence was not met by NeuroPixel in this pilot. |
| Long-delay learned retention | Correct recall at the prespecified longer horizons | Conditional main panel not executed. |
| Robustness to interference | Recall after a controlled conflicting distractor relative to a matched blank history | Conditional main panel not executed. |
| Consolidation under further training or restart | An old learned capability survives later changes or reconstruction | Not established by this item. |
| Attractor convergence or indefinite retention | A suitable stability property of the actual dynamics | Not established by finite state differences or finite outputs. |

The distinction between information in activations and in weights is explicit in [Hochreiter and Schmidhuber (1997), LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf), §1. Their delayed-signal experiments in §5.2 require an earlier signal to support a trained later response despite intervening inputs. [Hopfield (1982)](https://www.cctbio.ece.umn.edu/wiki/images/4/41/Hopfield_Neural_Networks_and_Physical_Systems_with_Emergent_Collective_Computational_Abilities.pdf), equation 2, stores patterns through a connection-matrix prescription and retrieves through recurrent dynamics. Those results do not transfer an attractor or stability guarantee to NeuroPixel.

[Neural Turing Machines (2014)](https://arxiv.org/pdf/1410.5401), §§3–4, separate the controller from a writable memory matrix and reset dynamic state between episodes. Copy and associative-recall tasks test different behaviors; associative recall is not merely copying the query. [The DNC paper (2016)](https://gwern.net/doc/reinforcement-learning/model-free/2016-graves.pdf), system overview and Figure 1, specifies read/write, allocation and temporal-link mechanisms. A recurrent canvas or stored expert weights alone do not establish an equivalent external episodic memory.

## 2. A — Source and historical evidence

### Current execution paths

At the audited source, `NeuroPixel.forward` initializes state from the current token input. Its iterative update reinjects that input; the default method does not internally retain the previous return value as the next call's initial state. An explicit external hook can alter a running state, and must be attributed to the caller.

The existing `phase3.np_stream` does carry state from one frame to the next inside a single sequence call. A new sequence call initializes anew. The existing GRU frame path also starts a fresh recurrent episode when called without a hidden-state argument. The strict research helper reproduces the intended native path and adds explicit initial-state, boundary-zero and boundary-permutation interfaces with validation and actual before/after snapshots.

The historical stream uses `zip(frames, steps)`, which silently truncates a mismatch. The new research helper rejects unequal lengths. The original production helper was not silently changed. This prevents a malformed research fixture from looking like a completed history.

Grounding buffers `g_rgb` and `g_mask` are registered as nonpersistent. They affect the effective grounded dictionary but are absent from `state_dict`. A reproducible grounded checkpoint therefore needs the constructor and relevant buffer metadata in addition to parameters.

Sources: [model.py](https://github.com/Agnuxo1/NeuroPixel/blob/cef4e67358ca5c344cd4154a7f641feefeb406ae/neuropixel/model.py), [phase3.py](https://github.com/Agnuxo1/NeuroPixel/blob/cef4e67358ca5c344cd4154a7f641feefeb406ae/neuropixel/phase3.py), [strict helper](https://github.com/Agnuxo1/NeuroPixel/blob/cef4e67358ca5c344cd4154a7f641feefeb406ae/neuropixel/research/stream_memory.py), and [historical source audit](https://github.com/Agnuxo1/NeuroPixel/blob/cef4e67358ca5c344cd4154a7f641feefeb406ae/docs/research/11_historical_memory_audit.md).

### Historical table, preserved with its limits

The historical `results/phase3/memory.json` reports the following within-sequence accuracies.

| Blank frames | Historical NeuroPixel | Historical GRU |
|---:|---:|---:|
| 0 | 99.2% | 77.0% |
| 4 | 98.7% | 79.1% |
| 8 | 96.5% | 78.5% |
| 12 | 86.1% | 78.1% |
| 16 | 69.5% | 76.8% |
| 24 | 44.0% | 74.6% |
| 32 | 31.6% | 72.9% |

That table is evidence of an archived claim, not a new reproduction. The audited inventory did not contain the original per-example outputs and learned checkpoint needed to independently recount or replay it. The script uses one selected seed and a fixed evaluation seed for the curve; horizons are not independently sampled populations. Its four facts also have a fixed role order. Historical parameter counts were 29,824 and 49,067, and recorded elapsed training times were 6,079 and 565 seconds. Neither parameter counts nor computation were matched in those historical measurements. The old model also predates the effective PAD-dictionary correction, so a new run on the present source would not reproduce the historical source byte for byte.

The README shorthand “99% up to 8” was corrected in the isolated research branch to the actual archived values, 99.2% at zero blanks and 96.5% at eight, with explicit within-stream and historical provenance. No historical raw file was replaced and no new result was substituted for it.

Sources: [historical JSON](https://github.com/Agnuxo1/NeuroPixel/blob/bf8e4829ef609df3d7487bfe18ceaf6be24f949f/results/phase3/memory.json), [historical driver](https://github.com/Agnuxo1/NeuroPixel/blob/bf8e4829ef609df3d7487bfe18ceaf6be24f949f/scripts/phase3.py), [documented correction](https://github.com/Agnuxo1/NeuroPixel/blob/bf8e4829ef609df3d7487bfe18ceaf6be24f949f/README.md), and the source audit linked above.

## 3. B — Frozen untrained mechanism diagnostic

The [diagnostic protocol](https://github.com/Agnuxo1/NeuroPixel/blob/bf8e4829ef609df3d7487bfe18ceaf6be24f949f/docs/research/11_probe_protocol.md) and [execution plan](https://github.com/Agnuxo1/NeuroPixel/blob/bf8e4829ef609df3d7487bfe18ceaf6be24f949f/docs/research/11_probe_plan.json) were frozen at **2026-10-07 12:55:25 UTC**, before the diagnostic. The plan SHA-256 is `e93c42d0a6c62f4870f26d360dbdc9d877fe0d5196e5d207db86ce1b65db8bc5`. Source: `bf8e4829ef609df3d7487bfe18ceaf6be24f949f`.

The fixture uses a 3×3 canvas, vocabulary eight, six cue symbols 2–7, PAD zero and query one. Cue and query locations are fixed at (0,0) and (2,1), with readout (2,2). A cue or blank frame receives three local updates and the query ten. Delays 0, 4, 16 and 32 therefore imply 13, 25, 61 and 109 local updates. There is no optimizer and no learned checkpoint.

The diagnostic model has 29,392 parameters, 48 state channels, a 16-dimensional identity dictionary and update hidden width 128. Since the ordinary final update projection starts at zero, a separate declared diagnostic seed makes that projection nonzero. This enables an activity counterfactual; it does not train memory. Main initialization uses seed 110001 and the private projection initializer uses 110002.

The executed checks establish:

1. Five distinct sequences of earlier independent calls produce exactly the same subsequent query state and logits when their returned states are discarded.
2. Native within-sequence streaming and the strict helper agree; splitting at the pre-query boundary and explicitly supplying the returned state reproduces continuation.
3. Zeroing the actual pre-query state gives the query-only reference.
4. Permuting the actual pre-query states produces the donor trajectories within the frozen floating-point tolerance.
5. Before/after boundary tensors independently show that the intended zero and permutation operations happened.
6. Filling a returned state with the finite sentinel 3.25 does not cause a fresh independent call to inherit it.
7. Saved parameter/buffer hashes and the global random-state records remain consistent across the diagnostic.
8. Explicit grounding-buffer reconstruction restores the grounded dictionary and tested grounded computation; parameter-only reconstruction does not restore the effective dictionary.

Native and chunk comparisons were exact in this fixture. Swap-to-donor comparisons use absolute tolerance 1e-6 and relative tolerance 1e-5; the largest reported delay-32 logit difference was approximately 1.91e-6 and met that combined tolerance. This numerical transport check must not be described as correct symbol recall.

| Blank frames | Total local updates | Maximum absolute cue-versus-no-cue state difference |
|---:|---:|---:|
| 0 | 13 | 1.699513793 |
| 4 | 25 | 1.823159814 |
| 16 | 61 | 3.925979615 |
| 32 | 109 | 13.595387459 |

Nonzero differences show dependence on the earlier cue in the carried state. Increasing magnitude is not a memory-capacity score, and it is not evidence of stable storage. With the same subsequent query and dynamics, a complete state permutation should transport subsequent computation regardless of whether any predicted symbol is correct.

The producer reported **47 passing diagnostic checks**, ten condition records and 18 scientific payloads, including 17 NPZ files. Contract execution collected **eight parent test methods and 31 subtests**, giving 39 passing call reports, with no skip or failure. The independent diagnostic auditor recounted **504 checks** from saved arrays and recorded no issue. The parameter-only `.pt` file was hash checked by the independent auditor; the save/load witness itself comes from the original executed diagnostic, not an independent checkpoint replay.

Raw source: [B archive](https://github.com/Agnuxo1/NeuroPixel/tree/ae82b5154c8d3d7da132c4b95ff05009be253437/results/research/11_cloud_runs/37624600542-1-probe). Independent recount: [B audit](https://github.com/Agnuxo1/NeuroPixel/blob/0a8403b1001881acc8626369a1e4f9dde83cc8e2/results/research/11_cloud_runs/37627489876-1-audit/probe_analysis/probe_audit.json).

## 4. C — Learned recall protocol and the completed negative preflight

### Prespecified design

The [learned-memory protocol](https://github.com/Agnuxo1/NeuroPixel/blob/cef4e67358ca5c344cd4154a7f641feefeb406ae/docs/research/11_memory_protocol.md) and [preflight plan](https://github.com/Agnuxo1/NeuroPixel/blob/cef4e67358ca5c344cd4154a7f641feefeb406ae/docs/research/11_preflight_plan.json) were frozen at **2026-10-07 13:11:49 UTC**. Plan SHA-256: `dbdd2e02da175522cff93323b32a89538b842423fe15eaef2c53d11a35aa25d5`. Recipe SHA-256: `b3c397b106c9c40776f13db7c5f45f1455d821be27474aa845f772c8a5c7e82f`. Source: `cef4e67358ca5c344cd4154a7f641feefeb406ae`.

The task has six cue labels and eight possible cue positions, giving a known **48-case census** in target-outer, row-major-position-inner order. Each history presents one cue, removes it, presents blank frames, and finally presents a query. The query does not supply the target symbol. Cue positions include the query position; the output position is excluded. Training samples census rows uniformly with replacement in batches of 32 and a uniform delay from 0–2 for each minibatch. A private PCG64 stream, seed 100000 plus the model seed, draws the complete sample-index matrix followed by the delay vector; both arrays are preserved. Model initialization and training firing have separately recorded seeds.

NeuroPixel uses 29,392 parameters and 432 float32 state values per example, or 1,728 bytes of state payload. The GRU uses 29,336 parameters and 70 state values, or 280 bytes. These are approximately parameter-matched models with different state tensor sizes and different per-history computation. NeuroPixel performs 13+3d local updates on nine cells; the GRU receives d+2 frames. Parameter matching must not be called compute matching.

The four pilots cross two families with rates 0.001 and 0.003, all at seed 69 and **1,024 updates**. Training uses AdamW, weight decay 0.0001, gradient clipping 1.0 and answer cross-entropy only. Development uses the full census at delays three and four. Selection ranks mean development accuracy first, mean cross-entropy second, then the lower rate. Near-delay two competence is a separate admission check: both selected models must achieve at least 95%, meaning 46/48.

Had admission passed, the fixed main design would have used five fresh paired seeds 70–74, 2,048 updates and the selected rates. Its single primary contrast was the NeuroPixel-minus-GRU accuracy difference at delay eight, summarized across the five training pairs with the declared t interval. Its secondary panel included longer delays, cue erasure, state zero/permutation and matched blank/distractor cases. Conflicting distractors and same-label re-exposure were to be reported separately. All of this remains **planned, not executed**.

### All pilot recall outcomes

Every entry below uses 48 census cases. These are development/competence results at one initialization seed, not final independent-content generalization or five scientific replications.

| Family | Learning rate | Delay 2 correct | Delay 3 correct | Delay 4 correct | Mean DEV accuracy |
|---|---:|---:|---:|---:|---:|
| neuropixel | 0.001 | 40/48 | 36/48 | 31/48 | 69.791667% |
| neuropixel | 0.003 | 38/48 | 36/48 | 33/48 | 71.875000% |
| gru | 0.001 | 48/48 | 48/48 | 48/48 | 100.000000% |
| gru | 0.003 | 48/48 | 48/48 | 48/48 | 100.000000% |

NeuroPixel's rate 0.003 is selected because its mean development accuracy, 71.875%, exceeds the 69.791667% of rate 0.001. The larger cross-entropy of rate 0.003 does not override the prospectively ordered accuracy criterion. The GRU's accuracies tie and its rate 0.003 has lower development cross-entropy. Both selected rates are therefore 0.003.

The selected NeuroPixel's delay-two accuracy is **79.166667% (38/48)** and fails the 46/48 threshold. Its other rate also fails at **83.333333% (40/48)**. The selected GRU achieves **100% (48/48)**. The audited admission result is consequently **false**. There was no outcome-driven retuning, extension, substitute selection or main-study launch.

### Cross-entropy and limits on diagnosis

These are mean negative log likelihoods in natural-log units, independently recomputed from the saved logits.

| Family | Learning rate | Delay 2 CE | Delay 3 CE | Delay 4 CE |
|---|---:|---:|---:|---:|
| neuropixel | 0.001 | 322.4639791 | 1911.668396 | 9100.976827 |
| neuropixel | 0.003 | 1793.748441 | 11152.39928 | 65938.79564 |
| gru | 0.001 | 0.001112260411 | 0.001163846879 | 0.001227374082 |
| gru | 0.003 | 0.0002864148680 | 0.0002858566714 | 0.0002863728627 |

The high NeuroPixel values are finite and survive the independent recount. An accuracy percentage can coexist with very large cross-entropy when wrong predictions have very poor target likelihood. This does not by itself identify why the model produced such logits. Training acquisition, evaluation dynamics, numerical sensitivity and representation remain possible lines of investigation; this item does not assign their causal contribution. A sampled final training loss also cannot replace intact evaluation competence.

**Failure of this preflight is evidence about this frozen configuration and budget. It is not a proof that all NeuroPixel variants cannot learn memory.** Conversely, the untrained state-transport diagnostic cannot compensate for the failed learned prerequisite. Without competent intact recall, errors in an interference condition would not isolate forgetting or interference. The conditional panel was withheld for exactly that reason.

Sources: [selection.json](https://github.com/Agnuxo1/NeuroPixel/blob/1b1759775adba1ab0d289ba34b6b306f9ad547c2/results/research/11_cloud_runs/37626660573-1-preflight/selection.json), [preflight summary](https://github.com/Agnuxo1/NeuroPixel/blob/1b1759775adba1ab0d289ba34b6b306f9ad547c2/results/research/11_cloud_runs/37626660573-1-preflight/preflight_summary.json), and the independent scientific audit linked above. The latter includes all per-target/per-position metrics and an independently recounted CSV.

## 5. Execution, preservation and independent checks

All item-11 numerical execution used bounded standard public CPU runners. The local executor remained unavailable; it was not silently represented as having run these tests. No GPU or paid compute was requested. Diagnostics and audits used one numerical thread; learned pilots used two and one inter-op thread. The aggregate CPU cap was four and admission/supervision required at least 8 GiB available RAM. Each worker had a finite total deadline with a reserved final-archive interval; the earlier worker/workflow deadline controlled.

| Stage | Run / job | Final results commit | Files / bytes, excluding manifest | Worker seconds before final archive |
|---|---|---|---:|---:|
| B mechanism diagnostic | 37624600542 / 112803038520 | ae82b5154c8d3d7da132c4b95ff05009be253437 | 36 / 10,315,839 | 14.016487243 |
| C learned preflight | 37626660573 / 112810057008 | 1b1759775adba1ab0d289ba34b6b306f9ad547c2 | 72 / 12,153,630 | 92.894473800 |
| Independent B+C audit | 37627489876 / 112812914654 | 0a8403b1001881acc8626369a1e4f9dde83cc8e2 | 24 / 10,623,568 | 10.522987138 |

The B runner had six sampled resource observations with a minimum available RAM of 14.475296020507812 GiB. The C runner had 85 observations with a minimum of 14.306175231933594 GiB across contract and scientific stages. Its scientific-stage monitor alone had a minimum of 14.386096954345703 GiB. The audit runner's four parent monitor observations had a minimum of 14.46136474609375 GiB; the independent C audit also recorded its own later observations around 14.4396 GiB. These are different observation series, all above the admission floor, not conflicting claims about an identical sample set.

All eight learned-study contract methods passed, without subtests, skips or failures. Across B and C there are 16 collected parent methods and 31 B subtests, not 47 independent experiments. The 47 B producer assertions, 504 independent B checks, 13,517 independent C checks and 6,409 archive/operational checks are separate inventories and should not be conflated.

The independent wrapper recovered both complete result trees, including their manifests; checked file sizes/hashes, actual Git/source ancestry and ZIP contents; confirmed source bindings and clean execution receipts; recounted collected tests and actual call events; and inspected stage runtime and resource-monitor evidence. Its two scientific auditors ran as separate NumPy processes without importing Torch, the producer or learned checkpoints. The C auditor reconstructed the PCG64 streams and deterministic census, recalculated predictions/negative log likelihoods from saved arrays, checked all subgroup scores and independently applied the selection/admission rule.

The audit ran at source `f50ef0e55397ce4fb8e85360f5be8c12003bd6ee`, under plan SHA-256 `2d34eb9492f9a8eccef24e575bf1cca4e10fc695b02de2b09712c807e9516742`, frozen **2026-10-07 13:18:16 UTC**. It completed before report closure. Hashes and execution receipts establish a traceable computational record; they do not replace replication by an external research group or prove every physical operation independently.

### Pilot cost observations

| Pilot | Training seconds | Total pilot seconds |
|---|---:|---:|
| pilot_neuropixel_lr0001_s69 | 36.360916 | 37.209651 |
| pilot_neuropixel_lr0003_s69 | 35.777296 | 35.900298 |
| pilot_gru_lr0001_s69 | 2.789552 | 2.820553 |
| pilot_gru_lr0003_s69 | 2.788262 | 2.819256 |

Each pilot made 32,768 sampled example presentations and processed 98,144 frame examples. Their index/delay streams are identical by design. For a NeuroPixel pilot, the saved schedule implies 4,714,272 forward state-cell update positions; the GRU has 98,144 recurrent example steps. These labels count different operations and are not interchangeable FLOPs, energy or whole-pipeline costs. The observed training-time difference is descriptive for this runner and implementation, not a universal efficiency ratio.

### Preserved corrections and interruptions

Before the first learned pilot, source review corrected two implementation defects: a complete negative scientific admission had been raised as an operational failure, and a prospective final-admission loader could accept an empty development mapping. Negative admission now remains a completed result; all three development outputs are mandatory. Contract fixtures test positive and negative admission semantics. The independent auditor was aligned before execution. Prior candidate bytes, correction receipts and review records are preserved under `results/research/11_validation/` and `source_drafts/`; none is an outcome-driven rescue.

Two internal review-service capacity interruptions were recorded separately. They did not execute numerical trials and are not counted as model failures. The diagnostic, four pilots and offline audit each completed successfully as operations; the negative result is the learned scientific admission decision.

## 6. Achieved, missing and next action

| Work or claim | Status at closure | What would change the status |
|---|---|---|
| Exact definition of the storage object and API lifetime | Achieved for audited paths | Broaden only if a different interface is actually introduced. |
| Historical within-stream claim traced and README precision corrected | Achieved, with limited historical provenance | Recover original exact source/configuration/raw predictions/checkpoint for a historical reproduction. |
| Validated explicit continuation, zero and swap interfaces | Achieved for finite contract/diagnostic fixtures | Additional applications must test their own shape/device/dynamic assumptions. |
| Finite untrained history dependence | Demonstrated in the declared fixture | It remains a mechanism observation, not trained behavioral competence. |
| Four frozen learned pilots and complete preservation | Achieved | Already executed; no additional outcome-selected pilot belongs to this protocol. |
| Independent saved-array/source/operational recount | Achieved internally | External replication remains a separate requirement. |
| Near-delay competence sufficient for the conditional main panel | Not achieved by NeuroPixel | A new, explicitly prospective study must establish adequate intact recall before testing its loss. |
| Main five-pair delay-eight contrast | Not executed because its prerequisite failed | New admission evidence and a new frozen protocol would be necessary. |
| Learned interference resistance and longer-delay retention | Not established | Competent intact recall, matched histories and independent replications. |
| Automatic independent-call/restart episodic persistence | Not implemented in the tested default path | A defined storage/restore interface with fidelity, capacity and behavior tests. |
| Indefinite retention, consolidation or attractor stability | Not established | Appropriate dynamical analysis and tests targeted to those distinct claims. |
| Nobel-level scientific significance | Not supported by this item | A new, important, reproducible scientific result with independent validation and demonstrated significance. |

**Closure means the authorized, scientifically interpretable investigation for this point has been completed. It does not mean its aspirational capability passed.** Item 12 may open only after this report and its closure ledger are committed. No later item was executed within the item-11 trials. Historical H1 remains closed and not supported.

## Reproduction and artifact entry points

- [Frozen B protocol](https://github.com/Agnuxo1/NeuroPixel/blob/bf8e4829ef609df3d7487bfe18ceaf6be24f949f/docs/research/11_probe_protocol.md) and [B plan](https://github.com/Agnuxo1/NeuroPixel/blob/bf8e4829ef609df3d7487bfe18ceaf6be24f949f/docs/research/11_probe_plan.json).
- [Frozen C protocol](https://github.com/Agnuxo1/NeuroPixel/blob/cef4e67358ca5c344cd4154a7f641feefeb406ae/docs/research/11_memory_protocol.md) and [C plan](https://github.com/Agnuxo1/NeuroPixel/blob/cef4e67358ca5c344cd4154a7f641feefeb406ae/docs/research/11_preflight_plan.json).
- [Frozen audit plan](https://github.com/Agnuxo1/NeuroPixel/blob/f50ef0e55397ce4fb8e85360f5be8c12003bd6ee/docs/research/11_audit_plan.json).
- [B raw tree](https://github.com/Agnuxo1/NeuroPixel/tree/ae82b5154c8d3d7da132c4b95ff05009be253437/results/research/11_cloud_runs/37624600542-1-probe).
- [C raw tree](https://github.com/Agnuxo1/NeuroPixel/tree/1b1759775adba1ab0d289ba34b6b306f9ad547c2/results/research/11_cloud_runs/37626660573-1-preflight).
- [Independent audit tree](https://github.com/Agnuxo1/NeuroPixel/tree/0a8403b1001881acc8626369a1e4f9dde83cc8e2/results/research/11_cloud_runs/37627489876-1-audit).
- [B workflow run](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37624600542), [C workflow run](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37626660573), and [audit workflow run](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37627489876).
- Local paths in this report are repository-relative. All numerical claims refer to the immutable source/results commits above.
