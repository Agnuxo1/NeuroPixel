# Item 9 — Reviewed Stage-A results and Stage-B admission recommendation

**Recommendation: admit the unchanged ten-run Stage-B study after root completes the prospective freeze of the independent analysis code, tests and analysis runtime.** Stage A completed both memorization diagnostics and all four pilots; the scientific recount and archive audit found no discrepancies. The selected learning rates are **NeuroPixel 0.001** and **RelativeTransformer 0.003**, exactly as the frozen validation-binding ranking requires. The resource projection accommodates all ten 4,096-update runs without altering the scientific recipe.

This is a reviewed development result, not a final capability finding. Both models memorized the 32-row training fixture, but the pilots did not establish reliable two-event binding. No final performance collection was generated or accessed. Historical item-5 H1 remains closed and not supported; Stage B has not been launched or completed by this report.

## 1. Provenance and verification

The experiment used source [e63764ccb924ab23d65c2deb94b477387d102c82](https://github.com/Agnuxo1/NeuroPixel/commit/e63764ccb924ab23d65c2deb94b477387d102c82), frozen at 2026-10-07 07:45:55.891523 UTC, in [Actions run 37589908205](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37589908205), attempt 1, job 112688648396. The worker ran from 07:51:50.481646 to 07:57:18.234401 UTC. The stopped raw evidence is preserved under [archive commit 8311c052796aeef5a52484b597cf5d9675ca315e](https://github.com/Agnuxo1/NeuroPixel/tree/8311c052796aeef5a52484b597cf5d9675ca315e/results/research/09_cloud_runs/37589908205-1-preflight).

Two complementary checks support this report:

- [Scientific audit](../../results/research/09_validation/preflight_audit_01.json): **138,074 checks, zero issues**. It independently parsed saved scenes, recounted saved logits/predictions and selection, and checked input-stream records without running models or replaying RNGs.
- [Archive audit 03](../../results/research/09_validation/preflight_archive_audit_03.json): **861 checks, zero issues**. It authenticated the Git archive, all 71 manifested files, the complete 312-file source ZIP, 49 frozen bindings, runtime, tests and execution receipts. Earlier auditor versions and receipts remain preserved; their counts are not added together.

The cloud suite passed **34 collected methods with zero skips/failures**. JUnit's aggregate counter is 137, which includes its additional accounting and is not 137 distinct collected methods. Training/inference used Python 3.12.8, CPU Torch 2.6.0, NumPy 2.2.6, two intra-op threads and one inter-op thread. Available RAM had a recorded minimum of **14.200397491455078 GiB** across sampled observations; this is not peak process RSS or a continuous utilization guarantee.

The authenticated [Stage-A handoff receipt](../../results/research/09_validation/stage_a_evidence_receipt.json) contains the complete original selection and run records. Its SHA-256 is `a1d53e0b064e812317d93e6e8749bae4728730fecfbf359edf0249f2a72f6ee4`. It binds selection hash `16ac3fc99d8ff9a193465ae9eb2f6d0f760f54fa8dc715b1e4b30055cb5b3961` and complete run-record hash `f7cade3f1ee03361c9c46a8e1cfe376ec72adf2440f6fa6a275fa915e169cb77` to the audited source/archive. The scientific audit's SHA-256 is `1ee4f118ed846d767dae6a7ed58910f15412946611ebd196a31141c01f3d7974`.

## 2. Development data and controls

The fixed two-event task uses eight explicit event–role–filler triples on a 10×8 grid, with four distinct nouns, two verbs and two places. The audited collections contain **2,048 training bags**, **128 validation bags**, a **64-bag training probe**, and a **four-bag memorization fixture**. Validation has 1,024 base-query rows; the probe has 512; memorization has 32. Probe and memorization bags are training-derived and are not held-out evidence.

The canonical bag hash assigns partitions before sampling. All assignments, layouts, queries and transformations of a bag stay in its partition. The ledger prospectively excludes **100 inspected fixture candidates, including 31 constructed fixture groups**. The nominal 623,700-bag universe precedes those exclusions. The source-defined 70/15/15 buckets are procedural proportions, not guaranteed exact finite counts. Individual tokens and smaller combinations intentionally recur across partitions; new bags do not constitute new grammar rules or new event counts.

The independently checked validation controls gave these global correct-answer probabilities:

| Control | Global validation score | Interpretation |
|---|---:|---|
| Symbolic visible-input lookup | 1.0 | Exact grammar-informed lookup |
| Role-only matching | 0.5 | Ignores which event is queried |
| Event-aware category matching | 0.75 | Does not distinguish the two noun roles |
| Global bag/category matching | 0.375 | Ignores event and noun-role associations |
| Training-only role majority | 0.091796875 | Fixed from training answers; no validation fitting |

The corresponding analytical binding expectations are 0.5 for role-only and event-category, and 0.25 for bag-category. Uniform controls are scored as exact answer probabilities, not Monte Carlo predictions. Symbolic correctness validates this constrained interface; it is not a trained, capacity-matched neural result. The archive contains no final-data directory, final-access receipt or final predictions.

## 3. Memorization diagnostics

Both families used initialization 38, LR 0.003 and the same fixed 32 training rows, with a maximum of 4,096 updates. The criterion required two consecutive checks, 128 updates apart, with 32/32 correct and mean CE ≤0.05. Both completed successfully:

| Family | Qualifying checks | Stopping update | Final correct | Final binding / all-eight | Final CE | Run wall seconds |
|---|---|---:|---:|---|---:|---:|
| NeuroPixel | 1,280 and 1,408 | 1,408 | 32/32 | 1.0 / 1.0 | 0.000014673004459641427 | 100.254104 |
| RelativeTransformer | 128 and 256 | 256 | 32/32 | 1.0 / 1.0 | 0.002273512198441699 | 4.488987 |

The first qualifying CEs were approximately 0.00005796461823 and 0.008334721069, respectively; the original full-precision checks are retained. Every final role accuracy was 1.0. The final logits were independently recounted; intermediate stopping checks were verified from their saved metrics, not by rerunning their checkpoints.

These results show that both implementations can fit this tiny repeated training fixture under the diagnostic recipe. They do not establish generalization across randomized assignments, unseen bags or counterfactual conditions. The different stopping times are single diagnostic observations, not an architecture-efficiency estimate. A failure to reach this criterion would have been a flag under the protocol; a technical failure would have blocked the normal handoff. Neither occurred here.

## 4. All four pilots and the frozen selection

Every pilot completed 1,024 updates at initialization 39, with batch 32, answer cross-entropy, AdamW, weight decay 0.0001 and clipping at 1.0. Parameter counts were 29,856 for NeuroPixel and 30,157 for RelativeTransformer; proximity in count does not equate computation or optimization difficulty. All four saved 1,024-update input streams matched, with aggregate hash `1c7f0d66a000efd142bc04fbbce07dd983d63c245e1d2a461257126f2979769c`. This establishes agreement of recorded input hashes; no training canvases, initializations or firing masks were regenerated by the audit.

Accuracies below are proportions, not percentages. Displayed CEs are rounded to six decimals; selection used the original unrounded values. Validation role denominators are 256 each, binding 512 and global 1,024.

| Family | LR | Agent | Action | Patient | Place | Global | Binding | CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NeuroPixel | **0.001** | 0.10546875 | 0.12890625 | 0.09375 | 0.16015625 | 0.1220703125 | **0.099609375** | 2.422488 |
| NeuroPixel | 0.003 | 0.10546875 | 0.1171875 | 0.08203125 | 0.16015625 | 0.1162109375 | 0.09375 | 2.386589 |
| RelativeTransformer | 0.001 | 0.2421875 | 0.4921875 | 0.23046875 | 0.5078125 | 0.3681640625 | 0.236328125 | 1.358217 |
| RelativeTransformer | **0.003** | 0.25390625 | 0.4921875 | 0.234375 | 0.5078125 | 0.3720703125 | **0.244140625** | 1.388715 |

The fixed training-probe outcomes and complete run costs were:

| Family | LR | Probe global | Probe binding | Probe CE | Binding below 0.95 | Run wall seconds |
|---|---:|---:|---:|---:|---|---:|
| NeuroPixel | 0.001 | 0.115234375 | 0.09375 | 2.408765 | Yes | 78.684197 |
| NeuroPixel | 0.003 | 0.109375 | 0.08984375 | 2.375870 | Yes | 77.594396 |
| RelativeTransformer | 0.001 | 0.359375 | 0.23046875 | 1.347595 | Yes | 23.587887 |
| RelativeTransformer | 0.003 | 0.37109375 | 0.25 | 1.399262 | Yes | 23.948400 |

All four pilots had **zero all-eight-correct scenarios** on both validation and probe. Full role/probe metrics and arrays remain in the linked scientific audit and raw archive.

The ranking is binding descending, CE ascending only on a binding tie, then LR ascending. Therefore NeuroPixel selects 0.001: 0.099609375 exceeds 0.09375. RelativeTransformer selects 0.003: 0.244140625 exceeds 0.236328125. **CE favors the opposite rate in both families**, but neither comparison is a binding tie, so CE cannot override the primary ranking. No rate was chosen using speed, probe or memorization performance.

NeuroPixel's pilot binding lies below the 0.25 bag-category expectation; the Transformer's lies near it. Together with the zero all-eight rates, these observations do not establish reliable event–role conjunction. They also do not identify why learning is limited. The four runs use one development initialization and selected validation data; no training-population interval or final superiority claim follows. All four low-probe flags are retained without changing the planned budget.

## 5. Operational admission cost

The [prospective cost record](../../results/research/09_validation/stage_b_cost_projection.json), SHA-256 `27d74fa8b9738f27cf758b366655e9f70b55c879a9dca9611740d2b0eebf3a09`, uses the slower observed per-update time and the larger endpoint overhead **across both rates within each family**. It preserves 40,960 optimizer updates and 122,880 nominal final query rows.

| Projection component | NeuroPixel seconds | RelativeTransformer seconds |
|---|---:|---:|
| Five primary runs, including endpoint proxy | 1,559.865946 | 474.133536 |
| Five final panels, approximate inference/persistence proxy | 37.448265 | 12.891921 |

Total model-work projection is 2,084.33966663 seconds. Applying a factor of two, then adding 900 seconds for tests, 300 for data/source/setup, 600 for intermediate archival and 180 for final archival yields **6,148.67933326 seconds**, about **102.48 minutes**. This includes the reserve and fits the 13,800-second worker limit; the corresponding execution portion also fits its 13,620-second pre-reserve window.

The slower timing rule avoids selecting favorable observations. Endpoint overhead includes checkpointing and 1,536 probe/validation rows; extrapolating it eightfold to 12,288 final rows is a planning approximation, not measured final inference. The factor two and allowances are operational margins, not confidence bounds. RAM/deadline guards remain authoritative. Stage A's total worker duration was 327.752899868 seconds before final archive, not an isolated model-training benchmark.

The scientific helper also reports a **2,020.04269056-second selected-rate median-delta projection**, excluding several costs. That number is descriptive only and is not the admission calculation. The cost record's `performance_data_recount_pending=true` reflects its earlier creation at 08:02 UTC; the subsequently completed 08:09 scientific audit resolves that pending step without rewriting the original record.

## 6. Admission conditions and remaining limits

The source/scientific audits and code reviews have no open blocker. Proceed with both families, selected rates **0.001/0.003**, seeds **40–44**, **4,096 updates each**, and the unchanged data, supervision, inference and final-condition inventory. First freeze the reviewed analysis implementation/tests, exact Stage-A receipt and Stage-B plan. Do not extend budgets, replace seeds or select checkpoints after seeing results.

The offline analysis runtime is separately pinned prospectively to **Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0**, one numerical thread. It does not change neural training/inference. Its declared PCG64 seed 94001, 2,000 shared scenario resamples, saved exact index artifact and linear percentile endpoints identify the actual analysis; no cross-version sampling identity is assumed. See the [analytical addendum](09_analytical_addendum.md).

Primary uncertainty concerns five paired **training realizations**, varying initialization and input/firing streams, with a paired t interval at df=4. Scenario-bootstrap intervals are conditional on fixed checkpoints and answer a different question. The competence screen and six-condition diagnostics remain future final outcomes. All ten completed checkpoints must be inventoried and archived before the consume-once final gate opens. Current checkpoint audit coverage is byte/container verification, not newly deserialized tensor equivalence. This is an internal team validation, not external replication or custody, and it establishes no capability beyond the explicitly bounded task.
