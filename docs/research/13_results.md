# Item 13 — Vocabulary expansion and distractors: completed feasible investigation

**Outcome:** the native vocabulary-allocation and battery-helper defects have been corrected and passed the declared regression tests. The historical “new word” evaluation does not identify learning a new meaning: its positive-only new panel admits a complete presence shortcut. An exhaustive transformed-triple census also shows why the original partition does not establish disjointness after a noun is replaced. The saved-output distractor analysis preserves the previously observed weak binding competence.

This closes the feasible investigation of item13. Semantic acquisition from a few examples and competent generalization with novel distractors remain unestablished scientific capabilities. Item5's frozen H1 remains closed and not supported. No work on item14 is included here.

## Evidence and execution identity

| Record | Identity |
|---|---|
| Source frozen before execution | 9618ba8aae3b83d95a2813f99dd72c9d2e1d148a |
| Freeze | 2026-10-07 14:55:04 UTC |
| Protocol | [13_probe_protocol.md](13_probe_protocol.md) |
| Plan SHA-256 | 60278a7580dabaaa1fb338a0276819d0c653a9052ed33c51d5793a14edd627da |
| Source bindings | 29 |
| Authorized public CPU run | [37640756510](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37640756510), job 112858763837, successful |
| Final immutable result commit | e16d0ab07a5aff93c66c4057a6509cc31b744160 |
| Result directory | results/research/13_cloud_runs/37640756510-1-probe |
| Final manifest SHA-256 | 425234cd242d2c2aac53d1ea149a5d2e76ffec7ee381d3977028337f7209c1c3 |
| Completion and verification receipt | [probe_completion_receipt.json](../../results/research/13_validation/probe_completion_receipt.json) |

The source changed only on the isolated research branch. One numerical workflow ran; no new task-training study, checkpoint loading, confidence interval or hypothesis selection was performed. Small optimizer steps in the regression contracts are software checks and are accounted for separately from the untrained native diagnostic.

The [final archive](https://github.com/Agnuxo1/NeuroPixel/tree/e16d0ab07a5aff93c66c4057a6509cc31b744160/results/research/13_cloud_runs/37640756510-1-probe) contains 27 files totaling 10,981,004 bytes before its 4,356-byte manifest. It includes complete contract logs, source ZIP, native raw JSON, the saved-prediction recount and its independent audit. The source ZIP is 9,296,916 bytes, SHA-256 897b15a4ef902c7cbfdf561dc1c6f802e4a26bfba94e51e5d284375bcd937275.

## 1. Vocabulary allocation: reproduced defects and verified corrections

The historical phase3 helper was preserved as a complete source preimage. Its relevant function was extracted by AST and run against the current core. This isolates the allocator defect; it is not a replay of the historical training environment.

| Check | Preserved legacy result | Corrected result and scope |
|---|---|---|
| Existing grounding | Two enabled grounding-mask rows became zero enabled rows after expansion; the old effective dictionary changed by a maximum absolute 2.4441168308258057 in the fixture. | Old RGB and mask prefixes preserved exactly; appended rows ungrounded and zero. |
| Float64 NP | Forward raised a mismatch between float input and double bias. | Old-input forward works with the original float64 modules/tables. |
| Float64 Transformer | Forward raised a Double/Float matrix mismatch. | Embedding, head and bias preserve their original dtype. |
| Old parameters/configuration | The phase3 helper copied most old parameters, while its new tables lost dtype/grounding. | Old table prefixes, non-vocabulary parameters, model-level training flag and configuration preserved in declared NP/TF fixtures. |
| Battery reconstruction | Nondefault NP and rectangular/nondefault Transformer raised state-dictionary configuration mismatches. | Battery delegates to native expansion and preserves the supplied model configuration, model-level training flag, device, dtype and grounding. |
| Fresh optimizer with zero decay | This is the supported historical caller pattern. | One Adam step changed appended rows while all old rows and other tensors stayed exactly fixed. |
| Optimizer-decay negative control | Zero old-row gradients do not prevent decay. | AdamW with deliberately nonzero decay still changed old rows, as expected; this limitation is documented. |

The actual legacy exceptions and labels are in [contract_tests.log](https://github.com/Agnuxo1/NeuroPixel/blob/e16d0ab07a5aff93c66c4057a6509cc31b744160/results/research/13_cloud_runs/37640756510-1-probe/contract_tests.log). Successful tests retain these witnesses because the worker ran pytest with output capture disabled.

The corrected native function retains its return API: copied model, tensors admitted to adaptation, and first appended ID. Boolean row masks follow the gradient's device. Repeated expansion and a conversion to half precision were checked using a gradient-only fixture. This does not establish half-precision convolution or GPU behavior.

The original phase3 float32 initialization and RNG consumption remain equal in the tested valid native paths. Battery deliberately migrates from its former reconstruction-by-defaults routine to that policy; its older initializer/RNG sequence is not claimed identical.

Grounding buffers remain nonpersistent. A state dictionary alone still cannot reconstruct their values, and reconstruction for further adaptation must reestablish gradient-hook policy. Existing optimizers are not migrated and still reference their original parameters. Fresh optimizers with no decay or inherited momentum are the documented scope of the old-row gradient restriction.

The saved newword callers reconstructed models without explicit grounding. The grounding defect is therefore a verified implementation problem, but it is not attributed to their archived scores. Also, core n_params counts entire tensors marked requires_grad, whereas the hook admits gradients only for appended rows. Neither storage nor arithmetic automatically shrinks to the number of new-row scalars.

## 2. The stale-target branch is corrected

The battery helper previously changed one noun filler while preserving the original target when ask_new=False. A manual census enumerated all four queries and both possible insertion roles, for eight combinations.

- Legacy targets: [5,17,6,27,5,17,6,27].
- Visible correct targets after replacement: [35,17,6,27,5,17,35,27].
- Legacy incorrect rows: 0 and 6, or 2/8.
- Corrected incorrect rows: 0/8.
- The question itself was preserved in the unforced-query branch.
- The forced-query branch retained exactly the legacy inputs and targets in the same fixture.

This is a complete manual query-role/insertion-role census. It is not a random dataset score, and its 2/8 result should not be reported as an observed error rate in historical trained-model outputs. Historical t2_new_word used the default ask_new=True branch.

The corrected false branch means “retain the original question.” It can legitimately ask about the newly inserted token, so it is not, by itself, a pure irrelevant-token control. The separate diagnostic constructs and verifies that control explicitly.

## 3. What the historical five-example result establishes

The historical newword JSON records the selected np_lens30k/norma/k5 condition as:

| Historical quantity | Recorded value |
|---|---:|
| New-token accuracy |98.6% |
| Old-token accuracy after adaptation |92.6% |
| Old-token accuracy before adaptation |97.95% |
| Old-accuracy change |−5.35 percentage points |

These are preserved reported aggregates. No archived per-example prediction set or adapted checkpoint was available for an independent recount of that condition.

Other recorded settings matter to interpretation. Plain five-shot adaptation on the np_lens30k base was 90.0% new / 81.3% old; on the np_reposo base it was 92.4% new / 94.8% old. In the scaling driver's five adaptation trials, one 30k NP base reported 84.68% new, range 74.8–94.7%, while another reported 59.8%, range 44.1–77.7%. The 44k Transformer base reported 61.22%, range 40.1–86.7%.

The README's selected 98.6% uses norm-constrained adaptation, while its Transformer comparison draws on the separate plain scaling adaptation procedure. Those numbers do not form a matched method comparison. Five adaptations of a fixed base also are not five independently trained base models. Scaling checkpoints were saved before adaptation, so their presence cannot reproduce the subsequent five adapted predictions by simple recount.

Most decisively, both historical phase3 new-word drivers insert nid, force the query to the role containing it, and set every new-panel target to nid. Thus:

- A constant nid predictor solves 100% of the positive-only new panel.
- A presence gate that answers nid whenever it appears can solve that panel while leaving an existing predictor's outputs unchanged on the separate old panel, which contains no nid.
- Neither score identifies the role-binding rule used by the actual trained model.

This is a constructive failure of identification. It does not prove that NeuroPixel actually chose the shortcut. A nontrivial test needs queries whose answers remain old while the new token is present, multiple competing new tokens, and verified compositions that were not used in adaptation.

Historical evidence: [newword.json](../../results/phase3/newword.json), [scale_np30k.json](../../results/phase3/scale_np30k.json), [scale_np30k_s1.json](../../results/phase3/scale_np30k_s1.json), [scale_tf44k.json](../../results/phase3/scale_tf44k.json), and the frozen [driver](https://github.com/Agnuxo1/NeuroPixel/blob/9618ba8aae3b83d95a2813f99dd72c9d2e1d148a/scripts/phase3.py).

## 4. Constructed controls expose the missing distinctions

The frozen native diagnostic constructed 32 scenes, all four role questions and six conditions: 768 examples. A parser scanned the visible unique role-marker/filler pairs and verified all 768 gold labels independently of the generator metadata. Unpaired old/new tokens occupied blank cells outside the query/output row and preserved every gold.

The table shows literal baseline scores, not learned model performance. Each condition has 128 questions.

| Condition | Constant new token 35 | Presence 35 gate with symbolic fallback | Symbolic solver |
|---|---:|---:|---:|
| Old-token base |0/128 |128/128 |128/128 |
| New token as agent; all four questions |32/128 |32/128 |128/128 |
| New token as patient; all four questions |32/128 |32/128 |128/128 |
| Two distinct new tokens in agent/patient roles |32/128 |32/128 |128/128 |
| Unpaired old-token distractor |0/128 |128/128 |128/128 |
| Unpaired new-token distractor |0/128 |0/128 |128/128 |

Restricting a single-new-role condition to the 32 questions whose target is 35 yields 32/32 for the constant predictor—the same favorable property as the historical positive-only panel. Keeping the other 96 questions exposes failure to answer old targets in the new token's presence. With two new tokens, the same predictor answers only 32/64 new-target questions correctly.

The fallback is deliberately symbolic: it constructs a predictor compatible with success on old-only and new-positive panels. Its 100% base score is not assigned to a trained NP. Similarly, no trained model was newly evaluated on the unpaired-token fixtures, so these validated labels establish a usable control set rather than resistance to those perturbations.

Raw canvases, targets, controls and counts are in [native/report.json](https://github.com/Agnuxo1/NeuroPixel/blob/e16d0ab07a5aff93c66c4057a6509cc31b744160/results/research/13_cloud_runs/37640756510-1-probe/native/report.json).

## 5. Replacing a noun can destroy the original split distinction

The current legacy split recipe contains 1,056 training triples and 264 test triples, with no original overlap. The exhaustive projection census produced:

| Transformation of original triple | Unique TRAIN keys | Unique TEST keys | Shared unique keys |
|---|---:|---:|---:|
| Replace agent |120 |104 |104 |
| Replace patient |120 |110 |110 |
| Replace both nouns |10 |10 |10 |

Every transformed TEST key in this full-pool census has a counterpart in transformed TRAIN. Erasing a noun collapses original triples that were distinct before the transformation.

These keys use the task's internal noun/action indices plus replacement sentinels 35/36. They are not literal canvas-token triples. This is a statement about the complete possible triple pools under the current recipe; it is not evidence that particular historical five-shot support canvases and test canvases were identical. The actual five examples, places, layouts, software version and old random draws were not reconstructed. A future test must split or audit the transformed semantic objects and retained examples themselves.

## 6. Preserved old logits do not preserve the expanded distribution

Let Z be the sum of exponentiated old logits and A the corresponding sum for appended logits. With unchanged old logits:

p'_i = p_i Z/(Z+A), and ΔNLL = log(Z+A) − log Z.

Reconditioning on the old output classes recovers their earlier distribution. The order among old classes remains the same, while the full argmax can move to an appended class. This is an output-space effect and applies to untied as well as tied readouts.

The two literal float64 fixtures supplied four rows in total. Their NLL increases were:

- 0.091177481793009 and 1.436419351814674 in the one-new-class fixture;
- 7.651464075448163 and 1.6835965184923976 in the two-new-class fixture.

New-class probability masses ranged from 0.08714431874203257 to 0.9995246523255441. Three of the four full argmax answers moved to a new class. All old logits were unchanged, and the declared identities passed.

These literal values are arithmetic counterexamples, not NeuroPixel accuracy or training outcomes. No embedding-average initialization theorem is claimed for the special PAD readout. Hewitt's author technical note derives the normalizer issue; this report extends the displayed identity to several new classes under the stated unchanged-logit condition.

## 7. Renaming is a checked representation identity

The diagnostic jointly permuted noun IDs, input IDs and all corresponding model tables. PAD/role IDs remained fixed. NP grounding RGB and mask rows were also permuted; TF input/output tables and bias rows followed the same permutation.

In the two-input float64 fixtures, the maximum aligned-logit differences were 5.551115123125783e−17 for NP and 1.1102230246251565e−16 for TF. NP state was exactly identical. Both met the declared 1e−12 logit tolerance.

These were tiny untrained models. NP's residual layer was manually activated with small nonzero values, avoiding a quiescent update fixture. No semantic correspondence was learned. The result verifies consistent coding and table migration; it supplies no evidence that an unchanged model understands an unfamiliar symbol.

## 8. Existing irrelevant facts: independent raw-prediction recount

The original item9 scientific source is 83fe135e301f76bc0c74e30c66bb18e067ca5959, with raw archive b64d0e2ba222df76caf72bc9870c7602873e7236. Twelve selected raw files totaled 23,928,770 bytes. Both new programs reopened those exact NPZ files; neither loaded model checkpoints.

The selected transformation exchanges agent and patient only in the event that is not queried. Independent visible-scene parsers verified the exact intervention, same query/layout, group/record identities and unchanged gold. The base already includes that other event. This is perturbation of existing irrelevant facts, with 2,048 paired questions per saved training realization and 256 shared bags.

The following percentages are means over the five saved training realizations in each family. “Retained if base correct” is the mean of the five conditional ratios, not a ratio of pooled counts.

| Family | Questions | Base correct% | Transformed correct% | Both correct% | Same prediction% | Same wrong prediction% | Retained if base correct% |
|---|---|---:|---:|---:|---:|---:|---:|
| NeuroPixel | all | 10.9863 | 10.9180 | 10.4688 | 93.2617 | 82.7930 | 95.4003 |
| NeuroPixel | binding | 9.6289 | 9.5508 | 9.0820 | 90.8203 | 81.7383 | 94.4062 |
| Relative Transformer | all | 73.5156 | 73.5059 | 73.0762 | 98.9648 | 25.8887 | 99.4027 |
| Relative Transformer | binding | 49.3555 | 49.3555 | 48.5352 | 98.0273 | 49.4922 | 98.3446 |

For binding questions, NeuroPixel retains about 94.41% of its initially correct answers, but only about 9.63% were initially correct. Around 81.74% of all binding pairs have the same wrong answer in both versions. Prediction stability therefore coexists with poor binding competence. The Transformer retains about 98.34% of initially correct binding answers, with 49.36% base accuracy; it also remains limited on binding. These descriptive results show no learned robustness advantage for NP in this contrast.

### Exact per-realization binding counts

Each row has N=1,024. CC/CW/WC/WW mean correct→correct, correct→wrong, wrong→correct and wrong→wrong. The latter includes wrong answers that change, while “same wrong” counts only identical wrong answers.

| Run | Base correct | Transformed correct | CC | CW | WC | WW | Same wrong | Retained if base correct% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NP 40 | 100 | 100 | 97 | 3 | 3 | 921 | 887 | 97.0000 |
| NP 41 | 103 | 101 | 98 | 5 | 3 | 918 | 854 | 95.1456 |
| NP 42 | 94 | 94 | 91 | 3 | 3 | 927 | 871 | 96.8085 |
| NP 43 | 105 | 104 | 93 | 12 | 11 | 908 | 705 | 88.5714 |
| NP 44 | 91 | 90 | 86 | 5 | 4 | 929 | 868 | 94.5055 |
| TF 40 | 490 | 488 | 482 | 8 | 6 | 528 | 524 | 98.3673 |
| TF 41 | 513 | 513 | 510 | 3 | 3 | 508 | 508 | 99.4152 |
| TF 42 | 519 | 516 | 500 | 19 | 16 | 489 | 485 | 96.3391 |
| TF 43 | 509 | 512 | 505 | 4 | 7 | 508 | 501 | 99.2141 |
| TF 44 | 496 | 498 | 488 | 8 | 10 | 518 | 516 | 98.3871 |

The complete 60-row table additionally contains all-role and each individual-role results, all numerators/denominators, corrected-error rates and null handling. No new confidence interval, significance test or final-set selection was added. Same bags across variants and seeds remain shared data; the rows are not independent new datasets.

See [secondary/report.json](https://github.com/Agnuxo1/NeuroPixel/blob/e16d0ab07a5aff93c66c4057a6509cc31b744160/results/research/13_cloud_runs/37640756510-1-probe/secondary/report.json), [transitions.csv](https://github.com/Agnuxo1/NeuroPixel/blob/e16d0ab07a5aff93c66c4057a6509cc31b744160/results/research/13_cloud_runs/37640756510-1-probe/secondary/transitions.csv) and [independent audit](https://github.com/Agnuxo1/NeuroPixel/blob/e16d0ab07a5aff93c66c4057a6509cc31b744160/results/research/13_cloud_runs/37640756510-1-probe/secondary_audit/audit.json).

## 9. Verification, resources and remaining limits

All 12 parent contract methods and 20 additional subtest calls passed: 32 successful call reports, zero failures and zero skips. The JUnit aggregate reports 32 tests, with 12 parent testcase elements; these are two representations of the same run. Contract time in JUnit was 1.131 seconds.

The native program passed eight declared checks. Its saved JSON was independently recounted by the frozen JavaScript auditor: 13,215 checks, 88 numeric comparisons, no issues, maximum arithmetic discrepancy 2.6645352591003757e−15. This recount checks labels, counts, split intersections, softmax and aligned saved outputs; it does not rerun Torch inference.

The fresh independent NumPy audit completed 32,954 checks with no issues. It revalidated raw NPZ identities, visible transformations, predictions, targets, counts, summaries, CSV and saved paired outcomes. It used an event-start parser, Counter transitions and centered logaddexp NLL, independently of the producer's row parser and vectorized calculations. Maximum selected-NLL difference was 2.220446049250313e−15.

Root also reconstructed 60 transition rows from the previously verified item9 projection and compared the new producer and auditor: 2,641 checks, 2,257 numeric comparisons, zero difference. This is additional agreement with a prior audit, not independent external replication.

The root rehashed all 25 textual files in the current archive, matching every byte count and SHA-256, and checked the complete Git run-file inventory, modes and binary sizes. Binary descriptors for source.zip and paired_outcomes.npz were matched to the manifest. The root V8 environment did not independently unzip or deserialize those binary payloads. The standalone NumPy auditor did validate the paired NPZ against the original selected raw arrays. The original entire item9 archive custody audit was not repeated.

The worker took 20.74881701000001 seconds before final archival; the probe controller took 4.74119417 seconds. Its three serial component processes took 1.068145454, 1.676591634 and 1.169377399 seconds. These are diagnostic wall times, not comparable model training/inference benchmarks.

Eleven one-second supervisor observations showed a minimum available RAM of 14.216365814208984 GiB. Numerical and interop threads were 1 under an aggregate active-CPU ceiling of 4. No GPU or paid computation was requested. All stages finished within the frozen budgets and the runner completed successfully. Source and selected-input hashes stayed unchanged.

Before execution, reviewers corrected an optional-import test default, centered the independent audit's NLL, and made root JSON object comparison insensitive to key insertion order. Their preimages and correction receipts remain in results/research/13_validation. Those were static defects; they were not observed failures in the saved scientific predictions. The legacy failures exercised in regression tests remain visible as expected controls.

## 10. Achieved and still missing

| Item13 objective | Assessment |
|---|---|
| Preserve native old rows/configuration/grounding during expansion | Verified on the declared CPU fixtures; code corrected. |
| Preserve valid float32 phase3 initialization behavior | Verified for the declared two native model paths. |
| Correct labels after an unforced new-token insertion | Verified by the complete eight-case manual role census. |
| Validate controlled new-token/irrelevant-token gold |768 constructed examples verified. |
| Distinguish old logits from expanded softmax probabilities | Verified analytically and in literal fixtures. |
| Consistent simultaneous ID/table renaming | Verified in both tiny model families; representation identity. |
| Audit transformed triple split | Exhaustive census completed; original split is insufficient after replacement. |
| Recount previously observed irrelevant-event performance | Completed and independently checked; weak initial binding retained in the interpretation. |
| Acquire a new semantic correspondence from a few examples | Unestablished; positive-only scores do not identify it. |
| Generalize that correspondence to withheld compositions with competing new tokens | Unestablished; requires retained supports, transformed-object split checks and a competent base. |
| Resist new distractors while preserving high task competence | Unestablished; validated control labels are available, but no such learned result was demonstrated. |
| Reproduce the historical five-shot adapted models exactly | Unestablished; aggregate summaries do not replace missing adapted weights/predictions. |
| Independent external replication or transformative scientific utility | Not established by this internal code-and-evidence audit. |

The next investigation in the agreed sequence is item14, assisted repair versus stored memory. It opens only after this point's closure is recorded.

## Primary-source context

The following sources were read directly on 2026-10-07, with a focused review, without a claim of systematic coverage:

- [Press & Wolf, 2017, section 3](https://arxiv.org/pdf/1608.05859): input/output weight tying couples the two uses of a vocabulary table. Its language-model results do not establish a NeuroPixel advantage.
- [Hewitt, 2021](https://www.cs.columbia.edu/~johnhew/vocab-expansion.html): author's technical note explaining probability changes after vocabulary expansion. It is not a peer-reviewed demonstration of semantic acquisition.
- [Lake & Baroni, 2018, experiment 3](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf): learning a primitive mapping and composing it in previously unseen contexts are distinct evaluations.
- [Shi et al., 2023, sections 3.1–3.2](https://arxiv.org/pdf/2302.00093): verify that irrelevant additions preserve the target and retain grouping of variants.
- [Mirzadeh et al., ICLR 2025, arXiv v2 dated 2025-08-27](https://arxiv.org/pdf/2410.05229): separate types of symbolic transformations and validate their targets; its model-specific results are not NeuroPixel baselines.

No priority claim or Nobel-level discovery follows from vocabulary allocation, coordinate renaming, passing software contracts or recounting low-competence predictions.
