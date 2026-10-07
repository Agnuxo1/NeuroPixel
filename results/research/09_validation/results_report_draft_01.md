# Item 9 — Complex role binding: completed study

**Evidence status (2026-10-07):** Stage A and Stage B are complete. The full archive, final-access gate and scientific saved-array audits are verified with zero reported issues. This report is prepared for independent editorial review and archival closure of item 9; it does not initiate item 10.

## 1. Scope and main findings

This study asks whether two small neural models can answer explicit event–role queries when a scene contains two events and eight distinct fillers, and whether their answers behave appropriately under controlled changes to the scene or query. It compares NeuroPixel with a relative-position Transformer under a fixed training and evaluation recipe. The task extends the earlier single-event setting; it does not test natural-language parsing, repeated entities, anaphora or recursive structure.

The development stage established that both implementations can memorize the same 32-example training fixture. All four scheduled learning-rate pilots completed, received identical training inputs, and were audited. The frozen selection rule chose learning rate **0.001 for NeuroPixel** and **0.003 for the Transformer** using development binding accuracy. Development performance remained far below the declared competence thresholds. These observations justified proceeding with the unchanged study after its technical and cost checks; they did not establish generalization or explain the reason for low pilot accuracy.

The final study consists of five paired training realizations per family, each trained for 4,096 updates. Its primary comparison is base-condition macro accuracy on the agent and patient roles. **NeuroPixel achieved mean base binding accuracy 0.0962890625, compared with 0.4935546875 for the Transformer. The paired difference was −0.397265625, with a t 95% interval of [−0.4125410251515038, −0.3819902248484962] across five training realizations.** Neither family demonstrated high competence on this task. All three components of the prespecified NeuroPixel competence screen failed. The item-5 H1 decision remains closed and is not reconsidered here.

The evidence establishes successful execution of the declared study, a consistent advantage for the Transformer on the primary comparison under these recipes, and substantial remaining errors in resolving agent/patient roles. It does not establish a generally superior architecture, an internal variable-binding mechanism, or an external replication.

## 2. Task, grouping and controlled interventions

Each 10 × 8 symbolic canvas contains eight explicit event–role–filler facts: two events, each with agent, action, patient and place roles. The query specifies an event and a role. Four nouns, two verbs and two places are distinct within each scene, so switching a queried event or exchanging its agent and patient has an unambiguous target consequence. The vocabulary has 37 entries, including the two event markers. Fact rows and horizontal positions vary; the output location is fixed.

The partition unit is the canonical category bag of four nouns, two verbs and two places. There are 623,700 possible bags. A deterministic SHA256-based assignment allocates hash buckets 0–69 to training, 70–84 to development and 85–99 to final evaluation. These are procedural allocation proportions, not a claim that every finite sample has exactly 70/15/15 membership. All assignments, layouts, queries and interventions of one bag remain in the same partition.

The recipe selects 2,048 training bags, 128 development bags and 256 final bags. Training varies assignments, layouts and queries within its fixed bag set. Development selection uses only the 1,024 base queries from its 128 bags. A 64-training-bag probe supplies 512 queries. The four-bag memory fixture supplies 32 fixed training queries. A prospectively recorded fixture-exposure ledger excludes 100 candidate bags, including 31 constructed fixtures, from final selection. This protects the particular seed-91003 final collection against the documented fixtures; it does not make the task generator, vocabulary or project history unknown.

The final panel contains all eight queries for every bag in all six conditions: 12,288 nominal rows per checkpoint.

| Condition | Intervention | Expected target relation |
|---|---|---|
| Base | Original scene and query | Reference answer |
| Queried-event agent/patient swap | Exchange the two noun fillers in the event selected by the query | Agent/patient answers change; action/place answers remain the same |
| Other-event agent/patient swap | Exchange the two noun fillers in the other event | All queried answers remain the same |
| Global event relabeling | Exchange event names in both facts and query | Answer remains the same |
| Query-event switch | Change only the event named in the query | Every answer changes because the fillers are distinct |
| Layout permutation | Move facts while preserving their content and the query | Answer remains the same |

**Interpretive clarification added after launch and before final outcomes were inspected:** the frozen renderer chooses the affected event separately for each query in both swap conditions. Each such condition combines four queries from each of two transformed fact grids. Its all-eight score is therefore joint correctness across eight **query-conditioned swap cases within a bag**, not eight answers about one fixed transformed scene. Base all-eight concerns one fixed scene, and the competence screen's all-eight threshold applies only to base. Targets, arithmetic, pairing and bag resampling remain unchanged. The separately preserved [interpretation guide](09_interpretation_guide.md) records the clarification's provenance.

Query-event switch also permutes the eight base queries. It contributes paired tests of whether predictions change appropriately, but it is not an independent set of new scenes. The expected dataset structure is 40 unique inputs per bag rather than 48: 32 occur once and eight twice. The saved-array audit confirmed 10,240 unique inputs: 8,192 occur once and 2,048 occur twice. For every checkpoint, all 2,048 repeated inputs had identical predicted classes and a maximum absolute logit difference of zero. This verifies duplicate consistency rather than adding independent accuracy evidence.

The declared controls are a visible-fact symbolic parser and marginal analytical references. A role-only rule that ignores event has expected global and binding accuracy 0.5. An event-plus-category rule that ignores agent/patient distinction has expected global accuracy 0.75 and binding accuracy 0.5. A category-only bag guess has expected global accuracy 0.375 and binding accuracy 0.25. A majority control is fitted only from training data with a fixed tie rule. The probabilistic controls specify marginal expectations, not fabricated predictions or unprescribed all-eight joint probabilities.

## 3. Models, selection, exposure and statistical estimands

NeuroPixel uses the corrected current implementation, with 16 token-embedding channels, 48 state channels, a hidden width of 128 and 16 recurrent updates. Training uses a 0.5 firing probability; evaluation updates all cells. PAD state is zeroed as specified. The Transformer uses width 32, two layers, four heads, feed-forward width 128, relative positional attention, key padding masks and zero dropout. Parameter counts are 29,856 and 30,157, respectively. Similar parameter counts do not equate architecture, recurrent work or optimization difficulty.

Both families use answer cross-entropy, AdamW, batch size 32, weight decay 0.0001 and gradient clipping at 1. Stage B uses the development-selected rates and the final scheduled checkpoint after 4,096 updates. There is no final-data choice of checkpoint, rate, budget or preferred seed.

Seeds 40–44 identify five paired training realizations. Within each seed, both families receive the same sampled training inputs. Initialization, the private minibatch stream and NeuroPixel firing vary with the realization; consequently, the five runs do not isolate initialization variance. Model parameter values and internal noise are not matched between families. Family differences describe the complete frozen training procedures on this task.

The Stage-A receipt authenticates full copies of the selection and preflight run records. Stage B validates their hashes and common recipe/data/context, recomputes the prescribed selection rule, and checks the rates against its plan. Final access requires all ten scheduled trainings to complete and the complete inventory to be archived before the exclusive final-access record. These software checks support an auditable workflow; they are not independent data custody or adversarial access control.

The primary score is base-condition macro agent/patient accuracy. All five family scores and the five paired NeuroPixel-minus-Transformer differences must be reported. The mean difference receives the prespecified paired t interval at 95%, with sample standard deviation and four degrees of freedom. This interval concerns variation across the five training realizations conditional on the selected recipes and the fixed final panel. Its small-sample assumptions and sensitivity to individual runs remain material.

The separate scenario bootstrap uses 2,000 PCG64 resamples with seed 94001, preserving all queries, conditions and checkpoints within each of the 256 bags. The same sampled bag indices support paired contrasts. This is conditional uncertainty across the sampled bags for the fixed trained checkpoints, not ten new training replicates or 12,288 independent scenarios. The exact index artifact, percentile convention and runtime must be preserved.

For each checkpoint and condition, global accuracy has 2,048 query trials, binding accuracy has 1,024 agent/patient trials, each role has 512 trials, and all-eight has 256 bag trials. A calculation that first averages within bag may instead expose a denominator of 256; its reported unit must remain explicit. For invariance/equivariance diagnostics, the denominator is the eligible unchanged-target or changed-target subset. No eligible cases yields null, not a perfect score.

Counterfactual success requires the base and transformed answers both to be correct. Equal predictions on invariant pairs, or unequal predictions on changed-target pairs, are insufficient by themselves: an incorrect model can satisfy either relation. The report therefore presents ordinary accuracy, paired joint correctness and target-conditioned prediction relations separately, with their denominators.

A constant binary bag metric can yield a degenerate percentile interval such as [0,0] or [1,1]. That is the prespecified empirical bootstrap summary; it does not establish an exactly zero or one population probability or guaranteed nominal boundary coverage. Numerators and denominators must accompany these cases without introducing a new retrospective test.

The descriptive competence screen requires every one of the five NeuroPixel runs to attain base binding accuracy at least 0.95, base all-eight at least 0.90, and binding accuracy at least 0.90 in every condition. It is distinct from the paired family comparison and from the previously closed H1. Secondary condition and counterfactual analyses are a complete diagnostic panel, not multiplicity-adjusted discovery claims. Full details remain in the [protocol](09_protocol.md), [recipe](09_experiment_recipe.json), [preflight admission review](09_stage_b_admission_review.md) and [interpretation guide](09_interpretation_guide.md).

## 4. Audited development results

### 4.1 Memory fixture

Both families met the frozen stopping rule: 32/32 correct with cross-entropy at most 0.05 on two consecutive checks, scheduled every 128 updates. These were repeated evaluations of the training fixture, not held-out generalization.

| Family | Initialization seed | Learning rate | Qualifying update checks | Stopped at update | Correct training queries | Final fixture cross-entropy | Recorded run wall, s |
|---|---:|---:|---|---:|---:|---:|---:|
| NeuroPixel | 38 | 0.003 | 1,280 and 1,408 | 1,408 | 32/32 | 0.000014673004459641427 | 100.254103828 |
| Transformer | 38 | 0.003 | 128 and 256 | 256 | 32/32 | 0.002273512198441699 | 4.488987127 |

This demonstrates that both implementations can fit this small, fixed fixture under the stated optimizer. The different stopping times do not measure general sample efficiency, and fixture success does not establish role-binding generalization.

### 4.2 All four learning-rate pilots

Every pilot used seed 39 and 1,024 updates. The following accuracies are proportions, without percentage conversion; displayed cross-entropy values are rounded to six decimals, with full precision preserved in the linked records.

| Family | LR | Dev agent | Dev action | Dev patient | Dev place | Dev global | Dev binding | Dev CE | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeuroPixel | 0.001 | 0.10546875 | 0.12890625 | 0.09375 | 0.16015625 | 0.1220703125 | 0.099609375 | 2.422488 | Yes |
| NeuroPixel | 0.003 | 0.10546875 | 0.1171875 | 0.08203125 | 0.16015625 | 0.1162109375 | 0.09375 | 2.386589 | No |
| Transformer | 0.001 | 0.2421875 | 0.4921875 | 0.23046875 | 0.5078125 | 0.3681640625 | 0.236328125 | 1.358217 | No |
| Transformer | 0.003 | 0.25390625 | 0.4921875 | 0.234375 | 0.5078125 | 0.3720703125 | 0.244140625 | 1.388715 | Yes |

| Family | LR | Training-probe global | Training-probe binding | Training-probe CE | Recorded run wall, s |
|---|---:|---:|---:|---:|---:|
| NeuroPixel | 0.001 | 0.115234375 | 0.09375 | 2.408765 | 78.684196716 |
| NeuroPixel | 0.003 | 0.109375 | 0.08984375 | 2.375870 | 77.594395589 |
| Transformer | 0.001 | 0.359375 | 0.23046875 | 1.347595 | 23.587887495 |
| Transformer | 0.003 | 0.37109375 | 0.25 | 1.399262 | 23.948400293 |

Development and probe all-eight were zero in every pilot. All four probes triggered the recorded below-0.95 binding flag. The flag describes the observed probe outcome; it does not establish insufficient budget as its cause. The train-fitted majority control attained development global accuracy 0.091796875.

The selection rule ranks development binding accuracy first, then cross-entropy only for ties, then the smaller rate. It therefore selected NeuroPixel 0.001 and Transformer 0.003 even though cross-entropy alone preferred the opposite rate in each family. These were not discretionary selections. All four training-input streams had SHA256 `1c7f0d66a000efd142bc04fbbce07dd983d63c245e1d2a461257126f2979769c`.

### 4.3 Admission, checks and cost

The independent development data/metric audit passed 138,074 checks with zero issues. The corrected archive verifier passed 861 checks with zero issues. These are heterogeneous audit checks, not additional examples or experimental replications. The cloud suite passed all 34 test methods with no failures or skips. Its JUnit counter of 137 reflects the emitted test representation and must not be described as 137 distinct planned methods.

The cost admission calculation used the slower observed pilot rate in each family rather than only the selected-rate timings. It projected 2,084.33966663 seconds for the ten scheduled trainings, doubled that estimate, and added fixed test, setup, archive and reserve allowances: **6,148.67933326 seconds** in total, approximately 102.48 minutes. The helper's selected-rate median projection of 2,020.04269056 seconds was descriptive and was not the controlling admission estimate. No budget was changed in response to pilot accuracy.

The development worker occupied 327.752899868 seconds before its final archive step, from 07:51:50.481646 to 07:57:18.234401 UTC on 2026-10-07. Its recorded minimum available RAM was 14.200397491455078 GiB. Recorded per-run wall times above have a different scope from total worker elapsed time and must not be added to that total as separate costs.

These results supported admitting the unchanged ten-run study after freezing its final analysis code, tests and plan. A flag that a training fixture or probe failed a competence threshold would have been scientific information under the declared design; missing runs, invalid evidence or safety failures would have required operational resolution. Stage A met the actual memory criterion and all four pilots completed.

The complete development account is [09_preflight_results.md](09_preflight_results.md). Its evidence is [the data/metric audit](../../results/research/09_validation/preflight_audit_01.json), [archive audit 03](../../results/research/09_validation/preflight_archive_audit_03.json), [Stage-A evidence receipt](../../results/research/09_validation/stage_a_evidence_receipt.json) and [cost projection](../../results/research/09_validation/stage_b_cost_projection.json).

## 5. Final study results

### 5.1 Completion, integrity and controls

All ten scheduled training realizations completed 4,096 updates, for 40,960 total updates, and all ten final prediction files were retained. Each paired seed received the same training-input stream across families; five separate stream hashes verify that record. The final panel contains 256 bags, 12,288 nominal rows and 10,240 unique canvases per checkpoint. The scientific audit independently verified visible targets, transformations, group membership, fixture exclusions, repeated-input identities, saved metrics and the final-access hash links.

The final-access gate audit verified the recorded sequence: completed training and its final inventory preceded the intermediate archive receipt; that receipt preceded final execution and consumption of the final-access record. The intermediate archive contained no final outcomes. Its verification is evidence about the preserved workflow, not proof of external blindness or the absence of unrecorded access.

The symbolic control's verified visible-fact answers were correct on all 12,288 rows. The train-fitted role-majority control had global accuracy 0.10888671875 and binding accuracy 0.087890625. Analytical marginal references remained 0.375 global/0.25 binding for bag-category guessing, 0.5/0.5 for role-only guessing, and 0.75/0.5 for event-plus-category guessing. These analytical references do not create sampled model predictions or an independent confidence interval.

### 5.2 Primary comparison and all five realizations

The primary endpoint is base macro agent/patient accuracy. Values in the following table are exact proportions; differences are NeuroPixel minus Transformer.

| Paired training seed | NeuroPixel base binding | Transformer base binding | Paired difference |
|---|---:|---:|---:|
| 40 | 0.09765625 | 0.478515625 | -0.380859375 |
| 41 | 0.1005859375 | 0.5009765625 | -0.400390625 |
| 42 | 0.091796875 | 0.5068359375 | -0.4150390625 |
| 43 | 0.1025390625 | 0.4970703125 | -0.39453125 |
| 44 | 0.0888671875 | 0.484375 | -0.3955078125 |

NeuroPixel's mean was **0.0962890625**, sample SD **0.005802129209965996**, and range **[0.0888671875, 0.1025390625]**. The Transformer's mean was **0.4935546875**, sample SD **0.011771528285393877**, and range **[0.478515625, 0.5068359375]**. All five paired differences were negative.

The mean paired difference was **−0.397265625**, with sample SD of differences **0.012302362131463691**. The prespecified paired t 95% interval with n = 5 and df = 4 was **[−0.4125410251515038, −0.3819902248484962]**, or approximately **−39.727 percentage points [−41.254, −38.199]**. This supports a difference favoring the Transformer for the complete frozen procedures on this task; it is not an isolated causal effect of any architectural component.

The next table shows absolute base performance and the separate conditional bag-bootstrap intervals. Percentages are rounded to three decimals and cross-entropy to six decimals; full-precision values and all interval endpoints remain in the [complete scientific report](../../results/research/09_offline_runs/37603398040-1-offline/scientific_audit_01.json). These checkpoint intervals resample bags, whereas the t interval above summarizes differences across training realizations.

| Seed | Family | Base global, % | Base binding, % | Conditional bag-bootstrap 95% interval for binding, % | All-eight, bags / 256 | CE |
|---|---|---:|---:|---|---:|---:|
| 40 | NeuroPixel | 10.498 | 9.766 | [8.301, 11.331] | 0/256 | 2.366164 |
| 40 | Transformer | 69.141 | 47.852 | [46.484, 49.121] | 0/256 | 0.609672 |
| 41 | NeuroPixel | 11.719 | 10.059 | [8.398, 11.719] | 0/256 | 2.359583 |
| 41 | Transformer | 75.049 | 50.098 | [49.414, 50.879] | 0/256 | 0.396346 |
| 42 | NeuroPixel | 10.352 | 9.180 | [7.617, 10.645] | 0/256 | 2.361200 |
| 42 | Transformer | 75.342 | 50.684 | [49.707, 51.660] | 0/256 | 0.472443 |
| 43 | NeuroPixel | 11.719 | 10.254 | [8.691, 11.819] | 0/256 | 2.371649 |
| 43 | Transformer | 74.854 | 49.707 | [48.730, 50.781] | 1/256 | 0.433884 |
| 44 | NeuroPixel | 10.645 | 8.887 | [7.422, 10.352] | 0/256 | 2.381879 |
| 44 | Transformer | 73.193 | 48.438 | [47.461, 49.316] | 0/256 | 0.518980 |

Mean global accuracy was **0.10986328125** for NeuroPixel and **0.73515625** for the Transformer. NeuroPixel scored 0/256 entirely correct base bags in every run. The Transformer scored 1/256 in seed 43 and 0/256 in the other four runs. All zero all-eight cases had empirical bootstrap intervals [0,0]; Transformer seed 43 had [0,0.01171875]. These are observed bag counts and the frozen empirical intervals, not claims that the population probability is exactly zero.

![All five base binding differences and their mean with paired t interval](../../results/research/09_offline_runs/37603398040-1-offline/figures/base_paired_difference.png)

### 5.3 Complete condition panel and competence screen

The following summaries include every condition and family. Means, sample SDs and ranges are across five training realizations and are descriptive; they are not the scenario-bootstrap interval. Every all-eight count uses 256 original bags per seed. The swap rows concern eight query-conditioned cases, as explained in Section 2.

| Condition | Family | Mean global, % | Mean binding, % | Binding sample SD, pp | Binding seed range, % | Mean CE | All-eight counts, seeds 40–44 (each /256) |
|---|---|---:|---:|---:|---|---:|---|
| Base | NeuroPixel | 10.986 | 9.629 | 0.580 | [8.887, 10.254] | 2.368095 | 0, 0, 0, 0, 0 |
| Base | Transformer | 73.516 | 49.355 | 1.177 | [47.852, 50.684] | 0.486265 | 0, 0, 0, 1, 0 |
| Queried-event swap | NeuroPixel | 10.947 | 9.863 | 0.694 | [8.789, 10.645] | 2.368886 | 0, 0, 0, 0, 0 |
| Queried-event swap | Transformer | 73.516 | 49.375 | 1.620 | [46.973, 50.781] | 0.486200 | 0, 0, 0, 2, 0 |
| Other-event swap | NeuroPixel | 10.918 | 9.551 | 0.554 | [8.789, 10.156] | 2.367513 | 0, 0, 0, 0, 0 |
| Other-event swap | Transformer | 73.506 | 49.355 | 1.167 | [47.656, 50.391] | 0.487319 | 0, 0, 0, 1, 0 |
| Global relabeling | NeuroPixel | 11.152 | 9.902 | 1.097 | [8.984, 11.719] | 2.368492 | 0, 0, 0, 0, 0 |
| Global relabeling | Transformer | 73.369 | 48.887 | 1.600 | [46.387, 50.684] | 0.485865 | 0, 0, 0, 0, 0 |
| Query switch | NeuroPixel | 10.986 | 9.629 | 0.580 | [8.887, 10.254] | 2.368095 | 0, 0, 0, 0, 0 |
| Query switch | Transformer | 73.516 | 49.355 | 1.177 | [47.852, 50.684] | 0.486265 | 0, 0, 0, 1, 0 |
| Layout permutation | NeuroPixel | 11.025 | 9.531 | 0.551 | [8.887, 10.156] | 2.368003 | 0, 0, 0, 0, 0 |
| Layout permutation | Transformer | 73.438 | 49.199 | 0.982 | [48.145, 50.586] | 0.484252 | 0, 0, 0, 0, 0 |

All sixty binding scores are preserved explicitly below in seed order 40, 41, 42, 43, 44. No condition or run was selected for reporting.

| Condition | NeuroPixel binding, seeds 40–44 (proportions) | Transformer binding, seeds 40–44 (proportions) |
|---|---|---|
| Base | 0.09765625, 0.1005859375, 0.091796875, 0.1025390625, 0.0888671875 | 0.478515625, 0.5009765625, 0.5068359375, 0.4970703125, 0.484375 |
| Queried-event swap | 0.1015625, 0.1005859375, 0.0966796875, 0.1064453125, 0.087890625 | 0.4697265625, 0.50390625, 0.5029296875, 0.5078125, 0.484375 |
| Other-event swap | 0.09765625, 0.0986328125, 0.091796875, 0.1015625, 0.087890625 | 0.4765625, 0.5009765625, 0.50390625, 0.5, 0.486328125 |
| Global relabeling | 0.0986328125, 0.0986328125, 0.0908203125, 0.1171875, 0.08984375 | 0.4638671875, 0.4921875, 0.5068359375, 0.49609375, 0.4853515625 |
| Query switch | 0.09765625, 0.1005859375, 0.091796875, 0.1025390625, 0.0888671875 | 0.478515625, 0.5009765625, 0.5068359375, 0.4970703125, 0.484375 |
| Layout permutation | 0.0927734375, 0.0927734375, 0.1005859375, 0.1015625, 0.0888671875 | 0.4814453125, 0.498046875, 0.505859375, 0.48828125, 0.486328125 |

The three NeuroPixel screen checks were all false: not all five base binding scores reached 0.95; not all five base all-eight scores reached 0.90; and not every condition in every seed reached binding 0.90. Indeed, no individual NeuroPixel binding score approached these thresholds. The Transformer was substantially more accurate but likewise did not demonstrate high role-binding competence: its binding scores across these conditions remained approximately 0.464–0.508, and entirely correct bags were rare. The formal prespecified screen applies to NeuroPixel; the Transformer statement describes its absolute scores rather than introducing a second decision rule.

The full role breakdown helps distinguish the source of high global accuracy from successful agent/patient discrimination. Values below are means across seeds, in percent.

| Condition | Family | Agent, % | Action, % | Patient, % | Place, % |
|---|---|---:|---:|---:|---:|
| Base | NeuroPixel | 9.648 | 11.055 | 9.609 | 13.633 |
| Base | Transformer | 49.258 | 99.805 | 49.453 | 95.547 |
| Queried-event swap | NeuroPixel | 9.609 | 10.742 | 10.117 | 13.320 |
| Queried-event swap | Transformer | 49.727 | 99.805 | 49.023 | 95.508 |
| Other-event swap | NeuroPixel | 9.531 | 11.133 | 9.570 | 13.438 |
| Other-event swap | Transformer | 49.180 | 99.766 | 49.531 | 95.547 |
| Global relabeling | NeuroPixel | 9.727 | 11.094 | 10.078 | 13.711 |
| Global relabeling | Transformer | 48.906 | 99.844 | 48.867 | 95.859 |
| Query switch | NeuroPixel | 9.648 | 11.055 | 9.609 | 13.633 |
| Query switch | Transformer | 49.258 | 99.805 | 49.453 | 95.547 |
| Layout permutation | NeuroPixel | 9.727 | 11.250 | 9.336 | 13.789 |
| Layout permutation | Transformer | 48.711 | 99.844 | 49.688 | 95.508 |

The Transformer's base action and place means were 99.805% and 95.547%, while its agent and patient means were 49.258% and 49.453%. Its global/binding pattern is numerically compatible with the event-plus-category analytical reference of 75%/50%, which need not resolve the two noun roles. This is a descriptive comparison, not proof that the trained Transformer implements that heuristic or a statistical equivalence claim. NeuroPixel's low absolute scores were numerically near the train-majority control and below the category-guessing references; no extra hypothesis test or causal explanation is inferred from that comparison.

![All six conditions and all five seeds for both families](../../results/research/09_offline_runs/37603398040-1-offline/figures/binding_by_condition.png)

### 5.4 Counterfactual correctness and invariance

The next table summarizes all five interventions. Entries are arithmetic means of the five checkpoint proportions, in percent, rather than pooled independent observations. Every checkpoint's numerator, denominator, role breakdown and paired bag-bootstrap interval is retained in `analysis.final_recount[run].conditional_bag_diagnostics` in the full report.

Both-correct all-role rates use 2,048 base/transformed pairs per checkpoint; the agent/patient summary uses 1,024. For queried-event swap, prediction equality applies to 1,024 unchanged-target action/place pairs and prediction inequality to 1,024 changed-target agent/patient pairs. Other-event swap, relabeling and layout each have 2,048 invariant pairs and no changed-target pairs. Query switch has 2,048 changed-target pairs and no invariant pairs. A zero eligible denominator is reported as null.

| Intervention | Family | Both correct, all roles, % | Both correct, agent/patient, % | Equal predictions among invariant pairs, % | Different predictions among changed-target pairs, % |
|---|---|---:|---:|---:|---:|
| Queried-event swap | NeuroPixel | 6.260 | 0.879 | 95.469 | 9.180 |
| Queried-event swap | Transformer | 50.996 | 4.375 | 99.902 | 8.594 |
| Other-event swap | NeuroPixel | 10.469 | 9.082 | 93.262 | null (n = 0) |
| Other-event swap | Transformer | 73.076 | 48.535 | 98.965 | null (n = 0) |
| Global relabeling | NeuroPixel | 10.596 | 9.023 | 94.922 | null (n = 0) |
| Global relabeling | Transformer | 70.488 | 43.535 | 93.789 | null (n = 0) |
| Query switch | NeuroPixel | 0.039 | 0.039 | null (n = 0) | 4.980 |
| Query switch | Transformer | 59.805 | 24.141 | null (n = 0) | 96.895 |
| Layout permutation | NeuroPixel | 7.715 | 6.250 | 73.955 | null (n = 0) |
| Layout permutation | Transformer | 71.689 | 45.977 | 96.299 | null (n = 0) |

The queried-event swap exposes an important limitation that marginal accuracy alone conceals. Both models usually failed to change the agent/patient answer when those targets changed: the mean prediction-change rates were 9.180% for NeuroPixel and 8.594% for the Transformer. Mean agent/patient both-correct rates were only 0.879% and 4.375%, respectively. The Transformer's 50.996% all-role both-correct rate is dominated by the invariant action/place queries and should not be presented as successful noun-role swapping.

NeuroPixel's high prediction-equality rates on other-event swaps and relabeling coexisted with roughly 10% all-role both-correct rates. This is concrete evidence that prediction invariance alone is insufficient for this study's success claim. Layout permutation reduced its mean both-correct rate to 7.715% even though marginal global accuracy changed little.

Query-switch marginal scores exactly equaled base scores for every checkpoint, as expected from the permutation of the same queries. The paired diagnostics contain additional information: the Transformer changed predictions on 96.895% of those pairs but answered both noun-role queries correctly on only 24.141%; NeuroPixel's corresponding values were 4.980% and 0.039%. Prediction change is therefore not equivalent to correct role resolution. These findings are behavioral diagnostics on the fixed panel, not an identification of internal mechanisms.

### 5.5 Recorded development/probe outcomes and resources

Every scheduled checkpoint was retained; there was no test-driven extension or checkpoint selection. All ten training-probe binding scores were below 0.95, and every probe had all-eight 0/64. The table gives exact development/probe binding proportions and the preserved wall times rounded to six decimals.

| Seed | Family | Development binding | Training-probe binding | Below 0.95 probe flag | Recorded run wall, s | Final evaluation wall, s |
|---|---|---:|---:|---|---:|---:|
| 40 | NeuroPixel | 0.080078125 | 0.10546875 | true | 469.730621 | 13.633897 |
| 40 | Transformer | 0.5 | 0.47265625 | true | 123.641545 | 3.249480 |
| 41 | NeuroPixel | 0.109375 | 0.07421875 | true | 465.568185 | 13.590479 |
| 41 | Transformer | 0.498046875 | 0.4921875 | true | 122.333759 | 3.235173 |
| 42 | NeuroPixel | 0.0859375 | 0.09765625 | true | 465.731873 | 13.604109 |
| 42 | Transformer | 0.5 | 0.48828125 | true | 122.441491 | 3.258163 |
| 43 | NeuroPixel | 0.103515625 | 0.109375 | true | 466.081880 | 13.626961 |
| 43 | Transformer | 0.498046875 | 0.48046875 | true | 121.691464 | 3.230428 |
| 44 | NeuroPixel | 0.08203125 | 0.10546875 | true | 465.999222 | 13.682361 |
| 44 | Transformer | 0.48828125 | 0.484375 | true | 121.969962 | 3.249981 |

The sum of recorded run wall times was **2,945.1900032210006 seconds**. This includes model construction, training, checkpoint writing and development evaluation; it is not pure optimizer time. The final-evaluation wall times summed to **84.36103189800042 seconds**, covering prediction/output/metrics while excluding model load and later summary writing. Neither quantity measures energy or isolated inference latency.

The study worker lasted **3,061.636433142 seconds before its final archive**, from 08:26:51.997378 to 09:17:53.632580 UTC. Its sampled minimum available RAM was **14.214973449707031 GiB**, above the 8 GiB guard. No scientific training retry or resource interruption was recorded in this completed attempt. The measured duration exceeded the uninflated pilot extrapolation but stayed within the prospectively budgeted margin; the recipe was unchanged.

The separate offline host performed archive/gate verification, the frozen saved-array analysis and figure production without training, model inference or generation of a new scientific dataset. Its worker lasted **27.802332284000016 seconds before its final archive**. Fifteen supervision samples had minimum available RAM **14.436992645263672 GiB**; 86 scientific-audit samples had minimum **14.435539245605469 GiB**. These are sampled available-memory observations, not peak resident memory or continuous utilization. The scientific-audit child stage lasted 11.452072082000015 seconds including shutdown.

The below-threshold probes demonstrate limited achieved accuracy under the fixed recipe. They do not identify insufficient updates, learning rate, supervision or an architectural mechanism as the cause. The successful 32-example memory test and weak held-out binding can coexist; neither observation removes the other's scope.

## 6. Provenance, verification and reproduction

Stage A used frozen source commit [e63764ccb924ab23d65c2deb94b477387d102c82](https://github.com/Agnuxo1/NeuroPixel/commit/e63764ccb924ab23d65c2deb94b477387d102c82), Actions run 37589908205, attempt 1, job 112688648396. Its [raw archived evidence](https://github.com/Agnuxo1/NeuroPixel/tree/8311c052796aeef5a52484b597cf5d9675ca315e/results/research/09_cloud_runs/37589908205-1-preflight) is anchored to commit 8311c052796aeef5a52484b597cf5d9675ca315e. The receipt binds selection, preflight records, source/data context and their complete copies.

| Stage-A evidence | SHA256 |
|---|---|
| Data/metric audit | `1ee4f118ed846d767dae6a7ed58910f15412946611ebd196a31141c01f3d7974` |
| Archive audit 03 | `2113accdce1cf74e01b638b03c865d5a64f4386ff7b2c48fdc2b1a758ce61d2a` |
| Stage-A evidence receipt | `a1d53e0b064e812317d93e6e8749bae4728730fecfbf359edf0249f2a72f6ee4` |
| Cost projection | `27d74fa8b9738f27cf758b366655e9f70b55c879a9dca9611740d2b0eebf3a09` |

Stage B used frozen scientific source commit [83fe135e301f76bc0c74e30c66bb18e067ca5959](https://github.com/Agnuxo1/NeuroPixel/commit/83fe135e301f76bc0c74e30c66bb18e067ca5959), run 37593731891 and job 112701154832. The [intermediate pre-final archive](https://github.com/Agnuxo1/NeuroPixel/commit/2e58f5deb47f5676e8663d666d9db7ea98e3e080) is G = `2e58f5deb47f5676e8663d666d9db7ea98e3e080`; the [complete raw archive](https://github.com/Agnuxo1/NeuroPixel/tree/b64d0e2ba222df76caf72bc9870c7602873e7236/results/research/09_cloud_runs/37593731891-1-study) is F = H = `b64d0e2ba222df76caf72bc9870c7602873e7236`. Here H denotes the supplied results-branch head at recovery; it is not a separate replication. The intermediate archive manifests 99 files, with its manifest as a 100th Git file, and no final artifacts. The complete archive manifests 134 files totaling 56,105,745 bytes, excluding the manifest.

The local executor became unavailable after collection. Analysis moved to a separate CPU host using the unchanged frozen scientific analyzer and its declared analysis runtime. Offline run 37603398040, job 112732965039, produced archive [O = 2b8203f15ec5f6fe190876c80bf29232034603ce](https://github.com/Agnuxo1/NeuroPixel/tree/2b8203f15ec5f6fe190876c80bf29232034603ce/results/research/09_offline_runs/37603398040-1-offline). The operational driver source was `87e1f1c0b388190474e8190952c148c0501c8abf`; this must not be confused with the scientific source. The offline archive manifests 33 files totaling 17,761,993 bytes, excluding its manifest. Operational workflow validation failures and recovery receipts are preserved; they were not additional training or final-data trials.

The [full archive audit](../../results/research/09_offline_runs/37603398040-1-offline/full_archive_audit_01.json) passed 1,100 checks with no issues, including 348 source files, bound inventory/source identities, resources and 34 passing test methods with zero skips. The JUnit counter remained 137. The separate [gate audit](../../results/research/09_offline_runs/37603398040-1-offline/gate_archive_audit_01.json) passed 690 checks with no failures or issues. The scientific saved-array audit was verified with no issues and records 104 input-file identities. Root's subsequent [report-value review](../../results/research/09_validation/stage_b_report_values.json) passed 462 arithmetic/denominator checks with no issues. These counts describe different verification procedures and are not combined into a sample size.

The training environment was Python 3.12.8, PyTorch 2.6.0+cpu, NumPy 2.2.6, SciPy 1.15.1, pytest 9.1.1, psutil 7.2.2 and Pillow 12.3.0, with two intra-operation threads, one inter-operation thread and an 8 GiB minimum available-memory guard. The completed analysis used the separately pinned Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 environment with one numerical thread. This does not assert identical random sampling across NumPy versions. The actual 2,000 × 256 bootstrap index array was retained and its 512,000 integer entries were independently checked against the recorded little-endian int64 digest.

| Final analysis artifact | Bytes | SHA256 |
|---|---:|---|
| Full scientific report, including bootstrap indices | 7,142,199 | `2ffcd15900dd583f796f158bac6de9dee1761ba74cf7f4d9ae9ca700d217316e` |
| Readable projection, omitting only those indices | 715,535 | `97d5e1c348343e7a08856d46a5bd67baa5858cdfd27e70e155988cbb869fee89` |
| Bootstrap indices serialized as little-endian int64 | 4,096,000 | `95f989b32c09be3185420ba911b4bcafd0291a9e50aea3fcf88075039f0718c6` |

The bootstrap serialization is a digest representation of the retained index array, not a claim that an additional binary file was saved. Root verified that the projection equals the full report after only the disclosed indices removal. This report's author retrieved the projection and independently matched its UTF-8 byte count and SHA256 before using its values. The [projection review](../../results/research/09_validation/stage_b_full_report_projection_review.json), [figure visual review](../../results/research/09_validation/stage_b_figure_visual_review.json) and [offline completion receipt](../../results/research/09_validation/offline_completion_receipt.json) are preserved in closure provenance. Both displayed PNGs were visually reviewed by root without identified problems; SVG counterparts and figure hashes remain in O.

All full-precision metrics, all sixty checkpoint-condition records, paired checkpoint intervals, role-specific counterfactual numerators/denominators and every bootstrap endpoint are available in the [full scientific report](../../results/research/09_offline_runs/37603398040-1-offline/scientific_audit_01.json) and its [readable projection](../../results/research/09_offline_runs/37603398040-1-offline/scientific_audit_summary.json). The root [report-value extraction](../../results/research/09_validation/stage_b_report_values.json) provides integer-count and summary cross-checks. This report's tables round only where their labels explicitly specify percent or display precision; decisions used the saved values.

For reproduction of the saved-array recount, use the scientific source at `83fe135...` and its `scripts/research_complex_binding_audit.py`, the raw `study/` directory in F, the frozen `docs/research/09_experiment_recipe.json`, and the separately pinned analysis environment. The archived offline status records the exact invoked command and source-root relationship. Preserve the selection receipt, archive, bootstrap array and access chronology. Do not rebuild selection from rounded report tables or silently regenerate a new final panel. A fresh training rerun would be a distinct reproduction attempt, not the original execution.

The audits did not deserialize checkpoint tensors, rerun models or verify every arithmetic operation during training. They verify saved bytes, independent visible-data logic and saved-prediction calculations. Git ancestry and logs corroborate the retained ordering and hash links; they do not independently certify when every remote push arrived or establish external holdout custody.

## 7. Interpretation, relation to prior work and limits

The scientific target is accurate use of explicit role and event markers under this finite two-event grammar. Smolensky's tensor-product framework provides a formal distinction between role and filler representations; success here would not show that either trained model implemented tensor products or a particular unbinding operation. SCAN and COGS demonstrate the value of separating supported combinations and roles from held-out combinations, while also showing why success on one split does not establish arbitrary systematic generalization. CFQ makes the distinction between primitive and compound distributions explicit; this study's bag partition is not a DBCA measurement and should not be described as one.

CLEVR motivates controlled scene generation, unambiguous questions and diagnostic breakdowns, but this experiment bypasses visual perception. HANS illustrates that a heuristic can look successful until supporting and conflicting cases are distinguished. The present swaps, relabeling and layout changes test named shortcuts; surviving them would not prove the absence of every shortcut. These are methodological precedents, not a claim of priority.

Six primary sources underpin the existing [methodological evidence note](09_methodological_evidence.md), which records reading scope and access on 2026-10-07:

1. Smolensky, P. (1990). *Tensor product variable binding and the representation of symbolic structures in connectionist systems*. Artificial Intelligence 46, 159–216. [Primary PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/Smolensky-90-AI-Tensor-Product-Variable-Binding-and-the-Representation-of-Symbolic-Structures-in-Connectionist-Systems.pdf); [DOI](https://doi.org/10.1016/0004-3702(90)90007-M).
2. Lake, B. M., and Baroni, M. (2018). *Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks*. ICML, PMLR 80, 2873–2882. [Primary PDF](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf).
3. Kim, N., and Linzen, T. (2020). *COGS: A compositional generalization challenge based on semantic interpretation*. EMNLP, 9087–9105. [Primary PDF](https://aclanthology.org/2020.emnlp-main.731.pdf).
4. Keysers, D., et al. (2020). *Measuring compositional generalization: A comprehensive method on realistic data*. ICLR; arXiv:1912.09713, version 2. [Primary PDF](https://arxiv.org/pdf/1912.09713.pdf).
5. Johnson, J., et al. (2017). *CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning*. CVPR; arXiv:1612.06890, version 1 consulted. [Primary PDF](https://arxiv.org/pdf/1612.06890.pdf).
6. McCoy, R. T., Pavlick, E., and Linzen, T. (2019). *Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference*. ACL, 3428–3448. [Primary PDF](https://aclanthology.org/P19-1334.pdf); [DOI](https://doi.org/10.18653/v1/P19-1334).

The following limits constrain the completed results:

- The grammar, vocabulary, explicit labels and two-event capacity are fixed. Distinct fillers remove repeated-entity ambiguity. There is no evidence here about recursion, anaphora, a third event, natural-language understanding or visual recognition.
- Bag separation controls membership for this study's collection. The generator and prior programme are known; public seeds and hashes support reproduction, not evaluator blindness or an externally secured holdout.
- Five paired training realizations provide limited information about procedure-level variability. Thousands of queries and bootstrap resamples do not increase that number.
- Scenario-bootstrap intervals are conditional on these trained checkpoints and the sampled bag population. Query-switch duplicates and query-conditioned swaps retain dependence inside each bag.
- Similar parameter counts, identical input streams and a common update budget do not establish equal computation or equal optimization opportunity. Development-selected rates and family-specific dynamics are part of the procedures being compared.
- A competence-screen failure would bound performance under this recipe, not establish universal inability. A pass would support this explicit task and its declared interventions, not a general theory of cognition or a verified internal mechanism.
- The audits independently recount this project's saved evidence. They are not external laboratory replications. No item-5 H1 decision is reopened, and no later programme item follows from this draft.

Under this frozen two-event task and budget, NeuroPixel did not demonstrate successful complex role binding. The Transformer achieved a substantial and consistent comparative advantage but also failed to resolve the noun roles reliably, particularly under queried-event swaps. These results support a bounded negative conclusion about the claimed competence under this recipe. They neither establish universal architectural inability nor identify the causal reason for the failures. The earlier H1 remains closed.
