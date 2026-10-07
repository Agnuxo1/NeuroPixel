# Item 10: externally authored relational question answering

**Investigation completed on 7 October 2026. Capability claim: not established. Historical H1 remains closed and not supported.**

## 1. Outcome and decision

NeuroPixel was trained from scratch on an externally authored task, the English 1k bAbI QA4 “two argument relations” benchmark. Five fixed main training realizations achieved mean official-TEST accuracy **50.40%**, compared with **88.66%** for the parameter-matched relative-position Transformer. The primary NeuroPixel-minus-Transformer contrast is **−38.26 percentage points**, with a nominal paired Student-t 95% interval **[−52.3857, −24.1343] points** over five seeds. Every paired difference is negative.

Every NeuroPixel seed also falls below the prospective fact-frequency control that excludes answer tokens named in the question, which obtains **65.70%** on the same 1,000 rows. NeuroPixel fails the declared competence screen in all five seeds. Applying the same screen descriptively to the Transformer gives two individual passes, but no all-five pass. This result does not support robust high competence or an advantage for the evaluated NeuroPixel recipe.

The experiment does establish a working literal adapter, TRAIN-derived representation, prospective development selection, ten completed fresh main runs, authenticated prediction/checkpoint archives, a final-access gate, and a separate saved-array and Git-archive audit. Completion of this investigation is distinct from achievement of the proposed capability. The result concerns learning within an externally authored synthetic benchmark; it is not zero-shot transfer, natural-language competence, or generalization beyond the bAbI generator.

Evidence: [complete main archive](https://github.com/Agnuxo1/NeuroPixel/tree/0e135ad10518e1ccd9b7adc562265aa4e2d94d2b/results/research/10_cloud_runs/37615277149-1-study); [independent scientific reconstruction](https://github.com/Agnuxo1/NeuroPixel/blob/bb2323ac0e0ba028975a4a4c89d2a97a20c7ff6d/results/research/10_cloud_runs/37618268225-1-audit/analysis/scientific_audit.json); [actual Git/archive reconstruction](https://github.com/Agnuxo1/NeuroPixel/blob/bb2323ac0e0ba028975a4a4c89d2a97a20c7ff6d/results/research/10_cloud_runs/37618268225-1-audit/offline_archive_audit.json). The main scientific source is `f619ac2150960ff658a797e3ac8594f49c8d96c1`; all reported main outcomes refer to that frozen source.

## 2. Main results: all realizations and denominators

Counts are correct/eligible population, with no selection of the best seed. “Novel” here means absent under exact normalized raw-input and frozen encoded-input identity. It does not mean a new concept or linguistic construction. In the last two columns, NeuroPixel precedes Transformer.

| Seed | NeuroPixel, all official | Transformer, all official | NP minus TF, pp | Novel vs optimization TRAIN: NP; TF | Novel vs declared full exposure: NP; TF |
|---:|---:|---:|---:|---:|---:|
| 60 | 540/1000 (54.00%) | 966/1000 (96.60%) | -42.60 | 451/886; 852/886 | 441/866; 832/866 |
| 61 | 594/1000 (59.40%) | 877/1000 (87.70%) | -28.30 | 509/886; 763/886 | 498/866; 746/866 |
| 62 | 467/1000 (46.70%) | 761/1000 (76.10%) | -29.40 | 387/886; 647/886 | 381/866; 634/866 |
| 63 | 621/1000 (62.10%) | 972/1000 (97.20%) | -35.10 | 524/886; 858/886 | 513/866; 839/866 |
| 64 | 298/1000 (29.80%) | 857/1000 (85.70%) | -55.90 | 251/886; 743/886 | 248/866; 727/866 |

| Population | Rows per seed | NeuroPixel mean accuracy | Transformer mean accuracy |
|---|---:|---:|---:|
| All official TEST | 1000 | 50.40% | 88.66% |
| Novel relative to optimization TRAIN | 886 | 47.90% | 87.20% |
| Novel relative to complete declared TRAIN/fixture exposure | 866 | 48.06% | 87.25% |

The primary five differences are −42.6, −28.3, −29.4, −35.1 and −55.9 points. Their sample standard deviation is 11.376423 points. The interval uses five paired training realizations and four degrees of freedom, conditional on this task, split, fixed update budget and development-selected rates. It is not a confidence interval formed by treating the 1,000 questions as independent model replications. The t approximation with only five pairs has limited distributional assurance; the complete pair table makes that limitation inspectable. No new significance threshold or historical-H1 replacement was introduced after seeing the outcomes.

Mean official cross-entropy is 1.988533 for NeuroPixel and 0.722467 for Transformer. These are arithmetic means over the five runs, not the cross-entropy of an ensemble. In this acquired population, all rows are encodable and all gold labels are supported, so accuracy and cross-entropy denominators coincide. The audit retains the masks and separate CE denominators so this equality is not assumed for arbitrary future data.

### Competence screen

The prospectively declared NeuroPixel screen requires at least 95% accuracy in every seed on all official rows and on the nonempty strict-novel subset. No NeuroPixel seed passes either population threshold. Transformer seeds 60 and 63 pass both; seeds 61, 62 and 64 do not. The Transformer calculation is a descriptive use of the same rule. Neither family has an all-five pass.

This screen is a prerequisite for a competence claim under this recipe, not a causal test of local dynamics and not an acceptance criterion for a Nobel-level discovery. The negative comparison remains meaningful as a result of the fixed experiment, while the cause of NeuroPixel's shortfall remains unresolved.

## 3. Prospective controls and optimization evidence

| Control | All official, n=1000 | Novel vs TRAIN, n=886 | Strict novel, n=866 |
|---|---:|---:|---:|
| bow_memory | 412/1000 (41.20%) | 319/886 (36.00%) | 311/866 (35.91%) |
| bow_naive_bayes | 169/1000 (16.90%) | 152/886 (17.16%) | 151/866 (17.44%) |
| exact_memory | 263/1000 (26.30%) | 149/886 (16.82%) | 148/866 (17.09%) |
| fact_answer_frequency | 343/1000 (34.30%) | 311/886 (35.10%) | 309/866 (35.68%) |
| fact_answer_frequency_excluding_query | 657/1000 (65.70%) | 589/886 (66.48%) | 579/866 (66.86%) |
| majority | 171/1000 (17.10%) | 149/886 (16.82%) | 148/866 (17.09%) |
| question_only | 193/1000 (19.30%) | 159/886 (17.95%) | 157/866 (18.13%) |
| symbolic_raw | 1000/1000 (100.00%) | 886/886 (100.00%) | 866/866 (100.00%) |

The exact-memory advantage on all official rows disappears on the novelty-filtered populations, where its default is the TRAIN majority answer. The fact-frequency-excluding-query rule exceeds every NeuroPixel seed on every reported population. This provides a direct warning that above-majority accuracy alone is insufficient evidence of solving argument roles.

The symbolic raw-text rule obtains 100% in all populations. It explicitly encodes the direct/inverse cardinal-relation grammar and is a task-specific positive control. It shows that the benchmark is answerable by the declared rule, rather than validating a learned neural mechanism. Raw rules are not parameter- or compute-matched neural baselines and must not be represented as such. Gold answers were never replaced with this rule's outputs.

The complete optimization-TRAIN and DEV endpoint predictions were saved for every run:

| Run | TRAIN, n=895 | DEV, n=105 | Official TEST CE | Training-loop seconds |
|---|---:|---:|---:|---:|
| NeuroPixel 60 | 689/895 (76.98%) | 50/105 (47.62%) | 2.125765 | 218.945 |
| Transformer 60 | 895/895 (100.00%) | 102/105 (97.14%) | 0.148552 | 33.292 |
| NeuroPixel 61 | 680/895 (75.98%) | 61/105 (58.10%) | 1.660564 | 217.336 |
| Transformer 61 | 895/895 (100.00%) | 83/105 (79.05%) | 0.694855 | 32.544 |
| NeuroPixel 62 | 599/895 (66.93%) | 41/105 (39.05%) | 1.899792 | 216.537 |
| Transformer 62 | 894/895 (99.89%) | 81/105 (77.14%) | 1.663847 | 34.347 |
| NeuroPixel 63 | 749/895 (83.69%) | 51/105 (48.57%) | 2.446656 | 215.497 |
| Transformer 63 | 895/895 (100.00%) | 102/105 (97.14%) | 0.124681 | 33.158 |
| NeuroPixel 64 | 332/895 (37.09%) | 39/105 (37.14%) | 1.809887 | 217.736 |
| Transformer 64 | 895/895 (100.00%) | 89/105 (84.76%) | 0.980400 | 33.832 |

All five NeuroPixel full-TRAIN accuracies remain below 95%, ranging from 37.09% to 83.69%. All runs actually visited all 895 TRAIN records, corresponding to 820 distinct normalized and encoded inputs. Therefore the poor final result cannot be attributed solely to previously unseen TEST inputs. Optimization, representation and finite training budget have not been separated. Four Transformers fit every TRAIN record; seed 62 misses one. Their TEST range still spans 76.10%–97.20%, so near-perfect training fit is also insufficient for a uniform held-out competence claim.

The small memorization preflight should be read at its actual scale. Both families fitted the same 32 distinct TRAIN inputs at two consecutive saved checks: NeuroPixel at updates 1,920 and 2,048; Transformer at 256 and 384. Earlier NeuroPixel isolated passes and subsequent failures were retained. This establishes the ability to fit that tiny fixture, not adequate optimization of the full TRAIN task.

Four fixed pilots selected LR 0.003 for both families by DEV accuracy first, then CE, then lower LR. Their DEV correct counts were 16 and 36 for NeuroPixel at 0.001 and 0.003, and 74 and 79 for Transformer. The lower Transformer pilot CE at LR 0.001 did not override the declared accuracy-first rule. Pilot and memorization weights were not reused as main weights. [Preflight report](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/docs/research/10_preflight_results.md); [main admission](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/docs/research/10_main_admission.md).

## 4. Population identity and conditional resampling

The official TEST member contains 1,000 rows, 933 distinct normalized inputs and 751 connected source components. Component sizes have histogram 559 of size one, 144 of size two, 41 of size three, six of size four and one of size six. Connectivity is defined from original episode, complete normalized fact story or identical visible input, joined transitively. The 67 additional duplicate-input rows have exactly identical predictions and saved logits within every model; maximum duplicate logit difference is zero. They are not 67 additional independent examples.

There are 114 row overlaps with optimization TRAIN and 134 with the complete official TRAIN plus the declared fixture ledger; raw and encoded overlap counts agree here. These are row counts, not counts of distinct overlap identities. The corresponding novel denominators are 886 and 866. There are zero conflicting raw-input labels, zero unencodable official TEST rows and zero unsupported official gold answers. Six intentionally oversized negative fixture contexts remain in the raw exposure ledger, although no valid encoded identity exists for them. Their presence does not remove or truncate any official TEST row.

The 149-context fixture ledger covers declared development fixtures. Known bibliographic examples and any additional historical exposure are outside that ledger. Consequently the exact-input novelty masks establish neither complete historical novelty nor blinding. The original data are public, and an internal read-access gate does not turn them into externally held secret evaluation data.

The separate conditional bootstrap resamples the 751 connected TEST components, holding all ten checkpoints fixed. It uses 2,000 common PCG64 index draws with seed 104001. All 45 endpoints—30 model/population endpoints and 15 paired-seed/population differences—have 2,000 defined draws and zero undefined draws. The all-official paired intervals, in percentage points, are:

| Seed | NP minus TF, pp | Conditional component-bootstrap 95% interval, pp |
|---:|---:|---:|
| 60 | -42.60 | [-45.9408, -39.2179] |
| 61 | -28.30 | [-32.1863, -24.6974] |
| 62 | -29.40 | [-33.5888, -25.2559] |
| 63 | -35.10 | [-38.3631, -31.7670] |
| 64 | -55.90 | [-59.4222, -52.0981] |

Every paired interval in all three populations has an upper endpoint below zero. These are correlated descriptive intervals sharing the same component draws and checkpoints, not 15 independent confirmations or a multiplicity-adjusted set of tests. They capture conditional sensitivity to the observed component composition, not training variability or sampling from all natural language.

The index matrix has shape 2000×751 and file SHA-256 `1f2716461bddc1d006e3e37ffa5861156eb76adb844455240ea9f1636f43bd03`; its little-endian int64 content SHA is `dd493d635d596b78b6e007c4d05d3b23ff757f6387b1335d71e11840aaed593e`. Saved draw archive SHA is `f6e4a727069956a8b8d1769937d8f215ee0e29d7052f9559f3bf81e001b4aa03`. Full individual intervals and denominators are in the [scientific audit](https://github.com/Agnuxo1/NeuroPixel/blob/bb2323ac0e0ba028975a4a4c89d2a97a20c7ff6d/results/research/10_cloud_runs/37618268225-1-audit/analysis/scientific_audit.json).

## 5. Frozen methods

### Scope, provenance and representation

The experiment trains two architectures from scratch on the original English 1k bAbI task 4, “two argument relations.” The benchmark was authored externally to NeuroPixel, and its raw stories and questions are adapted directly. It remains a synthetic question-answering benchmark. This is neither zero-shot transfer of the item-9 checkpoints nor evidence about general real-world language understanding.

Acquisition retained the failed original-author download and the successful documented mirror, pinned by file digests and artifact commit `19c8c40ab50141ba128623e7bbbc04cfb2460f31`. No original-host byte comparison or independently authenticated publisher checksum was established by this acquisition.

Parsing preserves the facts available before each current question, in their original order. Numeric line identifiers are stripped from model input. Tab-separated gold answers and supporting-fact indices are retained as metadata but never supplied to the neural encoder. Earlier questions and their answers are excluded. Lowercased, punctuation-preserving tokens retain sentence order: one fact per row, then a fixed question row and an empty output row. The adapter neither resolves relations nor converts them into the item-9 event grammar.

The 1,000 official TRAIN records contain 919 distinct normalized inputs. Connected components join records sharing an original episode, a complete normalized fact story, or a normalized visible input. The resulting 752 groups are assigned using the prospectively fixed salt `neuropixel-item10-dev-v1` and development fraction 0.1, yielding 895 optimization-TRAIN and 105 DEV records. Grouping is input based; conflicting labels cause admission failure. Vocabulary and geometry are fitted only on optimization TRAIN. The acquired split has no normalized-input overlap, input OOV or unsupported answer.

The fitted canvas is H4×W8, with readout at (3,7), and the vocabulary contains 18 entries including PAD and UNK. The output head covers the full TRAIN-derived vocabulary without an answer-category mask. A ledger includes 149 development fixture contexts, retaining raw exposure identities even for unencodable negative fixtures.

### Models and matching conditions

NeuroPixel uses the current core model directly: identity dimension 16, state width 48, hidden width 128, 16 recurrent local updates, and training firing probability 0.5. Evaluation uses dense updates. The effective PAD identity is zero, which does not imply zero recurrent activity or computation. Local radius grows by at most one per step; the greatest readout distance is seven. Sixteen steps therefore permit geometric access without proving learned transport.

The relative-position Transformer uses two layers, width 32, four heads and feedforward width 144. It has learned two-dimensional relative attention bias, masks empty key positions, retains the output cell as a query, and has no dropout. Both models return one categorical answer at the same output location.

Parameter counts are 29,552 versus 29,562. Transformer feedforward width 144 replaced the width-128 draft after TRAIN geometry was known but before neural outcomes. This matches parameters, not compute, locality, stochasticity or optimization. The comparison concerns two fixed recipes and does not isolate one architectural mechanism.

### Preflight, selection and main training

Six preflight runs were retained. Each family first trained on the same 32 distinct TRAIN inputs, chosen by record-ID order, with seed 58 and LR 0.003. Saved checks every 128 updates required two consecutive accuracy-1.0/CE≤0.05 results, within 4,096 updates. Failed checks and interrupted streaks remain available. Passing establishes tiny-fixture fitting, not held-out competence.

Four pilots used seed 59, 1,024 updates and LR 0.001/0.003 per family. DEV accuracy descending, CE ascending and LR ascending selected 0.003 for both. All six runs were authenticated before main admission; their checkpoints do not replace main models.

The frozen main inventory contains five paired seeds, 60–64, per family, each trained from scratch for exactly 4,096 updates. Training uses batches of 32, AdamW with weight decay 0.0001, gradient clipping at norm 1, and answer cross-entropy only. There is no school/lens loss, early stopping, final-result-driven model choice or checkpoint selection. Full optimization-TRAIN probes and DEV predictions are saved at the fixed endpoint.

Private NumPy generators seeded with 100000 plus the run seed produce identical paired, saved sampling-with-replacement minibatch streams. Torch uses the run seed for initialization and one subsequent reset to 110000 plus that seed for firing. Internal random-number consumption can differ. Planned/consumed stream identities, configurations, weights, logs and predictions are retained.

Training uses Python 3.12.14/Torch 2.6.0+cpu/NumPy 2.2.6/SciPy 1.15.1, two numerical threads, one interop thread and an 8 GiB RAM floor. Contracts/train/final have 300/2,700/300-second ceilings within worker 3,600 seconds, including archival reserve 180 seconds; workflow ceiling is 70 minutes. Shared deadlines prevail.

### Final-access gate, populations and controls

The scientific source for the main job is `f619ac2150960ff658a797e3ac8594f49c8d96c1`. Before final access, all ten completed configurations, checkpoints and summaries must form a durable training manifest. The worker confirms a normal results-branch archive and records its commit and manifest hash. The final controller reauthenticates the inventory and writes an exclusive consumed-access receipt before reading the exact pinned QA4 TEST member. Failed access remains consumed. These receipts provide internal provenance, not external custody or proof of network publication time.

All official TEST rows remain in the primary denominator. Overflow and unsupported gold count wrong; overflow is never truncated or replaced by fabricated input. Input OOV maps to UNK. Separate masks record encodability, output-vocabulary support and NLL eligibility; CE uses only encodable rows with supported gold and reports its denominator.

Three populations are reported: all official rows; inputs novel relative to optimization TRAIN; and a stricter population novel relative to the complete official TRAIN file, including DEV, and the declared fixture ledger. Novelty requires absence of the normalized visible-input identity and, when encodable, absence of the encoded-canvas identity. Unencodable but raw-novel rows remain in the appropriate subset and count wrong. These are exact-identity filters, not guarantees of new relational concepts, new templates, or absence of semantically related stories. Empty subsets have null accuracy and CE, not a successful score.

Eight prospective controls are retained: TRAIN majority, exact-input memory, question-only memory, exact bag-of-tokens memory, bag-of-words naive Bayes, fact-answer-token frequency, frequency excluding query-mentioned answer tokens, and a direct/inverse cardinal-relation symbolic solver. Learned tables and answer categories use optimization TRAIN. Frequency counts facts only. The symbolic solver uses no annotations or transitive closure and abstains on unsupported/ambiguous cases. Raw controls can answer beyond neural shape/vocabulary limits and encode strong task-specific priors; they are not matched-capacity neural baselines.

### Estimands and uncertainty

The primary contrast is the mean, across the five paired main seeds, of NeuroPixel minus Transformer all-official accuracy. Its nominal 95% interval is the paired Student-t interval with four degrees of freedom; the report retains all five raw pairs. It describes training-realization variation on one task/source/split, without treating questions as independent training replicates.

The declared operational screen requires NeuroPixel accuracy of at least 0.95 in every main seed on all official rows and on the strict-unseen subset when that subset is nonempty. An empty strict subset makes that component not established. This exploratory competence screen does not replace the historical H1 criterion or establish novelty.

A separate conditional bootstrap assesses sensitivity to observed TEST source components while holding the ten checkpoints fixed. Components transitively join the same original episode, complete normalized story or normalized visible input and are ordered by canonical group ID. One saved PCG64 index matrix, seed 104001 and 2,000 replicates, is shared across models and subsets. Each draw computes total selected correct divided by total selected population across resampled components, preserving unequal component sizes. Percentile intervals use NumPy's linear 2.5% and 97.5% quantiles. Fewer than two components makes the bootstrap unavailable; any zero-denominator draw makes that endpoint's interval null while preserving its undefined-draw count. Analysis uses Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 with one numerical thread.

The bootstrap neither resamples training realizations nor adjusts multiplicity or establishes real-world population sampling. Known bibliographic exposure and any additional historical exposure lie outside the declared TRAIN/fixture ledger; its exact-input masks do not establish blinding. Success supports bounded ability to learn and perform this externally authored synthetic task under the declared recipe; failure concerns representation, optimization and budget jointly, not universal incapacity.

Source anchors: [frozen main plan](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/docs/research/10_study_plan.json), [controller](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/scripts/research_babi_study.py), [data contract](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/neuropixel/research/babi_qa.py), [independent auditor](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/scripts/research_babi_audit.py).


## 6. Acquisition, publication order and independent verification

### Source provenance

The failed original-author HTTP download returned 404 and remains archived. Acquisition then used the independently documented HTTPS S3 mirror. The compressed archive is 11,745,123 bytes with SHA-256 `84f5296ab9a1ad0dc9464e08c491d65cd08830fca3acae9ab86f75e0fb81573c`. The exact TRAIN member is `tasks_1-20_v1-2/en/qa4_two-arg-relations_train.txt`, 117,470 bytes with SHA-256 `c9370502d85fa382f32d1fea2111e817f81ae9e3e79eb2924a1a5392a90417fa`. Dataset-card metadata declares CC BY 3.0; acquisition did not inspect an embedded license document or establish an independently authenticated publisher checksum.

Only the TRAIN payload was selected, extracted and parsed during acquisition and preflight. TAR headers for other members were traversed, and gzip traversal can decompress bytes belonging to other members. Thus “TEST not parsed or selected in those phases” is the supported statement, rather than a claim that no TEST-compressed byte ever passed through a decompressor. The later audit reads already saved final data, without triggering another neural final evaluation. [Acquisition report](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/docs/research/10_acquisition_results.md).

### Immutable evidence chain

| Stage | Scientific or operational source | Results archive | Actions run |
|---|---|---|---:|
| Acquisition | `99ecaeefdd00dffd3dae444a781ad009c19dff5d` | `19c8c40ab50141ba128623e7bbbc04cfb2460f31` | 37609533224 |
| Preflight | `071be375dce45d0d7fabce3d095bd1003d53ed36` | `707be7d4c1df58bde0706d965e6678c0be6a2cfa` | 37612623870 |
| Preflight independent audit | `6e7cad49305d3d042ba931acaa13bd49a554c2f3` | `83a4f36f07ee05d8fb34fabe37b61b41fbdbe0f8` | 37614761119 |
| Main | `f619ac2150960ff658a797e3ac8594f49c8d96c1` | `0e135ad10518e1ccd9b7adc562265aa4e2d94d2b` | 37615277149 |
| Final independent audit | `3417e8287f76a13e94293f0affcf2c10887fe768` | `bb2323ac0e0ba028975a4a4c89d2a97a20c7ff6d` | 37618268225 |

All five jobs completed successfully. Scientific main plan SHA-256 is `571dd747cb9649bc435ffb006afc6558132b82e0d1c40e7b251827c51821b6f0`, frozen at 11:36:23 UTC; recipe SHA is `8492892f58a4065a92df99dbbbf30f0bd7340111c853cf5533a7102109fe8d19`. Final-audit plan SHA is `670c1999a60a4a0d213ca430a633613d43b10f4aa36db7192760ba578e5244bc`, frozen at 12:02:19 UTC. That later operational audit does not modify the scientific recipe.

The ten completed training inventories were durably published in actual results commit **G=`e8f5cb4913574fd0e27ca52f986da9153bd6996d`**, confirmed at 11:59:30.212256 UTC. The gate receipt followed at 11:59:30.213019; the consumed final-access receipt is at 11:59:31.895875. Training-manifest SHA is `5152d71f9f0e527d4c4eae921649db7ba7addf8975103125d8693177e62be078`. The archive auditor established that G is a strict ancestor of F and that all ten training subtrees are byte-identical at G and F. G has an interim manifest and no final predictions or consumed-access record. This establishes actual Git lineage and the recorded internal order; it is not external custody or independent certification of wall-clock publication time.

Main F contains 141 files totalling 30,293,136 bytes excluding its manifest. Final audit O contains 25 files totalling 23,951,462 bytes excluding its manifest. Source ZIP contents were checked against the actual Git source inventories, including all 449 source files for the main source. Earlier acquisition and preflight source inventories contained 403 and 432 files respectively. Final, gate, preflight and acquisition inventories, including each own manifest, were checked in full.

### What was independently checked

The final archive audit records **10,988 checks, zero failures and zero issues**. The fresh scientific process records **306 checked input files, 35 grouped checks and zero issues**. It reconstructs raw-record identity, grouping, literal encoding, controls, TRAIN/DEV and final metric counts, supported-label NLL, paired differences and component bootstrap from saved data. It imports no Torch, constructs no model, performs no inference and does not deserialize checkpoints. Checkpoint byte identity is verified; weight tensor finiteness is not independently asserted by this audit.

The separate preflight audit recorded 5,819 archive checks and zero issues, including all six preflight runs, all 19 saved memorization checks, selection and actual source archives. The main contract phase executed 32 unique parent test methods and 87 passing subtests, producing 119 passing call reports, with zero failures, errors or skips. Test methods, subtests, archive checks and scientific replications are different units and must not be added into a single scientific sample count.

Root rehashed six final audit output files and compared 2,166 numerical fields against saved raw records with zero discrepancies. A separate reviewer rehashed 21 text files and completed 259 JSON/configuration checks with zero issues, including reconstruction of all ten configurations from preflight selection. Root additionally reconstructed the df=4 t quantile algebraically in JavaScript; the interval differs from the SciPy result by at most 2.22×10⁻¹⁶ in accuracy units. This verifies arithmetic, not the small-sample assumptions.

The final scientific report is 458,002 bytes, SHA-256 `4eeb8937fbc37a585d9102f23e6d4062aef5b29f674137d646ef59df4f24f5c7`; the actual Git/archive report is 2,201,961 bytes, SHA-256 `1732a7ce70e7380f76c9b4d5fdf61392b73ae4491300ce9dc548c50b54e9ed82`. The complete audit completion receipt and provider-decoded logs are retained in `results/research/10_validation/`. This constitutes independent implementation/recount within the same project workflow, not an external laboratory replication.

## 7. Resource costs, failed accesses and limits of measurement

NeuroPixel training-loop time totals 1,086.050522 seconds, mean 217.210104 seconds; Transformer totals 167.173937 seconds, mean 33.434787 seconds. These are observed elapsed loop times on this CPU recipe. Matching parameter counts did not match update cost, FLOPs, parallelism or stochastic behavior. The 131,072 sampled records per main run divided by the 895-record TRAIN population gives about 146.45 population equivalents; sampling was with replacement, so this is not 146.45 shuffled epochs.

The main scientific training process took 1,259.143942 seconds; the supervised training stage took 1,261.389100 seconds, within its 2,700-second ceiling. Scientific final evaluation took 4.770527 seconds; its supervised stage took 6.503731 seconds. The main worker took 1,284.266404 seconds before final archival. The acquisition and preflight worker times were 12.401777 and 262.467098 seconds. The preflight and final offline-audit worker times were 11.168775 and 15.924905 seconds.

During the main job, the one-second parent supervisor retained five contract samples, 1,261 training samples and seven final samples. Minimum sampled available RAM was 14.359989, 14.273090 and 14.450592 GiB respectively, above the 8 GiB floor. The final audit used one numerical thread and recorded eight parent samples plus 82 scientific boundary samples. The minimum during the eight-sample audit-stage monitor was 14.464985 GiB; a later worker boundary before final archival recorded 14.460808 GiB. All runners were released. These are available-RAM samples, not continuous minima, peak process RSS, CPU utilization or energy measurements. No joule, carbon, photonic or hardware-efficiency claim follows.

The acquisition's original-author 404 remains a data-access failure with a documented fallback. A root read-only lookup asked for `archive_manifest.json` at an interim main snapshot and received 404; the observed Git tree identified `interim_manifest.json`, after which the four contract artifacts were retrieved and rehashed. That was a metadata lookup issue, not a failed numerical experiment. Prospective unexecuted implementation drafts are retained as development history and are not counted as model runs. Local execution remained unavailable; successful computation and audits used the authorized bounded standard public CPU workflows.

## 8. Relation to prior work and remaining claims

The original bAbI publication deliberately supplies prerequisite toy tasks; QA4 tests argument order and relations. External authorship therefore expands evaluation beyond NeuroPixel's own event generator while preserving a narrow synthetic setting. The original author loader documents the dataset member and source URL. The author generator repository warns that its rewritten generator does not reproduce the published bytes, supporting the choice to pin the existing archive rather than silently regenerate a replacement. [Weston et al., bAbI](https://arxiv.org/abs/1502.05698); [official dataset loader](https://huggingface.co/datasets/facebook/babi_qa/blob/main/babi_qa.py); [author generator documentation](https://github.com/facebookarchive/bAbI-tasks/blob/master/README.rst).

End-to-End Memory Networks reports bAbI experiments under different version and training/selection conditions. Its historical QA4 numbers are not a matched replication or a ranking against the present models. That context was read after the preflight freeze and did not alter the fixed recipe. [Sukhbaatar et al.](https://arxiv.org/html/1503.08895v5). Deliberate compositional holdouts such as SCAN and COGS address different generalization questions; merely filtering repeated QA4 inputs does not implement those protocols. [Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a.html); [Kim and Linzen, 2020](https://aclanthology.org/2020.emnlp-main.731/).

This item leaves the following claims unestablished: high and stable neural competence across seeds; separation of optimization failure from representational limits; transfer of learned states or weights to a different task; novel vocabulary or deliberate compositional splits; broad natural-language or multimodal performance; persistence of activation memory between calls; independent external replication; and practical or Nobel-level impact. These are limitations, not negative universal theorems.

No post-result retuning was performed to rescue the item-10 outcome. The feasible item-10 investigation closes with the complete negative and positive evidence preserved. Item 11 may next investigate durable memory and interference under its own prospective scope; later items were not opened by this report.
