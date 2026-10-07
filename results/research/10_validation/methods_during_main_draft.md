# Item 10: methods and interpretation boundaries

Draft for the final report, prepared during the main job without inspecting its TEST payload or outcomes. This text specifies the frozen procedure; it does not assert that the main job or final audit has completed.

## Scope, provenance and representation

The experiment trains two architectures from scratch on the original English 1k bAbI task 4, “two argument relations.” The benchmark was authored externally to NeuroPixel, and its raw stories and questions are adapted directly. It remains a synthetic question-answering benchmark. This is neither zero-shot transfer of the item-9 checkpoints nor evidence about general real-world language understanding.

Acquisition retained the failed original-author download and the successful documented mirror, pinned by file digests and artifact commit `19c8c40ab50141ba128623e7bbbc04cfb2460f31`. No independently authenticated publisher checksum was available.

Parsing preserves the facts available before each current question, in their original order. Numeric line identifiers are stripped from model input. Tab-separated gold answers and supporting-fact indices are retained as metadata but never supplied to the neural encoder. Earlier questions and their answers are excluded. Lowercased, punctuation-preserving tokens retain sentence order: one fact per row, then a fixed question row and an empty output row. The adapter neither resolves relations nor converts them into the item-9 event grammar.

The 1,000 official TRAIN records contain 919 distinct normalized inputs. Connected components join records sharing an original episode, a complete normalized fact story, or a normalized visible input. The resulting 752 groups are assigned using the prospectively fixed salt `neuropixel-item10-dev-v1` and development fraction 0.1, yielding 895 optimization-TRAIN and 105 DEV records. Grouping is input based; conflicting labels cause admission failure. Vocabulary and geometry are fitted only on optimization TRAIN. The acquired split has no normalized-input overlap, input OOV or unsupported answer.

The fitted canvas is H4×W8, with readout at (3,7), and the vocabulary contains 18 entries including PAD and UNK. The output head covers the full TRAIN-derived vocabulary without an answer-category mask. A ledger includes 149 development fixture contexts, retaining raw exposure identities even for unencodable negative fixtures.

## Models and matching conditions

NeuroPixel uses the current core model directly: identity dimension 16, state width 48, hidden width 128, 16 recurrent local updates, and training firing probability 0.5. Evaluation uses dense updates. The effective PAD identity is zero, which does not imply zero recurrent activity or computation. Local radius grows by at most one per step; the greatest readout distance is seven. Sixteen steps therefore permit geometric access without proving learned transport.

The relative-position Transformer uses two layers, width 32, four heads and feedforward width 144. It has learned two-dimensional relative attention bias, masks empty key positions, retains the output cell as a query, and has no dropout. Both models return one categorical answer at the same output location.

Parameter counts are 29,552 versus 29,562. Transformer feedforward width 144 replaced the width-128 draft after TRAIN geometry was known but before neural outcomes. This matches parameters, not compute, locality, stochasticity or optimization. The comparison concerns two fixed recipes and does not isolate one architectural mechanism.

## Preflight, selection and main training

Six preflight runs were retained. Each family first trained on the same 32 distinct TRAIN inputs, chosen by record-ID order, with seed 58 and LR 0.003. Saved checks every 128 updates required two consecutive accuracy-1.0/CE≤0.05 results, within 4,096 updates. Failed checks and interrupted streaks remain available. Passing establishes tiny-fixture fitting, not held-out competence.

Four pilots used seed 59, 1,024 updates and LR 0.001/0.003 per family. DEV accuracy descending, CE ascending and LR ascending selected 0.003 for both. All six runs were authenticated before main admission; their checkpoints do not replace main models.

The frozen main inventory contains five paired seeds, 60–64, per family, each trained from scratch for exactly 4,096 updates. Training uses batches of 32, AdamW with weight decay 0.0001, gradient clipping at norm 1, and answer cross-entropy only. There is no school/lens loss, early stopping, final-result-driven model choice or checkpoint selection. Full optimization-TRAIN probes and DEV predictions are saved at the fixed endpoint.

Private NumPy generators seeded with 100000 plus the run seed produce identical paired, saved sampling-with-replacement minibatch streams. Torch uses the run seed for initialization and one subsequent reset to 110000 plus that seed for firing. Internal random-number consumption can differ. Planned/consumed stream identities, configurations, weights, logs and predictions are retained.

Training uses Python 3.12.14/Torch 2.6.0+cpu/NumPy 2.2.6/SciPy 1.15.1, two numerical threads, one interop thread and an 8 GiB RAM floor. Contracts/train/final have 300/2,700/300-second ceilings within worker 3,600 seconds, including archival reserve 180 seconds; workflow ceiling is 70 minutes. Shared deadlines prevail.

## Final-access gate, populations and controls

The scientific source for the main job is `f619ac2150960ff658a797e3ac8594f49c8d96c1`. Before final access, all ten completed configurations, checkpoints and summaries must form a durable training manifest. The worker confirms a normal results-branch archive and records its commit and manifest hash. The final controller reauthenticates the inventory and writes an exclusive consumed-access receipt before reading the exact pinned QA4 TEST member. Failed access remains consumed. These receipts provide internal provenance, not external custody or proof of network publication time.

All official TEST rows remain in the primary denominator. Overflow and unsupported gold count wrong; overflow is never truncated or replaced by fabricated input. Input OOV maps to UNK. Separate masks record encodability, output-vocabulary support and NLL eligibility; CE uses only encodable rows with supported gold and reports its denominator.

Three populations are reported: all official rows; inputs novel relative to optimization TRAIN; and a stricter population novel relative to the complete official TRAIN file, including DEV, and the declared fixture ledger. Novelty requires absence of the normalized visible-input identity and, when encodable, absence of the encoded-canvas identity. Unencodable but raw-novel rows remain in the appropriate subset and count wrong. These are exact-identity filters, not guarantees of new relational concepts, new templates, or absence of semantically related stories. Empty subsets have null accuracy and CE, not a successful score.

Eight prospective controls are retained: TRAIN majority, exact-input memory, question-only memory, exact bag-of-tokens memory, bag-of-words naive Bayes, fact-answer-token frequency, frequency excluding query-mentioned answer tokens, and a direct/inverse cardinal-relation symbolic solver. Learned tables and answer categories use optimization TRAIN. Frequency counts facts only. The symbolic solver uses no annotations or transitive closure and abstains on unsupported/ambiguous cases. Raw controls can answer beyond neural shape/vocabulary limits and encode strong task-specific priors; they are not matched-capacity neural baselines.

## Estimands and uncertainty

The primary contrast is the mean, across the five paired main seeds, of NeuroPixel minus Transformer all-official accuracy. Its nominal 95% interval is the paired Student-t interval with four degrees of freedom; the report retains all five raw pairs. It describes training-realization variation on one task/source/split, without treating questions as independent training replicates.

The declared operational screen requires NeuroPixel accuracy of at least 0.95 in every main seed on all official rows and on the strict-unseen subset when that subset is nonempty. An empty strict subset makes that component not established. This exploratory competence screen does not replace the historical H1 criterion or establish novelty.

A separate conditional bootstrap assesses sensitivity to observed TEST source components while holding the ten checkpoints fixed. Components transitively join the same original episode, complete normalized story or normalized visible input and are ordered by canonical group ID. One saved PCG64 index matrix, seed 104001 and 2,000 replicates, is shared across models and subsets. Each draw computes total selected correct divided by total selected population across resampled components, preserving unequal component sizes. Percentile intervals use NumPy's linear 2.5% and 97.5% quantiles. Fewer than two components makes the bootstrap unavailable; any zero-denominator draw makes that endpoint's interval null while preserving its undefined-draw count. Analysis uses Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 with one numerical thread.

The bootstrap neither resamples training realizations nor adjusts multiplicity or establishes real-world population sampling. Public exposure outside the ledger remains unknown. Success supports bounded transfer to this external synthetic task; failure concerns representation, optimization and budget jointly, not universal incapacity.

Source anchors: [frozen main plan](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/docs/research/10_study_plan.json), [controller](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/scripts/research_babi_study.py), [data contract](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/neuropixel/research/babi_qa.py), [independent auditor](https://github.com/Agnuxo1/NeuroPixel/blob/f619ac2150960ff658a797e3ac8594f49c8d96c1/scripts/research_babi_audit.py).
