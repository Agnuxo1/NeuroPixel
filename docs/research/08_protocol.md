# Item 8 — Metric and invariant correction protocol

## Status and scientific question

Item 8 opened on 2026-10-07T06:27:00.782435+00:00 after the archived closure of item 7 at commit `ef25ed6497cd3554d8eb78a0ef278f7b589e1e1f`. Items 9–30 remain unopened. This protocol records the source-level diagnoses and locally observed regressions before the pinned Torch integration run. The accompanying JSON cloud plan will be frozen only after all implementation hashes and the declared inventory are final.

The question is whether four concrete contracts are implemented correctly: effective PAD identity, descriptive expert-routing summaries, classification denominators and invalid-input rejection, and Soil CDF/fold evaluation. These are software and mathematical correctness questions. They do not test a model-performance advantage, a new task capability or a Nobel-level contribution. Item-5 H1 remains not supported. Item-6 and item-7 conclusions are not reclassified.

## Evidence policy

Preserve the exact source before each substantive correction, the diagnostic that fails on it, every recorded unsuccessful regression, the corrected source and its tests. Local runs use the actual source functions, sometimes extracted by AST to avoid the unavailable local Torch dependency; such extraction is not full-module integration. A separately frozen standard CPU run supplies actual Torch integration. No old result JSON is rewritten and no missing historical predictions are reconstructed from an insufficient summary statistic.

The four focused audit notes give code-level details and the receipts under `results/research/08_validation/` retain raw transcripts. The final report will separate observed facts, implementation decisions and unresolved empirical questions. Successful synthetic contracts do not validate a hidden competition scorer or an unavailable real dataset.

## A. Effective PAD identity

**Diagnosis.** The core stores `nn.Embedding(..., padding_idx=0)` but obtains its effective dictionary tensor and calls functional embedding without the padding argument. The same dictionary is reused for identity reinjection and tied lens readout. A zero initial row therefore does not by itself guarantee zero gradient along either path. The original source is preserved with SHA-256 `564045fff799185e15b921873225d344014a4d4bcae6df6deacaf42f6e4bd701`.

**Intervention.** Project row zero of the effective dictionary to a constant zero after grounding, preserving all other effective rows, parameter names and checkpoint shapes. Existing nonzero stored PAD weights and optimizer momentum may remain in historical checkpoint state; the contract concerns the effective identity used by inference and gradient propagation. Do not assert that every raw stored weight remains zero. `ResearchNCA` has its own dictionary and forward implementation and an explicit `freeze_pad` setting; preserve that historical research recipe.

**Falsifiable checks.** In a fixed four-token, two-state-channel, two-update 2×2 fixture, the preserved source must exhibit a nonzero PAD gradient and changed effective row after one SGD step through each of the answer-reinjection and lens-readout routes. The correction must keep its effective PAD zero while non-PAD rows learn. Additional tests cover a nonzero old checkpoint, resumed AdamW momentum, grounding, unchanged non-PAD behavior in the entirely occupied, answer-only fixture with no PAD objective, the streaming path, observed camera input and the explicit ResearchNCA compatibility policy. Zero identity or zero initial state does not imply zero later recurrent activity.

**Scope.** These are deterministic optimizer fixtures, not a learning benchmark. The cloud demonstration repeats the two tiny routes also covered by the regression suite; report this duplication openly rather than describing the whole item as only two optimizer steps.

## B. Expert-routing summaries

**Diagnosis.** The old `t_grow2` scalar is the mean of chosen slot identifiers. Arbitrary nominal identifiers do not define a quantity of expert-selection quality. The preserved counterexample `[1,1]` versus `[0,2]` has the same old mean 1.0 and different distributions. The old source and its expected failing witness are retained.

**Intervention.** Emit counts and fractions for every declared slot, unused slots included; modes and a nullable unique mode; optional contingencies for actual known topic IDs; and exact score-tie counts only when finite aligned score rows are supplied. An optional explicit topic-to-slot map permits a named mapped-slot agreement. That value is neither answer accuracy nor an oracle-expert score. Pool examples before summarizing so unequal batch sizes are weighted by examples. Do not infer missing ties or mappings.

**Falsifiable checks.** Preserve the counterexample, check empty/invalid inputs, unused slots, nominal relabeling, ties, mapping completeness and unequal batch sizes. Verify the integrated drivers retain model construction, training, argmax and prediction gathering. The first-maximum argmax rule is order-sensitive on tied scores; a relabeling argument for fixed assignments must not be presented as a guarantee for reordered tied score columns.

**Scope.** Historic three-slot frequencies cannot be recovered from their scalar mean. Adaptive drivers gain corrected descriptive outputs; this item neither retrains them nor changes their model-selection policy.

## C. Classification metrics

**Diagnosis.** The previous summary could accept unknown role IDs, causing the global denominator to include examples absent from every declared role denominator. It also accepted a scalar, short or incorrectly shaped NLL array as a mean over a different population. Nonfinite/negative NLL values and some inappropriate count/interval argument types were insufficiently rejected at the function boundary.

**Intervention.** Validate nonempty equal-length integer prediction/target/role vectors, the exact declared role inventory, nonnegative predictions and non-PAD targets. If supplied, NLL must have exactly one finite nonnegative real value per prediction. Validate binary counts and confidence; validate actual boolean interval requests and bootstrap seed/repetition arguments before constructing an RNG. Keep existing micro accuracy, all-role macro accuracy, agent/patient macro accuracy, float64 NLL mean, Wilson and bootstrap formulas for valid inputs.

**Falsifiable checks.** A fixed balanced four-row fixture must agree exactly with the previous valid output. A separate unbalanced-role fixture is checked against independently calculated expectations to demonstrate the distinct micro and macro denominators. Reject each malformed input class and ensure invalid bootstrap arguments do not create an RNG. Locally, actual source-function ASTs are executed without Torch. The declared cloud run must also execute the real-module integration case, which is locally skipped.

**Scope.** The helper cannot infer the model vocabulary from the observed predictions; upper vocabulary bounds and semantic correctness remain the caller's dataset/model contract. Nothing in the local witnesses establishes that item-5 or item-6 recorded evaluations supplied invalid arrays. No earlier score is numerically revised by inference.

## D. Soil CDF and cross-validation evaluation

**Diagnosis.** The original mean-CDF baseline used all labels before forming folds, so held-out labels influenced the reference predictor. Cumulative error arrays were printed with per-fold field names. Invalid decreasing curves could be silently clipped/renormalized in CDF-to-PDF conversion. The first correction then exposed two integration defects: missing meaningful evidence before the first completed fold, and a no-photo uniform fallback whose cumulative floating-point endpoint could be slightly above 100. Every observed local failure and intermediate source/test snapshot is preserved.

**Intervention.** Fit the mean-CDF baseline from training-fold labels only and pass only those labels to the training call. Record raw per-sample target, prediction and reference curves, scores, normalized and raw sample identity, fold membership and photo/device metadata. Distinguish current-fold summaries from pooled-to-date summaries and pool equal-weight samples. Reject nonfinite, malformed, out-of-range or decreasing CDFs; accept only a declared endpoint roundoff tolerance of 1e-4 percentage points. Construct the uniform fallback with an exact final value of 100 before validation; do not relax bounds. Write complete initial provenance before training, persist every completed sample, and retain structured failure context before re-raising ordinary exceptions.

**Metric.** Preserve the project's existing trapezoidal area of absolute CDF differences over its log10 diameter grid, with CDF values expressed in percentage points. NumPy documents the composite trapezoid rule; SciPy documents the distinct distributional Wasserstein distance. A trapezoidal approximation at eleven finite grid points is not automatically the exact discrete 1-Wasserstein distance. The official hidden competition scorer has not been independently inspected in full, so no identity with it is asserted. The historical prediction function still contains its own clamp; the new validator itself performs no clipping or monotonic projection.

**Falsifiable checks.** Cover CDF validity and PDF conversion; exact metric examples including a distinction from discrete Wasserstein; identity collisions; complete nonoverlapping group folds; train-only reference fitting; unequal-fold sample aggregation; unrounded score records; and the actual CV main body extracted by AST with inert model/data substitutes. A failure injected into the second prediction must leave the first sample and planned initial evidence available. Evaluate the actual submission fallback constructor expression for the missing-photo case; this does not execute the entire submission control flow.

**Scope.** These fixtures use no competition images and measure no model quality. Random sample folds do not establish a holdout by phone/device or external geographic/material domain. The current prediction path includes mirroring in CV and submission; a historical coordination description of another run is not evidence that today's code lacks CV mirroring. Dataset hashes, real competition reruns and external-domain validation remain unresolved.

## Integration recipe and decision rule

Use the exact committed source and JSON plan, standard public Ubuntu 24.04, Python 3.12.8, Torch 2.6.0+cpu and the existing pinned `requirements/research-cloud.txt`. Run the PAD legacy demonstration and the entire declared test suite once in fresh bounded child stages. Full-suite execution is justified by changed core dictionary and research metric integration, including compatibility with prior ablation and split contracts; it is not a request for additional scientific training or final task performance.

Configure one numerical and one inter-op thread. The existing device-selection compatibility fixture explicitly requests two numerical threads; allow that bounded fixture within the aggregate ceiling of four and restore the declared controls. Keep at least 8 GiB available RAM at admission and during sampled supervision, a 900-second limit per stage and 35-minute workflow limit including setup/archival. Runtime versions and actual child controls must be recorded. No GPU, paid runner, cache or Actions artifact-storage upload is requested.

The expected inventory is the complete set of frozen test files with pytest parameter expansion, plus the separate two-route PAD witness. Optional CIFAR data absence is the only anticipated test skip; the classification real-module integration must run. Every other error, failure, skip or inventory discrepancy requires an explicit diagnosis and preserved evidence before closure. If a correction is needed after this freeze, retain the failed run, document the change and freeze a new source/plan revision. Do not silently rerun until green.

The full-suite historical-source fixture originally copied the live model and metric source while asserting item-6 historical hashes. Its rejection after the legitimate item-8 edits was reproduced locally with the actual historical gate. The fixture now copies the two hash-pinned preserved pre-8 sources and retains all gate assertions. The historical controller and recovery hash policy remain unchanged and correctly refuse the modern source as a historical replay. A separate narrow test-setup change guards the process-global inter-op initializer so independently authored test classes can share one pinned test process. These are integration maintenance changes, not changes to earlier study evidence.

Completion requires: the declared legacy PAD violations reproduced; corrected contracts and existing integration checks accepted; exact source and raw-output archive verified; all observed failures retained; independent review of material corrections; and a report that keeps empirical limitations. Closing this item means the feasible correctness investigation is complete. It does not certify all possible inputs or promote the project's scientific claims.

## Primary methodological sources

1. scikit-learn developers, Common pitfalls, data leakage: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage . Accessed 2026-10-07. The training-only fitting rule supports the baseline correction; the project's actual defect is established by its own source and counterexample.
2. NumPy 2.3, `numpy.trapezoid`: https://numpy.org/doc/2.3/reference/generated/numpy.trapezoid.html . Accessed 2026-10-07. Definition of the composite trapezoidal operation; not an endorsement of a particular competition metric.
3. SciPy 1.16.2, `scipy.stats.wasserstein_distance`: https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.stats.wasserstein_distance.html . Accessed 2026-10-07. Distinguish discrete-distribution Wasserstein distance from this repository's preserved numerical approximation.

Primary implementation sources are the archived original and corrected NeuroPixel files and the exact Torch release used for the numerical fixture. Attempts to open the versioned generated PyTorch embedding documentation in the research interface were unsuccessful; no unsupported quotation from those pages is used.
