# Item 9 — Prospective Stage-B admission and statistical review

**Status: design review only; Stage B is not admitted by this document.** Actual Stage-A outcomes, archive verification and cost estimates are pending. This note was prepared without reading pilot outcomes, generating evaluation data, running models or modifying the 49 files bound by the Stage-A freeze. It creates no new performance threshold, arm, stopping rule or permission to inspect final examples.

The source anchor supplied by root is commit `e63764ccb924ab23d65c2deb94b477387d102c82`, with preflight freeze `2026-10-07T07:45:55.891523+00:00`. The authoritative scientific recipe remains [09_experiment_recipe.json](09_experiment_recipe.json), SHA-256 `7450b1ede40542c5ff940842e42eb2725ac4420946baad6bac920e779a214b97`; the frozen [protocol](09_protocol.md) has SHA-256 `d47689aab66101cc479fdee5f60dae59ec1941ea489a5e32b04659a8b70577d8`. This note supplements those sources without amending them. The [trainer review](09_trainer_review.md) records the earlier source checks and resolved handoff defects.

## 1. What admits or blocks Stage B

The frozen workflow separates technical completion from the strength of a diagnostic score. A poor but valid result is evidence to retain, not a newly introduced exclusion criterion.

| Stage-A observation | Required treatment under the frozen design |
|---|---|
| Cloud contract suite completes with its exact 34-case inventory, correct classes/runtime and no skips or failures | Required technical evidence. Local skipped Torch cases do not substitute for it. |
| Either memorization run fails technically, is incomplete, contains nonfinite values or is interrupted by a resource/deadline guard | Blocks the normal Stage-A selection path. Preserve the attempt and its failure; do not substitute a different run silently. |
| A memorization run completes its permitted budget without reaching its competence criterion | Record its failure-to-reach flag. This is **not** a protocol-defined blocker, not a claim of architectural impossibility and not authorization to enlarge the scientific budget. |
| Memorization reaches the criterion | Check that two consecutive scheduled checks each have all 32 answers correct and mean CE ≤0.05. It is a training-fixture diagnostic, not primary weights, LR-selection evidence or held-out generalization. |
| All four pilot runs complete exactly 1,024 updates at initialization 39, one per family/rate combination, with finite valid validation metrics and common source/data contexts | Required. Retain both selected and unselected rates. An incomplete or duplicated inventory cannot select a winner. |
| All pilot validation scores are low, rates tie, or a probe binding score is below 0.95 | Apply the declared ranking unchanged; retain scores and flags. There is no minimum pilot/probe score for admission and no declared futility rule. |
| A pilot's validation result is unexpectedly strong | Apply the same ranking. It neither removes primary replications nor authorizes final access, early stopping or a larger claim. |
| Source, runtime, dataset, checkpoint, copied selection or archive identities fail verification | Blocks admission until the discrepancy is resolved through explicit preserved/versioned evidence. A hash field cannot be replaced merely to make an unexplained mismatch pass. |
| The complete Stage-B workload cannot reasonably fit the approved resources/deadline | Do not launch an inventory likely to be truncated. Record resource infeasibility and obtain a prospective operational decision. Do not reduce updates, seeds, families or final conditions silently. |
| The actual final study collection was generated or accessed outside the prescribed gate | Stop and document exposure. Reusing the same instances or changing the seed cannot silently restore the promised history. |

The memorization criterion's earliest normal stopping opportunity is the second scheduled check; a single qualifying last check is insufficient. Comparisons use saved unrounded values. A returned `status="completed"` with `memorization_criterion_reached=false` is different from a run failure. The preflight controller requires both memorization runs and all four pilots to complete before writing its selection; its competence flags do not alter `select_pilot`.

For each family, selection is descending **base validation binding**, then ascending **base validation CE**, then ascending LR among 0.001 and 0.003. The validation panel has 128 bags × eight queries = 1,024 rows. Final conditions, probe performance, memorization success, wall time and final outcomes do not enter the ranking. Stage-A seed 39 is a development realization, not an additional primary replicate.

## 2. Root-to-Stage-B provenance obligations

Before freezing Stage B, root and the independent archive review must establish the following chain using saved evidence rather than a remembered winning rate:

1. Identify the actual Stage-A workflow attempt, frozen source commit, final stopped archive commit and complete archive manifest. Verify the claimed files and distinguish final evidence from periodic incomplete snapshots. Confirm completed worker/scientific-stage status and the actual pinned Python 3.12.8 / CPU Torch 2.6.0 runtime and resource records.
2. Recount the four pilots' saved base-validation predictions against the authenticated 1,024-row validation panel. Verify targets, roles, group/record alignment, logits/NLL finiteness and the exact binding/CE values used for selection. Verify the two memorization outcomes and flags against their saved checks; do not turn those flags into another selection variable.
3. Authenticate development group identities, counts and support, the complete manifest, fixture-exposure ledger and its prospective exclusions. The declared ledger contains 100 candidate bags, including 31 constructed fixture groups. Shared primitive tokens across partitions are intentional. Membership-only candidate inspection is not a claim that a performance model trained on those candidates.
4. Construct the versioned Stage-A receipt from the exact complete `pilot_selection.json` and `preflight_runs.json` copies. Preserve their original hashes/byte counts and source/run/archive provenance. Verify the receipt against the archive before binding `receipt_path` and `receipt_sha256` into plan B. Canonical reconstruction is accepted as exact only if it matches the archived original bytes/hash.
5. Recompute the frozen family-wise selection and require equality with both the copied selection and plan-B rates. Require the same recipe hash and development identities. The implemented validator enforces internal consistency; the separate archive audit establishes the claimed historical provenance.
6. Freeze the complete Stage-B source, receipt, dataset recipe and primary inventory. Preserve all ten runs: both families × seeds 40–44, 4,096 updates each, final scheduled checkpoints, common batch 32 and the unchanged optimizer/loss/model settings. Added analysis or operational files do not justify silently changing any scientific definition.

Stage B regenerates development data and requires exact expected identities before training. Its final child must match the training context. Successful admission to training is **not admission to final access**: all ten primary runs must first complete and be authenticated, their full inventory must be archived, and the consume-once access record must be written and fsynced before final seed 91003 is used. Any failed or incomplete primary panel blocks that ordinary final path.

## 3. Projecting cost without changing the scientific budget

The fixed primary workload is **40,960 optimizer updates**: 5 × 4,096 per family. Final neural scoring contains **122,880 nominal query rows**: ten checkpoints × 12,288 rows. These are accounting counts, not FLOPs, energy estimates or observed timings. Each primary run also saves its checkpoint and scores the fixed 512-row probe and 1,024-row base validation panel.

Use Stage-A timing/resource evidence from the same pinned runtime and model/grid/batch settings. For each family, inspect both LR pilots and retain the slower valid timing when forming an admission estimate, rather than choosing a favorable run. A transparent conservative planning decomposition is:

\[
\widehat T_B = 5\sum_f(4096\,u_f + h_f + q_f) + T_{\mathrm{setup/tests/archive}} + M.
\]

Here `u_f` is a documented per-update estimate from completed pilot training logs; `h_f` covers one primary run's initialization, checkpointing, probe and validation overhead; `q_f` covers its 12,288-row final inference and persistence; and `M` is an explicit operational uncertainty margin. Record the exact extraction rule and all components before admission. The pilot's `wall_seconds` includes work beyond training, while the last training-log elapsed value precedes endpoint evaluation; their difference is an overhead proxy, not a clean inference benchmark. If that proxy is used, state the approximation. The final panel has eight times as many rows as probe plus validation combined; multiplying by eight is only a planning approximation and does not account automatically for data generation, serialization or archive growth.

Do not run or load final data to calibrate this estimate. Do not extrapolate from a short memorization run as though its repeated fixed batch represented randomized training cost. Include contract tests, development reconstruction, periodic/final archival, source checks and shutdown reserve. The current worker's study policy is finite: 13,800 worker seconds, 240 workflow minutes and a 180-second final-archive reserve. Thus the nominal worker execution window is at most 13,620 seconds, shared by tests, training, final evaluation and intervening operations; it is not 13,620 seconds of model training alone. Compare the full estimate to that window and the observed ≥8 GiB RAM constraint. A projection is not a guarantee against host variation or later resource failure.

The estimate may determine whether resources admit the unchanged study. It must not select a better-performing architecture, LR, checkpoint or seed, nor quietly replace 4,096 updates or five replications. A resource-only change, if required and authorized, needs a new explicit pre-execution plan/source binding and a preserved reason; otherwise record non-admission.

## 4. Statistical interpretation fixed before outcomes

For final checkpoint `s` and scenario `g`, base binding averages the four agent/patient answers across two events. Averaging these scenario values gives the balanced macro endpoint `b_fs`. For paired seed `s`, define `d_s = b_NP,s − b_TF,s`. Report all five values per family and all five differences, plus mean, sample SD and range. The declared two-sided 95% interval is

\[
\bar d \pm t_{0.975,4}\,s_d/\sqrt{5},\qquad
s_d^2 = \frac{1}{4}\sum_{s=1}^5(d_s-\bar d)^2.
\]

These are **five paired training realizations**. The seed changes initialization and the private minibatch stream, plus NCA firing; it does not isolate initialization variance. Pairing is appropriate because corresponding families use the same minibatch stream and final scenarios. It does not equate their parameterizations or stochastic computation. The interval is conditional on this dataset, recipe and selected rates, uses a small-sample normal-theory assumption, and omits dataset-generation and LR-selection uncertainty. Do not infer five × 256 independent training replicates or append the pilot to obtain a larger `n`. Missing declared pairs cannot be imputed or silently replaced with an altered df=4 analysis.

The scenario bootstrap has a different estimand. Use the declared 2,000 resamples and seed 94001, sharing each sampled vector of 256 bag indices across all checkpoints, conditions and eight queries. Preserve the resampling artifact. Percentile 95% intervals are conditional on the saved trained checkpoints and sampled scenario distribution. They do not replace the five-realization interval or incorporate total pipeline uncertainty. Zero eligible counterfactual rows must produce `n=0` and an undefined/null rate, not artificial perfect consistency.

For each checkpoint/condition, retain global and four-role accuracy, binding, mean CE, all-eight correctness and complete predictions. In each final condition, global denominators are 2,048 query rows; each role has 512, binding has 1,024, and all-eight correctness has 256 scenarios. Paired counterfactual eligibility is:

| Variant | Changed-gold denominator | Invariant denominator |
|---|---:|---:|
| Queried agent/patient swap | 1,024 noun queries | 1,024 action/place queries |
| Other-event agent/patient swap | 0 | 2,048 |
| Global event relabel including query | 0 | 2,048 |
| Query-event switch | 2,048 | 0 |
| Layout permutation | 0 | 2,048 |

Within a role, its corresponding nonzero denominator is 512. Base/variant joint correctness uses the explicitly reported paired set and accompanies equality on invariant cases or inequality on changed-gold cases. Conditions contain dependent or duplicate inputs, and cannot multiply the effective sample size. The independent analysis author confirmed this denominator design, shared bag resampling and null treatment; this statement records coordination, not approval of code that has not yet been reviewed.

Uniform controls retain exact correct-answer probabilities rather than sampled pseudo-predictions. Their global/binding expectations are role-only 0.5/0.5, event-category 0.75/0.5 and bag-category 0.375/0.25. The majority control uses only training answers. Control expectations are mathematical references, not evidence that a learned model resolves roles.

The competence screen requires every one of the five NeuroPixel runs to have base binding ≥0.95, base all-eight accuracy ≥0.90 and binding ≥0.90 in each declared condition. Apply it to unrounded metrics and report every component. It is a final descriptive screen, **not a Stage-B admission rule**, a significance test, a Transformer-superiority criterion or a reopened H1. Secondary comparisons remain descriptive without a family-wide discovery claim. Failure does not prove the capacity is unattainable under every training recipe.

## 5. Actual admission fields — pending evidence

| Required field | Current state |
|---|---|
| Stage-A workflow/run/attempt and final archive commit | PENDING — not inspected in this design review |
| Archive/file/source/runtime audit and exact cloud test outcomes | PENDING |
| Two memorization statuses, completed updates, criteria and flags | PENDING |
| Four complete pilot validation binding/CE scores and independent recount | PENDING |
| Selected LR per family, including exact tie-break evidence | PENDING |
| Authenticated development identities, manifest and exposure ledger | PENDING |
| Exact Stage-A receipt path/hash and root archive-provenance approval | PENDING |
| Per-family timing inputs, overhead/margin, RAM observations and total projection | PENDING |
| Stage-B plan/source freeze preserving all ten runs and final conditions | PENDING |
| Admission conclusion with explicit reasons | **NOT YET DETERMINED** |

No external literature was needed to resolve these implementation-specific obligations. General methodological support and its limits remain in the frozen [evidence note](09_methodological_evidence.md); this review does not introduce a new benchmark, numerical simulation or subsequent programme item.
