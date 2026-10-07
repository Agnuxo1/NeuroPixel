# Item 8 — Results: metric and invariant corrections

**Scope:** correctness of four concrete software/mathematical contracts, with preserved failures, fixed-source CPU integration and independent checks of saved evidence. This item establishes no new model-performance advantage. Item-5 H1 remains not supported.

## 1. Result and evidence boundary

The corrected implementation passed all **58 new regression methods**. The complete frozen suite collected **187 cases**: **186 passed, one declared optional CIFAR-data test skipped, zero failures and zero errors**. The actual classification-module integration ran in Torch; its local dependency skip did not carry into the cloud result. Both separate cloud stages completed on the first frozen attempt. These figures are verified from individual collected IDs, JUnit case elements and thread-boundary records, rather than trusting a single CI badge.

| Area | Demonstrated problem or contract | Implemented and verified outcome | Remaining empirical boundary |
|---|---|---|---|
| Effective PAD identity | The old dictionary row could acquire gradients through identity reinjection and tied lens readout despite its zero initialization. | Project the effective PAD row to constant zero after grounding; ten deterministic tests cover both paths and compatibility cases. | Changing the constraint can alter learning; no benchmark improvement is established. |
| Routing summaries | Mean nominal slot ID discarded distributional information. | Counts, fractions, modes, optional known-topic contingencies, explicit mapping agreement and score-tie information; 16 tests. | Old scalar summaries cannot reconstruct missing three-slot frequencies. |
| Classification | Unknown roles and misaligned/invalid NLL arrays could produce inappropriate summaries. | Explicit input/denominator contracts while preserving valid formulas; 11 tests, including actual module integration. | No evidence shows earlier studies supplied those malformed arrays. |
| Soil evaluation | The mean-CDF baseline accessed held-out labels; pooled errors were presented as fold errors; invalid CDFs could be silently repaired. | Train-only reference fitting, explicit fold/pooled metrics, CDF and identity validation, unrounded sample evidence and failure preservation; 21 tests. | No competition dataset was available and no new competition score was measured. |

The four audit notes give the complete inventories: [PAD](08_padding_audit.md), [routing](08_routing_audit.md), [classification](08_classification_audit.md) and [Soil](08_soil_audit.md). The [protocol](08_protocol.md) was reviewed before cloud execution. Closing the feasible investigation is distinct from demonstrating the project's larger scientific claims.

## 2. Frozen implementation and execution

| Anchor | Exact identity |
|---|---|
| Item opened after item-7 closure | 2026-10-07T06:27:00.782435+00:00; parent `ef25ed6497cd3554d8eb78a0ef278f7b589e1e1f` |
| Cloud recipe frozen | 2026-10-07T06:52:42.874308+00:00 |
| Plan | `08_cloud_validation_plan.json`; SHA-256 `46b7f19063325b78e593ea53fbbc9bf4372a3f81e7d503a8ac58ee301c15455b` |
| Executed source | `d01e20625068a2ec19bf25554e106c1fc2cc4420` |
| Executed source tree | `03b0549f6ebb39beef4b4160aaaa66292d626059` |
| Actions run / job / attempt | `37584119610` / `112670244618` / `1` |
| Final raw archive | `505a7098a077f90421e4a5ed0cd4c06882e98211` |
| Raw result directory | `results/research/08_cloud_runs/37584119610-1` |

[Inspect the actual Actions run](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37584119610). [Inspect the immutable raw archive](https://github.com/Agnuxo1/NeuroPixel/tree/505a7098a077f90421e4a5ed0cd4c06882e98211/results/research/08_cloud_runs/37584119610-1). The source branch remains isolated; no main merge or external scientific submission is part of this result.

The cloud plan binds **95 files**, including every executable Python source then present, the complete test inventory and direct protocol/runtime inputs. Their hashes agree before and after execution. The archived ZIP contains **272 regular tracked files and 33 directory entries**, with **13,713,224 uncompressed file bytes**; all regular files match the frozen worktree and Git tree. Its 305 total entries must not be described as 305 source files.

The final manifest covers **14 files and 7,819,552 bytes**. Including that manifest, the raw directory contains **15 files and 7,821,952 bytes**. The independent archive audit performed **1,494 checks with no unresolved issues**. These are integrity checks, not independent experimental repetitions.

## 3. Failures were preserved and used to define the fixes

| Saved attempt | Verified local outcome | Interpretation |
|---|---|---|
| Classification before correction | 11 methods: three passed, seven had failures, one skipped; 20 failed subtests plus one direct method failure. | Twenty-one failure entries do not mean 21 failed methods out of 11. |
| Classification after correction | Ten passed, one local Torch-dependent skip. | Actual-function AST validation succeeded; cloud later executed all 11 methods. |
| Routing legacy counterexample | One passed, one expected failure. | Demonstrates that two different routing distributions have the same old scalar. |
| Routing correction | 16 passed. | Deterministic stdlib contracts and driver integration checks. |
| Soil original witnesses | Three failed. | Held-out baseline dependence, fold/pooled ambiguity and silent CDF repair reproduced. |
| First Soil correction | 19 passed. | Subsequent review identified missing coverage of two integration edges. |
| Soil review witnesses | 18 passed, one failure and two errors across 21 methods. | Missing meaningful initial/partial evidence and an out-of-bounds uniform endpoint were exposed. |
| Final Soil correction | 21 passed. | Both reviewed edges and the earlier contracts passed. |

The [local evidence audit](../../results/research/08_validation/local_evidence_audit.json) checks 156 file/receipt/count relationships across eight saved transcripts and finds no contradictions. It is an independent recount of saved evidence within this assistant team; its author also implemented Soil, so this is not external replication or independent custody.

Three first-correction Soil source versions had not been separately snapshotted when superseded. They were recovered later from exact retained edit payloads and accepted only after matching the already recorded SHA-256 values. Their later recovery is explicitly recorded. The Soil review receipt binds its log but lacks contemporaneous hashes for its preserved source/test snapshots. The local historical-source fixture has JSON receipts and a traceback but no separate saved extraction harness or stdout transcript. Those are real local evidence limits. The final frozen cloud source, full tests and raw outputs have the stronger complete binding described above.

## 4. PAD: the old numerical violation and the precise correction

The core originally declared `nn.Embedding(..., padding_idx=0)`, but its effective dictionary was subsequently used through functional embedding and tied readout. A zero initial parameter row did not enforce a zero effective identity after updates. The preserved source has SHA-256 `564045fff799185e15b921873225d344014a4d4bcae6df6deacaf42f6e4bd701`.

The two deliberately small fixtures use explicit parameters, a four-token vocabulary and a 2×2 canvas. The answer route takes two deterministic recurrent updates. Each separate demonstration performs one SGD update with learning rate 0.1. All displayed quantities belong to the old source; they are not task accuracies.

| Route | Saved loss | Saved PAD gradient, component 0 | Saved PAD value after SGD, component 0 |
|---|---:|---:|---:|
| answer_reinjection | 4.326562881470 | 3.681760549545 | -0.368176072836 |
| lens_readout | 0.746567308903 | 0.174371495843 | -0.017437150702 |

All other displayed PAD-vector components are exactly zero. The answer's PAD output class is masked, so its nonzero gradient in this fixture comes through repeated identity reinjection. The lens fixture reaches the dictionary row through its unmasked local readout.

An independent standard-library calculation reconstructs these values from explicit softmax expressions. If `x` is the first PAD component, the answer fixture has read value `u=2(1+x)`, logits `(-10000,u,-u,u/2)` and target index 2. At `x=0`, its loss is `4 + log(1 + exp(-4) + exp(-1))`; differentiating gives `2(1+p1-p2+p3/2)`. The lens has logits `(x,1,-1,0.5)` and target index 1, so its PAD derivative is `p0`. Averaging four identical lens positions cancels their multiplicity.

The [analytical recount](../../results/research/08_validation/padding_analytic_recount.json) verifies **26/26 saved scalar components**, including **20 exact structural zeros**, without Torch, model execution or sampling. Its maximum absolute discrepancy is **2.402022563074979×10⁻⁷** and maximum relative discrepancy among nonzero values is **1.1113327935842836×10⁻⁷**. The declared comparison envelope is `rtol=16×2⁻²³`, `atol=0`; it is a practical float32 comparison envelope, not a universal kernel error theorem. The maximum ULP-scaled discrepancy is about **1.040374 float32 ULP**, in the lens PAD value after SGD. The maximum absolute discrepancy instead occurs in the answer loss and is about **0.503741 ULP** at that reference value. The [recount script](../../scripts/research_verify_item8_padding.py) and [derivation](08_padding_derivation.md) preserve this calculation.

The correction constructs a constant zero effective row **after grounding**. It preserves parameter names and checkpoint shapes. Tests cover both gradient paths, other-row learning, historical nonzero PAD weights, resumed AdamW momentum, grounding, forward/lens/stream behavior, camera observations and explicit research-model compatibility.

The guarantee concerns the effective dictionary. A historical raw parameter or optimizer momentum need not be zero. Empty identity and zero initial state also do not imply zero later recurrent activity. Unchanged non-PAD forward/gradient behavior is checked in the entirely occupied, answer-only fixture with no PAD objective; general training gradients can change. `ResearchNCA` retains its own implementation and explicit `freeze_pad` option, preserving the earlier research recipe. These qualifications prevent a software fix from being mistaken for a retroactive experiment or a measured performance gain.

## 5. Routing and classification now state what their numbers mean

### Routing

The old mean is exactly 1.0 for both chosen-slot lists `[1,1]` and `[0,2]`. Their counts across three slots are `[0,2,0]` and `[1,0,1]`. The mean therefore loses information needed to describe expert usage.

New `tema*_routing` and pooled `routing` records contain counts/fractions including unused slots, all modes and a nullable unique mode. They can include contingencies when actual topic IDs are supplied, and exact tie counts when aligned finite score vectors are supplied. An explicit complete topic-to-slot map permits `mapped_slot_agreement`; it measures agreement with that map, not answer accuracy or the best possible expert choice. The adaptive drivers do not fabricate this mapping.

Pooling concatenates example assignments, avoiding unweighted averaging of unequal batch summaries. Nominal relabeling permutes the counts of fixed assignments; the retained first-maximum argmax rule can still change a tie decision when score columns are reordered. The routing drivers retain their training calls, task-construction expressions, argmax and answer-gathering policy. The shared core PAD correction can affect their actual future training and inference trajectories; those effects were not measured here. Historical three-slot frequencies cannot be recovered from their scalar average.

### Classification

The helper now rejects mismatched/empty vectors, ID arrays with noninteger dtypes (including boolean arrays), missing/unknown roles, negative predictions, PAD targets and an NLL population different from the prediction population. The ID dtype check applies after `np.asarray` conversion; mixed Python boolean/integer lists can be converted to integer arrays, so original scalar Python types are not independently retained or rejected by that check. NLL must contain finite nonnegative real values. A PAD prediction is still a possible incorrect answer. Binary counts, confidence, interval requests and bootstrap arguments have explicit types/ranges; invalid bootstrap options are rejected before an RNG is constructed.

The scoring formulas for valid inputs are preserved. A balanced four-row fixture matches the previous complete output exactly. A separate unbalanced fixture independently checks the following arithmetic:

| Quantity in the unbalanced synthetic fixture | Value |
|---|---:|
| Total correct / examples | 4 / 8 |
| Micro accuracy | 0.5 |
| Role accuracies | 1/3, 1, 1/2, 1/2 |
| Macro across all four roles | 7/12 ≈ 0.5833333333 |
| Macro agent/patient accuracy | 5/12 ≈ 0.4166666667 |
| Mean NLL | 1.75 |

Different denominators legitimately give different averages. Unknown roles must not enter the global count while disappearing from the declared role counts. The helper cannot infer the model vocabulary's upper bound from observed predictions; that remains a separate dataset/model contract. The audit does not establish that earlier stored study arrays violated these new checks, so it makes no retrospective numerical correction to their results.

## 6. Soil: train-only references, explicit metric and durable sample records

The baseline now estimates its mean curve solely from training-fold sample labels. The model-training call also receives only that fold's label dictionary. The observed leakage concerned the old reference curve; access to a broader dictionary alone was not used as proof that the neural model trained on held-out examples. Folds require nonempty, complete and nonoverlapping normalized sample identities. Duplicate/colliding raw identities are rejected, and photo/phone/scale metadata are recorded.

The project's numerical metric is preserved. Let `Fj` and `Gj` be CDF values in percentage points at diameter thresholds `dj`, with `Δj=|Fj−Gj|`. It is the composite trapezoid sum

`sum(j=1..10) 0.5 × (Δj + Δ(j+1)) × log10(d(j+1)/dj)`.

This integrates over the logarithmic diameter coordinate. It is not automatically the exact discrete 1-Wasserstein distance between point masses. The [NumPy trapezoid documentation](https://numpy.org/doc/2.3/reference/generated/numpy.trapezoid.html) and [SciPy Wasserstein documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.stats.wasserstein_distance.html) clarify the distinct operations. The complete hidden competition scorer was not inspected, so identity with that scorer is not asserted.

CDF checks require eleven finite entries, values within `[0,100]`, nondecreasing order and an endpoint within `10⁻⁴` percentage points of 100. Bounds and monotonicity remain strict. The new validator does not clip or project curves. The historical prediction function still has its own clamp; that is explicitly retained. The uniform no-photo fallback is constructed with an exact final value of 100 before validation, avoiding a cumulative floating-point value of `100.00000000000001`.

Each CV record preserves unrounded target/prediction/reference curves, sample and fold identity, photo metadata and three scores. Current-fold summaries and pooled-to-date summaries have separate names, and the pooled result weights samples equally. Complete provenance and planned memberships are written before training; every finished sample is persisted; an ordinary exception records fold/sample/type/message before being re-raised. The injected-failure fixture confirms that the first sample survives failure on the second prediction. The CV main body is executed through AST with inert model/data substitutes; submission coverage evaluates its actual fallback constructor expression, not the complete submission workflow.

The [training-only fitting principle](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) supports the baseline correction. This item does not supply a phone/device holdout, external material/geographic validation, dataset artifact hashes or actual competition reruns. The shared current prediction function mirrors images in both CV and submission; a historical note about another run is not treated as evidence about today's path. Future Soil runs also consume the corrected core PAD implementation, so their source version and model behavior must be distinguished from historical results.

## 7. Integration, reporting units and resource observations

| Frozen suite component | Cases passed | Skips |
|---|---:|---:|
| New PAD invariants | 10 | 0 |
| New classification contracts | 11 | 0 |
| New routing contracts | 16 | 0 |
| New Soil contracts | 21 | 0 |
| Prior integration and research contracts | 128 | 1 |
| **Complete suite** | **186** | **1** |

The sole skip is `tests/test_core.py::test_cartilla_layout`, reason `sin CIFAR-10 local`. No dataset was downloaded to turn an unrelated optional test into a mandatory new experiment.

The raw JUnit suite header declares `tests=341` although the collection and actual XML contain 187 distinct cases. This reporting-unit difference is preserved and explained in [junit_reporting_audit.json](../../results/research/08_validation/junit_reporting_audit.json). The frozen tests contain 154 successful nested `subTest` outcomes: classification 27, PAD 2, development sampler 19, routing 31, Soil 22 and split governance 53. The tagged pytest 9.1.1 sources show that each nested successful call increments passed statistics while the XML reporter reuses the parent case node. Thus **341 = 187 + 154**. No separate per-subtest raw event stream was saved; 154 is a source-based reconstruction consistent with completed parents and the header, not 154 additional independently collected cases. The installed package version was recorded; the explanatory source inspection used the published tag rather than separately archived installed module bytes. [Pytest's own documentation](https://docs.pytest.org/en/9.0.x/how-to/subtests.html) distinguishes execution-time subtests from collection-time parameterized cases.

Two narrow test-maintenance fixes made the complete suite composable. The historical-source test now takes the two changed files from hash-pinned originals, preserving the unchanged item-6 rejection policy. A separate setup guard avoids initializing the process-global inter-op setting twice; its four test methods and assertions are unchanged. Neither repair makes a historical controller accept current source as the original experiment.

The actual runtime was Python **3.12.8**, Torch **2.6.0+cpu**, NumPy **2.2.6**, SciPy **1.15.1**, pytest **9.1.1**, psutil **7.2.2** and Pillow **12.3.0**. Normal test boundaries had one numerical and one inter-op thread. The exact safety fixture intentionally requested two numerical threads and the worker restored one afterward. The legacy process initially reported its pre-initialization inter-op value of two, then 1/1 after the CLI's own configuration. These are boundary observations, not continuous utilization measurements.

| Cloud observation | Value |
|---|---:|
| PAD witness stage, including supervision | 3.001269452 s |
| Complete pytest stage, including supervision | 8.002703984 s |
| JUnit-reported pytest duration | 6.433 s |
| Worker duration before archival | 11.713040590 s |
| Worker available-RAM observations | 15 |
| Minimum observed available RAM | 14.328044891 GiB |

Worker execution ran from **06:55:01.805627 UTC** to **06:55:13.516990 UTC** on 2026-10-07. Durations exclude dependency setup and final evidence archival where stated; the stage durations include parent supervision and therefore differ from JUnit's internal timer. The worker stayed above the declared sampled 8-GiB floor and recorded no resource stop. The local analytical recount observed 9.126377106 GiB available RAM. None of these numbers measures peak process RSS, physical energy or an architecture's efficiency.

## 8. Reproduction and interpretation

Use the executed commit and its pinned dependencies when reproducing this result. The [worker note](08_worker_audit.md) documents the precise two-stage invocation, controls, collection gates and failure preservation. The numerical source freeze remains unchanged after execution. The later analytical audit is a separate closure artifact: run `scripts/research_verify_item8_padding.py` from the closure checkout, with `--source-root` pointing to the exact clean `d01e206...` checkout, `--input` pointing to the saved `legacy_padding.json`, and a new `--output` path. That verifier intentionally is not inside the earlier execution ZIP.

The [cloud audit](../../results/research/08_validation/cloud_archive_audit.json), [local audit](../../results/research/08_validation/local_evidence_audit.json), [PAD analytical receipt](../../results/research/08_validation/padding_analytic_recount.json) and [JUnit interpretation](../../results/research/08_validation/junit_reporting_audit.json) retain the detailed evidence. Two operational read mistakes during local retrieval/report preparation are also recorded: a concurrent `FETCH_HEAD` read and a relative-path calculation. Both failed before changing scientific artifacts and were corrected with explicit sequential source identities and resolved paths. No scientific run was repeated to hide either error.

**Achieved in item 8:** four tested correctness improvements, preservation of observed counterexamples, verified source/runtime/output provenance, explicit historical compatibility and an analytically checked PAD diagnosis. **Still required for stronger scientific claims:** a prospectively specified benchmark of the corrected source, real-data/domain validation, durable memory and other capability experiments, complete efficiency/energy measurements and external replication. Those belong to subsequent agenda items and have not been opened early to obtain a favorable outcome here.

The evidence supports these bounded software contracts. It does not support an accuracy improvement, a rescue of the closed primary hypothesis, an external replication or Nobel readiness. The next sequential investigation is item 9, complex role binding, after this report and its closure are archived.
