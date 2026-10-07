# Item 9 — Independent static review of the staged trainer

Review date: 2026-10-07 UTC. Scope: item 9 only. This reviewer read the controller, protocol helpers, recipe, relevant generator/metric/model paths, tests and the existing local protocol-test receipt. No datasets, predictions or models were generated or executed by this review. No implementation files were edited. This is a source review, not a successful execution receipt or independent performance replication.

## Status and findings

The focused reread of the corrected source found **no remaining concrete scientific blocker** in the reviewed controller. The initial snapshot below exposed a Stage-A-to-Stage-B provenance gap; that gap and the separately identified development-manifest anchor were corrected before scientific execution. The initial and corrected hashes are kept separately. Actual cloud execution and the future Stage-A archive audit remain distinct requirements, not outcomes certified by this static review.

| Finding | Review resolution | Consequence |
|---|---|---|
| Revalidate all four run artifacts when consuming final access | Resolved in source | `claim_final_access` authenticates each run summary and rechecks checkpoint, training log, probe predictions and validation predictions before writing the access receipt. |
| Distinguish unit fixtures from the eventual final performance collection | Resolved in protocol and generator | The protocol limits its no-generation claim to the final study collection with seed 91003. The fixture ledger includes constructed and membership-only candidate bags and supplies prospective group exclusions. |
| Authenticate the selected learning rates across the Stage-A/Stage-B boundary | Resolved in focused reread | The initial `train_primary` trusted rates copied into `plan.pilot_evidence`. The corrected path binds a receipt containing the selection and complete pilot records, verifies their identities/context/recipe, and recomputes the winner before accepting the plan's rates. |
| Bind the complete development manifest and fixture ledger into the run context | Resolved in focused reread after a separate review identified the gap | The context now authenticates the complete manifest. Final loading verifies it before opening arrays, then authenticates the fixture ledger and checks its identity and excluded groups. |

The latter two entries concern integrity of the intended recipe, not model performance. They were identified and corrected before any item-9 scientific outcomes. The corrected source and associated evidence are recorded below; no new scientific arms or selection rules were requested.

## Checks supported by source inspection

**Pilot selection and inventory.** `select_pilot` requires the complete four-combination family/rate inventory, rejects duplicates, failures, wrong phase/initialization/budget and mixed contexts, and requires finite bounded binding scores and nonnegative finite CE. Its ranking matches the recipe: binding descending, CE ascending, rate ascending, separately for each family. Memorization results do not enter that ranking. The primary inventory contains both families for each of five seeds 40–44 and retains all declared runs, with no primary early stopping or best-checkpoint selection.

**Matched input streams.** Each run resets a private `random.Random(92000 + initialization)` independently of Torch. Both families and both pilot rates call the same minibatch routine with the same bag pool and batch size. The routine's choices, assignment/layout draws and query draws do not depend on a model output. Thus the source specifies the same minibatch sequence for equal initializations. Per-update content hashes and an aggregate sequence hash permit a later comparison of actual saved runs. This conclusion is static; equality of actual run logs remains an execution audit. Equal seed values do not imply parameter equality between different architectures or pair a Transformer's computation with NCA firing masks.

**Initialization and update RNG.** The controller seeds Torch for model construction, then separately resets it to `93000 + initialization` for training updates. Data use the private Python stream. Evaluation does not draw stochastic NCA firing masks: `predict` sets `eval()`, uses `inference_mode()` and restores the previous training flag. The core NCA applies firing probability 0.5 only while training; evaluation performs all 16 updates densely. This is the declared inference dynamics, not an average over training masks or an estimate of inference stochasticity. The reference Transformer has no dropout.

**Training and model identity.** The factory constructs the current `NeuroPixel` directly, verifies its effective PAD identity is zero, and constructs the existing `RelativeTransformer` with the new grid/vocabulary. It does not route through the historical research-NCA factory. Both receive only the visible canvas. The trainer uses answer cross-entropy, AdamW, the stated weight decay and gradient clipping, and saves the final scheduled checkpoint. Memorization uses its separate fixed 32-row training fixture and declared consecutive-check stopping criterion; failure to meet that criterion is retained as a flag, not converted into a generalization result. Actual parameter counts and constructor configuration are saved. Approximate parameter matching does not establish equal compute or optimization opportunity.

**Denominators and saved evidence.** Development selection evaluates base queries only. Final rendering iterates six conditions, then bags, then the eight event/role queries. `point_metrics` requires eight rows per group with two queries per role, calculates agent/patient macro accuracy from the two noun-role means, and averages all-eight correctness over scenarios. Saved arrays retain logits, predicted IDs, per-row NLL, targets, roles, group/condition indices and record IDs; the data archive carries event/pair/group metadata. These fields permit independent checks of unique event-role coverage and cross-condition correspondence. The present metric helper does not itself replace that full array audit. Queries/conditions and their legitimate duplicate canvases remain correlated measurements, not additional independent scenarios.

**Final-access ordering.** The initial controller creates the complete checkpoint inventory after all ten primary runs succeed. The worker then publishes an archive, verifies the copied inventory hash, and writes its receipt before starting the separate final child. `final_evaluation` verifies context/development evidence, consumes and fsyncs access, and only then requests the final collection. Existing final data or an already consumed access cannot silently be reused. Every gated checkpoint receives every condition. These statements concern the reviewed control flow; actual event ordering, archive commits and array identities must still be audited after execution. The local gate is neither an adversarial security boundary nor external custody.

**Failure and resource scope.** Per-update loss/input records, started/configuration records and completed or failed run summaries are retained. Ordinary run failures prevent a complete final inventory; a resource stop propagates to stop the phase. The worker supplies admission, source checks before numerical imports, deadlines, monitoring and archival. Standalone invocation of the trainer is not a substitute for the declared supervisor. No successful resource or timing outcome is inferred from reading these paths.

## Existing local test evidence read, not rerun

The saved [protocol-test receipt](../../results/research/09_validation/complex_binding_protocol_test_01.json) and [transcript](../../results/research/09_validation/complex_binding_protocol_test_01.log) report 12 unittest methods: **11 passed and one skipped**, with no failure. The skipped method is actual model integration because local Torch was absent. The recorded local Python version is 3.12.14, which is not the pinned cloud Python 3.12.8; this local receipt therefore does not certify the production numerical runtime. The transcript's 0.077 seconds is unittest's elapsed value; the receipt's approximately 0.215 seconds is wrapper elapsed time. They are different measurements, not inconsistent counts or timings.

The receipt's stored log hash matches the reviewed transcript, `10ef015e5607bc68ff8342c949cf77fbdfd5b3bb0fe02c6442ee1a78965200fb`. Its source hashes match the initial controller/protocol/metric/test/recipe snapshot below. The subsequently corrected handoff must be assessed against its own source version and receipts. The intended cloud contract suite and actual classes still require execution under the pinned environment; this reviewer did not execute them.

## Initial reviewed source identities

SHA-256 values were calculated from exact local bytes after source reading. This is the **initial review snapshot**, not a claim that these files are the eventual frozen execution revision.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `scripts/research_complex_binding.py` | 32258 | `abd171a1ca082c96cdf90f3162bab48a52a3c275e569d2d6b939068c4f4953fc` |
| `scripts/research_item9_worker.py` | 30188 | `a9012e556337c51d49a1fa95db60f166d6dec78669440fdca1a084f90cb190d5` |
| `neuropixel/research/complex_binding_protocol.py` | 9617 | `6b0e3087eca7f6019372cd8818d04cd555c3349696eff806a51faa64ff9d9ff0` |
| `neuropixel/research/complex_binding_data.py` | 16945 | `67a5b2649acbd612458706796fdbd110a54cfa74a82e107cdfbc5777d6a0b1b6` |
| `neuropixel/research/complex_binding_metrics.py` | 1617 | `70116cebe7279f61b74a3e7299ec57faf7c4d2d117a0be1d5a263f4dbf73c8e5` |
| `neuropixel/research/classification_contracts.py` | 3148 | `29be6bd5ccac6c95ae3ea61f32f80ec3f1c32a993f93c564cae8b3bbe1c9fdd5` |
| `neuropixel/model.py` | 6895 | `677f3aab045fc6e0ae73ccc19e7093c1b8211e939dcadbe36db8a793130cd345` |
| `neuropixel/research/models.py` | 17438 | `646810a33a8ee97a8e66cafb61b418b3fee544394ef71123229e6ab0342c54ee` |
| `tests/test_complex_binding_data.py` | 14651 | `4bd206ed35407d05607df4a958b117fa92bc3e14e7d5c497b3625542631ce296` |
| `tests/test_complex_binding_protocol.py` | 10968 | `62f01403bd1f37cd3eb49252c4ed8e0e60a4f41035d37116dd405f9de7dc65d8` |
| `docs/research/09_experiment_recipe.json` | 4079 | `7450b1ede40542c5ff940842e42eb2725ac4420946baad6bac920e779a214b97` |
| `docs/research/09_protocol.md` | 12366 | `419970aa7268475eded80cc207e99f9229fc3f5ca428c103d5c13035469df3b7` |

## Focused correction review

`validate_pilot_evidence` now checks the evidence schema/item/recipe, canonical-byte hashes of both copied artifacts, selection source/recipe/context, absence of final-performance generation, the declared selection rule, and the selection's reference to the complete pilot records. It requires the pilot contexts to agree with the selection, recomputes `select_pilot`, and rejects a stored winner that disagrees. Before regenerating any development collection, `train_primary` verifies the receipt's file hash against the Stage-B plan and requires the recomputed rates and data identities to equal that plan. The new deterministic fixture changes the winner to another admissible rate and recomputes the altered selection's hash; the validator still rejects it because it is not the winning rate.

Both preflight and primary contexts now include the exact development-manifest artifact. The fixture ledger is a manifest file and a development identity; exclusions must equal its contents. Final loading authenticates the manifest before reading its subordinate data files. This closes the previously unauthenticated metadata path. The gated run summaries also bind that context.

The [second local receipt](../../results/research/09_validation/complex_binding_protocol_test_02.json) and [transcript](../../results/research/09_validation/complex_binding_protocol_test_02.log) contain **14 methods: 12 passed, two skipped, no failures**. The skipped methods require Torch: the actual model-class integration and the new one-update-per-family serialization/prediction-roundtrip fixture. This reviewer read the source and transcript without running either method. The cloud inventory is now 20 data-contract plus 14 protocol/integration cases; that prospective count is not a claimed cloud pass. The second transcript hash is `ab5713b058b7a8083e5b254e467be3d23f4df6770ad1e9d69c5c01349873d12f`; its receipt hash is `defc58ce97a9df24131b8619cc83427257059f0c28786d457e8e41d4e6f7eb46`.

| Corrected path | Bytes | SHA-256 |
|---|---:|---|
| `scripts/research_complex_binding.py` | 33708 | `e76ce37640c70a0c3632cf4a2078371a94fdde229cff4aa9bc0f5daf36e7ee05` |
| `neuropixel/research/complex_binding_protocol.py` | 11780 | `7c8a23d2c867702f627c9da14984bc62fa736c8bebd2403f0ac704e1a09769c2` |
| `tests/test_complex_binding_protocol.py` | 15395 | `32bcfc789a12ae46b2ceb376649c6fb9864372f01ab5864058f2928c53d83307` |

The [source-recovery receipt](../../results/research/09_validation/protocol_attempt01_source_recovery.json) preserves the three initial source versions under distinct attempt-01 snapshot paths. Their bytes and SHA-256 values were checked against the original receipt and the initial table above, with all three matching. This was later recovery of exact source bytes by reversing retained patches; it was not a contemporaneous snapshot or a repeated execution. Test attempts are kept separate and their pass counts must not be pooled.

Before Stage B, a read-only audit must still establish that the embedded selection and pilot records actually match the archived Stage-A run and commit, then bind the real receipt in the frozen plan. The validator establishes internal consistency and exact byte identity against that chosen receipt; it cannot establish remote historical provenance by itself. Subsequent independent recount must verify saved input streams, source/checkpoint identities, gate chronology and all declared prediction arrays. These are execution/closure obligations, not a reason to reopen the scientific recipe or infer performance from this source review.
