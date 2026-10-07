# Item 10 neural controller: independent static review

Review date: 2026-10-07 UTC. Scope: the prospective raw-text bAbI QA4 controller and eight bounded contract methods before neural execution. No official TEST example was opened. No Python code was compiled or executed during this review, and no model was trained or evaluated by the reviewer.

## Source identities

| Component | Git blob | UTF-8 bytes | SHA-256 |
|---|---|---:|---|
| Reviewed corrected trainer | `410a7c491f81f704598053e19e5d37890315f0ce` | 52564 | `ce6ed6e4f5822fac9a2b0df455d1e9c347878f17e2160adbebcabb8787161782` |
| Proposed eight controller contracts | `d9b4652db16896e44546456450ed2ab43d16e9af` | 38625 | `0de47098be56e8920a5110f082ede552e8e8990ec9f88fa4033fbba7e9c69154` |
| Pinned-input recovery helper | `b8ce35a0ff6125fa04c635767239f4abc99ec635` | 4766 | `2cc0d5dd96fc7140a5480b1337632d79d96a3efb0f453f44e0947d4ed47a3102` |

The worker `604275ff000e95d75aafc9d7fa97f02cf2f857e4` was reviewed previously. Core NeuroPixel and the relative Transformer were read at acquisition source `99ecaeefdd00dffd3dae444a781ad009c19dff5d`, through blobs `b66e2f575c6ae937d8cad807b4dd0e48531850f7` and `930cb7e44f7e978c21b605d50f978852c00c47ac`. The earlier trainer `12b03cdbde916a7ce10c73518fd59343e9c12454` remains a historical development candidate, not the recommended freeze version.

## Resolved findings before freeze

The earlier `load_selection` checked saved identities and prediction recounts but did not reconstruct all six expected preflight configurations. A different seed, update budget or other operational configuration could therefore evade that semantic check if its enclosing hashes were consistently updated. The corrected trainer shares `preflight_specs(data)` between execution and admission. It compares every complete configuration, canonical configuration hash and saved configuration JSON with the reconstructed expectation; pilots must complete exactly 1,024 updates, while memorization completion must follow the declared 128-update schedule and remain within 4,096. Saved memorization predictions authenticate the consecutive-check criterion. This was a prospective source-review defect, not an observed training failure.

Root separately identified an operational collision: the worker and trainer both tried to create the same phase environment/status paths exclusively. The corrected trainer writes `study_{phase}_environment.json` and `study_{phase}_status.json`, including the failure branch; the worker retains the unprefixed paths. The proposed tests include success and failure through the real CLI orchestration with inert phase work and pre-existing worker-owned files.

No further concrete blocker was found in the reviewed code. This conclusion remains conditional on the actual frozen contract-test/runtime/source receipts; static inspection is not a test-pass claim.

## Scientific and controller behavior checked

- Raw facts and the question supply the canvas. Targets and supporting-fact annotations remain separate. The selected acquisition artifacts and fitted encoder are hash-bound and inputs are re-encoded to check their correspondence with the saved raw records.
- The parameter comparison is prospective and based on authorized TRAIN geometry and vocabulary. For H=4, W=8 and V=18, the declared NP configuration has 29,552 parameters and the Transformer with feed-forward width 144 has 29,562. Similar parameter totals do not match computation, locality, optimization or representational capacity.
- Each run records distinct initialization, sample and firing seeds. A private NumPy generator constructs and saves minibatch indices. The primary panel pairs the complete sample-index streams between families for each training realization. The NP firing stream is stochastic during training; the Transformer has no equivalent stochastic firing. These seeds do not make the architectures' parameter initializations or computational trajectories identical.
- `predict` switches the model to evaluation mode and restores the previous mode. In the reviewed core, stochastic firing is conditional on training mode; evaluation therefore uses deterministic full updates, not Monte Carlo averaging of firing masks. Both models suppress the PAD output class.
- Development selection uses accuracy descending, then answer cross-entropy ascending, then the smaller learning rate. All four pilots are required. Selection is recomputed from saved prediction arrays and is linked to the same recipe and data identity before the ten-run primary panel is constructed.
- Primary training uses exactly 4,096 updates for each of five paired realizations. The durable manifest includes all ten checkpoints and their source/configuration/data/selection identities. Final access authenticates this graph and the worker's prior archive receipt, then exclusively persists a consumed-access record before reading the declared TEST member. A failed final read cannot silently retry through the same output directory.
- Unencodable inputs and unsupported answers remain in exact-match denominators. Cross-entropy is reported only for encodable inputs with an output-vocabulary target, with its separate denominator. Empty subsets return null accuracy/CE. Raw-text controls may still answer an input that does not fit the neural grid; their input advantage must be reported.
- Novelty masks are formed before model predictions using normalized raw and encoded inputs, with optimization-TRAIN separated from all declared development exposure, including the complete fixture ledger. Exact novelty is not semantic or compositional independence. No final subset is selected by a model's correctness.

## Eight proposed contract methods

1. Instantiate both actual model families; check declared parameter totals, finite backward gradients, one bounded update, exact weight serialization/reload, deterministic evaluation and the NP effective PAD dictionary.
2. Run two fixture updates per family through the actual trainer; authenticate private sampled indices, unchanged NumPy global state, paired streams and distinct firing-seed behavior.
3. Exercise prediction support masks, unencodable rows, full accuracy denominator, conditional CE, empty subsets and train/eval-mode restoration.
4. Exercise every development ranking tie-break, missing/duplicate pilot admission, a valid saved six-run preflight graph, and a configuration change rejected even after all surrounding hashes are recomputed.
5. Recount saved predictions and reject NLL drift of 1e-8, argmax changes, support-mask changes and record-order changes.
6. Authenticate the ten-checkpoint gate and reject mutations of each checkpoint, dependent artifacts, run inventory, LR/selection identity, gate timing and completed update budget.
7. Preserve consumed access before an injected final-read failure, reject retry, and check worker/trainer output namespaces through successful and failed inert CLI phases.
8. Read only an invented temporary TAR to check exact member identity, duplicate/missing-member rejection, declared exposure masks and preserved overflow/unsupported denominators.

The tests reuse and re-export `test_babi_qa.py`'s existing `DEVELOPMENT_FIXTURES`; no new raw input contexts are introduced. Opaque checkpoint bytes in file-graph fixtures are never deserialized and are not asserted to be trained weights. Actual tensor serialization is tested separately. The tests deliberately distinguish synthetic identity-graph checks from real neural forward/backward checks.

## Interpretation boundaries

This is a from-scratch external synthetic-QA architecture comparison, not zero-shot transfer of item-9 weights, a real-world benchmark claim, or reopening item-5 H1. Five signed model differences describe paired training realizations, not five independent TEST corpora. Any later cluster bootstrap must remain conditional on the fixed checkpoints and declared source clusters, separate from the between-realization t interval. Successful memorization of 32 development examples is an admission diagnostic, not proof of learning the complete task. A .95 probe or competence threshold cannot by itself identify the cause of success or failure.

The archive and TEST file are public and share a compressed source; the gate supplies documented workflow ordering, not independent blind custody. The future saved-artifact audit remains separately responsible for checking raw arrays, populations, selection provenance and the actual completed execution.

## Final test-source addendum before freeze

The final eight-method test source is Git blob `1c45503ef00e41958c9f578cbea390dc8e618b05`, 38,951 UTF-8 bytes, SHA-256 `fb2a33ba5c2ded36499157813c3423a93b525a93249220323baa40c8177e3e26`. It preserves the reviewed `d9b4652db16896e44546456450ed2ab43d16e9af` candidate with one additional assertion inside existing test 08: when the frozen `docs/research/10_preflight_plan.json` exists, its complete `recipe.development_fixtures` must equal the re-exported `DEVELOPMENT_FIXTURES` list canonically. There remain exactly eight methods and zero additional raw input contexts.

The architecture reviewer completed a separate static reading of all eight methods and helpers against trainer `410a7c491f81f704598053e19e5d37890315f0ce` and reported no concrete blocker. That reviewer also accepted the final fixture-ledger equality check. Neither review executed or compiled the tests. The complete frozen cloud run must supply the actual pass/fail evidence.
