# Item 7 — Source audit of split semantics and final-evaluation gates

**Audited source commit:** `34ec75bad57825ae71cc7772ed236efb17cfcc94`.
**Scope:** source inspection for item 7 only, after item 6 was closed. No training, model execution, checkpoint loading, new outcome analysis or historical-source edits were performed. The package is `neuropixel/research/`, not a root-level `research/` directory. Line references below refer to this commit. Current SCI-007 coordination authorizes the inspection and a separate prospective governance implementation; this note does not open item 8.

The existing research drivers enforce a useful execution order: development completes before final examples are generated in those drivers. They do **not** make the test generator inaccessible, establish that a test population was never exposed in another experiment, or certify that no unlogged call occurred. These distinctions should survive any new wrapper or registry.

## 1. What the partitions actually mean

`RoleTask.__init__` enumerates ordered `(agent noun, verb, patient noun)` triples with unequal nouns, shuffles them using a private Torch generator, assigns the first 20% to test and the remainder to training ([task.py, lines 60–75](https://github.com/Agnuxo1/NeuroPixel/blob/34ec75bad57825ae71cc7772ed236efb17cfcc94/neuropixel/task.py#L60)). With the present vocabulary, there are 1,320 triples: 264 test and 1,056 legacy training triples. The split unit is a **directed triple**, not an individual rendered canvas, a token, an unordered noun pair, a location, or a task/domain.

`ResearchRoleTask.__init__` calls that constructor with the same 20% boundary, retains its test pool, takes the first 132 legacy-training triples as validation, and retains the remaining 924 as training ([data.py, lines 25–43](https://github.com/Agnuxo1/NeuroPixel/blob/34ec75bad57825ae71cc7772ed236efb17cfcc94/neuropixel/research/data.py#L25)). Therefore, for the same permutation and seed:

| Research population | Permutation positions, zero-based and half-open | Relationship to legacy 80/20 |
|---|---|---|
| Test, 264 triples | `[0, 264)` | Exactly the same legacy test pool |
| Validation, 132 triples | `[264, 396)` | A subset of the legacy training pool |
| Training, 924 triples | `[396, 1320)` | The remainder of legacy training |

This is an exact source-level relationship, conditional on the same vocabulary enumeration and Torch permutation implementation. A new sampling seed changes sampled triples/layouts/places/queries, not pool membership. A new split seed repartitions the same universe; it does not by itself establish a fresh, globally unexposed population. This note does not substitute Python or NumPy permutation algorithms for `torch.randperm`, and does not claim numerical cross-seed overlap counts without the separate exact-permutation audit.

Each sampled canvas contains all four role/filler pairs, distinct pair rows, a random place and layout, and one query; the target is the visible filler for that query ([task.py, lines 98–115](https://github.com/Agnuxo1/NeuroPixel/blob/34ec75bad57825ae71cc7772ed236efb17cfcc94/neuropixel/task.py#L98); [data.py, lines 76–98](https://github.com/Agnuxo1/NeuroPixel/blob/34ec75bad57825ae71cc7772ed236efb17cfcc94/neuropixel/research/data.py#L76)). All generated variants of a given directed triple remain in that split within one task. Individual tokens, places, query roles and rendering rules remain shared. This tests withheld compositions in the same generator, not unseen vocabulary, unseen renderers, or an external distribution.

The growth constructor filters each already assigned pool to a four-noun topic, preserving split membership, and records disjointness and 120 distinct triples per topic ([growth_ablation.py, `make_topic_task` and `partition_manifest`, lines 122–156](https://github.com/Agnuxo1/NeuroPixel/blob/34ec75bad57825ae71cc7772ed236efb17cfcc94/neuropixel/research/growth_ablation.py#L122)). It does not repartition after filtering. Its test pools are subsets of the previously used split-0 test pool. Different sample sizes and seeds in growth create a different example realization, not a new untouched triple population. Disjoint topic nouns also give visible lexical domain cues; this is a task property, not evidence that a topic ID was passed to the model.

## 2. Enforced properties versus procedural conventions

| Surface | What the source enforces | What it does not establish |
|---|---|---|
| Legacy `RoleTask.sample` | Chooses training only for the exact string `"train"`; every other name selects test (`task.py:77–96`) | Strict split names; separate validation; authorized final access |
| Research `sample` | Exact train/validation/test names; explicit CPU generator; validated batch/query/place IDs; CPU draws before transfer (`data.py:45–98`) | That the caller is authorized to request test; historical non-exposure |
| Research partition attributes | Tuple membership and separate cached CPU tensors at construction (`data.py:29–43`) | Deep immutability: attributes and `_triples_cpu` can be reassigned, as the growth filter intentionally does |
| `frozen_dataset` | Private sampling generator, equal role blocks, fixed construction recipe (`data.py:102–122`) | A gate or capability check; it can generate `"test"` directly |
| `dataset_artifact` | Saves arrays, file SHA-256 and canonical tensor-content digest (`experiment.py:68–90,234–240`) | Complete semantic dataset identity; write-once storage; untouched status |
| `evaluate` | Uses evaluation mode, checks finite logits and restores the previous mode (`experiment.py:210–231`) | Split provenance or authorization: it accepts an arbitrary `(canvas,target,roles)` tuple |
| `train_one` | Fresh configured initialization; samples only training; evaluates validation and a training probe after the fixed budget; saves the final checkpoint (`experiment.py:270–347`) | A technical barrier to external test calls; instrumentation proving its `final_test_accessed=False` declaration |
| Research controllers | Validate prescribed development artifacts and write a gate/selection record before their final loader | Protection against a different script, a new output directory, deleted receipts, or deliberate changes to the process |

**Concrete inherited-API trap.** `ResearchRoleTask` inherits `RoleTask.sample_loop` without overriding it. That method uses `train_triples if split == "train" else test_triples` ([task.py, lines 141–147](https://github.com/Agnuxo1/NeuroPixel/blob/34ec75bad57825ae71cc7772ed236efb17cfcc94/neuropixel/task.py#L141)). Thus `research_task.sample_loop(..., split="validation")` samples test. None of the inspected research drivers calls this method; this is a prospective API risk, **not evidence that items 4–6 used test for validation**. A new governed surface must cover or explicitly reject inherited helpers, not only validate the main sampler. The inherited camera helper delegates to `self.sample`, so it must remain inside the same authorization boundary if supported.

**Concrete artifact-identity trap.** `dataset_artifact` names files using split seed, split name, count and sample seed; it omits generator class/source, geometry, topic, vocabulary and pool digest (`experiment.py:234–239`). `atomic_binary` uses `os.replace` (`experiment.py:35–49`), so a later call can replace an existing same-named artifact. Item 6 growth avoids topic collisions with separate topic directories, and the core checks expected content digests. The generic helper alone does not prevent a caller from colliding different datasets in a common directory. Hashing records the resulting bytes; it does not by itself prevent replacement.

## 3. Development selection and final access by driver

### Historical 80/20 scripts

`scripts/train.py` uses one command-line seed for model initialization and the triple split, and seed+1 for training sampling (lines 111–129). It evaluates both train and test every `eval_every` updates, with a fixed evaluation sampling seed 123 (lines 28–35,151–159). It reports the maximum observed `best_test` and the final score, but saves the final weights rather than a best-test checkpoint (lines 156–167). Therefore the printed best score is selected over repeated test observations, and the test curve is visible during development. The code does not prove that a person selected a particular hyperparameter from that curve; it does prove the access path and best-score statistic exist. `scripts/sweep.py:54–85` runs multiple configurations/seeds through that script and collects both final and best scores.

Additional public retrospective routes reuse test: camera evaluation and traces in `train.py:38–64`, the battery's `test_set` with sample seed 5 in `scripts/battery.py:38–40`, and `RoleTaskFar.sample` preserves its parent's split pools while changing layout (`neuropixel/phase3.py:84–108`). A changed layout or sampling seed is not a new triple partition. Historical results should retain their retrospective/exposed designation.

### Item-4 pilot and fixed validation selection

`scripts/research_train.py:53–89` refuses an output directory with selection/final-phase evidence, completes every family/LR configuration, checks common source/environment, and writes the selection before constructing final examples at lines 90–97. `select_pilot` checks complete family/rate membership and chooses each LR by validation binding, then validation CE, then lower LR; the reference family is selected by validation binding, CE, parameter count and name (`experiment.py:361–380`). Final evaluation hashes selected checkpoint files before loading them (`research_train.py:99–114`). The selection helper consumes recorded validation fields; it is not a provenance or access-control system for arbitrary caller-supplied records.

The exact frozen protocol uses **pilot initialization 7**, split 0, validation sampling seed 61001 and final sampling seed 61002, with validation/final sizes 2,048/4,096 (`protocol.json:6–23,25–49`). The primary repeat panel uses initialization 10–14 at split 0; the separate split panel uses initialization 10 at split 0/101/202 (`protocol.json:57–76`). Changing initialization while retaining a test set produces model repetitions conditional on that test set, not new data exposure units.

The recovered item-5 report describes completion of all 28 trainings before its final gate and a frozen item-4 selection ([05_results_recovered_20261007.md, lines 24–58](05_results_recovered_20261007.md#L24)). **Current source limitation:** `scripts/research_replicate.py`, its original raw CPU archive and original selection evidence are not present in this continuation checkout. Its claimed historical gate is therefore report-level recovery evidence for this audit, not a newly inspected implementation. The recovery manifest explicitly records unavailable historical raw material; do not reconstruct missing code silently or upgrade this evidence category.

### Item-6 core gate

`load_execution_plan` checks the frozen 26-run/34-evaluation inventory, implementation hashes and recovered historical-source hashes (`scripts/research_ablations.py:188–214`). `run_ablations` rejects existing final-phase files/events, checks each training configuration, source, resolved model and validation/probe content hashes, and requires all 26 distinct runs and a common environment (`:225–303`). It verifies development checkpoint/prediction files, writes the gate and authorization event, and only then calls the final dataset loader (`:304–331`). Checkpoints are checked again before each evaluation (`:335–357`). There is no final-score configuration selection in this controller.

These are enforceable checks in the normal controller path. The gate does not make split-0 final examples new: the controller deliberately uses the protocol's same sample seed/count and checks the expected final content hash. Item 6 is explicitly exploratory; freezing new interventions before its own run does not erase earlier exposure to the same task/test setting.

### Item-6 growth gate

`run_growth` refuses any nonempty output directory (`scripts/research_growth_ablation.py:129–137`). Its `enter_final_phase` checks six exact trajectories, 18 complete stages, ten banks, eight completed train-only gate fits and two construction checks before invoking authorization and the final-data callback (`:61–114`). The authorization closure checks source/checkpoint hashes and records all development artifact hashes; the final datasets are created only afterwards (`:495–530`).

Growth diagnostic inputs and supervised router-fit targets both come from TRAIN (`:313–337`); all eight router fits finish before the gate (`:478–494`). The validation pools are recorded but not used to select a growth router or threshold in this implementation. The resonance threshold is derived from the prescribed post-A training diagnostic, not from final outcomes. Gate-fitting examples can overlap the experts' eligible training population; this is additional supervised fitting, not an independent validation set. Separate final scoring remains necessary.

The complete test triple memberships are already constructed and written in `partitions.json` before growth training (`:313–315`). Thus “first final access” in the controller means first creation/scoring of the **final sampled examples**, not first availability of the test membership list. That definition must be explicit for a public deterministic synthetic task.

## 4. Exposure and possible leakage: appropriate classifications

1. **Historical test exposure is known at the procedure level.** The 80/20 test curve and best-test statistic are available during legacy runs. The new split retains that same test pool for an identical split seed. New validation membership was eligible for old training. Actual minibatch occurrence or exact historical canvas identity requires original streams/artifacts; pool eligibility alone does not prove that every member was seen.
2. **Within-run partitioning is different from cross-study exposure.** New research models are freshly initialized and trained on their declared 924-triple pool. A triple that was used by another historical model does not mechanically enter these new weights. It can nevertheless inform human architecture/protocol choices; preserve that distinction rather than calling all overlap direct gradient leakage or all new runs pristine.
3. **Cross-seed overlap is possible and must be measured by group identity.** Seeds 0/1/2 and 101/202 reorder a common finite universe. A training group under one seed can be test under another. Report the panels separately and retain the union of prior exposure; a new directory or sampling seed does not reset that history.
4. **The chosen group definition limits the generalization claim.** Directed-triple disjointness permits shared nouns, verbs and places, reversed `(patient,verb,agent)` triples and closely related compositions across splits. If future claims require separation of unordered pairs, templates, source entities or symmetry families, that grouping must be declared and enforced before assignment. Such stronger grouping is a different study; the present directed split is not automatically invalid for its stated compositional target.
5. **No data-fitted normalization or vocabulary discovery was found in the inspected adjacent-task path.** Vocabulary, rendering and topic filters are predefined; embeddings and router weights are learned from training inputs/targets. The target is intentionally recoverable from the visible role/filler arrangement. This task shortcut is not accidental label leakage into model arguments. Future fitted transforms must record their fit population and cannot learn parameters from selection/final data without an explicitly different design.
6. **Public synthetic labels cannot be made secret by a Python token.** Inputs expose the information needed to derive targets. Prospective gates can make ordinary misuse visible and preserve a declared order; they are not secure custody, independent blinding or proof of absence of unlogged inspection.

## 5. Minimal prospective contract, without changing historical behavior

Implement a new, opt-in governance module/adapter and leave the seven recovered historical scientific files byte-identical. The following are minimum guarantees for that new surface, not claims about existing helpers:

| Contract element | Required behavior / important failure case |
|---|---|
| Canonical dataset manifest | Include generator/source identity, vocabulary, geometry, transforms, RNG/runtime recipe, ordered pool membership and hash, declared group key, sample recipe and content hash when materialized. Reject conflicting content under the same identity. Paths, split labels and seeds alone are insufficient. |
| Group ownership | Require pairwise disjoint declared groups for fit/selection/final within a study; verify that each example resolves to its manifest group. Preserve all augmentation/layout/query descendants in that group. A conflicting duplicate ID must fail even when its bytes differ. |
| Explicit exposure registry | Record study/purpose/time and identities of opened example sets and their source pools. Distinguish eligible group membership, sampled input/target exposure, metric exposure and unknown historical coverage. Never relabel previously exposed or unknown material “untouched” through a new study ID. |
| Authorized sampler surface | Keep strict split enumeration. Cover or reject `sample_loop`, camera helpers and direct dataset constructors; a blocked inherited helper must not silently select test. Training-only handles should not hand out an unrestricted object containing a usable test sampler. Document the boundary against callers deliberately bypassing it. |
| Exact selection inventory | Require unique declared candidate IDs and complete finite validation results tied to the same population, metric definition and denominators. Freeze tie-breaking before selection; reject missing, duplicate, foreign-split or post-final entries. A metric JSON with the string `validation` is not sufficient provenance. |
| Immutable selection receipt | Bind protocol/manifest/selector/config/source/checkpoint hashes and the full candidate inventory. Refuse an edited receipt or substituted artifact, even if a display name is unchanged. |
| Consume final access before loading | Persist a write-once final-opened receipt before calling any final loader. A loader/evaluator exception still consumes exposure. Distinguish a completed score from opened-but-failed access, and require an explicit recovery record rather than silent re-opening. |
| Declared evaluation inventory | Permit multiple predeclared paired checkpoints/cases against the same frozen final set; do not mistake those planned comparisons for repeated adaptive selection. Freeze cases and checkpoint hashes together and reject undeclared additions after access. |
| Safe persistence and scope | Refuse silent overwrite; write consistent state transitions atomically. State clearly that local receipts prevent accidental reuse within the governed workflow, not deletion/tampering, independent scripts or unobserved external access. |

Critical verification cases for the new implementation are wrong split names; inherited validation-to-test fallback; overlapping group identities despite different canvas bytes; same-name/different-content manifests; duplicate/missing/nonfinite or foreign-population selection entries; changed weights/config/source after freeze; a final loader that fails after access is consumed; and a new output directory pointing at already exposed data. These can be verified without model training. Item 7 should report what the new code actually enforces and keep historical exposed populations and H1 status unchanged.

## 6. Audit receipt and limits

The inspection read `AGENTS.md`, current SCI-007 coordination and the source/functions cited above. Initial measured available RAM was 9.13 GiB; only short source/JSON reads and single-process standard-library inspection were used. No Torch import, statistical simulation, dataset generation, tests or scientific execution was needed for this note. No Git refs, coordination ledger or historical source were changed.

Source inspection establishes the intended and enforced normal execution path. It does not independently recover missing item-5 artifacts, enumerate actual historical minibatches, verify cryptographic custody, or prove the absence of actions outside that path. Those limits are separate from the already completed item-6 saved-array audits.
