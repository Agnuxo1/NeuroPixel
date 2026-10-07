# Item 9 — Complex role binding: prospective protocol

Status: design agreed before any item-9 performance run; executable plans will bind the exact source before their respective phases. This is a new, bounded exploratory investigation. Item-5 H 1 remains closed and not supported.

## Question and operational scope

Can the current corrected NeuroPixel retrieve a filler using both an explicit event identifier and an explicit role when two complete events compete on a grid? How does its behavior compare with the existing relative-position Transformer under the declared answer-only training recipe? Are predictions responsive to relevant changes and stable under irrelevant changes?

The existing adjacent and far generators contain one event and four unique roles. Greater spatial separation in RoleTaskFar does not itself test multiple events. This investigation adds an explicitly scoped two-event lookup task. It does not test natural-language parsing, implicit event identification, coreference, repeated entities, recursive structure, more than two events, real-world reasoning or general systematicity. Completion of this investigation is not certification of a Nobel-level contribution.

## Representation and grouping

A 10×8 integer grid contains eight horizontal triples [EVENT][ROLE][FILLER], one triple in each of eight distinct rows chosen from 0…8. Starts are in 0…5. Row 9 contains query event at column 5, query role at 6, and the blank output at 7. There are 37 token IDs: the existing 35-token vocabulary and two event identifiers. Every event contains agent, action, patient and place. A scenario uses four distinct nouns, two distinct verbs and two distinct places; no filler repeats anywhere. There are no camera inputs or RGB grounding.

A group is the canonical unordered bag of four noun IDs, two verb IDs and two place IDs. The finite group universe contains C(12,4)×C(10,2)×C(8,2)=623,700 bags. Canonical SHA-256 modulo 100 assigns residues 0…69 to training,70…84 to validation and 85…99 to final. These are procedural bucket proportions, not guaranteed exact realized proportions in the universe. All assignments, event/role exchanges, layouts, queries and declared transformations of a bag inherit its group and partition. Shared individual tokens between partitions are intentional; held-out bags are not held-out single-event triples or a new grammar.

The actual sampled group sets, source, seeds, scenarios, support counts and content digests will be saved. The deterministic fixture ledger prospectively excludes its 100 inspected candidate bags, including 31 constructed fixture groups, from every performance collection; the nominal universe above precedes those exclusions. Draw unique bags without replacement within each partition. Keep 2048 training bags,128 validation bags and 256 final bags. Training resamples assignments, layouts and one event/role query from its fixed bags. Validation selection uses the base condition only:128×8=1024 query rows. The fixed training probe uses 64 training bags×8=512 rows. Four training bags×8=32 fixed rows form the separate memorization diagnostic. These training-derived fixtures never establish generalization.

## Conditions and units

Every evaluation scenario supplies all eight event-by-role queries. Final evaluation contains six conditions in a fixed order,256×8×6=12,288 nominal rows per checkpoint. Some rows are exact duplicates, particularly base queries and their selector-switched counterparts. They are paired measurements, not independent data or additional replications.

| Condition | Visible change | Expected answer relation to base |
|---|---|---|
| base | None | Reference |
| swap_queried_agent_patient | Exchange agent/patient fillers within the queried event | Changes for the four noun queries; invariant for action/place |
| swap_other_agent_patient | Exchange those fillers within the unqueried event | Invariant for all eight queries |
| relabel_events | Exchange both event labels in all facts and in the query | Invariant for all queries |
| query_switch | Change only the queried event identifier | Changes for all queries because same-role fillers differ |
| layout_permutation | Reposition the same facts while preserving their labels | Invariant for all queries |

A group/scenario is the resampling unit. Query rows and conditions are not independent units. A single scenario/assignment is sampled per held-out bag in the scored arrays. The fixed two-event template, all vocabulary, rendering grammar and transformation rules remain known.

## Controls

The exact symbolic control must recover the unique target from the visible canvas without labels or generator metadata. It is a grammar-informed algorithmic control, not a trained or parameter-matched neural baseline. A parser failure or inconsistent target blocks performance execution.

Uniform categorical controls are scored by their exact answer probability, without Monte Carlo draws. Role-only matching ignores the event: expected 50% globally and 50% for agent/patient binding. Event-aware category matching ignores the agent/patient distinction:75% globally and 50% binding. Global bag/category matching ignores both associations:37.5% globally and 25% binding. These rates depend on the deliberately distinct fillers. A train-only per-query-role majority control uses only training-bag answers and a deterministic lowest-token-ID tie break. No final label is used to fit a control. Report actual scores as well as the stated expectations.

## Neural models, training and staged freeze

Use the current core NeuroPixel class directly, including its effective zero-PAD dictionary: c_id 16, state 48, hidden 128,16 recurrent updates and training fire rate 0.5; evaluation updates all cells. Use the existing RelativeTransformer with width 32, two layers, four heads and feed-forward width 128 on the same 10×8 grid. No model sees event/query metadata apart from the rendered tokens. Record actual trainable parameter counts at runtime; parameter proximity is not equal compute, equal expressivity or equal optimization opportunity.

Common optimization: AdamW, batch 32, weight decay 0.0001, gradient norm clipping 1.0, answer cross-entropy only, no school/lens loss, no curriculum, no test-time search, and the final scheduled checkpoint. Numerical work uses the pinned CPU runtime, two intra-op threads, one inter-op thread and at least 8 GiB available RAM. Input sampling has its own Python RNG stream, independent of Torch initialization and stochastic firing. Matched primary initializations share the same minibatch stream across model families; firing masks are not paired with an attention model.

Stage A freezes the preflight source, recipe and runtime before creating performance datasets. Meaningful contract tests run first. Each family then receives a separate 32-row fixed memorization diagnostic with LR 0.003 and at most 4096 updates. Check every 128 updates; stop only after two consecutive evaluations with all 32 answers correct and mean CE≤0.05. A failure to reach this criterion is retained as an optimization/capacity diagnostic, not hidden or automatically converted into architecture impossibility. It does not select primary weights or hyperparameters.

The pilot fits both families at both predeclared rates 0.001 and 0.003, each for 1024 updates with initialization 39. For each family select LR by descending base validation agent/patient macro accuracy, then ascending validation CE, then ascending LR. Retain all four runs. The final performance collection defined by seed 91003 is not generated, fitted or evaluated in Stage A. Contract fixtures may inspect other bags in the final bucket, and rejection sampling inspects candidate bag identities. A saved fixture ledger records development exposure; those recorded fixture groups are excluded prospectively from all three performance collections. This is not a claim that the final hash bucket is never inspected. The pilot is part of development and is not a fifth/sixth primary replication.

Stage B freezes the selected LR pair, exact source, all dataset recipes/hashes and the complete primary inventory after reviewing only Stage-A evidence. Its plan binds a receipt containing the exact Stage-A selection and complete pilot records. The trainer authenticates those copies, recipe, contexts and data identities, recomputes the winning rates, and rejects any mismatch with the rates copied to Stage B. A separate archive audit establishes the receipt's claimed Stage-A Git provenance before this freeze. Train each family from scratch with initializations 40,41,42,43,44 for 4096 updates. These ten runs are the entire primary inventory. There is no early stopping, best-checkpoint selection, post-final tuning or automatic extension. Save losses, per-update input digests, train probe/validation predictions, final weights and finite/resource status. A failed run remains a failed declared run; incomplete panels must not be presented as complete paired evidence.

Before creating final data, persist and fsync a complete inventory of all ten eligible checkpoints and their hashes, model configurations, training/validation identities, source and the fixed selected LRs. A consume-once final-access record must exist before final sampling/loading. A hash/contract/resource failure cannot be bypassed by changing a flag; any repair requires a new explicit version, preserves the failed evidence and cannot turn already exposed examples into a new holdout. This is a local software gate, not external custody or a security boundary.

The final recipe seed is declared before the pilot; its actual instances remain ungenerated until the gate. Every selected checkpoint is evaluated on every final condition, with logits/predictions and labels saved for independent recount. This new test does not restore historical H 1 blindness or permit a retrospective change to H 1.

## Estimands, uncertainty and decision language

Primary descriptive endpoint: base-condition macro agent/patient accuracy, equally weighted across the two noun roles and events. Record each of five seeds, mean, sample SD, range and paired NeuroPixel-minus-Transformer differences. Report a two-sided 95% t interval for the mean paired difference across five initialized runs (df 4), conditional on this dataset, recipe and selected rates. Five seeds offer limited information about the distribution of training outcomes; normal-theory coverage is not guaranteed and this interval does not include dataset-generation or selection uncertainty.

For each fixed checkpoint and condition retain global accuracy, four-role accuracy, binding, mean cross-entropy, per-scenario all-eight correctness and prediction tables. Counterfactual diagnostics retain base-and-variant joint correctness, prediction equality for expected invariant cases, and prediction inequality for expected changed-gold cases, with separate eligible denominators. A constant answer can have perfect invariance while being wrong, so equality/inequality alone never establishes successful binding.

Bootstrap only scenario/group rows, preserving all paired queries, conditions and models. Use 2000 fixed seeded resamples and percentile 95% intervals for declared conditional diagnostics. Save the resampling indices or a byte-identical reproducible artifact and recount from saved predictions. These conditional scenario intervals and the five-seed interval answer different questions; neither is a complete uncertainty estimate for architecture superiority. Secondary comparisons are descriptive; no selective significance claims or unadjusted broad discovery claims.

An operational high-competence screen, stated before performance: all five NeuroPixel checkpoints achieve base binding≥0.95 and base all-eight-scenario correctness≥0.90, with binding≥0.90 in every declared condition. Report each part separately. Even meeting the screen would demonstrate only this constrained two-event setting. An advantage over the neural reference is reported with its paired uncertainty, not required or inferred merely from the competence screen. Failure leaves the associated capability unestablished under the frozen budget, without proving it unattainable.

## Closure requirements

Preserve all failed and successful attempts, source and runtime identity, split/group audits, controls, tests, selected and unselected pilot outcomes, all primary outcomes, frozen gates, exact saved predictions and independent checks. Distinguish measured findings, source-derived facts, analytical expectations and untested hypotheses. Review the report and release resources before opening item 10.

Primary methodological sources and reading limits are documented separately in 09_methodological_evidence.md. The protocol applies their lessons without claiming to reproduce SCAN, COGS, CFQ, CLEVR, HANS or tensor-product binding.
