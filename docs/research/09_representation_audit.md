# Item 9 — Representation, spatial reach and shortcut audit

This is a source audit and prospective design recommendation, not a new performance result. The inspected source is item-8 closure commit **`6b2f2554b56c3ac2d7811ea4f6732befd7a93817`**, before implementation of the item-9 generator. The item-9 coordination entry permits this bounded audit; items 10 onward remain pending. No model, training loop or test suite was run. Short standard-library reads and arithmetic used one process; admission showed 9,560,456 KiB available RAM, above the 8-GiB floor.

The proposed input can specify two competing events without an information ambiguity. A 10×8 canvas and T16 remove the obvious *insufficient spatial reach* obstruction for the symbolic NCA, but do not prove that the chosen weights, width or optimizer can learn the required computation. The task is explicitly tagged relational retrieval. It is not a test of unmarked event segmentation, anaphora, hierarchical syntax or recursive reasoning.

## 1. Existing tasks and evidence must remain distinguishable

| Existing implementation | Actual representation and query | Consequence for item 9 |
|---|---|---|
| `RoleTask`, `neuropixel/task.py:60–116` | One event; four distinct rows contain adjacent `[ROLE][FILLER]` pairs. The final row contains one queried role and an empty output cell. | The query role uniquely determines one fact. Repeating a role for another event without adding a selector would make this interface ambiguous. |
| `RoleTaskFar`, `neuropixel/phase3.py:84–108` | Inherits the same event, fillers, target and query. Moves each role to columns 0–2 and its filler at least two columns right on the same unique row. | Already implements distant composition. The exact relation is “same row,” not a second event or a repeated-role ambiguity. |
| `MemoryTask` and `np_stream`, `neuropixel/phase3.py:15–32,53–80` | Presents the four facts of one event in separate frames, then blank frames and its role query; recurrent state survives between frames. | A temporal-retention task exists, but it is not the simultaneous two-event task proposed here. It does not justify adding a new memory programme inside item 9. |
| `ResearchRoleTask`, `neuropixel/research/data.py:16–122` | The original adjacent grammar with explicit CPU RNG and 924/132/264 train/validation/test triples. | Its vocabulary, fixed 1,320-triple universe and four-role sampler cannot become the new generator merely by changing canvas dimensions. |

The historical Far driver uses 12×12, T24 for NeuroPixel, and, in the rest-state configurations, training horizons 18–36 with damage probability 0.5. It supplies school loss 0.3 to NeuroPixel and none to the historical `TinyTransformer` (`scripts/phase3.py:339–369`). Its 2,000-example test set, sampling seed 5, also appears in the recorded training probe curve (`scripts/phase3.py:58–102,358–359`); these are already exposed historical measurements, not a fresh final gate.

The committed Far records are:

| Record under `results/phase3/` | Parameters | Global accuracy | Agent | Patient |
|---|---:|---:|---:|---:|
| `far_neuropixel_s0.json` | 29,824 | 65.90% | 38.85% | 37.39% |
| `far_np_reposo_s0.json` | 29,824 | 70.70% | 40.13% | 41.18% |
| `far_np_big_reposo_s0.json` | 108,224 | 74.30% | 45.22% | 46.22% |
| `far_tf_small_s0.json` | 48,323 | 56.40% | 7.43% | 8.40% |
| `far_tf_big_s0.json` | 319,907 | 56.10% | 7.43% | 7.14% |

These are stored rounded metrics, not recomputed predictions. `README.md` and `docs/FASE3_CONCLUSIONES.md` describe this historical “66% → 74%” panel. The largest NeuroPixel record has perfect action/place scores; its global score does not establish accurate agent/patient binding. An input-category control has expected 75% global and 50% noun-role accuracy on that single-event generator, while an exact row parser can solve it completely. Those expectations are structural deductions, not a new control execution or a significance test against the rounded 74.3% result. The old neural references and supervision also differ from the later research references.

Item 1 specified these limits; item 2 required richer generators to receive their own prospective specification. Item 6 subsequently found low absolute binding and budget sensitivity on the adjacent task, with optimization and mechanism attribution unresolved (`docs/research/06_results.md`, sections 4.3–4.5 and 7). Neither its negative outcomes nor the old Far headline predetermine the new task's outcome. Old checkpoints require their actual source and PAD policy: current core execution would incorporate the item-8 correction.

## 2. Recommended visible grammar

Use the existing 35-token vocabulary, appending two public event tokens, `E0` and `E1`, for **V=37**. Keep event IDs distinct from role and filler IDs. Choose four distinct nouns, two distinct verbs and two distinct places; assign them to the two events without linking an event label to any lexical subset. This deliberate distinctness makes response-changing contrasts observable, but excludes repeated entity identities and identical fillers across events.

Each event has one agent, action, patient and place. Encode the eight facts as horizontal `[EVENT][ROLE][FILLER]` triples. Choose eight distinct rows from 0 through 8, one unused fact row, and independently choose each triple's first column from 0 through 5. Randomize fact-to-row assignment independently of event, role, filler and query. Reserve row 9: event selector at (9,5), role selector at (9,6), output PAD at (9,7). There are 26 occupied input cells: 24 fact tokens and two query tokens. Output stays empty.

The full canvas now visibly defines a function

`answer(canvas) = filler of the unique fact matching (query_event, query_role)`.

An exact parser must reconstruct it from the canvas, public vocabulary and public query/output positions. It must reject missing or duplicate event–role keys, malformed triples, illegal filler categories, query/output collisions and unexpected occupied cells; it must not read generator metadata or targets. Rejecting malformed scenes avoids silently selecting the first of two contradictory facts. An independently implemented parser is also a check of generated gold labels.

One fact per row makes parsing simple and gives the models an explicit spatial grouping convention. This is intentional task syntax, not a hidden leakage path. It measures selection among tagged relational records, not autonomous discovery of what constitutes an event. Random layouts prevent a fixed row or half-canvas from standing in for event identity. The fixed query boundary is public syntax and gives all compared models the same information.

## 3. Group separation and six paired conditions

The proposed canonical semantic group is the sorted bag of four noun IDs, two verb IDs and two place IDs, with category boundaries in its serialization. There are

`choose(12,4) × choose(10,2) × choose(8,2) = 623,700`

possible bags. The group identity must exclude event/role assignment, query and layout. All allocations and transformed descendants of that bag remain in the same train/development/final partition. A deterministic SHA rule needs a versioned canonical encoding, recorded split salt/domain and exact interval thresholds. Hash thresholds yield approximately, not exactly, the requested split proportions. A group-level manifest must check actual membership rather than assuming those proportions imply disjointness.

This is a new compositional holdout within a familiar vocabulary. Individual words and some constituent single-event combinations were used historically. It is not lexical novelty, historically untouched knowledge, or external-domain generalization. Distinct bag partitions are stronger than splitting rendered canvases alone, but still share many constituents; the claimed generalization unit must be the complete bag under this generator.

For each base scenario, evaluate all eight event×role queries in each of these six declared conditions:

| Condition | Input change | Required gold relation |
|---|---|---|
| Base | Original rendered scene and selected query | Exact parsed answer. |
| Swap agent/patient in queried event | Exchange its two noun assignments; query unchanged | Answer changes for agent/patient queries; action/place answers stay fixed. |
| Swap agent/patient in unqueried event | Exchange only the distractor event's noun assignments | Answer stays fixed for every queried role. |
| Globally relabel E0/E1 | Swap event labels in all facts **and** the query | Answer stays fixed; event names carry no substantive meaning. |
| Query switch | Switch only the query event selector | Answer changes for all four roles because corresponding fillers are distinct. |
| Layout permutation | Re-render the same facts into another valid arrangement; query syntax fixed | Answer stays fixed. |

The event relabel and query switch preserve the **scene/filler bag**, not the complete multiset including the changed query token. Do not describe all variants as identical full token bags. Role swaps and pure layout changes do preserve their token multisets. Declare the layout permutation algorithm and its handling of an accidental identity before evaluation.

Queries and transformations are not independent experiments: six conditions times eight queries give 48 rows per scenario, and equivalent rendered inputs can recur in a transformation orbit. Keep group ID, scenario ID, condition and query keys in saved arrays; report duplicate counts rather than promoting duplicates to new sample size. Bootstrap complete scenarios, or complete semantic groups when a bag occurs in more than one scenario, with all their descendants together. Initialization variation remains a separate level. The existing `classification_metrics(..., intervals=True)` resamples examples by role (`experiment.py:201–209`), so its interval recipe does not implement this clustering.

The primary base-condition macro agent/patient score is a sensible bounded endpoint. Report per-event and per-role scores alongside it. Exact-all-eight-query success, strict success across all conditions, correct paired responses and invariant-response checks are useful secondary endpoints. Define strict metrics at scenario level. A changed prediction is not automatically the *correct* changed prediction; invariance can be achieved by a constant wrong answer. Paired correctness and correctness-conditioned consistency prevent either property from masquerading as binding.

## 4. Controls that expose partial solutions

All controls see only public input information. Uniform probability controls have the following exact expectations under the proposed distinct-filler grammar and balanced eight-query evaluation:

| Control | Information used | Global expected accuracy | Agent/patient expected accuracy |
|---|---|---:|---:|
| Exact symbolic parser | Event, role and the corresponding triple | 100% | 100% |
| Role-only | Role association; ignores event, chooses uniformly between the two role fillers | 50% | 50% |
| Event-aware category-only | Correct event and token category; ignores agent versus patient within that event | 75% | 50% |
| Bag-of-fillers/category | Query category and visible filler bag; ignores event and role assignments | 37.5% | 25% |

For the last control, noun queries choose among four nouns, while action/place queries choose among two category fillers: `(1/4 + 1/2 + 1/4 + 1/2)/4 = 3/8`. These are declared control expectations, not measured neural baselines. If the generator later allows repeated fillers, recompute from the actual support and multiplicities; do not retain these percentages by habit.

Keep the probability assigned to gold separate from empirical accuracy of a seeded draw or deterministic tie-break. For paired tests, controls whose admissible information is unchanged must reuse the same prediction/randomness; seeding separately by condition can create artificial “responsiveness.” In particular, event-aware category-only cannot correctly answer both members of a noun-role swap using one unchanged choice, and event-ignorant role-only cannot correctly answer both members of a query-switch pair. Exact controls validate representation and labels; they are not evidence that a 30k neural architecture or its optimizer has solved the task.

## 5. Spatial reach, representational capacity and cost

The core uses pointwise seeding, a depthwise 3×3 perception, pointwise update mixing, local identity reinjection and pointwise final readout (`model.py:50–54,58–108`; `research/models.py:81–133`). With fixed weights, fixed firing masks and no nonlocal hook, an output after T updates depends only on the initial input within Chebyshev radius T. Local reinjection does not enlarge this bound. The induction and its conditions remain those of item 1, section 6; the new PAD projection is pointwise and does not alter the proof.

The farthest possible input cell from output (9,7) on 10×8 has distance 9. Therefore T16 includes the whole symbolic canvas in its *potential* dependency region. On 12×8 that maximum becomes 11; T16 still covers it, at greater cost without adding a representational requirement for this eight-fact grammar. T1/T4 failure would confound restricted reach with binding, as item 6 already cautioned. A query-to-fact-to-output round trip is one possible computation, not a mandatory architecture-wide lower bound: facts can communicate local combinations toward the output before matching its nearby query.

Full reach is only a necessary obstruction check, not a constructive proof that the selected finite-width NCA realizes the parser. Combining event, role and filler requires actual nonlinear computation and transport. Training firing at 0.5 can impede paths compared with full-firing evaluation; radius T is not guaranteed propagation speed, and no simple half-speed theorem follows. T16 is a defensible starting horizon, not a performance-optimized or sufficient-learning claim.

There is no obvious input-level collapse: event tokens are distinct, triple order is spatially visible, and the output is a 37-class answer. The tied decoder has a 16-dimensional latent factor; rank alone does not limit it to 16 selectable token classes. Neither that observation nor the availability of 48 state channels proves learnability. The ConvGRU's reset-plus-candidate route can span two state-neighbor radii per update (`research/models.py:167–172`); the relative Transformer has global attention with positional bias and two encoder layers (`research/models.py:192–309`). Equal T, parameter totals or examples therefore do not equalize receptive fields, latency or optimization difficulty.

The following parameter counts are **source-derived arithmetic**, not a fresh model instantiation, for V37 and the unchanged research widths:

| Model | Parameters on 10×8 | Change with 12×8 |
|---|---:|---:|
| Tied NCA, c48 / hidden128 / identity16 | 29,856 | None |
| Untied factorized NCA | 30,448 | None |
| Spatial ConvGRU, c24 / identity16 | 27,917 | None |
| Two-layer relative Transformer, d32 / heads4 / ff128 | 30,157 | 30,637 |

The NCA has `29,264 + 16V` parameters; an untied decoder adds `16V`. ConvGRU has `26,400 + 41V`. The Transformer has `25,472 + 65V + 8(2H−1)(2W−1)` because both layers own per-head relative-bias tables. Runtime preflight should confirm the actual selected configurations, including grounding and PAD policy, before any result.

NCA/ConvGRU state/update work scales approximately with B×H×W×T at fixed widths; increasing 8×8 to 10×8 multiplies those terms by 1.25, and to 12×8 by 1.5. For the NCA, perception and two update convolutions alone use `18c + (3c+d)hidden + hidden*c = 27,488` multiply-accumulates per cell/update: 35,184,640 per example at 10×8/T16. This excludes seeding/readout, activations, masks, reductions, memory traffic, backward/optimizer work and any school lens. It is not a measured FLOP, latency or energy result. Dense attention pair terms scale with `(HW)^2`: 1.5625× and 2.25× the 8×8 pair count, respectively; linear/projection terms scale differently. Empty-key masking does not avoid constructing the dense score matrix in this implementation.

Forty-eight evaluations per scenario also multiply evaluation work and storage; they do not multiply independent model replications. T, batch size, selected widths, school supervision and any tuning budget must be frozen prospectively with actual bounded resource admission.

## 6. Immediate integration and interpretation recommendations

1. Implement a new versioned generator/controller rather than relabeling `RoleTaskFar` or patching historical datasets. `train_one` directly instantiates `ResearchRoleTask` (`experiment.py:276`); changing `height`, `width` or `vocab` alone cannot train on the new grammar.
2. Declare the corrected PAD policy explicitly. `build_model('neuropixel')` returns `ResearchNCA`, whose default remains `freeze_pad=False` (`research/models.py:317–343`). Use explicit `freeze_pad=True` for a corrected-source study, or clearly label a deliberate legacy-policy condition. Do not silently reinterpret the old checkpoints or source hashes.
3. Keep canvas-only exact/control decoding, transformation gold relations and group-disjoint governance as pretraining correctness gates. All eight queries and six variants must be predeclared; do not select the easiest final subset after inspection.
4. Use validation and a fixed training probe to distinguish failure to fit from failure to generalize. Persistent low probe accuracy supports a bounded failure-to-learn statement; it does not prove insufficient representation. High probe/low held-out accuracy points to a different issue but still does not identify its cause without controls. A finite micro-fixture can diagnose execution/fitting without replacing the complete task result.
5. Preserve every selected configuration, initialization, failure and paired condition. No final-driven horizon/budget change or stronger claim follows from the historical Far scores. All group-level intervals and run-level uncertainty need explicit units; consistency scores alone are insufficient.

The design is suitable for asking whether a learned model selects the right role filler from one of two explicitly tagged competing events. Success would establish that bounded capability under this grammar and split. It would not establish natural-language compositionality, recursive structure, persistent memory, efficiency, or a rescue of the closed item-5 hypothesis.

## Source identity

All paths above refer to the inspected commit. Stable source links: [task](https://github.com/Agnuxo1/NeuroPixel/blob/6b2f2554b56c3ac2d7811ea4f6732befd7a93817/neuropixel/task.py), [core](https://github.com/Agnuxo1/NeuroPixel/blob/6b2f2554b56c3ac2d7811ea4f6732befd7a93817/neuropixel/model.py), [research models](https://github.com/Agnuxo1/NeuroPixel/blob/6b2f2554b56c3ac2d7811ea4f6732befd7a93817/neuropixel/research/models.py), [phase-3 components](https://github.com/Agnuxo1/NeuroPixel/blob/6b2f2554b56c3ac2d7811ea4f6732befd7a93817/neuropixel/phase3.py), [phase-3 driver](https://github.com/Agnuxo1/NeuroPixel/blob/6b2f2554b56c3ac2d7811ea4f6732befd7a93817/scripts/phase3.py).

| Inspected source | SHA-256 |
|---|---|
| `neuropixel/task.py` | `e088d35937040f9924264b227e70e0ba9aaf23ede1b1c6bc9f8d6c78821a156f` |
| `neuropixel/model.py` | `677f3aab045fc6e0ae73ccc19e7093c1b8211e939dcadbe36db8a793130cd345` |
| `neuropixel/research/models.py` | `646810a33a8ee97a8e66cafb61b418b3fee544394ef71123229e6ab0342c54ee` |
| `neuropixel/phase3.py` | `57c177757b333ee5ba9fdc953cbbd7ef284892edee612e234cca1df83b6814d0` |
| `scripts/phase3.py` | `b07487fb641d56f922b90927974a2a1ef48b5e55f72dea21e6e7480af9ad5615` |
