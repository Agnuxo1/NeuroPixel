# Item 9: independent complex-binding data contract

This contract accompanies `neuropixel/research/complex_binding_data.py`. It is
written before any item-9 performance dataset or model run. The module uses only
the Python standard library. It does not import the historical task, models or
Torch, and it does not use Python's global random stream. The experiment recipe
and execution plans must bind the exact module bytes and Python version.

## Geometry and semantics

The canvas is a list of ten rows, each containing eight integer token IDs. PAD
is 0; agent/action/patient/place are 1/2/3/4; nouns are 5–16, verbs 17–26, places
27–34, and event labels 35/36. The eight facts comprise every event/role pair
exactly once. Each occupies a different row chosen from 0–8, with consecutive
`[EVENT, ROLE, FILLER]` cells beginning at column 0–5. Exactly one of those nine
rows is unused. The last row is `[0,0,0,0,0,QUERY_EVENT,QUERY_ROLE,0]`.

There are four distinct noun fillers, two distinct verb fillers and two distinct
place fillers across the complete scene. Thus the two events never share a filler.
The parser rejects invalid shapes, IDs, role/category combinations, duplicated
event/role facts, repeated fillers, missing facts, extra occupied cells, malformed
queries and nonblank outputs. A model receives only the canvas.

The semantic target is always obtained by `symbolic_answer(canvas)`, which parses
the visible triples and visible query. It accepts no target or generator metadata.
The test suite separately specifies eight manual triples and their answers; it
does not establish correctness solely by comparing the renderer to itself.

## Canonical grouping and split assignment

`canonical_group(bag)` accepts exactly the keys `nouns`, `verbs`, `places` and
returns sorted lists, after validating category membership and distinctness.
`group_id(bag)` is the lowercase SHA-256 hex digest of that object encoded with
`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False).encode("ascii")`. For example, a canonical byte sequence is
`{"nouns":[5,6,7,8],"places":[27,28],"verbs":[17,18]}`.

`split_for_group(bag)` converts the entire hex digest to an integer, reduces it
modulo 100, and uses residues 0–69 for `train`, 70–84 for `validation`, and 85–99
for `final`. This is a deterministic 70/15/15 bucket rule, not a claim that exact
universe counts have those proportions. The universe contains
`C(12,4) C(10,2) C(8,2) = 623700` bags. All assignments, layouts, queries and
transformations of a bag retain its group. Shared individual tokens and shared
smaller combinations across partitions are permitted; held-out bags are not a
new vocabulary, grammar, or necessarily held-out single-event triples.

`generate_scenarios(split, count, seed, *, max_attempts=None, excluded_groups=())` uses its own
`random.Random(seed)`. It draws category bags uniformly by sampling without
replacement inside each bag, rejects wrong-partition or previously accepted bags,
and creates one random assignment and layout for each accepted bag. It returns a
list with `count` distinct groups. The default attempt bound is
`max(1000, count*100)`; exhaustion raises an error rather than returning a partial
population. It is a bounded sampler, not a full-universe enumerator. Sequences
are tied to the pinned Python runtime; cross-version RNG identity is not promised.

`excluded_groups` accepts a collection of lowercase SHA-256 group IDs; an excluded
bag is rejected before constructing its facts or layout. Exclusion changes later
RNG consumption and therefore the sampled population, so the exact exclusion
ledger must be frozen identically for development and the primary study.

`fixture_exposure_ledger()` reconstructs only the small declared unit-fixture
draws. It lists their seeds, requested counts, candidate group IDs, constructed
scenario group IDs, the manual fixture, and the expected bounded-sampling
failure. Its sorted `excluded_groups` union includes every inspected candidate,
including rejected bags inspected only for membership. This is a conservative
exclusion set, not a claim that all its members were rendered or scored. It also
covers the explicit exclusion regression, whose RNG consumption differs after
skipping a bag. Repeated calls return the same ledger without consuming global
random state. The study must retain the ledger, bind its hash, and pass its
`excluded_groups` list to every performance-population generation call.

Every scenario is a plain JSON object with exactly these fields:

| Field | Meaning |
|---|---|
| `schema_version` | Integer 1 |
| `group_id` | Canonical bag SHA-256 |
| `bag` | Canonical category bag |
| `facts` | Eight objects with `event` in 0–1, `role` in 0–3, and `filler` token ID |
| `layout` | Eight corresponding objects with `row` in 0–8 and `start` in 0–5 |

The generated fact order is event-major, then role-major. Rendering validates the
semantic inventory and can also accept a reordered fact list if the corresponding
layout entries are reordered with it. Scenario IDs bind the serialized scenario,
including that order; group IDs intentionally ignore it.

`randomize_scenario(scenario, rng)` requires an explicit `random.Random` instance,
validates the original, and returns a new assignment/layout within its unchanged
bag. It neither changes the input object nor changes its partition. This supplies
the approved training augmentation. Sampling which training bag and which query
to use remains the new trainer's responsibility and must use its declared stream.

## Rendering and pairing API

`render_scenario(scenario, event_idx, role_idx, condition="base",
layout_seed=None)` uses zero-based event and role indices. `CONDITIONS` fixes the
order below. Layouts are shared across the eight queries of a scenario. For the
layout condition, the private default layout seed is derived from the complete
scenario and the literal purpose `layout_permutation`, without any query index.
An explicit integer override is allowed and must be recorded by the caller. A
sampled layout identical to the original falls back to a rotation of the eight
distinct slots, guaranteeing a changed fact grid without additional RNG draws.

| Condition | Intervention | Gold relation to base |
|---|---|---|
| `base` | None | Same |
| `swap_queried_agent_patient` | Exchange the two noun fillers in the original queried event | Different for agent/patient; same for action/place |
| `swap_other_agent_patient` | Exchange noun fillers in the other event | Same |
| `relabel_events` | Exchange both visible event labels in facts and query | Same |
| `query_switch` | Exchange the event label only in the query | Different for every role |
| `layout_permutation` | Reassign fact rows and starting columns | Same |

Every record contains `canvas`, `target`, `base_target`, `condition`, and
`changed_gold` (the exact target inequality); `query_role` is zero-based.
`base_query_event` retains the original pair anchor; `query_event` records the
effective visible query after the transformation. Both are zero-based.

`scenario_id` hashes the complete original scenario. `pair_id` hashes the scenario
ID, original event query and role query; all six paired conditions share it.
`record_id` additionally binds condition and rendered canvas. `group_id` retains
the coarser split/resampling unit. All are SHA-256 strings with the same JSON
encoding convention. An external ordered dataset manifest should bind the record
objects, not just their IDs, because targets and all analysis fields must also be
covered by its content digest.

Repeated canvases across conditions are legitimate. In particular, a query-switch
row equals the other event's base query. Their record and pair identities still
represent their declared roles in the paired design. Query-switch preserves the
bag of fact tokens, but changes the extra event token in the query: it does not
preserve the bag of every token in the entire input.

## Input-only control distributions

`control_weights(canvas, control)` accepts one of `CONTROLS` and returns a
dictionary from filler token ID to exact probability. No RNG, target or metadata
is accepted. The common parser validates all grammar constraints; the selector
then uses only the information declared below.

| Control | Selector information | Binding answer probability | All-role mean answer probability |
|---|---|---:|---:|
| `symbolic` | Event and role | 1 | 1 |
| `role_only` | Role association, ignoring the event query | 1/2 | 1/2 |
| `event_category` | Queried event and filler category, ignoring noun-role association | 1/2 | 3/4 |
| `bag_category` | Filler category only; event query and binding ignored | 1/4 | 3/8 |

The reported probabilities are expectations for these randomized categorical
controls, evaluated analytically as `weights.get(target, 0)`. They are not a new
set of sampled discrete predictions. Agent and patient are the two binding roles.
The exact symbolic control is a grammar-informed reference, not a parameter- or
optimization-matched neural baseline. A train-fitted majority control belongs to
the trainer/scorer and is not implemented in this data module.

## Study boundaries and validation scope

The currently approved study uses 2048 training bags, 128 validation bags with
base-only eight queries (1024 rows), a fixed 64-training-bag probe (512 rows),
and 256 final bags with all six conditions (12288 nominal rows). The separate
memorization diagnostic uses four training bags and all eight queries (32 rows).
Those populations are not created by the local unit tests. Tests use only small
manual fixtures and separately named fixture seeds in the 9109xx range. Their
small `final`-partition membership examples are grammar fixtures, not the future
performance population. The fixture ledger excludes those bags and all other
inspected fixture candidates from the study, but does not justify a claim of
universal nonexposure outside the recorded scope.

The callable generator itself is public and can generate any partition. The
development controller must restrict calls to train/validation and keep the
declared final recipe unexecuted until the complete frozen checkpoint inventory
and consume-once final-access receipt exist. This is local workflow protection,
not external blind custody. Item-5 H1 and its datasets/controllers remain closed
and unchanged.

The group/bag is the dataset resampling unit. Queries, transformations, duplicate
rows and answers within a group must not be counted as independent replications.
Across-model comparisons require the same ordered records and targets. An
independent saved-array recount should verify content hashes, reconstruct truth
from visible inputs, and preserve paired/group identities for diagnostics.
