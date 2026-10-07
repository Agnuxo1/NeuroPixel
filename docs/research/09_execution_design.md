# Item 9: technical execution design

This source review preceded new item-9 model execution. The agreed scientific
recipe is in `09_protocol.md` and `09_experiment_recipe.json`; this note explains
the compatibility boundaries and required artifacts. It does not authorize a
run or replace either phase's exact frozen execution plan.

## Reuse the operational design, keep the scientific controller new

The item-8 worker is a useful operational template: the coordinator does not
import Torch; checks committed source/plan bytes before numerical imports;
admits and monitors at least 8 GiB available RAM; bounds its own child session
and Git subprocesses; captures stage logs and runtime/source hashes; archives a
source ZIP and all success/failure evidence using a separate results worktree;
and does not change source HEAD. Preserve those properties in a new item-9 worker.

The item-8 executable itself must not drive this study. Its plan expressly
forbids scientific training and final scoring and fixes item-8 stages, fixture
inventory, and thread exceptions. Its test-only allowance for one two-thread
safety fixture is not a study resource policy. The new recipe explicitly allows
two intra-op numerical threads and one inter-op thread, with sequential model
runs, at least 8 GiB available RAM, and a bounded public CPU workflow. Freeze
these actual controls and record both requested and observed settings.

The old `experiment.train_one` constructs the old `ResearchRoleTask` internally,
uses old data IDs/probes, and records the historical source inventory. The old
factory's NeuroPixel arm returns a legacy-compatible `ResearchNCA` with
`freeze_pad=False` by default. Neither is the new study's training path. Construct
the current `neuropixel.model.NeuroPixel` directly (effective zero PAD), and the
existing `RelativeTransformer` directly with the new 37-token, 10×8 geometry.
Train from scratch and record explicit constructor arguments, class/source,
parameter count, initialization and optimizer recipe. Do not change historical
model defaults, H1 selection rules, data, or controllers to accommodate this run.

`experiment.evaluate` saves predictions/confidences/NLL but not full logits and
its optional intervals independently resample query rows. Those intervals are
inappropriate for paired multi-query groups. A new evaluator should save full
float32 logits and use the new ordered record inventory; a separate recount can
calculate summaries and group bootstrap intervals afterward. The numerical
global/four-role accuracy definition remains usable, but old role-wise Wilson
intervals or example bootstrap must not be presented as group-level uncertainty.

## Grouped scenarios and delayed final generation

The stdlib generator's contract is in `09_data_contract.md`. Its canonical group
is the category bag, covering every assignment, layout, query and intervention.
Persist the exact training and validation scenario JSONL files and their ordered
record manifests before fitting. Preserve finite training bag IDs, augmentation
stream seed, each update's ordered input digest, and the fixed probe/diagnostic
records. A seed alone is insufficient evidence of which runtime-specific samples
were used. IDs, token vocabulary, rendering rules, source hash and ordered
record content must all be bound.

The old `DevelopmentRoleTask` is specific to the historical task; it is not a
generic development gate. The new development path must explicitly accept only
train/validation. It may declare the final recipe and grouping rule before the
pilot, but must not generate final scenarios or open final targets at that phase.
Public generator code is not a security boundary; document the scope honestly.

`split_governance.make_manifest` expects an existing final artifact hash and
ordered records. `FinalStudy` freezes one selected candidate and permits one
evaluation callback. It does not directly implement a panel of ten checkpoints
with final examples generated only after selection. Do not fill its hash fields
with placeholders, call it repeatedly as if it were a panel gate, or silently
relax the API. The new panel controller needs two explicit, exclusive receipts:

1. A frozen complete model inventory after all declared training, validation,
   source/config/checkpoint checks; a consume-once access record must be fsynced
   before any final generation or loading.
2. A finalized dataset identity after generating the frozen recipe exactly once,
   then exact-byte evaluation by every frozen checkpoint on that one inventory.

Bind final ordered record IDs, groups, inputs, targets, transformations, effective
and original queries, seeds/runtime and file hashes. Any failed attempt remains
failed with its artifacts. Repair or recovery is an explicit new version, not an
overwrite or an undisclosed rereading of a newly exposed final set.

## Bounded pilot and primary panel

The agreed Stage A is development-only: contract tests, two fixed 32-row
memorization diagnostics, and four 1024-update pilot runs (two families × two
rates, initialization 39). Keep all pilot outcomes. Select each family's learning
rate only by base validation binding, validation CE and LR tie-break order. The
memorization diagnostic has a predeclared 4096-update ceiling and a fixed stop
criterion; it is not a source of primary checkpoints or evidence of held-out
generalization. Save its complete stopping evidence and elapsed cost.

Stage B requires a second source/recipe/inventory freeze, with rates selected
before final access. It comprises two families × five initializations 40–44,
4096 updates each. This is a new exploratory panel, not a rerun or extension of
historical H1. A single run checkpoint is the final scheduled update. No hidden
early stopping, best-final selection, or extension is permitted.

Whether this workload fits the allowed CPU workflow is an empirical feasibility
question. Stage A supplies measured update, validation, peak/available-resource
and archival costs before Stage B admission. Use those costs with explicit
overhead and shutdown/archive reserve to set finite per-run and whole-workflow
deadlines. Do not infer a defensible wall budget solely from the earlier 8×8
fixture run. If the approved inventory cannot fit, retain the pilot and obtain a
prospective resource/recipe decision before launching; do not quietly cut runs
or use final results to revise the budget. Admission must recheck RAM before
each numerical child and monitoring must stop only owned child processes.

## Artifacts needed for independent recount

Keep plain UTF-8 JSON/JSONL metadata and non-object numeric arrays loadable with
`allow_pickle=False`; no model import is needed to recount predictions.

| Artifact | Minimum bound content |
|---|---|
| Recipe/source | Exact committed plan bytes, code/config SHA-256 map, source ZIP, actual HEAD before/after, all constructor and optimization arguments |
| Scenarios | Canonical bags, facts/layouts, group/scenario IDs, partition rule, sampling seed/runtime, ordered file hash |
| Evaluation records | Canvas, target, role, anchor/effective event, condition, changed-gold flag, record/pair/group IDs and ordered digest |
| Training record | Bag/probe identities, init/data/firing stream seeds, update count, input digest per update, finite losses/gradient summaries, optimizer and status |
| Checkpoint | CPU state dictionary, checksum, actual parameter count, source/config/dataset identities; no implicit legacy PAD interpretation |
| Predictions | Ordered float32 logits `[N,37]`, argmax prediction, target, confidence, NLL and matching record/group/pair indices; exact first-index argmax tie rule |
| Gate | Complete declared candidate inventory, LR-selection evidence, checkpoint/source hashes, fsynced pre-access event, frozen final recipe, completed/failed access status |
| Runtime | Actual package versions, CPU/thread settings, resource samples, bounded stage times/logs, failures and independent archive manifest |

Use separate private data and Torch initialization/firing streams. The matched
primary model families share the same ordered minibatch inputs; no equivalence
between stochastic cellular masks and Transformer updates is implied. Effective
NeuroPixel PAD remains zero through its current dictionary; no old learned PAD
checkpoint is imported.

The recount verifies hashes and ordered alignment, reconstructs targets from
input, recomputes stable log-sum-exp NLL and argmax/global/per-role scores,
validates symbolic/shortcut controls, and calculates paired transformation
diagnostics with their appropriate eligible denominators. Resample whole bags
with common indices across conditions and models. Across five initializations,
report raw paired differences, mean, sample SD/range and the declared df-4 t
interval separately from checkpoint-conditional bag intervals. Do not pool the
12288 correlated rows as independent replications, and do not reinterpret the
old H1 decision from this study's results.
