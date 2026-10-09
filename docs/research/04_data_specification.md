# Item 4 — Frozen training-data implementation

## Scope fixed before neural training

This implementation specification was recorded on 2026-10-06, after item 3 was
closed at `f835b0a` and before the item 4 neural pilot. It implements the data
portion of frozen protocol `NP-SCI-20261006-v1`; it does not amend the frozen
protocol files or change the original `neuropixel/task.py` generator.

`ResearchRoleTask` provides the adjacent task and the existing public attributes
`v`, `h`, `w`, `out_pos`, `query_pos`, and `role_ids`. Its constructor is
`ResearchRoleTask(h=8, w=8, seed=0)`. The constructor seed determines only the
triple partition. Sampling requires an explicit, independent CPU generator.

## Exact triple partitions

The original vocabulary contains 12 nouns and 10 verbs, with different agent and
patient nouns. There are exactly 1,320 valid triples. For split seed `s`, use the
original `RoleTask` triple enumeration and `torch.randperm` with a private CPU
generator seeded by `s`. The shuffled triple sequence is partitioned as follows:

| Partition | Positions in the shuffled sequence | Triples |
|---|---|---:|
| Final test | `[0:264]` | 264 |
| Validation | `[264:396]` | 132 |
| Training | `[396:1320]` | 924 |

The final test membership and order are exactly the original generator's 20%
test partition. Validation takes the first 132 triples from the original training
partition, and research training uses its remaining 924 triples. All three sets
are pairwise disjoint and their union is the complete valid vocabulary product.
The declared split seeds are **0, 101 and 202**.

Public `train_triples`, `validation_triples`, and `test_triples` are immutable
tuples. Each split has a separate cached CPU tensor. Unknown split names raise an
error: validation can never silently use the test pool through the original
generator's train-versus-everything-else conditional.

## Sampling and random-state separation

The method signature follows the original generator:

```python
sample(batch, split="train", generator=None, device="cpu",
       query_role=None, place_pool=None, meta=False)
```

The caller must provide a CPU `torch.Generator`. A missing or non-CPU generator
raises an error rather than falling back to global randomness. Every random draw
and every layout operation happens on CPU, in the original sampler's order:
triple, place, distinct fact rows, fact columns, and query role. Layout uniforms
use explicit float32 values. A forced query still consumes the original query
draw before replacing it, preserving the original CPU call sequence.

After sampling, the returned canvas, targets, and optional metadata move to the
requested destination. The sampler never generates random values on the output
device. Resetting the same CPU generator therefore creates the same examples
for a CPU or CUDA destination. Metadata retains the original `rows`, `cols`, and
`fillers` fields. The training stream uses seed **9002**; the runner owns and
advances this generator separately from initialization and update-mask RNGs.

## Frozen evaluation and optimization-probe datasets

`frozen_dataset(task, split, n, seed, device="cpu")` returns `(canvas, target,
roles)`. The final tensor contains query-role indices 0 through 3. `n` must be a
positive integer divisible by four. The helper creates one private CPU generator,
then samples `n/4` examples for each query role in the declared `ROLES` order:
AGENTE, ACCION, PACIENTE, LUGAR. It concatenates those blocks without shuffling.
It neither reads nor advances PyTorch's global random state.

| Dataset | Partition | Examples | Examples per role | Sampling seed |
|---|---|---:|---:|---:|
| Validation | validation | 2,048 | 512 | 61001 |
| Final evaluation | test | 4,096 | 1,024 | 61002 |
| Optimization probe | train | 512 | 128 | 61006 |

The neural runner owns atomic data export and content hashes. Model parameters
and initialization seeds cannot change these datasets or the training stream.
Data hashes must distinguish partition and split seed; a file name or model seed
is not a substitute for hashing the actual inputs, labels, and role indices.

## Validation boundary

Tests check the declared shuffle independently, complete coverage and disjointness,
unchanged original test membership, metadata truth, exact role balance/order,
separate split caches, and independence from global/model-initialization RNG.
Device equivalence is structurally enforced by CPU generation followed by transfer;
a direct CUDA equality test runs only when CUDA is available. This data work does
not train models, inspect new neural scores, or evaluate future programme items.

The data-specific suite completed before training: **23 passed, 1 skipped in
1.14 seconds**, using Python 3.12.14 and PyTorch 2.14.1+cpu with the CPU thread
limits set to two. The skip is the direct CUDA destination comparison because
this environment has no CUDA device. Five tests also confirmed that current
original and research test samples, including metadata, are exactly equal for
sampling seed 5 and either unforced or individually forced query roles. This
verifies the current software implementation, not archived historical tensors.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /workspace/scratch/61d6e3e6e688/neuropixel-env/bin/python \
  -m pytest tests/test_research_data.py -o addopts='-p no:nengo' -q
```

The runner should keep the training generator on CPU even when model tensors
live on CUDA:

```python
task = ResearchRoleTask(seed=0)
training_generator = torch.Generator(device="cpu").manual_seed(9002)
canvas, target = task.sample(64, "train", training_generator, device=model_device)
validation = frozen_dataset(task, "validation", 2048, 61001)
final_test = frozen_dataset(task, "test", 4096, 61002)
training_probe = frozen_dataset(task, "train", 512, 61006)
```
