# Item 8: bounded CPU validation worker

This note records implementation inspection before execution. The worker and
workflow have been compiled/inspected locally; Torch is not installed in that
environment. No cloud stage, model fixture, regression suite or scientific
experiment has been executed by this preparation step. Root freezes the exact
plan and source hashes before the declared public CPU job.

## Inventory and preservation

[`research_item8_worker.py`](../../scripts/research_item8_worker.py) adapts the
item-7 coordinator. Its exact stage order is:

1. `legacy_padding_demonstration`: execute the preserved-source PAD fixture
   through `tests/test_padding_invariant.py --legacy-demonstration OUTPUTJSON`;
2. `full_pytest`: collect and run the complete current `tests/` suite once,
   saving JUnit XML, collection node IDs and thread-boundary observations.

The plan path is `docs/research/08_cloud_validation_plan.json`. It must be
committed verbatim, marked frozen, identify item 8, bind the worker/workflow,
dependencies and all 13 test files, and declare the expected collected count
(187 at this inspection). The runtime collection checks the actual count,
unique node IDs and file inventory before running cases. The only permitted
optional skip is `tests/test_core.py::test_cartilla_layout`, for unavailable
local CIFAR data; no dataset download is requested.

An ordinary nonzero legacy-fixture result does not suppress the full regression
stage. Both ordinary failures are retained and make the worker fail. A source
drift, RAM stop or deadline stop prevents further work. These distinctions avoid
turning failure preservation into permission to exceed a resource boundary.

Evidence is written exclusively to
`results/research/08_cloud_runs/{run-id}-{attempt}`: exact plan bytes, `source.zip`
from the execution commit, before/after source hashes, stage statuses/log hashes,
actual runtime versions, environment/thread observations, fixture output when
produced, test collection/JUnit outputs, worker status and a final manifest.
Incomplete or failed evidence is also archived. A separate detached worktree
adds only that new path to the existing isolated results branch and uses a
normal push. The source worktree HEAD is not checked out, committed or reset.

## Runtime and resource contract

The standard public Ubuntu 24.04 workflow pins Python 3.12.8 and Torch
2.6.0+cpu and installs the existing `requirements/research-cloud.txt` unchanged.
The child verifies installed versions of Torch, NumPy 2.2.6, SciPy 1.15.1,
pytest 9.1.1, psutil 7.2.2 and Pillow 12.3.0. A CUDA Torch build is rejected.
Source/plan and RAM admission checks precede the Torch import. The coordinator
imports no numerical runtime.

Normal numerical work uses one intra-op and one inter-op thread, with common
BLAS/OpenMP environment controls set to one and pytest plugin autoload disabled.
The legacy demonstration runs in a fresh process and sets both Torch controls
at its own entry point. Its first environment record is explicitly *before*
that CLI initializes thread controls; its final record must show 1/1.

The full suite sets Torch to 1/1 before collection and restores one intra-op
thread between cases. The existing safety-compatibility fixture deliberately
calls `choose_device(..., threads=2)`. It is the sole allowed temporary
two-intra-op exception, identified by its full node ID in the worker, and is
recorded rather than hidden by monkeypatching production behavior. The other
tests begin and end at 1/1. Thread-boundary observations are not continuous CPU
utilization measurements; aggregate active CPU remains bounded by the declared
four-thread ceiling, with sequential child work and one-thread Git packing.

Admission and child monitoring require at least 8 GiB available RAM. The parent
samples at most one second apart during each child. Each stage has an execution
deadline of at most 900 seconds; termination gets bounded shutdown grace, which
is included in the separately named `wall_seconds_including_shutdown` field.
The first workflow step establishes a deadline 30 seconds before its 35-minute
timeout. The worker reserves up to 180 seconds for final archival, so a late
stage can be shortened or refused rather than consuming that reserve. Git and
transport processes use private sessions and finite remaining-time bounds;
temporary files avoid indefinite reads from inherited pipes. No child is alive
when final archival starts.

Dependency-setup failure before the worker starts is available in Actions logs,
not a worker receipt. Network/archive failure cannot be converted into a claim
that evidence was pushed successfully. The workflow uses no paid runner,
dependency cache or Actions artifact upload. Its push trigger is restricted to
its own YAML and the item-8 plan, so later reports do not relaunch validation.

## Static full-suite interaction correction

The item-7 `DevelopmentSamplerTests.setUpClass` unconditionally called
`torch.set_num_interop_threads(1)`. In a combined process an earlier PAD class
or worker can already have configured inter-op threads; Torch does not permit
unconditional repeated initialization. This was identified by source inspection,
not reproduced locally without Torch.

The exact earlier file is retained at
[`development_sampler_test_before.py`](../../results/research/08_validation/development_sampler_test_before.py),
SHA-256 `698c7a7184cf12a7b6abdf0fd71d5723c84d0ce87c1221ae163e6ed3147a764e`.
The only correction is to call the setter when `get_num_interop_threads() != 1`.
AST comparison confirms all four test methods are unchanged; sampler production
code and assertions are untouched. The corrected test file SHA-256 is
`0be8bce4d4a3ec513ef88590b810513fda3ed8800daf759a6cc7ea12a2b16e27`.

The legacy PAD witness and optimizer updates in regression fixtures are small
implementation checks, not a scientific training programme or new final-test
performance assessment. Historical fixture repairs coordinated separately by
root remain distinct from this worker preparation. Local preparation performed
syntax compilation and static inventory/AST inspection only; the eventual
runtime results must be taken from the frozen job's preserved evidence.
