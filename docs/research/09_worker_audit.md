# Item 9 — Operational worker contract

The new worker is `scripts/research_item9_worker.py`. It supervises existing child CLI contracts; it does not define model selection, losses, training budgets, metrics or final-example generation. The implementation is prospective and has not run the scientific study at the time of this note. Source/plan and resource admission must precede execution.

## Entry points and freeze

Run `python scripts/research_item9_worker.py --phase preflight` or `--phase study` only on the isolated public Actions source branch `research/scientific-validation-2026-10-07-cloud`. The fixed plans are `docs/research/09_preflight_plan.json` and `docs/research/09_study_plan.json` respectively. Both require schema version1, item9, status `frozen`, the matching phase and a past timezone-aware `freeze_utc`.

Every plan binds `docs/research/09_experiment_recipe.json` by `recipe_path`, `recipe_sha256` and the same entry in `implementation_sha256`. The source map must include all Python package files under `neuropixel/`, the scientific controller, this worker, both workflows, the two contract-test files, pinned dependency file and pytest configuration. Further source bindings are allowed. Paths must be relative, canonical and contain no `..` components. The exact test inventory is `tests/test_complex_binding_data.py`, then `tests/test_complex_binding_protocol.py`; the plan supplies a positive exact collected-case count.

The source guard checks actual SHA-256 values, tracked membership, HEAD against the Actions SHA, plan bytes against their committed Git blob, no tracked differences and no unexpected untracked files outside this owned output directory. Source is rechecked before every stage, inside each fresh child before numerical imports, and after execution. The worker saves the plan and recipe, source-before/source-after records and a ZIP of the exact committed tree. Root commits and reviews the concrete source/plan before the corresponding workflow can start.

The workflow push filter contains **only its corresponding plan path**, not the workflow path. Creating both workflows alone does not launch a study. Both use one shared concurrency group with cancellation disabled, a standard public `ubuntu-24.04` runner, pinned checkout/setup-python actions, no GPU, paid runner, cache or Actions artifact upload. The source/main branch is not changed by the worker.

## Stages and runtime

| Phase | Exact ordered stages | Scientific CLI phases |
|---|---|---|
| preflight | `contract_tests`, `preflight` | `preflight` |
| study | `contract_tests`, `train`, `final` | `train`, then `final` |

Each scientific stage invokes the unchanged interface `python scripts/research_complex_binding.py --phase PHASE --plan PLAN --output RUN_DIR/study`, through a fresh worker child and `runpy` after runtime configuration. Train and final share that study directory. Root's scientific controller remains responsible for the complete candidate/checkpoint inventory and the final-access receipt before loading/generating final examples. A failed test, scientific stage, source check, deadline or resource guard prevents later scientific stages. Existing output is retained, not retried or silently resumed.

The supervisor imports no Torch or NumPy. Each child verifies Python3.12.8, Torch2.6.0+cpu, NumPy2.2.6, SciPy1.15.1, pytest9.1.1, psutil7.2.2 and Pillow12.3.0, and requires `torch.version.cuda is None`. Before tests or the scientific CLI it sets **two numerical threads and one inter-op thread**. The trainer must not unconditionally call `set_num_interop_threads` a second time. Runtime records precede/follow the stage; thread-related environment variables are set to2. These are recorded settings/boundaries, not measurements of continuous utilization.

Contract tests run only the two new files. The embedded plugin saves collected node IDs and requires the declared count, unique IDs and exact file set; it saves JUnit XML, stdout, runtime and thread boundaries. No skipped or expected-failure test is permitted. Boundaries must remain2/1; unlike item8, no historical two-thread exception is needed because this suite starts with two numerical threads. JUnit aggregate counts may include subtests; collected IDs and individual case elements remain distinct reporting units.

## Exact `resource_policy` fields

The worker requires equality with this schema. The first twelve fields are common to both plans:

| Field | Value |
|---|---:|
| `numerical_threads` | 2 |
| `interop_threads` | 1 |
| `minimum_available_ram_gib` | 8 |
| `aggregate_active_cpu_threads` | 4 |
| `git_threads` | 1 |
| `supervisor_interval_seconds` | 1 |
| `heartbeat_interval_seconds` | 60 |
| `archive_interval_seconds` | 300 |
| `git_timeout_seconds` | 60 |
| `final_archive_reserve_seconds` | 180 |
| `contract_test_seconds` | 900 |
| `minimum_stage_start_seconds` | 60 |
| `workflow_minutes` | preflight120; study240 |
| `worker_seconds` | preflight6600; study13800 |
| `stage_seconds` | preflight6600; study13800 |

The worker deadline is the earlier of its phase budget and the workflow deadline established before dependency setup (workflow timeout minus30seconds). The execution deadline is another180seconds earlier, reserving final archival **inside** that total. Scientific stage caps are further shortened to this shared execution deadline; 900seconds caps contract tests. There must be at least60seconds of remaining execution time immediately before any child launch. This minimum is admission hygiene, not a prediction that a scientific phase will complete in60seconds; an incomplete stage is stopped and preserved.

An independent thread samples available RAM and checks the owned child deadline every second while main may be copying files or waiting on Git. It saves sampled-resource receipts and emits a heartbeat at most60seconds apart while monitoring. Below8GiB or at deadline it signals only the new child process group, escalating SIGINT→SIGTERM→SIGKILL with five seconds per step. Poll/wait/signals share a lock to avoid reaping races. The parent stops/reaps the child and joins the monitor before final archival. Git runs in its own session, with noninteractive stdin, one packing/index thread, a temporary output file and a timeout bounded by both60seconds and the current deadline; timeout kills only that owned Git session.

Two configured numerical threads plus one Git thread stay within the declared four-active-CPU allowance, with light Python supervision; this is not an OS-wide CPU-utilization cap. Available-RAM observations are sampled values, not a continuous mathematical guarantee or process peak RSS.

## Archives, durability and final access

Local and remote run paths are `results/research/09_cloud_runs/{run_id}-{attempt}-{phase}`. A new directory is exclusive; a results-branch path already present at worker startup is refused. Archives use a separate detached worktree from `research/scientific-validation-2026-10-07-cloud-results`. Only this owned run directory is staged. Pushes are normal fast-forwards, never forced; push confirmation checks the remote ref. Concurrent incompatible advancement fails conservatively and preserves local files.

There is an archive before scientific execution, after each successful stage, and periodic archive attempts at300second intervals while a stage runs. Each interim transport batch has a120second total deadline; the independent resource/heartbeat monitor continues during it. A network timeout is a recorded failure, not a promise that network delivery always completes by300seconds. Final publication has at most the remaining180second reserve and occurs after the child has stopped.

Copies of growing files are limited to the byte length observed when each file is opened. After copying completes, the manifest is computed **from the copied files in the results worktree**. It does not hash a later version of the active training log. Interim snapshots can include prefixes/incomplete live files and different observation times; `interim_manifest.json` explicitly labels them `interim_partial`. They are not completion evidence. Temporary `.tmp` files and the scientific writer's `_pending_` files are omitted and symlinks refused. Prior interim versions persist in Git history. The final stopped snapshot removes the interim manifest and adds `archive_manifest.json`, with `final: true` and an independent `attempt_status` of completed or failed. A final manifest is never overwritten. Failure to publish is recorded locally and in Actions stdout; it does not claim remote durability.

After a zero-return train stage, the worker publishes and confirms its archive. It checks that copied `study/final_inventory.json` has the same SHA-256 as the stopped local inventory and that all ten checkpoint/summary references and the primary-run index match the copied bytes/hashes. It then exclusively writes and fsyncs `study/gate_archive_receipt.json` and its containing directory:

```json
{
  "schema_version": 1,
  "item": 9,
  "source_commit": "actual frozen execution commit",
  "archive_commit": "confirmed results commit containing the inventory",
  "results_branch": "research/scientific-validation-2026-10-07-cloud-results",
  "final_inventory_sha256": "SHA-256 of the identical copied/local inventory",
  "archived_at_utc": "actual aware UTC timestamp"
}
```

These strings describe runtime values, not literal placeholder values to use in a run. An existing receipt is refused. The scientific final phase must validate that receipt and every expected checkpoint/summary, then consume its own access receipt before generating final examples. The archival receipt attests remote preservation of the declared inventory; it does not substitute for verifying its ten-checkpoint contents. No final phase starts if training or confirmed archival fails.

## Verification status and limits

This note specifies new operational code for root review. Local validation is limited to source inspection and compilation; no scientific child, pytest suite, model, Git push or workflow was executed by its author. Execution receipts and actual timing/resource outcomes must be reported after the frozen job. The worker is not a security sandbox, external custody mechanism, scientific replication, efficiency experiment or permission to open item10.
