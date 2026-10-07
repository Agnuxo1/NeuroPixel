# Item 9: independent operational source review

Review completed at **2026-10-07T07:41:51.593794+00:00** before execution-plan
freeze. **No unresolved concrete blocker was identified in the reviewed worker,
two workflows, or corrected development-manifest linkage.** This is a static
source review, not an executed worker test, performance run, timing estimate or
independent model replication.

The reviewer read the current worker/workflows and relevant trainer/gate
functions. No Torch or project model was imported; no scientific dataset was
generated, child stage executed, remote object accessed or Git ref changed.
Light local work used one Python process. The recorded available-RAM admission
was **9.126087188720703 GiB** under Python **3.12.14**. That local interpreter is
not the pinned cloud training runtime.

## Exact reviewed inputs

| File | SHA-256 |
|---|---|
| `scripts/research_item9_worker.py` | `a9012e556337c51d49a1fa95db60f166d6dec78669440fdca1a084f90cb190d5` |
| `.github/workflows/research-item9-preflight.yml` | `af5d60a9c6bf5385a88d71620ab6507c18c336040e46a3d3a6b9d4e9bca482de` |
| `.github/workflows/research-item9-study.yml` | `0215d2256f97b55abbe4c4d0bd90614db91a76d31e8677f68db147f2fd6cd8ac` |
| `scripts/research_complex_binding.py` | `e76ce37640c70a0c3632cf4a2078371a94fdde229cff4aa9bc0f5daf36e7ee05` |
| `neuropixel/research/complex_binding_protocol.py` | `7c8a23d2c867702f627c9da14984bc62fa736c8bebd2403f0ac704e1a09769c2` |
| `docs/research/09_worker_audit.md` | `fe723f0aba0ea6c4a11eeebbb2ce517e285c8a1a75db9f3172879f0d364458f5` |
| `docs/research/09_experiment_recipe.json` | `7450b1ede40542c5ff940842e42eb2725ac4420946baad6bac920e779a214b97` |

## Development identity: reported issue resolved

The initial trainer wrote `fixture_exposure.json` without listing it in its
artifact or identity map. The mutable development manifest also supplied the
final exclusion list without a separately anchored manifest digest. This was
reported before freeze; the trainer owner corrected it.

The reviewed correction includes the fixture ledger in both `files` and
`identities`, places the complete development-manifest artifact descriptor in
the run context, and verifies that descriptor before loading the manifest.
`load_development` verifies listed artifacts and requires equality of the saved
exclusion list and the verified ledger, plus the ledger's recorded SHA-256.
The final inventory binds that context. Pilot-to-primary comparison uses the
declared content identities, including the fixture ledger; each phase separately
anchors its actual complete manifest. The issue is resolved at the trainer hash
listed above.

The related API review found matching generator calls and no shared-scenario
mutation: rendering and augmentation return new objects; the training bag list
remains fixed; each run creates its own input RNG from the declared seed; matched
families receive the same generation stream. `controls_for` passes only canvas
to control selectors, then uses the recorded target for scoring. Final sampling
uses the same anchored exclusion list after access consumption.

## Operational checks

- **Source and runtime:** plans must be frozen item-9 plans with the declared
  stage/test/resource inventory. Source guards bind tracked file hashes, clean
  tracked bytes, allowed untracked output, actual Actions HEAD, and the exact
  committed plan blob. Guards run before each child and inside it before Torch
  import, and again after execution. The ZIP archives the actual source commit.
  Each child validates Python 3.12.8 and the pinned CPU dependency versions,
  sets two intra-op/one inter-op Torch threads, and records stage boundaries.
- **Owned process supervision:** a separate thread checks RAM and deadline every
  second while main may copy files or block on Git. Child poll/wait/signals are
  serialized. Termination targets the private child session, with bounded
  SIGINT/TERM/KILL escalation. Git has a separate owned session, no interactive
  stdin, temporary-file output and a timeout bounded by 60 seconds and the
  current absolute deadline. Later stages require fresh source/resource/time
  admission, and failures prevent later scientific stages.
- **Budgets:** preflight workflow/worker caps are 120 minutes/6600 seconds;
  study caps are 240 minutes/13800 seconds. Dependency setup also consumes the
  workflow budget. The execution deadline reserves 180 seconds for stopped
  archival; contract tests have a 900-second cap, and each stage is constrained
  by the remaining common execution budget. Sixty seconds of remaining time is
  a start guard, not a promise that a scientific stage fits. Stage-B admission
  should use measured Stage-A costs as described in the execution design.
- **Partial archives:** only the owned run directory is copied into a detached
  results worktree. Growing files are copied only through their size at open;
  hashes describe the copied prefix. `_pending_`/`.tmp` files are omitted and
  symlinks rejected. Interim manifests explicitly state partial status. The
  stopped final snapshot replaces the interim manifest and separately records
  successful or failed attempt status. A final manifest cannot be overwritten.
  Normal pushes are confirmed against the remote ref; source HEAD is unchanged.
- **Final gate ordering:** a zero-return complete training stage is archived
  first. The worker checks the confirmed copy of `final_inventory.json` and all
  ten checkpoint/summary references plus the primary index against their
  declared hashes and sizes. Only then does it exclusively write/fsync the
  archive receipt. The scientific final phase verifies that receipt and the
  frozen inventory, consumes a one-use access receipt, and only afterward
  generates final scenarios. Failed training or archival cannot start final.
- **Workflow scope:** both use the standard public Ubuntu 24.04 runner and one
  shared concurrency group without cancellation. Each push trigger contains
  only its own plan path on the isolated source branch. Creating reports or
  changing unrelated sources does not itself launch either study. No GPU,
  paid runner, dependency cache or Actions artifact-storage step is configured.

RAM records are sampled observations, not a continuous lower-bound proof or
peak-process-RSS measurement. Thread settings are not an OS-wide utilization cap.
Finite network timeouts bound attempts but cannot guarantee archival delivery;
failed delivery is retained locally and printed. This local gate does not create
external custody or blindness. Actual runtime, completion and evidence integrity
still require their execution receipts and independent saved-artifact checks.

The review introduced no change to scientific recipes, historical controllers,
H1, source refs or the worker itself. Items 10 and later remain outside scope.
