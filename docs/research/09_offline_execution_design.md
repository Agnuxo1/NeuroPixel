# Item 9: operational fallback for the frozen offline analysis

## Reason and scope

After 09:02:53 UTC on 2026-10-07, the isolated local executor stopped returning even read-only shell requests. Root observed no response or process session ID for source reads and repeated bounded health probes (including a non-login shell and a separate /tmp PTY). The waiting orchestration cells were terminated; none of those requests launched scientific analysis or training. GitHub retrieval and the already-running study job remained available. Preserve this as an operational interruption, not a failed scientific result.

The proposed fallback runs only the already-authored archive audit, actual-Git gate audit, frozen scientific recount and figure helper on one new standard public ubuntu-24.04 runner. It never trains, changes checkpoint selection, regenerates final data, or draws new scientific examples. Item 9 remains active until the complete results and evidence are reviewed. All later items remain pending.

The scientific source remains 83fe135e301f76bc0c74e30c66bb18e067ca5959. The independent analyzer and its 19 successful current tests were frozen before Stage B. Its SHA-256 remains b12ac084aed181b1a22aa96f1b28a1c467d55587e205e0de7f18bb7e32b80e05. Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 remain exact. The additional plotting/supervision dependencies are Matplotlib 3.10.8 and psutil 7.2.2. This changes the execution host, not the declared statistical procedure. Actual platform metadata and numerical residuals are still recorded; matching package versions does not promise hardware-independent bit identity.

The separate gate and figure helpers come from the auxiliary source checkpoint 21097fa8937a2bdcd358816558e5ebe0ae7ca886. They were authored after study launch and before final outcomes were inspected. Their respective SHA-256 values are f4212055fd181b2f34d447d5d5a2e96ca76ce81a3b323f6583ba5343de82774a and f6d1a9d9f03858de316036b6af5197550426f35d4a08a922c23861358d74e05a.

## Prospective operational admission

The new workflow is triggered only by docs/research/09_offline_plan.json on the existing isolated source branch. Publishing the driver and workflow without that plan does not launch a job. Before publishing the frozen operational plan, root must observe that the study job finished successfully, fix the complete archive F and an independently observed results-branch head H, retrieve the claimed intermediate commit G from the archive through the API, and preserve the complete decoded Actions job log plus the provider's response metadata. F, H, G and the study source must be recorded without substituting a later head silently.

The plan binds operational files using their Git blob identities and retains the frozen scientific plan hash. The full scientific binding map is checked at its original commit; all 83 files must remain unchanged. The current driver commit, scientific source commit and final raw archive commit are distinct provenance roles. The driver does not claim to be the source that trained the models.

The standard free public CPU runner uses one numerical thread, one Git packing thread, at most four aggregate active CPU threads, and at least 8 GiB available RAM. Its workflow limit is 30 minutes. Its worker has at most 1,380 seconds within the workflow remainder, including 180 seconds reserved for stopped final archival. Each new child requires at least 60 seconds remaining. Stage caps are 600 seconds for each archival audit, 900 for the scientific recount and 300 for plotting, all shortened by the shared execution deadline. The existing reviewed supervisor monitors only its owned child every second, writes heartbeats every 60 seconds, and terminates that child if the RAM floor or deadline is crossed. Intermediate archives occur at most every 300 seconds during long stages and after every completed stage. No GPU, paid compute, billing change, dependency cache or Actions artifact-storage upload is used.

## Data and failure preservation

The driver obtains complete Git objects and creates two detached worktrees: scientific source S and final raw archive F. It verifies that the previously observed H is an ancestor of the runner's newly observed head, retaining both identities. The final raw directory remains at F. The gate auditor receives the complete decoded job log and the original H anchor; it opens G independently through Git and checks ancestry and exact bytes. This is operational corroboration under shared custody, not external blinding or independent scientific replication.

Run the full archive audit, gate audit, scientific recount and figures in that order. Every output path is new. A failed return code, resource stop or failed receipt stops the sequence and is preserved; no audit receipt or scientific result is overwritten. Any later correction requires a separately identified attempt with its reason and source. The driver records its complete source ZIP, plan, source guards, runtime, original provider receipts, recovered object identities, full logs, resource samples and output hashes. The existing archive helper publishes ordinary fast-forward commits on the isolated results branch and confirms the returned head.

The scientific JSON remains complete and unchanged, including all 2,000 by 256 bootstrap indices. An additional gzip copy is lossless. A clearly labelled display projection omits only the duplicated bootstrap.indices array and points to the full file and its SHA-256; it preserves all other values without recomputing any statistic. A small operational cost table copies saved wall-time and parameter fields from verified raw records and labels their scope. It does not turn wall time into energy or pure inference latency.

Figures remain editorial renderings of saved values. Their receipt deliberately retains visual_review_pending until root actually inspects the generated images and records a separate review. Successful program execution alone does not establish visual quality.

## Availability check and current status

Root opened the official Python 3.12.14 release page and the actions/python-versions manifest on 2026-10-07. The manifest lists an ubuntu-24.04 x64 build for the exact requested version. This is an availability check, not execution evidence.

- https://www.python.org/downloads/release/python-31214/
- https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json
- https://github.com/actions/python-versions/releases/tag/3.12.14-31661455385

At this document's creation the driver is an unexecuted candidate, the most recently observed study job status was in progress, and no final performance outcome has been read. A separate review and the concrete final-archive inputs precede the operational plan. No local syntax test is claimed while that executor is unavailable.
