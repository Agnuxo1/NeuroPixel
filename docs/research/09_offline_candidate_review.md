# Item 9 offline candidate review record

This is root's record of independent review messages, not a byte-identical transcript or an execution result. The reviewed initial candidate is f8c477d1a294b3eed293466eb2fcccd6cd72d8d3. Both reviewers used GitHub file retrieval, not the unresponsive local executor, and did not inspect final performance outcomes.

The evidence reviewer checked the exact auditor interfaces, data roots, runtime, helper versions, status gates, figure input, report projection, gzip and cost-field mappings. No blocker was identified in those contracts. The scientific analyzer runs from S=83fe135e301f76bc0c74e30c66bb18e067ca5959, reads RUN/study, and keeps Python3.12.14, NumPy2.3.5 and SciPy1.17.0. The full archive and actual-Git gate auditors read RUN. All three must verify before figures run. The figure helper receives the complete original scientific JSON. The display projection copies the bootstrap dictionary before omitting only indices, while the original and lossless gzip remain complete. Cost fields and their measurement limits match the frozen trainer. This code review did not approve an operational plan that had not yet been written.

The operational reviewer identified three corrections before freeze:

1. Checkout, Python setup and installation lacked individual time limits; a bootstrap failure could occur before any worker archive existed. Bound those steps and explicitly preserve pre-worker failures through complete Actions logs and API status, without calling them completed scientific attempts.
2. The parent polling loop needed an immediate supervisor.failure check. Without it, a monitor that could not stop/reap its child might leave the parent waiting for the workflow timeout instead of entering the preserved failure path.
3. Restore the public-repository condition as well as the exact repository-name condition, to enforce the declared standard public-runner scope.

Root incorporated all three before the operational freeze. The workflow now caps deadline/checkout/setup/install steps at1/3/4/4 minutes, adds an always-step provider outcome record when the platform permits it, and retains the worker's finite shared deadline and180-second stopped archival reserve. The parent checks supervisor.failure in each polling iteration. The job requires the exact public repository.

The reviewer also noted that copied external receipts use basenames. Root added an explicit uniqueness check and refusal to overwrite a destination. The operational plan must bind every script, workflow, log and provider receipt actually used; the driver now checks that its required set is present and that both log bindings agree. The raw manifest must identify a completed final archive. These are input/operation checks, not altered scientific criteria.

The proposed host change itself is compatible with the frozen analysis. Exact package versions do not guarantee complete cross-host binary identity, so existing tolerances remain in force and any discrepancy remains a preserved failure. The new runner is an operational continuation under shared custody, not an independent experimental replication. It must keep S and F in separate clean worktrees, retain root's H anchor alongside any later observed head, run only the frozen analytical bootstrap, and never regenerate examples or retrain weights.

Final-plan review remains pending after these source corrections. The final-archive identities, exact operational blobs, complete decoded job log and external receipt bindings must be checked before the triggering plan is published. No local compilation, unit test or actual audit execution is claimed in this review record.
