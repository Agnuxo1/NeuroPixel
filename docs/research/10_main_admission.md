# Item 10: independent preflight audit and main-study admission

Recorded 2026-10-07T11:35:07+00:00. **The saved-artifact audit verified the preflight and the main study is admitted under the fixed finite budget below.** Execution of the main study is a separate phase and no main outcome is available at this timestamp.

## Independent evidence

The [audit run](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37614761119) used operational source `6e7cad49305d3d042ba931acaa13bd49a554c2f3` and the unchanged scientific auditor previously bound by the neural preflight source `071be375dce45d0d7fabce3d095bd1003d53ed36`. It completed successfully. Its [immutable archive](https://github.com/Agnuxo1/NeuroPixel/tree/83a4f36f07ee05d8fb34fabe37b61b41fbdbe0f8/results/research/10_cloud_runs/37614761119-1-audit) contains 19 files and 10,025,025 bytes excluding its manifest. The [completion receipt](../../results/research/10_validation/preflight_audit_completion_receipt.json) contains exact source identities, raw report hashes, resource records and limits.

The operational audit passed 5,819 checks with zero failed checks or issues. It verified all 91 preflight and 35 acquisition files including their manifests, their actual Git identities and ancestry, and every source-ZIP entry against the 432-file neural source and 403-file acquisition source inventories. It re-extracted only the declared 117,470-byte TRAIN member from the pinned compressed archive, recovering SHA-256 `c9370502d85fa382f32d1fea2111e817f81ae9e3e79eb2924a1a5392a90417fa`. Traversal of other TAR headers does not mean their payloads were selected or parsed.

The fresh scientific process used Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 with one numerical thread. It imported no model or Torch code and performed no model inference. It verified 105 input files and reported ten successful check groups, including raw TRAIN parsing/grouping/encoding, eight independently reconstructed controls, all six run configurations and preserved sampling/prediction arrays, all 19 memorization snapshots, and selection. Its admission result is true. All six DEV and full-TRAIN probe metrics agree exactly with the saved summaries in the 84-field root cross-check. Weight files were hashed, not deserialized.

The selected rates remain 0.003 for both families. The small-set diagnostic stops remain 2,048 updates for NeuroPixel and 384 for the Transformer, with the first two consecutive qualifying checks in each case. All lower, unstable and development results in the [initial preflight report](10_preflight_results.md) remain part of the evidence. That report's pending-audit statements describe its earlier timestamp and are resolved by this audit, without changing any original data or scientific source.

## Resource observations

The audit worker used 11.168775286 seconds before its final archive; the scientific child process used 0.665720720 seconds. Five parent-supervisor samples recorded a minimum available system RAM of 14.452686309814453 GiB. Five wrapper boundaries and 32 scientific-auditor boundaries were also sampled. These are sampled system-availability measurements, not continuous minima, peak model memory, energy or physical utilization. The standard public CPU runner is released.

## Admitted main recipe and operational bound

The scientific recipe is unchanged: five paired fresh seeds 60–64, both model families, 4,096 updates per run, batch size 32, AdamW with weight decay 0.0001, gradient clipping 1.0, and answer cross-entropy only. Both families use their DEV-selected learning rate 0.003. NeuroPixel has 29,552 parameters and the relative-position Transformer 29,562. Initialization, sample-index streams, data identity, vocabulary, grid and all input-only fixture declarations remain fixed. No pilot checkpoint initializes a primary run.

The [cost admission record](../../results/research/10_validation/main_cost_review.json) takes the worst observed pilot training time and per-run nontraining overhead within each family. Scaling to ten primary runs gives 1,290.110063305 seconds. A twofold safety multiplier plus 60 seconds gives 2,640.22012661 seconds. The main train-stage cap is 2,700 seconds; contract tests and final evaluation each have 300-second caps. A 180-second final-archive reserve is included in the 3,600-second total worker limit. The workflow cap is 70 minutes, with a 4,170-second deadline established at its first step. The earliest applicable deadline wins. These finite operational choices are not confidence limits or completion guarantees.

One standard public CPU job uses two numerical threads, inter-op one and Git one, under the aggregate CPU cap of four and an 8 GiB available-RAM floor. RAM is supervised every second and intermediate evidence is archived at 300-second intervals. A failed or incomplete attempt must retain its records and cannot unlock final evaluation. No optional tuning or result-dependent budget extension is part of this study.

## Final access and interpretation

All ten prescribed configurations, checkpoints and run artifacts must be completed and published to the results branch before the worker issues its pre-final archive receipt. The scientific controller then validates that inventory and writes an exclusive consumed-access marker before reading the exact QA4 TEST member. The input representation, vocabulary and selection remain based on optimization TRAIN and DEV. Unsupported answers and unencodable inputs remain in the accuracy denominator; no TEST-derived adaptation is allowed. Every main seed is retained.

The primary contrast is the mean paired NeuroPixel-minus-Transformer accuracy difference over five seeds on all official TEST rows, with the frozen paired t interval. Full results also retain TRAIN/DEV probes, shortcut controls and the two prespecified novelty masks. The common source-component bootstrap is conditional on the fixed checkpoints and its estimand is distinct from across-seed variation. Empty novelty subsets must be reported as unavailable.

This trains the architecture afresh on an externally authored synthetic benchmark. It does not test zero-shot transfer of the project's previous weights, natural language in the wild, or generalization outside the bAbI generator. Input novelty and deduplication do not by themselves establish systematic compositional generalization. The prospective access record is internal governance over public data, not external blind custody. Historical H1 remains closed and not supported. Item 10 is still active; items 11–30 remain unopened.
