# Item 10: completed neural preflight

Recorded 2026-10-07T11:31:58+00:00. This report records the six prespecified TRAIN/DEV executions. The independent saved-array/archive audit is the next required gate; the five-seed main study is not yet admitted. Item 10 remains active and items 11–30 remain unopened.

## Evidence and fixed design

Scientific source: `071be375dce45d0d7fabce3d095bd1003d53ed36`. Plan SHA-256: `f01846c1c6685328d49a5cdcb6d5c399222f99b8f160e37ccff62ee5e0418993`. Recipe SHA-256: `8492892f58a4065a92df99dbbbf30f0bd7340111c853cf5533a7102109fe8d19`. [Public CPU run](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37612623870) completed successfully at 11:18:29 UTC. [Immutable raw archive](https://github.com/Agnuxo1/NeuroPixel/tree/707be7d4c1df58bde0706d965e6678c0be6a2cfa/results/research/10_cloud_runs/37612623870-1-preflight) contains 90 files and 14,645,751 bytes excluding its manifest. Exact metrics, references and all memorization checks are retained in [the completion receipt](../../results/research/10_validation/preflight_completion_receipt.json).

The external bAbI QA4 TRAIN member and all derived data are fixed by the preceding [acquisition report](10_acquisition_results.md). There are 895 optimization-TRAIN records and 105 grouped DEV records. The input representation is a 4 by 8 grid with an 18-token TRAIN vocabulary. Two seed-58 diagnostics train on the same 32 input-deduplicated records; four seed-59 pilots compare learning rates 0.001 and 0.003 for each family, with 1,024 updates, batch size 32, answer cross-entropy and identical sampled input sequences across paired runs. Parameter counts differ by 10 (about 0.03384% of NeuroPixel's count). Parameter matching does not establish equal compute, memory, energy or inductive bias.

## All six outcomes

| Run | Completed updates | Parameters | Full TRAIN probe correct | DEV correct | DEV cross-entropy | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| mem_neuropixel_s58 | 2048 | 29552 | 173/895 | 13/105 | 46.695875 | 112.059 |
| mem_relative_transformer_s58 | 384 | 29562 | 287/895 | 28/105 | 5.8504168 | 3.206 |
| pilot_neuropixel_lr0001_s59 | 1024 | 29552 | 220/895 | 16/105 | 2.1456817 | 55.419 |
| pilot_neuropixel_lr0003_s59 | 1024 | 29552 | 381/895 | 36/105 | 1.5318370 | 55.705 |
| pilot_relative_transformer_lr0001_s59 | 1024 | 29562 | 722/895 | 74/105 | 0.63974895 | 8.496 |
| pilot_relative_transformer_lr0003_s59 | 1024 | 29562 | 780/895 | 79/105 | 0.66940223 | 8.599 |

All four pilots actually visited all 895 eligible TRAIN records, as recorded in their preserved sampling arrays and run summaries. The forthcoming array audit must independently verify those statements. The 32-record diagnostics each visited their own 32-record training population. Their separate 895-record probes are broader probes, not the populations they were trained to memorize.

The fixed selection rule prioritizes DEV accuracy, followed by lower DEV cross-entropy and then lower learning rate. It selects 0.003 for both families: 36/105 for NeuroPixel and 79/105 for the relative-position Transformer. The Transformer at 0.001 has lower DEV cross-entropy but fewer correct answers (74/105); this does not override the declared ordering. No TEST result was used for selection.

## Memorization trajectory and interpretation

The criterion required 100% accuracy and cross-entropy at most 0.05 on two consecutive checks 128 updates apart. NeuroPixel first met the two-check condition at updates 1,920 and 2,048, with cross-entropies 0.0000027958837362927973 and 5.810271319634134e-8. Earlier isolated passing checks at 768 and 1,024 were followed by failures; all 16 snapshots are retained. The Transformer met the condition at 256 and 384, with cross-entropies 0.0019354750325172325 and 0.0009372706738454208; all three snapshots are retained. Both runs stopped at the first successful two-check streak under the frozen rule.

These diagnostics establish that this implementation and optimizer can fit this small sample under the declared recipe. They do not establish reliable relational reasoning, full-task optimization sufficiency or generalization. Every full-TRAIN probe remains below 95%. The pilot difference is an observed development result, not a five-seed estimate or a confirmatory result. Historical H1 remains closed and not supported.

## Verification completed and still pending

All 32 uniquely collected parent test methods passed: 24 data contracts and eight controller contracts. There were an additional 87 passing subtests, producing 119 passing call reports and a JUnit header count of 119. There were no failures, errors or skips. The tests cover actual model factories/backward calls, strict input/denominator contracts, sampling and checkpoint inventory, receipt paths, and equality between the 149 declared invented input contexts and the test definitions.

The worker verified all 21 source/configuration bindings before and after execution and recorded clean tracked source. Root inspected and rehashed five interim test artifacts, five final summary/status artifacts and six run JSON files. A separate JSON-only review rehashed 25 text artifacts and passed 125 checks, including expected configurations, parameters, budgets, all 19 memorization-check metadata records, stopping behavior and the DEV selection rule. [That review](../../results/research/10_validation/preflight_json_review.json) explicitly excludes binary array or weight deserialization and full ZIP inspection.

Next, the unchanged prospectively bound scientific auditor will run in a fresh process with NumPy/SciPy only. The operational wrapper will recover the exact acquisition and preflight Git archives, verify complete file inventories and bytes, compare every source-ZIP entry with its source commit, and re-extract only the declared TRAIN member. The auditor will independently reconstruct inputs, sample exposure, configurations, all saved prediction metrics and the memorization criterion. This audit has not yet executed at this report's timestamp. No new model inference or official TEST-member extraction belongs to it.

## Resources and conditional next phase

The worker used 262.467097883 seconds before its final archive; the scientific preflight used 247.298092235 seconds. Four contract-stage and 249 neural-stage supervisor samples were retained. The minimum observed available system RAM in those child samples was 14.281620025634766 GiB, above the 8 GiB requirement. Numerical threads were two, inter-op and Git threads one, within the aggregate CPU cap of four. This is sampled available system memory, not continuous minimum or model peak memory. The public standard CPU runner is released. The local executor remains unavailable.

The [conditional cost review](../../results/research/10_validation/main_cost_review_candidate.json) uses the worse observed pilot timing within each family. It projects 1,290.110063305 seconds for ten runs including scaled training and observed per-run overhead. A factor of two plus 60 seconds for recovery/selection gives 2,640.22012661 seconds, inside a proposed 2,700-second train stage. This is an operational projection, not a statistical interval or a guarantee. The proposed 3,600-second worker contains test/train/final stage caps and a 180-second final-archive reserve; the proposed workflow cap is 70 minutes. Independent audit success remains necessary before actual main admission. The main recipe remains five fresh paired seeds 60–64, 4,096 updates each, with every checkpoint archived before the single consumed final-access gate.
