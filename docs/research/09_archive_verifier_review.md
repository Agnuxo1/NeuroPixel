# Item 9 — Static review of the independent archive verifier

Date: 2026-10-07 UTC. Scope: `scripts/research_verify_item9_archive.py` against the worker schema frozen in source commit `e63764ccb924ab23d65c2deb94b477387d102c82`. This reviewer read source and the three saved archive-audit summaries, checked their byte hashes and the two preserved auditor preimages, and wrote only this new document. The verifier was not executed by this reviewer. No scientific predictions, pilot scores, training outcomes, model objects or evaluation examples were opened or generated. No frozen source was modified.

**Conclusion: no remaining concrete schema or logic blocker was found in the corrected verifier for this frozen preflight archive.** Three acceptance gaps identified during review are resolved below. This conclusion concerns artifact/source/execution receipts; Stage-B admission remains pending the separate scientific recount, Stage-A provenance handoff and complete cost assessment.

## Findings and resolutions

| Finding | Original acceptance gap | Correction checked in current source |
|---|---|---|
| Complete Git coverage of raw evidence | The initial version checked every blob returned by `ls-tree` and every local manifest file separately, but did not require those path sets to coincide. A locally present ignored file could be in the manifest without being committed; a clean Git status alone does not exclude that case. | `raw_git_complete` requires a nonempty Git subtree and exact equality with all manifest paths plus `archive_manifest.json`. Local file equality, Git blob identities and SHA-256/byte counts are all checked. |
| JUnit identity, not just count | Equal case counts and unique XML names do not by themselves establish that the XML reports the collected tests. Root independently identified this gap. | `test_junit_identity` maps each frozen unittest node ID to its expected XML class/method pair and requires ordered equality. Collection, uniqueness, test-thread boundaries and absence of failure/error/skip elements remain separate checks. |
| Complete runtime boundary schema | Testing only that every present thread-environment value equals `"2"` accepted an empty or incomplete dictionary. Counting two runtime records did not identify which boundaries they represented. | The exact six keys and values are required: OMP, MKL, OpenBLAS, NumExpr, VecLib and BLIS thread controls. Record phases must be exactly `before_stage` then `after_stage_or_exception`. Torch thread counts, versions, Python version and CPU-only runtime are checked separately. |

These were holes in what the auditor would accept, not observed discrepancies in the first raw archive. The earlier passing receipt is preserved with its own code hash; it is not retrospectively attributed to the stronger verifier. No model recipe, data partition, test inventory or scientific outcome was changed by these corrections.

## Agreement with the frozen worker

The verifier uses only the standard library and Git subprocesses. It reads checkpoint files as opaque bytes for hashes; it does not unpickle them, import Torch or project scientific modules, instantiate RNGs, regenerate data or score models.

The source checks require the requested source/archival HEADs, clean checked-out trees, and ordinary file blobs. The raw archive's stopped manifest must be final and completed, identify the requested source/run, enumerate every local file and match every recorded byte count/hash. The source ZIP is checked for complete unique membership, CRC integrity and Git blob identity for every file, with byte equality against the source checkout. The copied plan and recipe are linked to the exact frozen source, and every implementation binding is verified. Before/after source-guard receipts must agree with those identities and report no source drift or unexpected files.

The execution checks match the worker's actual field names and phase inventories: `contract_tests` then `preflight` for Stage A; the other supported phase declares `contract_tests`, `train`, `final`. Stage status objects must equal their copies embedded in worker status, have zero return codes/completed states, lack supervision errors and match saved log hashes. Freeze, worker, stage and archive timestamps have a consistent order. Wall durations are compared with recorded admission budgets, including the stated bounded shutdown allowance. These checks corroborate saved receipts; they do not reconstruct execution from external instrumentation.

Sampled RAM records must be present, have no monitor failure, report the declared one-second monitor interval, and have a minimum matching their saved series and at least 8 GiB. Worker/stage admission and final observations are included. This validates the recorded samples, not a continuous memory guarantee, peak process RSS, energy use or an OS-wide CPU-utilization ceiling. The worker's final-archive reserve and network behavior are not converted into a claim that publication always completes by a fixed interval.

The preflight branch rejects the presence of the final study data/access/prediction directories. Its scope is a concrete absence check in the authenticated archive and the frozen control flow; it cannot establish adversarial secrecy or rule out every unrecorded activity outside that scope. Fixture exposure and the distinction between test-bucket candidates and the final study collection remain governed by the frozen protocol.

## Saved audit versions and evidence

All three saved audits name the same source commit above and final raw archive commit `8311c052796aeef5a52484b597cf5d9675ca315e`. They are successive checks of the same evidence, not three independent scientific replications. This reviewer read these saved results without rerunning the auditor or independently recounting predictions.

| Saved receipt | Recorded result | Auditor SHA-256 |
|---|---|---|
| [preflight_archive_audit_01.json](../../results/research/09_validation/preflight_archive_audit_01.json) | Verified; 859 checks; zero issues | `5dbed8a643d578667de234e18df3fa16fd5adf380322c778e9fcf3aac9e32ece` |
| [preflight_archive_audit_02.json](../../results/research/09_validation/preflight_archive_audit_02.json) | Verified; 861 checks; zero issues | `db383542069ed66570acb2589129bc828e2cc8060ac2fc50c02ebc0822b883db` |
| [preflight_archive_audit_03.json](../../results/research/09_validation/preflight_archive_audit_03.json) | Verified; 861 checks; zero issues | `2e83c184304ebaf15e9a7f13c1a081af9ca9551864fabca4dbd515af22ba5352` |

The latest receipt records 71 raw files excluding the manifest, 312 source ZIP files, 49 implementation-binding checks and the two preflight stages. It records **34 collected test cases** and successful identity matching to the XML cases. The XML suite counter is separately recorded as **137**; it must not be presented as 137 distinct collected test methods. The reviewer did not combine checks or test counts across audit attempts.

The latest receipt's sampled worker RAM minimum is 14.200397491455078 GiB and its worker wall duration before final archive is 327.752899868 seconds. These are read-only receipt fields, not newly measured quantities, isolated training timings or a Stage-B cost projection. No competence or scientific-performance inference is drawn from them.

Exact identities checked during this review:

| File | Bytes | SHA-256 |
|---|---:|---|
| Current `scripts/research_verify_item9_archive.py` | 14781 | `2e83c184304ebaf15e9a7f13c1a081af9ca9551864fabca4dbd515af22ba5352` |
| Preserved `archive_verifier_attempt01.py` | 13755 | `5dbed8a643d578667de234e18df3fa16fd5adf380322c778e9fcf3aac9e32ece` |
| Preserved `archive_verifier_attempt02.py` | 14479 | `db383542069ed66570acb2589129bc828e2cc8060ac2fc50c02ebc0822b883db` |

The three audit-receipt SHA-256 values are, respectively, `62ae849232a51a67ca39c0b273b8b6e3772d4499b1aaff40640f4b8d6c10f9e7`, `fb900e72724345540ea233b774458f681879331975483ae2e677b4edca9bfc87`, and `2113accdce1cf74e01b638b03c865d5a64f4386ff7b2c48fdc2b1a758ce61d2a`. Preserved source hashes match the corresponding receipt's auditor field. The frozen worker hash remains `a9012e556337c51d49a1fa95db60f166d6dec78669440fdca1a084f90cb190d5`.

## Remaining scope for admission

The archive verifier intentionally does not establish pilot ranking from logits, memorization criterion status, exact minibatch pairing, data/group semantics, checkpoint deserialization validity, counterfactual metrics or primary statistical conclusions. Those checks belong to the independent scientific audit and subsequent analysis. It also does not create or validate the future Stage-B receipt against a remembered rate. The [admission review](09_stage_b_admission_review.md) remains the design-level checklist for that separate decision.

The current verifier's support for a `study` phase is not an audit of study results that do not yet exist. Actual Stage-B and final-access chronology, ten-checkpoint completeness and all final arrays require their own stopped archive and scientific recount. Nothing in these 861 preflight checks opens the next programme item or restores historical H1 blindness.
