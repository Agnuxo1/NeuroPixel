# Item 10 acquisition: independent static review

Review date: 2026-10-07 UTC. Scope: the prospective acquisition phase only, before its execution. This review read source text through the GitHub blob API and compared interfaces. It did not compile or execute Python, run contract tests, download a dataset archive, inspect TEST examples, or certify a future training/final-scoring controller.

## Reviewed identities

| Role | Git blob | Bytes | SHA-256 of UTF-8 source |
|---|---|---:|---|
| Worker v1, retained failed design | `6e73d710141cb63dbb2255957a5480c78ad5058f` | 18463 | `b9564f63a7c959d37b74d49bf0b23467a607eb7597e5801acec7b7ee8b6c2f83` |
| Worker v2 | `31d26aea4211bbd1a867a91ed0a6b2044e2b05ea` | 20489 | `4f18244819f5c7a2226e6f5fe5d0d1446926dc0cd30f5cbab9f82aaafceb2f72` |
| Worker with context/resource fixes | `604275ff000e95d75aafc9d7fa97f02cf2f857e4` | 21190 | `6256f5f90cfa945c05829e23c30f4f24b05b0d92f0d7f4ac8830feac0e184537` |
| Acquisition workflow | `4e7301755828cc8e31c768af86219b257ecf0057` | 2951 | `fe6de5fabec311d88b809ee3f9f32310cb763b367bba8c35d8e84d40073d34df` |
| Acquisition CLI candidate | `5836f5a49060dd8d07b490120df28f08e1b43406` | 14303 | `c56f4e8324762193ac4afe3ed4d6611fed1e9b10b652d3cf8734bf7fc311ab77` |
| Raw-text QA module | `73170979b2b50282ab04648e231f1bff641f1235` | 25707 | `bea3b511b63a8107ae8eb6b6e0bf2d12cf9f76dd80083aae2aa967e42fe7c649` |

The reused item-9 transport/supervisor was read at item-10 opening source `a88276ec0038914f078fc51b1612d997677b6321`; it is not changed by this review.

## Resolved findings

1. Worker v1 did not enforce a phase-specific stage sequence. A malformed study plan could therefore reach a final stage without the claimed preceding training/archive gate. Worker v2 introduces exact phase inventories and requires completed contract-test/training stages plus the matching remote archive commit and training-manifest hash before a final child starts. This preserves the original defect as a source-review finding; it is not evidence that any invalid study ran. The acquisition sequence is exactly `contract_tests, acquire`.
2. The current worker restores the isolated Actions repository/branch and numeric run/attempt guards. It also records a final available-RAM observation and fails if that observation is below 8 GiB. Child supervision samples at approximately 1 Hz; archival work is not continuously monitored by that child supervisor.
3. The current workflow declares one numerical thread across all six thread variables, disables third-party pytest plugin autoloading, fixes UTF-8/unbuffered output, and leaves CUDA devices empty. Python 3.12.14, psutil 7.2.2 and pytest 9.1.1 are declared. These are source-level observations, not observed installation or runtime success.
4. The worker budget is 1,200 seconds total, with 180 seconds reserved for final archival and at most 1,020 seconds for execution before that reserve. The workflow declares 25 minutes and a 1,470-second absolute deadline from its initial step. Stage limits and the shared worker deadline are both applied; it would be incorrect to describe this as 1,200 execution seconds plus another 180.
5. Acquisition and module interfaces agree: the diagnostic development split accepts `allow_conflicts=True`, records raw-input label conflicts, and fitting is stopped if such conflicts exist. Optimization-TRAIN alone supplies vocabulary, canvas geometry and fitted shortcut controls. Head-vocabulary support is reported separately from whether a gold token occurred as an optimization-TRAIN label. The encoder accepts only visible facts/question; answer and support metadata remain outside that API.

## Findings still requiring a revised acquisition candidate

These findings apply specifically to CLI blob `5836f5a49060dd8d07b490120df28f08e1b43406`.

- `urlopen` follows redirects but preserves only the initial entry and final URL. It does not retain the intermediate redirect chain or enforce a declared redirect-admission policy. The two initial source URLs and selected TRAIN member are also trusted from the plan rather than explicitly admitted by the loader. The source record promised by this acquisition should name and retain those decisions before network access.
- The primary/mirror comparison is computed immediately before an assertion, then saved only after that assertion. If TRAIN payloads disagree, individual archive/member hashes and payloads survive, but the explicit comparison receipt is missing. Save the comparison before rejecting the disagreement.
- `train_audit_summary.json` is written with `status=completed` before the zero TRAIN/DEV-intersection assertion. An assertion failure would leave a completed summary beside a failed worker. Validate that guard before writing the completed summary, or preserve an explicit failed summary.

The per-download elapsed-time check occurs between reads; a blocking operation may additionally consume its socket timeout. The enclosing stage supervisor still supplies a bounded process deadline. Do not interpret the per-download field as an exact hard interruption time.

## Scope and limits of the review

The selected TAR member is returned through `extractfile` only after exact-name and regular-file/size checks. No `extract` or `extractall` writes archive paths into the filesystem. Traversing a compressed TAR can internally decompress or skip other member bytes; the defensible statement is that no TEST payload is extracted, decoded, tokenized, passed to a model or scored in this acquisition.

The raw archive and its public TEST entries share a download. Workflow gating is not independent blind custody. A documented mirror without an independently authenticated upstream checksum does not establish original-host byte identity. Full-archive and TRAIN-member comparisons should be reported separately if both sources succeed.

No obvious Python syntax defect was found by reading, but compilation and the frozen cloud contract tests remain required evidence. Current source checks do not constitute admission of a future neural study or a checkpoint-inventory/scientific gate. No new performance result, benchmark overlap count or model claim is made.
