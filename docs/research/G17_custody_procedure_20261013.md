# Commit-reveal custody for any new TEST (G17.1, 13 October 2026)

No new TEST may be opened until this procedure has run with an independent evaluator. It was rehearsed end to end with dummy material (`results/research/G17_REHEARSAL_20261013/rehearsal_receipt.json`, 9/9 steps passed); the rehearsal used no real model or data and is not custody.

1. **Evaluator commits the TEST specification.** The evaluator writes a JSON specification (generator `scripts/completion_g3_splits.py:make_family_test` at a pinned commit, a secret seed, the panel sizes and the SHA-256 of `results/research/NEWREL20261013/excluded_families.json`), runs `python -m completion_kit.commitments commit`, publishes the public record and keeps the opening.
2. **Team seals the cohort.** After step 1 is public, the team runs `commitments digest` over the directory of all selected models (DEV selector of G3) and publishes the commitment of that digest. Each publication needs external proof of time ordered after step 1 and before step 3.
3. **Reveal and verify.** The evaluator publishes the opening; anyone runs `commitments verify`, regenerates TEST with `make_family_test` from the revealed specification (deterministic; it rejects every development family) and scores only models whose current `directory_digest` equals the committed digest.
4. **Exposure log.** The access is recorded in `results/research/COMPLETION20261012/G3_split_registry.json`; the revealed TEST is never used for selection.

Rehearsal checks that must hold in a real run: openings verify (exit 0); a changed seed or salt fails (exit 1); TEST regeneration is byte-identical; a cohort file changed after sealing changes the digest and is refused.

External dependencies (owner approval required): designating an evaluator who is not an author, and an external time-stamping route (for example an OpenTimestamps proof or a dated public deposit). Claimed times inside the records are not proof of time. The evaluator package is `results/research/COMPLETION20261012/G17_evaluator_package.json`.
