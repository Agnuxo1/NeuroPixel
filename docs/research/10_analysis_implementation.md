# Item 10 independent saved-artifact audit contract

This is a prospective implementation note. The auditor and extractor have been authored and read as source, but have not been compiled or executed in this authoring environment. No official TEST payload or neural outcome was read while preparing them.

## Immutable review candidates

- scripts/research_babi_audit.py: blob 99853ff5c3556cefa154ceaa71e26d31ded59f53, 82,547 UTF-8 bytes.
- scripts/research_babi_fixture_ledger.py: blob 76e42c59b90414869ae4e81ed656faeb430f9977, 4,987 UTF-8 bytes.
- Scientific controller schema: scripts/research_babi_study.py blob 410a7c491f81f704598053e19e5d37890315f0ce. Scientific phase receipts use study_{phase}_environment.json and study_{phase}_status.json; worker receipts retain their separate namespace.
- The first auditor draft ec3c77d5ad988e1f7922d7a6e25e848f7ed224d0 and namespace-corrected b378bfa79fd23c63384dffdd5bc5738745d5bcb6 remain immutable preimages. The final listed candidate additionally permits a scientifically negative, complete preflight to be verified. These are source revisions, not reruns or test attempts.

## CLI and input roots

Run the independent auditor in a fresh process:

    python scripts/research_babi_audit.py --phase preflight --input PREFLIGHT_RUN_ROOT --development ACQUISITION_DATA --plan FROZEN_PREFLIGHT_PLAN --source-root FROZEN_SOURCE --output-dir NEW_AUDIT_DIRECTORY

For the later study:

    python scripts/research_babi_audit.py --phase study --input STUDY_RUN_ROOT --development ACQUISITION_DATA --preflight PREFLIGHT_RUN_ROOT --plan FROZEN_STUDY_PLAN --source-root FROZEN_SOURCE --output-dir NEW_AUDIT_DIRECTORY

All input roots are read-only. The output must be new and outside those roots. The auditor performs no network recovery and never imports project modules, Torch, a model or pickle. Checkpoint bytes are hashed without deserializing them. Operational recovery, remote ancestry, source ZIP verification and Git publication remain the worker/archive auditor's responsibilities.

The fixture exporter is a separate stdlib-only command:

    python scripts/research_babi_fixture_ledger.py --source-root FROZEN_SOURCE --test-sha256 EXPECTED_TEST_SHA256 --module-sha256 EXPECTED_DATA_MODULE_SHA256 --output NEW_FIXTURE_RECEIPT_JSON

It authenticates the test and data-module bytes, loads the reviewed test module's deterministic definitions, and exports DEVELOPMENT_FIXTURES without invoking any test method. Its receipt records raw and normalized context identities, the exact fixture sequence hash and coverage limits. Valid raw overflow fixtures remain in raw exposure; later encoding failures must be counted separately, without truncation. The eight controller tests reexport the same ledger and introduce no new raw contexts.

## Runtime binding and resources

The scientific plan must carry exactly:

    "analysis_runtime": {
      "python": "3.12.14",
      "numpy": "2.3.5",
      "scipy": "1.17.0",
      "threads": 1,
      "bootstrap_generator": "PCG64",
      "bootstrap_seed": 104001,
      "bootstrap_replicates": 2000
    }

This is separate from the training runtime, whose Python is also 3.12.14 but whose scientific packages remain the declared CPU Torch 2.6.0, NumPy 2.2.6 and SciPy 1.15.1 stack. Actual saved training/inference versions are compared with the plan.

The auditor sets numerical library thread environment variables to one before importing NumPy/SciPy, requires little-endian integer storage for stream-content digests, and requires MemAvailable at least 8 GiB before numerical imports and at periodic array/bootstrap boundaries. These are sampled admission checks, not a continuous resource trace or a peak-RSS measurement.

## Recount scope

The auditor independently parses the saved original TRAIN and TEST text. It validates original positive contiguous line IDs, prior-statement support metadata, raw-line/source identities and every QA's ordered visible prefix, gold and episode identity. Earlier QA answers and future statements are never encoded. The official gold is not replaced by a symbolic control.

It reconstructs connected TRAIN/DEV components using episode identity, identical complete normalized fact stories, and identical normalized visible inputs. It checks the declared 10% DEV salt, complete/disjoint assignment and the actual saved diagnostic allow_conflicts flag. Raw contradictory TRAIN labels block neural admission rather than being dropped. The TRAIN-only vocabulary, row layout, punctuation, OOV mapping and every saved canvas are reconstructed independently.

Every retained final record is checked, including unencodable inputs and unsupported output labels. The three masks are all_official, novel_vs_optimization_train and strict_novel_vs_full_exposure. Novelty requires both unseen normalized raw input and unseen frozen encoded input; an unencodable raw-novel row remains in its applicable subset and scores wrong. The full exposure set includes all official TRAIN, assigned DEV and the bound fixture ledger. Separate raw/encoded overlap counts clarify the conjunction. These are exact-input novelty tests, not unseen grammar, entities, relations or generator structure.

The eight control predictions and their denominators are recomputed from raw inputs and TRAIN-fitted categories. They include majority, ordered exact memory, question-only memory, unordered BOW memory, multinomial BOW Naive Bayes, fact-answer frequency, a separately declared query-exclusion frequency variant, and a literal direct-relation symbolic control. Raw controls may remain applicable where the neural grid or vocabulary fails; that difference is reported explicitly.

## Neural arrays, supervision and gates

The expected NPZ keys are logits, prediction_id, gold_id, can_encode, gold_supported, nll_supported, nll and record_id. The auditor checks exact row order, float32 logits, int64 IDs, boolean masks, finite values, first-index argmax and float64 NLL. Unsupported output labels and grid-overflow rows remain wrong in the full accuracy denominator.

Gold support is membership in the full nonreserved vocabulary. NLL eligibility is gold_supported AND can_encode, with a separate CE denominator. Unencodable records must retain zero logits, prediction ID -1 and no implied inference; ineligible saved NLL entries must be zero placeholders. Their reported loss is undefined, not zero loss.

NLL is recomputed using stable float64 logsumexp with the saved float32 logits treated as exact inputs. The predeclared tolerance is absolute 1e-10 plus relative 1e-12; no float32-scale arithmetic allowance is used for this double-precision calculation. Metrics retain full precision. Duplicate input prediction/logit agreement is described from actual arrays; predictions are not repaired or averaged.

All six preflight runs and all ten primary runs must be complete and match the exact expected recipes. The TF feed-forward width is 144 in the newly declared factory; historical model code remains unchanged. Checkpoint/config/log/array hashes, planned and consumed index hashes, unique sampled memberships, and matched-family input streams are checked. Stream matching does not claim that the declared PRNG seed was independently regenerated.

Every already-computed 32-row memorization checkpoint is recounted, including both checks establishing a passing streak. A complete negative memorization result must use the full 4096-update budget and must not have crossed the two-pass stop criterion earlier. A six-run negative preflight can have audit status verified with admission_passed false. Only the later study requires a passed preflight. Incomplete or corrupted runs fail the audit; no rows or runs are silently discarded.

The study audit verifies the ten-checkpoint training inventory and source/config/data/selection bindings, then verifies the local archive-gate and consumed-access receipts and their chronology before reading final populations or prediction arrays. A receipt and hashes alone do not independently prove remote publication; that evidence belongs to the separate operational archive audit.

## Statistical units and interpretation

The primary interval uses exactly five complete paired training realizations, seeds 60 through 64. For each seed d = accuracy_NeuroPixel - accuracy_RelativeTransformer on all official TEST rows. The report retains all five pairs, mean difference, sample SD with denominator four, range and mean plus/minus t(0.975,4) times SD/sqrt(5), without clipping or pre-rounding.

The descriptive NeuroPixel competence screen requires accuracy at least 0.95 in every seed on all official rows and on the strict-novel subset. If the strict subset is empty, that part is null and competence is not established. The same raw per-seed values are shown for the reference family. This screen is not a significance test, a mechanism attribution, or a replacement for historical H1.

The separate bootstrap uses 2000 PCG64 draws with seed 104001 over connected TEST components defined by the same episode/full-story/input identities. A single 2000-by-G int64 index matrix is shared across all ten checkpoints, all three subsets and all five signed paired differences. Each replicate divides pooled resampled component correct counts by pooled selected-row counts, retaining unequal component sizes.

With fewer than two components, the bootstrap is unavailable. For a particular metric/subset, any zero-denominator replicate makes its interval null; the undefined-replicate count is preserved, and other endpoints remain valid. Defined intervals use NumPy linear 2.5/97.5 percentiles. Saved draws and indices allow an external recount. This is conditional on the fixed checkpoints and observed source components; it does not treat 1000 rows as independent, estimate between-training uncertainty, or provide multiplicity-adjusted evidence.

The study outputs scientific_audit.json, bootstrap_cluster_indices.npy and bootstrap_accuracy_draws.npz. Undefined scalar draws use NaN only in the numeric NPZ; JSON intervals are null. Preflight produces scientific_audit.json without bootstrap draws. All outputs refuse overwrite; a failed audit preserves an explicit issue receipt.

Timing and cost fields are reported separately. They are saved measurements of update/training/total wall time and formula-checked parameter counts, not exclusive CPU work, peak memory, energy or continuous resource monitoring.

## Pre-freeze source-receipt correction

Independent static review found that the training and final processes produce distinct recorded_at_utc values in otherwise identical scientific source records. Candidate a8472480f6e3ab05b484749ebad2cd4bf17b1f34 incorrectly required those entire cross-phase objects to match. Its source is preserved as a preimage; no audit or neural run was executed from this candidate before correction. The current auditor compares stable source_commit, plan_sha256, implementation_sha256 and tracked_source_clean fields across phases, and binds final_access.json and final_metrics.json exactly to the final phase's source object in study_final_status.json. Both source timestamps remain separately reported; existing inventory/gate/access chronology is unchanged. No scientific recipe, selection rule, dataset or statistical criterion changed.
