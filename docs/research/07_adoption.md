# Item 7 — Adoption contract for future studies

This is prospective implementation guidance, not authorization to start item 8. It describes `neuropixel/research/split_governance.py` and `development_data.py` as frozen at commit **`0ed43bd5bb1d8bcb166b8e63e909195cbe69f769`**, read alongside their misuse tests and [the historical source audit](07_gate_source_audit.md). Historical samplers, studies and exposed populations retain their original provenance. The new API is opt-in; it is not automatically used by old training drivers.

## Adoption checklist

1. **Preregister before development feedback.** Archive the intended candidate IDs, data/group definition, source inventory, metric definitions, denominators, fixed budget and deterministic selection rule in a separately preserved plan. `FinalStudy.freeze()` receives completed validation metrics; it cannot prove that the inventory or rule was fixed before those metrics were seen.
2. **Construct a complete dataset manifest.** Use `make_manifest(dataset_id=..., specification=..., partitions=..., history=...)`. Supply the fields below from a documented preparation process. Do not create a nominally fresh dataset ID to hide previous exposure. Define the semantic groups and assign them to partitions first. Fit any data-dependent preprocessing only on the assigned training partition, and keep all generated or augmented descendants in their source group and partition.
3. **Give development only train/validation access.** Use `DevelopmentRoleTask(h=8, w=8, seed=split_seed)` and an explicit CPU `torch.Generator` for `sample()`. Exact accepted names are `"train"` and `"validation"`; `frozen_dataset(task, split, n, seed, device="cpu")` respects that restriction. `n` must be a positive multiple of four. `sample_loop` and `sample_camera` are unsupported for every split. Keep final paths, final loaders and unrestricted historical samplers out of development entry points. This is a code-organization rule, not an operating-system capability boundary: the inherited object still contains public test membership, and arbitrary code can bypass the adapter.
4. **Complete the whole candidate inventory on one validation population.** Supply exactly one completed record per preregistered ID, with finite metrics and matching validation artifact hash and role denominators. The API requires every candidate to be completed; it does not implement dropping failed candidates or choosing a reduced inventory after seeing results. Preserve a failure and resolve any changed study prospectively.
5. **Freeze, then pin the receipt externally.** `FinalStudy(workspace, directory).freeze(...)` creates `freeze.json` exclusively, binds all candidate configurations/checkpoints plus explicitly listed source files, and selects using the fixed rule below. Hash the actual receipt bytes with `file_digest()`, then commit or otherwise independently preserve that hash outside the mutable study directory before any final access. Source inventory completeness is the caller's responsibility; include the preparation, metric, model, controller and callback implementations and their dependencies as appropriate.
6. **Enter final evaluation once.** Supply the previously preserved hash to `evaluate(expected_freeze_sha256=..., final_artifact=..., callback=...)`; never recompute an expected hash from the current receipt at this point. The final path must be relative to `workspace`. The callback receives verified bytes and a detached freeze receipt, evaluates the selected frozen candidate under the preregistered metric contract, and returns a finite JSON dictionary. Do not resample final data or reopen the file in the callback.
7. **Preserve all outcomes and exposure.** `final_access.json` is written exclusively and flushed before final path resolution, size inspection or reading. Bad paths/bytes, loader or evaluator failures, and interrupts after this write consume access. Preserve `final_failure.json` or `final_result.json`; do not delete receipts or change directories to retry silently. Carry resulting exposure into subsequent manifests. The current API selects one candidate; multiple final cases require a separately preregistered controller/inventory rather than repeated calls to this one-use method.

## Manifest and selection fields

All hashes below are lowercase SHA-256 strings. Object schemas reject missing or unexpected fields. Paths for configurations, checkpoints and sources are relative to the declared workspace and cannot resolve outside it.

| Object | Required contents and checks |
|---|---|
| `specification` | `generator_id`, `generator_source_sha256`, `vocabulary_sha256`, `shape` (positive integer dimensions), `generation_parameters` (dictionary), `sampling_seed` (integer), `rng_runtime`, `grouping_rule`, `grouping_source_sha256`, `transformations` (list), `preprocessing_fit_partitions` (exactly `[]` or `["train"]`). Record relevant pool membership, split seed, sampling recipe and transform identities in the declared specification; the module does not infer them. |
| `partitions` | Exactly `train`, `validation`, `final`. Each has `artifact_sha256`, positive `artifact_bytes`, and ordered nonempty `records`. Each record has globally unique `record_id`, `group_id`, and `content_sha256` covering its label and all evaluation-relevant fields. Group or content overlap across partitions is rejected. Repeated content with distinct record IDs within one partition is retained. Record ordering contributes to identity. |
| `history` | `scope`, `coverage`, `evidence_sha256`, `exposures`. Coverage is `complete_within_declared_scope`, `partial`, or `unknown`. Each exposure has one unique `group_id` and a nonempty unique `uses` list drawn from `training_pool`, `validation_feedback`, `final_feedback`, `unknown`. The evidence hash anchors a caller-supplied history; the module does not independently authenticate its completeness. |
| `protocol` | `protocol_id`, `mode` (`exploratory` or `confirmatory`), unique nonempty `candidate_ids`, `selection_rule`, `validation_artifact_sha256`, `validation_denominators`. Selection rule must equal `list(SELECTION_RULE)`: descending binding accuracy, ascending cross entropy, ascending parameter count, then ascending candidate ID. |
| Each candidate | `candidate_id`, `status="completed"`, `configuration_path`, `configuration_sha256`, `checkpoint_path`, `checkpoint_sha256`, `validation_artifact_sha256`, `binding_accuracy`, `cross_entropy`, `parameter_count`, `denominators`. Binding lies in `[0,1]`, CE is finite and nonnegative, and parameter count is a positive integer. |
| Both denominator objects | Exactly `agent`, `patient`, `all`, each a positive integer, not a boolean. Every candidate must equal the protocol values, `all >= agent + patient`, and `all` must equal the manifest's number of validation records. |

`make_manifest()` returns detached nested values, partition hashes, exposure-derived fields and `identity_sha256`; `validate_manifest(manifest)` rebuilds and checks them. The API checks declared identities and denominators, not their semantic truth: preparation/audit code must verify that artifact rows, content hashes, roles, groups and metric calculations actually agree. It does not recompute validation predictions or binding/CE from submitted numbers. Keep the actual common validation artifact and an independent recount with the study evidence.

## Minimal call pattern

The following is an unexecuted integration template. Function arguments are real inputs supplied by future study preparation and evaluation code, not literal dummy hashes, invented results or a new runnable study. The two phases are deliberately separate. `specification`, `partitions`, `history`, `protocol` and `candidates` must contain the exact fields above; `source_paths` must name existing files.

```python
from pathlib import Path
from neuropixel.research.split_governance import (
    FinalStudy, make_manifest, file_digest,
)


def freeze_completed_development(
    workspace: Path, directory: Path, *, specification, partitions,
    history, dataset_id, protocol, candidates, source_paths,
):
    manifest = make_manifest(
        dataset_id=dataset_id, specification=specification,
        partitions=partitions, history=history,
    )
    study = FinalStudy(workspace, directory)
    receipt = study.freeze(
        manifest=manifest, protocol=protocol,
        candidates=candidates, source_paths=source_paths,
    )
    receipt_sha256 = file_digest(directory / "freeze.json")
    # Caller must preserve this hash in an external immutable execution record
    # BEFORE the separate final phase. Returning it alone is not external pinning.
    return receipt["selected_candidate_id"], receipt_sha256


def evaluate_once(
    workspace: Path, directory: Path, *, externally_pinned_freeze_sha256,
    final_relative_path: str, evaluate_selected_candidate,
):
    # evaluate_selected_candidate(payload: bytes, frozen_receipt: dict) -> dict
    # must use the selected checkpoint/configuration from the receipt and return
    # only the preregistered evaluation, without another loader or sampler call.
    return FinalStudy(workspace, directory).evaluate(
        expected_freeze_sha256=externally_pinned_freeze_sha256,
        final_artifact=final_relative_path,
        callback=evaluate_selected_candidate,
        max_artifact_bytes=256 * 1024**2,
    )
```

Pass an absolute `directory` within `workspace` consistently to both functions. Dataset preparation must already have supplied the final artifact's identity and size; the development function needs its manifest metadata, not its path or content. If a separate preparer or custodian is required by the scientific claim, establish that arrangement explicitly. This API does not provide one.

## History, chronology and custody limits

`mode="confirmatory"` is refused when declared history is partial/unknown or any listed exposed group intersects final groups, regardless of the listed use. Acceptance means **no conflict in the declared complete history**, not proof that the final population is pristine. `mode="exploratory"` permits such history while retaining the same within-study disjointness, candidate completeness, drift and one-use checks. Do not relabel existing exposed NeuroPixel pools as confirmatory merely because a new seed, layout, checkpoint or directory was used. Distinguish eligible pool membership from actual sampled examples, and distinguish both from human access to outcomes.

The externally pinned receipt protects against accidental receipt replacement only when the expected hash really comes from the prior independent record. Timestamps are local assertions; freeze-after-development does not implement prospective preregistration by itself. The callback is trusted to honor the selected-candidate and metric contract. Public Python code, visible synthetic labels, removable directories and caller-supplied history remain outside the security guarantee. These tools provide ordinary workflow checks and reviewable evidence, not blindness, tamper-proof storage, independent custody or proof of absence of unlogged access.

The implementation tests exercise overlap/identity changes, inventory and denominator misuse, nonfinite values, checkpoint/config/source drift, external hash pinning, receipt-before-callback ordering and consumed failures. Adapter tests cover exact development sampling compatibility and rejected final/inherited routes. Those checks support the stated API behavior; they do not certify the scientific claims or history of a future study.
