"""Small, standard-library artifact gates for the item-9 research study.

These gates prevent accidental incomplete, changed, or repeated evaluation in
the reviewed controller. They do not provide adversarial security or external
blinding. No numerical library, task generator, model, or RNG is imported.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


FAMILIES = ("neuropixel", "relative_transformer")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_new_json(path, value):
    """Write exclusively and flush before a dependent phase can proceed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != data:
        raise OSError("exclusive JSON persistence verification failed")
    return sha256(path)


def artifact(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def verify_artifact(record, root):
    """Verify exactly the expected local file under the declared output root."""
    if set(record) != {"path", "sha256", "bytes"}:
        raise ValueError("artifact record fields differ")
    root = Path(root).resolve()
    path = (root / record["path"]).resolve()
    path.relative_to(root)
    if (not path.is_file() or type(record["bytes"]) is not int
            or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]):
        raise ValueError(f"artifact identity differs: {record['path']}")
    return path


def expected_primary(recipe, selected_rates):
    if set(selected_rates) != set(FAMILIES):
        raise ValueError("both and only declared families require selected rates")
    rates = recipe["training"]["pilot_learning_rates"]
    for rate in selected_rates.values():
        if type(rate) not in (float, int) or not math.isfinite(rate) or rate not in rates:
            raise ValueError("selected rate is not a finite declared pilot rate")
    seeds = recipe["training"]["primary_initializations"]
    if seeds != [40, 41, 42, 43, 44]:
        raise ValueError("the five-initialization item-9 inventory differs")
    return [{"run_id": f"{family}_seed{seed}", "family": family,
             "initialization": seed, "learning_rate": float(selected_rates[family]),
             "updates": recipe["training"]["primary_updates"]}
            for seed in seeds for family in FAMILIES]


def select_pilot(records, recipe):
    """Select both rates only from a complete finite four-run pilot inventory."""
    expected = {(family, float(rate)) for family in FAMILIES
                for rate in recipe["training"]["pilot_learning_rates"]}
    observed = [(row["family"], row["learning_rate"]) for row in records]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("pilot inventory is duplicate, incomplete or unexpected")
    if len({canonical_bytes(row["context"]) for row in records}) != 1:
        raise ValueError("pilot source or dataset contexts differ")
    for row in records:
        if (row["status"] != "completed"
                or row["phase"] != "pilot"
                or row["initialization"] != recipe["training"]["pilot_initialization"]
                or row["updates_completed"] != recipe["training"]["pilot_updates"]):
            raise ValueError("pilot run is incomplete or uses another recipe")
        score, ce = row["validation"]["binding"], row["validation"]["cross_entropy"]
        if not (math.isfinite(score) and 0 <= score <= 1 and math.isfinite(ce) and ce >= 0):
            raise ValueError("pilot selection metrics must be finite and valid")
    chosen = {}
    for family in FAMILIES:
        rows = [row for row in records if row["family"] == family]
        winner = min(rows, key=lambda row: (-row["validation"]["binding"],
                                            row["validation"]["cross_entropy"],
                                            row["learning_rate"]))
        chosen[family] = float(winner["learning_rate"])
    return chosen


def validate_pilot_evidence(evidence, recipe, expected_recipe_sha256):
    """Authenticate copied Stage-A records and recompute the frozen selection.

The receipt file itself is hash-bound by the Stage-B plan. A separate read-only
archive audit must establish its claimed Git provenance before that plan freezes.
This function ensures that copying the wrong winning rate cannot pass silently.
"""
    if (evidence.get("schema_version") != 1 or evidence.get("item") != 9
            or evidence.get("recipe_sha256") != expected_recipe_sha256):
        raise ValueError("pilot evidence schema or recipe differs")
    selection, runs = evidence["selection"], evidence["preflight_runs"]
    for name, value in (("selection", selection), ("preflight_runs", runs)):
        if hashlib.sha256(canonical_bytes(value)).hexdigest() != evidence[f"{name}_sha256"]:
            raise ValueError(f"copied pilot artifact identity differs: {name}")
    if (selection["context"]["source_commit"] != evidence["preflight_source_commit"]
            or selection["context"]["recipe_sha256"] != expected_recipe_sha256
            or selection["final_performance_data_generated"] is not False
            or selection["rule"] != recipe["training"]["lr_selection"]):
        raise ValueError("pilot selection context, rule or final exposure differs")
    if (selection["pilot_records"]["sha256"] != evidence["preflight_runs_sha256"]
            or selection["pilot_records"]["bytes"] != len(canonical_bytes(runs))
            or selection["pilot_runs"] != [row["run_id"] for row in runs["pilots"]]):
        raise ValueError("selection does not reference the copied complete pilot records")
    if (any(row["context"] != selection["context"] for row in runs["pilots"])
            or selection["development_identities"] != selection["context"]["development_identities"]):
        raise ValueError("pilot dataset or source context differs")
    chosen = select_pilot(runs["pilots"], recipe)
    if selection["selected_learning_rates"] != chosen:
        raise ValueError("recorded pilot winners differ from recomputed selection")
    return chosen, selection["development_identities"]


def create_final_inventory(output, recipe, selected_rates, context):
    """Bind every declared trained checkpoint before any final-data access."""
    output = Path(output)
    if (output / "final_access.json").exists() or (output / "final_data").exists():
        raise ValueError("final data or access already exists")
    expected = expected_primary(recipe, selected_rates)
    index = read_json(output / "primary_runs.json")
    if index["status"] != "completed" or index["runs"] != [row["run_id"] for row in expected]:
        raise ValueError("primary run inventory is incomplete or reordered")
    entries = []
    for spec in expected:
        summary_path = output / "runs" / spec["run_id"] / "run.json"
        row = read_json(summary_path)
        for key in ("run_id", "family", "initialization", "learning_rate"):
            if row[key] != spec[key]:
                raise ValueError(f"primary configuration differs: {spec['run_id']} {key}")
        if (row["status"] != "completed" or row["updates_completed"] != spec["updates"]
                or row["context"] != context or row["phase"] != "primary"):
            raise ValueError("primary source, data or completion status differs")
        for name in ("checkpoint", "training_log", "probe_predictions", "validation_predictions"):
            verify_artifact(row["artifacts"][name], output)
        for name in ("probe", "validation"):
            score = row[name]
            if not all(math.isfinite(score[k]) for k in ("binding", "cross_entropy", "global")):
                raise ValueError("nonfinite primary endpoint")
        entries.append({**spec, "summary": artifact(summary_path, output),
                        "checkpoint": row["artifacts"]["checkpoint"]})
    gate = {"schema_version": 1, "item": 9, "created_at_utc": utc_now(),
            "context": context, "selected_learning_rates": selected_rates,
            "entries": entries, "final_data_generated": False,
            "primary_index": artifact(output / "primary_runs.json", output),
            "final_recipe": recipe["data"],
            "scope": "All ten checkpoints; local accidental-access guard, not external blinding."}
    write_new_json(output / "final_inventory.json", gate)
    return gate


def claim_final_access(output, recipe, selected_rates, context):
    """Consume access before sampling; failures after consumption cannot retry."""
    output = Path(output)
    if (output / "final_data").exists():
        raise ValueError("final data already exists")
    gate_path = output / "final_inventory.json"
    gate = read_json(gate_path)
    receipt = read_json(output / "gate_archive_receipt.json")
    if (gate["context"] != context or gate["selected_learning_rates"] != selected_rates
            or gate["final_recipe"] != recipe["data"] or gate["final_data_generated"] is not False):
        raise ValueError("frozen final inventory context differs")
    expected = expected_primary(recipe, selected_rates)
    if len(gate["entries"]) != len(expected):
        raise ValueError("final inventory has the wrong number of entries")
    for entry, spec in zip(gate["entries"], expected):
        if any(entry[key] != value for key, value in spec.items()):
            raise ValueError("final inventory entry differs from the declared inventory")
        summary = read_json(verify_artifact(entry["summary"], output))
        verify_artifact(entry["checkpoint"], output)
        for name in ("checkpoint", "training_log", "probe_predictions", "validation_predictions"):
            verify_artifact(summary["artifacts"][name], output)
    verify_artifact(gate["primary_index"], output)
    if (receipt.get("schema_version") != 1 or receipt.get("item") != 9
            or receipt.get("source_commit") != context["source_commit"]
            or receipt.get("final_inventory_sha256") != sha256(gate_path)
            or receipt.get("results_branch") != "research/scientific-validation-2026-10-07-cloud-results"
            or not isinstance(receipt.get("archive_commit"), str)
            or len(receipt["archive_commit"]) != 40
            or any(c not in "0123456789abcdef" for c in receipt["archive_commit"])):
        raise ValueError("remote archive receipt does not bind this final inventory")
    record = {"schema_version": 1, "item": 9, "claimed_at_utc": utc_now(),
              "final_inventory_sha256": sha256(gate_path),
              "archive_receipt": artifact(output / "gate_archive_receipt.json", output),
              "source_commit": context["source_commit"],
              "consumed_before_final_generation": True,
              "retry_allowed": False}
    write_new_json(output / "final_access.json", record)
    return gate
