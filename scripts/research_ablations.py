"""Execute the prospectively frozen core item-6 inventory.

The standard-library --describe path never imports Torch or creates datasets.
The execution path requires resource admission by the operational worker, keeps
the historical trainer unchanged, and opens final data only after all 26 runs
are complete. Growth has its own separately specified controller and gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Bypass package initializers on the stdlib-only inventory/pre-admission path.
_spec = importlib.util.spec_from_file_location(
    "item6_ablation_operations", ROOT / "neuropixel/research/ablations.py")
_operations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_operations)
core_evaluation_cases = _operations.core_evaluation_cases
core_training_configs = _operations.core_training_configs
evaluation_keep_mask = _operations.evaluation_keep_mask
evaluation_view = _operations.evaluation_view
training_factory_adapter = _operations.training_factory_adapter
training_metadata = _operations.training_metadata
training_model = _operations.training_model

PILOT_SELECTION_SHA256 = "a62300cfccc8d01f9d2777042c20a2620ba0ebfbd24d1b2b0fe4a7ea3463673b"
VALIDATION_CONTENT_SHA256 = "44873b31d17df696ceef3ff7c5fd2f1878921305d19623443780a443be364e0a"
PROBE_CONTENT_SHA256 = "f74bb2ae12e99b579eeea485bf6c262805a73f3c086a94421478466fc166b564"
FINAL_CONTENT_SHA256 = "23bca94c224b195da94ab1794801fdae131007a77bd594c2167a085227e7dba2"
REQUIRED_IMPLEMENTATION = {
    "neuropixel/research/ablations.py", "scripts/research_ablations.py",
    "docs/research/06_protocol_amendment.md",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_source(path):
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source paths must be repository-relative and contained")
    resolved = (ROOT / relative).resolve()
    resolved.relative_to(ROOT)
    return resolved


def load_protocol():
    """Verify the original protocol using only standard-library operations."""
    frozen = read_json(ROOT / "docs/research/protocol_frozen.json")
    for relative, expected in frozen["files"].items():
        if sha256(relative_source(relative)) != expected:
            raise RuntimeError(f"frozen protocol changed: {relative}")
    return read_json(ROOT / "docs/research/protocol.json")


def inventory(protocol):
    runs = core_training_configs(protocol)
    return {"planned_training_runs": len(runs), "planned_runs": runs,
            "planned_evaluation_cases": 34,
            "planned_evaluations": core_evaluation_cases(runs)}


def load_execution_plan(path):
    """Require the exact prospective inventory and all new implementation hashes."""
    path = path.resolve()
    path.relative_to(ROOT)
    if (ROOT / "source_snapshot.json").exists():
        raise RuntimeError("a stale root source snapshot would misidentify current Git HEAD")
    protocol, plan = load_protocol(), read_json(path)
    if plan.get("item") != 6 or plan.get("status") != "frozen":
        raise ValueError("the root-reviewed item-6 execution plan must be frozen")
    expected = inventory(protocol)
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("the frozen execution plan differs from the approved core inventory")
    implementations = plan.get("implementation_sha256", {})
    if not REQUIRED_IMPLEMENTATION <= set(implementations):
        raise ValueError("new code and the prospective amendment must enter source provenance")
    if path.relative_to(ROOT).as_posix() in implementations:
        raise ValueError("the execution plan cannot contain its own circular hash")
    for relative, expected_hash in implementations.items():
        if sha256(relative_source(relative)) != expected_hash:
            raise RuntimeError(f"frozen implementation changed: {relative}")
    selection_path = ROOT / "results/research/04_experiment/pilot/selection.json"
    if sha256(selection_path) != PILOT_SELECTION_SHA256:
        raise RuntimeError("the historical source anchor selection changed")
    historical = read_json(selection_path)["source"]["sha256"]
    for relative, expected_hash in historical.items():
        if sha256(relative_source(relative)) != expected_hash:
            raise RuntimeError(f"a historical scientific source changed: {relative}")
    controller = dict(implementations)
    controller[path.relative_to(ROOT).as_posix()] = sha256(path)
    return protocol, plan, controller, historical


def run_ablations(output, plan_path, device, minimum_ram_gib):
    import torch
    from neuropixel.research.data import ResearchRoleTask
    from neuropixel.research.experiment import (
        dataset_artifact, environment_record, evaluate, resource_sample,
        save_arrays, save_json, source_record, train_one, utc_now,
    )

    protocol, plan, controller, historical = load_execution_plan(plan_path)
    source = source_record()
    if source["git_commit"] == "unavailable" or source["sha256"] != historical:
        raise RuntimeError("actual Git HEAD and unchanged historical sources are required")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in
           ("evaluation_gate.json", "ablation_records.json", "evaluations")):
        raise RuntimeError("final-phase evidence exists; explicit recovery is required")
    configs = []
    for raw in plan["planned_runs"]:
        config = deepcopy(raw)
        config["controller_source"] = controller
        if not re.fullmatch(r"[a-zA-Z0-9_]+", config["run_id"]):
            raise ValueError("unsafe run identifier")
        configs.append(config)
    cases = plan["planned_evaluations"]
    if any(not re.fullmatch(r"[a-zA-Z0-9_]+", case["evaluation_id"]) for case in cases):
        raise ValueError("unsafe evaluation identifier")
    manifest_path, events_path = output / "run_manifest.json", output / "events.json"
    manifest = {"schema_version": 1, "item": 6, "panel": "core",
                "source": source, "controller_source": controller,
                "planned_runs": configs, "planned_evaluations": cases,
                "planned_training_runs": 26, "planned_evaluation_cases": 34,
                "historical_sources_unchanged": True,
                "selection_policy": "none; all prespecified configurations retained"}
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise RuntimeError("existing manifest differs; preserve the previous experiment")
    else:
        save_json(manifest_path, manifest)
    events = read_json(events_path) if events_path.exists() else []
    if any(row["event"].startswith("final_") for row in events):
        raise RuntimeError("final-phase events already exist; do not silently reopen test")
    if events:
        save_json(output / f"events_before_resume_{time.time_ns()}.json", events)
    events.append({"event": "ablation_started", "at_utc": utc_now()})
    save_json(events_path, events)

    def check_source():
        if source_record() != source:
            raise RuntimeError("historical source or Git HEAD changed during execution")
        if any(sha256(relative_source(path)) != digest for path, digest in controller.items()):
            raise RuntimeError("new implementation or the frozen plan changed during execution")

    runs = []
    for config in configs:
        check_source()
        with training_factory_adapter(config):
            row = train_one(config, output / config["run_id"], device, protocol,
                            minimum_ram_gib=minimum_ram_gib)
        if row.get("status") != "completed" or row.get("final_test_accessed") is not False:
            raise RuntimeError("a run did not complete development-only training")
        if row["config"] != config or row["source"] != source:
            raise RuntimeError("training recipe or source does not match the manifest")
        if row["resolved_model"] != training_metadata(config):
            raise RuntimeError("recorded model/objective/intervention differs from the actual adapter")
        if row["parameter_count"] != row["resolved_model"]["parameters"]:
            raise RuntimeError("actual parameter count differs from resolved metadata")
        if row["validation_dataset"]["content_sha256"] != VALIDATION_CONTENT_SHA256:
            raise RuntimeError("validation examples differ from the frozen adjacent split")
        if row["training_probe_dataset"]["content_sha256"] != PROBE_CONTENT_SHA256:
            raise RuntimeError("training probe differs from the original protocol")
        runs.append(row)
        events.append({"event": "training_completed", "run": config["run_id"],
                       "at_utc": utc_now()})
        save_json(events_path, events)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    check_source()
    if len(runs) != 26 or {row["config"]["run_id"] for row in runs} != {c["run_id"] for c in configs}:
        raise RuntimeError("the complete 26-run core inventory is required before final data")
    if any(row["environment"] != runs[0]["environment"] for row in runs):
        raise RuntimeError("the core runs do not share a measured execution environment")
    artifacts = {}
    for row in runs:
        directory = output / row["config"]["run_id"]
        actual_weights = sha256(directory / "weights.pt")
        actual_predictions = sha256(directory / "validation_predictions.npz")
        if actual_weights != row["weights_sha256"] or actual_predictions != row["validation_predictions"]["sha256"]:
            raise RuntimeError("development artifacts changed before the final gate")
        artifacts[directory.name] = {
            "training_sha256": sha256(directory / "training.json"),
            "weights_sha256": actual_weights, "validation_predictions_sha256": actual_predictions,
        }
    gate = {"at_utc": utc_now(), "complete_training_runs": 26,
            "run_manifest_sha256": sha256(manifest_path), "source": source,
            "controller_source": controller, "training_artifacts": artifacts}
    save_json(output / "evaluation_gate.json", gate)
    events.append({"event": "final_evaluation_authorized", **gate})
    save_json(events_path, events)

    # First final-data construction in this controller, strictly after the complete gate.
    resource_sample(device, minimum_ram_gib)
    task = ResearchRoleTask(8, 8, seed=0)
    final, final_record = dataset_artifact(
        task, "test", protocol["data"]["test_n"], protocol["data"]["test_sample_seed"],
        output / "datasets")
    if final_record["content_sha256"] != FINAL_CONTENT_SHA256:
        raise RuntimeError("final inputs differ from the frozen adjacent split")
    events.append({"event": "final_dataset_opened", "at_utc": utc_now(),
                   "dataset_content_sha256": final_record["content_sha256"]})
    save_json(events_path, events)
    by_id = {row["config"]["run_id"]: row for row in runs}
    masks, mask_records, records = {}, {}, []
    for case in cases:
        check_source()
        resource_sample(device, minimum_ram_gib)
        training = by_id[case["run_id"]]
        config = training["config"]
        checkpoint = output / case["run_id"] / "weights.pt"
        if sha256(checkpoint) != training["weights_sha256"]:
            raise OSError("checkpoint changed before its declared evaluation case")
        keep, mask_record = None, None
        if case["damage"] is not None:
            mask_key = json.dumps(case["damage"], sort_keys=True)
            if mask_key not in masks:
                mask = evaluation_keep_mask(len(final[0]), 8, 8, case["damage"])
                mask_name = "lesion_keep_" + hashlib.sha256(mask_key.encode()).hexdigest()[:16]
                mask_record = save_arrays(output / "datasets" / f"{mask_name}.npz", keep=mask.numpy())
                mask_record.update(damage=case["damage"], shape=list(mask.shape),
                                   ordering="example-major; complete mask generated once, then sliced")
                masks[mask_key], mask_records[mask_key] = mask, mask_record
            keep, mask_record = masks[mask_key], mask_records[mask_key]
        model = training_model(config).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        view = evaluation_view(model, config, case, keep)
        metrics, predictions = evaluate(view, final, device, intervals=True)
        if keep is not None and view.offset != len(final[0]):
            raise RuntimeError("evaluation did not consume exactly one mask per example")
        directory = output / "evaluations" / case["evaluation_id"]
        prediction_record = save_arrays(directory / "final_predictions.npz", **predictions)
        row = {
            "status": "completed", "config": config, "evaluation_case": case,
            "metrics": metrics, "predictions": prediction_record,
            "weights_sha256": training["weights_sha256"],
            "parameter_count": training["parameter_count"], "source": source,
            "controller_source": controller, "training_seconds": training["training_seconds"],
            "resolved_model": training["resolved_model"], "train_probe": training["train_probe"],
            "optimization_budget_limited": training["optimization_budget_limited"],
            "lesion_keep_mask": mask_record, "evaluated_at_utc": utc_now(),
        }
        save_json(directory / "final_evaluation.json", row)
        records.append(row)
        events.append({"event": "final_evaluation_saved", "evaluation": case["evaluation_id"],
                       "at_utc": row["evaluated_at_utc"]})
        save_json(events_path, events)
        print(json.dumps({"event": "ablation_final", "evaluation": case["evaluation_id"],
                          "binding": metrics["macro_agent_patient_accuracy"],
                          "global_accuracy": metrics["accuracy"]}), flush=True)
        del model, view
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    check_source()
    events.append({"event": "ablation_completed", "at_utc": utc_now()})
    save_json(events_path, events)
    result = {
        "schema_version": 1, "item": 6, "panel": "core", "status": "completed",
        "source": source, "controller_source": controller, "environment": environment_record(device),
        "dataset": final_record, "records": records, "events": events,
        "complete_training_runs": 26, "complete_evaluation_cases": 34,
        "total_training_seconds": sum(row["training_seconds"] for row in runs),
        "statistical_analysis": "pending separate analysis of saved predictions",
        "selection": "none; this exploratory item does not reopen or rescue H1",
        "inference_latency_seconds": None, "physical_energy_joules": None,
    }
    save_json(output / "ablation_records.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Print the approved inventory without importing Torch.")
    parser.add_argument("--plan", type=Path, default=ROOT / "docs/research/06_execution_plan.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--minimum-ram-gib", type=float, default=8.0)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(inventory(load_protocol()), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for execution")
    if args.minimum_ram_gib < 8 or not 1 <= args.threads <= 4:
        raise ValueError("the unchanged reservation requires RAM >= 8 GiB and one to four CPU threads")
    # Refuse below the RAM floor before importing Torch.
    import psutil
    if psutil.virtual_memory().available / 2**30 < args.minimum_ram_gib:
        raise RuntimeError("available RAM is below the declared floor; Torch was not imported")
    load_execution_plan(args.plan)
    from neuropixel.research.experiment import configure_runtime, environment_record, resource_sample
    device = configure_runtime(args.device, args.threads)
    print(json.dumps({"event": "preflight", "resources": resource_sample(device, args.minimum_ram_gib),
                      "environment": environment_record(device)}), flush=True)
    run_ablations(args.output, args.plan, device, args.minimum_ram_gib)


if __name__ == "__main__":
    main()
