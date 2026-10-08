"""Execute the frozen item-5 replication panels without further model selection."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from neuropixel.research.data import ResearchRoleTask
from neuropixel.research.models import model_config
from neuropixel.research.experiment import (
    configure_runtime, dataset_artifact, environment_record, evaluate, file_sha256,
    model_from_config, resource_sample, save_arrays, save_json, source_record,
    train_one, utc_now,
)
from research_train import load_protocol, run_name

PILOT_SELECTION_SHA256 = "a62300cfccc8d01f9d2777042c20a2620ba0ebfbd24d1b2b0fe4a7ea3463673b"
ALLOWED_CONFIG_CHANGES = {"phase", "init_seed", "split_seed", "controller_source"}


def controller_source():
    paths = ("scripts/research_replicate.py", "scripts/research_stage_worker.py",
             "docs/research/05_execution_plan.json")
    return {path: file_sha256(ROOT / path) for path in paths}


def load_selection(path):
    if file_sha256(path) != PILOT_SELECTION_SHA256:
        raise ValueError("Replication requires the exact selection frozen in item 4")
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    current = source_record()
    if current["sha256"] != selection["source"]["sha256"]:
        raise ValueError("A frozen model, dataset, training or pilot source file changed")
    return selection


def replication_configs(protocol, selection, controller):
    """Construct the set union of both panels; run the overlap only once."""
    families = protocol["training"]["families"]
    if set(selection["selected_configs"]) != set(families):
        raise ValueError("Pilot selection family inventory differs from the protocol")
    initializations = protocol["replication"]["initialization_panel"]
    splits = protocol["replication"]["split_panel"]
    pairs = [(seed, initializations["split_seed"]) for seed in initializations["init_seeds"]]
    pairs += [(splits["init_seed"], split) for split in splits["split_seeds"]
              if (splits["init_seed"], split) not in pairs]
    configs = []
    for seed, split in pairs:
        for family in families:
            base = selection["selected_configs"][family]
            config = copy.deepcopy(base)
            config.update(phase="item5_replication", init_seed=seed, split_seed=split,
                          controller_source=controller)
            if ({key: value for key, value in config.items() if key not in ALLOWED_CONFIG_CHANGES}
                    != {key: value for key, value in base.items() if key not in ALLOWED_CONFIG_CHANGES}):
                raise ValueError("A scientific setting changed beyond initialization/split")
            configs.append(config)
    if len(configs) != 28 or len({run_name(config) for config in configs}) != 28:
        raise ValueError("The frozen replication inventory must contain 28 unique runs")
    return configs


def run_replications(output, selection_path, device, protocol, minimum_ram_gib):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation_gate.json").exists() or (output / "replication_records.json").exists():
        raise RuntimeError("Final-phase evidence exists; explicit recovery is required")
    selection = load_selection(selection_path)
    source, controller = source_record(), controller_source()
    configs = replication_configs(protocol, selection, controller)
    execution_plan = json.loads((ROOT / "docs/research/05_execution_plan.json").read_text())
    for family, base in selection["selected_configs"].items():
        resolved = model_config(family, vocab=base["vocab"], h=base["height"], w=base["width"],
                                steps=base["steps"], **base["variant"])
        if resolved != execution_plan["resolved_models"][family]:
            raise RuntimeError("A resolved model setting differs from the frozen pilot")
    manifest_path = output / "run_manifest.json"
    manifest = {"item": 5, "source": source, "controller_source": controller,
                "pilot_selection_sha256": PILOT_SELECTION_SHA256,
                "primary_reference_family": selection["primary_reference_family"],
                "planned_runs": configs, "recipe_sources_match_pilot": True}
    events_path = output / "events.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise RuntimeError("The existing replication plan differs; preserve it")
    else:
        save_json(manifest_path, manifest)
    events = []
    if events_path.exists():
        previous_events = json.loads(events_path.read_text())
        if any(row["event"].startswith("final_") for row in previous_events):
            raise RuntimeError("Final-phase events already exist; do not reopen test")
        save_json(output / f"events_before_resume_{time.time_ns()}.json", previous_events)
        events = previous_events
    events.append({"event": "replication_started", "at_utc": utc_now()})
    save_json(events_path, events)
    runs = []
    for config in configs:
        row = train_one(config, output / run_name(config), device, protocol,
                        minimum_ram_gib=minimum_ram_gib)
        if row["resolved_model"] != execution_plan["resolved_models"][config["family"]]:
            raise RuntimeError("A trained model's resolved configuration differs from the pilot")
        if config["split_seed"] == 0:
            for name in ("validation_dataset", "training_probe_dataset"):
                if row[name]["content_sha256"] != execution_plan["primary_dataset_hashes"][name]:
                    raise RuntimeError("A primary development dataset differs from the pilot")
        runs.append(row)
        events.append({"event": "training_completed", "run": run_name(config), "at_utc": utc_now()})
        save_json(events_path, events)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if (any(row["source"] != source for row in runs)
            or any(row["environment"] != runs[0]["environment"] for row in runs)
            or controller_source() != controller):
        raise RuntimeError("Source, controller or environment changed during replication")
    gate = {"at_utc": utc_now(), "complete_training_runs": len(runs),
            "run_manifest_sha256": file_sha256(manifest_path),
            "pilot_selection_sha256": PILOT_SELECTION_SHA256,
            "source": source, "controller_source": controller}
    save_json(output / "evaluation_gate.json", gate)
    events.append({"event": "final_evaluation_authorized", **gate})
    save_json(events_path, events)
    datasets, dataset_records = {}, {}
    for split in sorted({config["split_seed"] for config in configs}):
        resource_sample(device, minimum_ram_gib)
        task = ResearchRoleTask(8, 8, seed=split)
        datasets[split], dataset_records[split] = dataset_artifact(
            task, "test", protocol["data"]["test_n"], protocol["data"]["test_sample_seed"],
            output / "datasets")
    if dataset_records[0]["content_sha256"] != "23bca94c224b195da94ab1794801fdae131007a77bd594c2167a085227e7dba2":
        raise RuntimeError("The primary final input content differs from the frozen pilot")
    records = []
    for training in runs:
        config = training["config"]
        path = output / run_name(config)
        if file_sha256(path / "weights.pt") != training["weights_sha256"]:
            raise OSError("Checkpoint changed before final evaluation")
        model = model_from_config(config).to(device)
        model.load_state_dict(torch.load(path / "weights.pt", map_location=device, weights_only=True), strict=True)
        metrics, predictions = evaluate(model, datasets[config["split_seed"]], device, intervals=True)
        prediction_record = save_arrays(path / "final_predictions.npz", **predictions)
        record = {"status": "completed", "config": config, "metrics": metrics,
                  "predictions": prediction_record, "weights_sha256": training["weights_sha256"],
                  "parameter_count": training["parameter_count"], "source": source,
                  "training_seconds": training["training_seconds"],
                  "train_probe": training["train_probe"],
                  "optimization_budget_limited": training["optimization_budget_limited"],
                  "pilot_selection_sha256": PILOT_SELECTION_SHA256, "evaluated_at_utc": utc_now()}
        save_json(path / "final_evaluation.json", record)
        records.append(record)
        events.append({"event": "final_evaluation_saved", "run": path.name,
                       "at_utc": record["evaluated_at_utc"]})
        save_json(events_path, events)
        print(json.dumps({"event": "replication_final", "run": path.name,
                          "binding": metrics["macro_agent_patient_accuracy"],
                          "global_accuracy": metrics["accuracy"]}), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    events.append({"event": "replication_completed", "at_utc": utc_now()})
    save_json(events_path, events)
    result = {"item": 5, "status": "completed", "source": source, "controller_source": controller,
              "environment": environment_record(device), "selection": selection,
              "pilot_selection_sha256": PILOT_SELECTION_SHA256,
              "primary_reference_family": selection["primary_reference_family"],
              "datasets": dataset_records, "records": records, "events": events,
              "total_training_seconds": sum(row["training_seconds"] for row in runs),
              "statistical_analysis": "pending separate analysis of saved predictions",
              "inference_latency_seconds": None, "physical_energy_joules": None}
    save_json(output / "replication_records.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "results/research/04_experiment/pilot/selection.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minimum-ram-gib", type=float, default=8.0)
    args = parser.parse_args()
    if args.minimum_ram_gib < 8:
        raise ValueError("The shared-machine RAM margin is at least 8 GiB")
    device = configure_runtime(args.device, args.threads)
    print(json.dumps({"event": "preflight", "resources": resource_sample(device, args.minimum_ram_gib),
                      "environment": environment_record(device)}), flush=True)
    run_replications(args.output, args.selection, device, load_protocol(), args.minimum_ram_gib)


if __name__ == "__main__":
    main()
