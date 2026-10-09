"""Run the frozen item-4 pilot; open final evaluation only after selection."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from neuropixel.research.data import ResearchRoleTask
from neuropixel.research.experiment import (
    configure_runtime, dataset_artifact, environment_record, evaluate, file_sha256,
    model_from_config, resource_sample, save_arrays, save_json, select_pilot,
    source_record, train_one, utc_now,
)


def load_protocol():
    protocol_path = ROOT / "docs/research/protocol.json"
    frozen = json.loads((ROOT / "docs/research/protocol_frozen.json").read_text())
    for relative, expected in frozen["files"].items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen protocol changed: {relative}")
    return json.loads(protocol_path.read_text())


def pilot_config(protocol, family, learning_rate):
    training = protocol["training"]
    return {"phase": "item4_pilot", "protocol_id": protocol["protocol_id"],
            "family": family, "vocab": 35, "height": 8, "width": 8,
            "split_seed": 0, "init_seed": training["pilot_init_seed"],
            "learning_rate": learning_rate, "updates": training["updates"],
            "batch_size": training["batch_size"], "steps": training["steps"],
            "weight_decay": training["weight_decay"], "gradient_clip": training["gradient_clip"],
            "school_weight": training["primary_school_weight"],
            "train_sample_seed": protocol["data"]["train_sample_seed"],
            "update_random_seed": training["update_random_seed"], "variant": {}}


def run_name(config):
    return (f"{config['family']}_s{config['split_seed']}_i{config['init_seed']}"
            f"_lr{config['learning_rate']:g}")


def run_pilot(output_dir, device, protocol, minimum_ram_gib):
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_summary = output_dir / "pilot_summary.json"
    if existing_summary.exists() or (output_dir / "selection.json").exists():
        raise RuntimeError("pilot selection or final results already exist; explicit recovery is required before reopening test")
    events_path = output_dir / "events.json"
    if events_path.exists():
        old_events = json.loads(events_path.read_text())
        if any(row["event"] in ("selection_frozen", "final_evaluation_opened", "pilot_completed") for row in old_events):
            raise RuntimeError("final-phase evidence already exists; do not silently reopen final evaluation")
        if old_events:
            save_json(output_dir / f"events_before_resume_{time.time_ns()}.json", old_events)
    source = source_record()
    events = [{"event": "pilot_started", "at_utc": utc_now()}]
    families, rates = protocol["training"]["families"], protocol["training"]["learning_rates"]
    runs = []
    for family in families:
        for learning_rate in rates:
            config = pilot_config(protocol, family, learning_rate)
            path = output_dir / run_name(config)
            runs.append(train_one(config, path, device, protocol, minimum_ram_gib=minimum_ram_gib))
            events.append({"event": "training_completed", "run": path.name, "at_utc": utc_now()})
            save_json(output_dir / "events.json", events)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    selection = select_pilot(runs, families, rates)
    if any(run["source"] != source for run in runs):
        raise RuntimeError("source changed between pilot runs; no final evaluation is authorized")
    if any(run["environment"] != runs[0]["environment"] for run in runs):
        raise RuntimeError("pilot runs do not share a common measured environment")
    selection["source"] = source
    selection_path = output_dir / "selection.json"
    save_json(selection_path, selection)
    selection_hash = file_sha256(selection_path)
    events.append({"event": "selection_frozen", "at_utc": utc_now(), "sha256": selection_hash})
    save_json(output_dir / "events.json", events)
    # This is the first construction/access of final examples anywhere in the pilot.
    resource_sample(device, minimum_ram_gib)
    task = ResearchRoleTask(8, 8, seed=0)
    final, final_record = dataset_artifact(task, "test", protocol["data"]["test_n"],
                                         protocol["data"]["test_sample_seed"], output_dir / "datasets")
    events.append({"event": "final_evaluation_opened", "at_utc": utc_now(),
                   "dataset_sha256": final_record["content_sha256"]})
    save_json(output_dir / "events.json", events)
    final_results = {}
    for family, config in selection["selected_configs"].items():
        path = output_dir / run_name(config)
        training = json.loads((path / "training.json").read_text())
        if file_sha256(path / "weights.pt") != training["weights_sha256"]:
            raise OSError(f"selected checkpoint changed: {path}")
        model = model_from_config(config).to(device)
        model.load_state_dict(torch.load(path / "weights.pt", map_location=device, weights_only=True), strict=True)
        metrics, predictions = evaluate(model, final, device, intervals=True)
        prediction_record = save_arrays(path / "final_predictions.npz", **predictions)
        row = {"family": family, "config": config, "metrics": metrics,
               "weights_sha256": training["weights_sha256"], "predictions": prediction_record,
               "selection_sha256": selection_hash, "evaluated_at_utc": utc_now(),
               "parameter_count": training["parameter_count"],
               "training_seconds": training["training_seconds"],
               "optimization_budget_limited": training["optimization_budget_limited"]}
        save_json(path / "final_evaluation.json", row)
        final_results[family] = row
        print(json.dumps({"event": "pilot_final_result", "family": family, "metrics": metrics}), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    events.append({"event": "pilot_completed", "at_utc": utc_now()})
    save_json(output_dir / "events.json", events)
    summary = {"schema_version": 1, "item": 4, "status": "completed", "protocol_id": protocol["protocol_id"],
               "source": source, "environment": environment_record(device), "selection": selection,
               "selection_sha256": selection_hash, "final_dataset": final_record,
               "results": final_results, "events": events,
               "total_training_seconds": sum(run["training_seconds"] for run in runs),
               "inference_latency_seconds": None, "physical_energy_joules": None,
               "limitations": ["A two-rate, 1024-update pilot is not exhaustive architecture optimization.",
                               "Equal examples/updates do not equalize FLOPs or elapsed time.",
                               "One initialization does not decide H1; item5 replication is still required.",
                               "All primary runs use answer supervision only; school is evaluated separately.",
                               "These tasks have exact elementary solutions demonstrated in item3."]}
    save_json(existing_summary, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/research/pilot")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minimum-ram-gib", type=float, default=8.0)
    args = parser.parse_args()
    if args.minimum_ram_gib < 8:
        raise ValueError("the declared shared-machine RAM margin is at least 8 GiB")
    protocol = load_protocol()
    device = configure_runtime(args.device, args.threads)
    print(json.dumps({"event": "preflight", "resources": resource_sample(device, args.minimum_ram_gib),
                      "environment": environment_record(device)}), flush=True)
    run_pilot(args.output, device, protocol, args.minimum_ram_gib)


if __name__ == "__main__":
    main()
