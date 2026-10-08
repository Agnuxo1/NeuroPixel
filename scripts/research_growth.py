"""Execute the prospectively specified item-6 growth and routing panel."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from neuropixel.research.experiment import (
    classification_metrics, configure_runtime, environment_record, file_sha256,
    resource_sample, save_arrays, save_json, utc_now,
)
from neuropixel.research.growth import (
    BATCH_SIZE, DECISION_PROBE_SEEDS, FINAL_DATA_SEEDS, FIRING_SEEDS,
    GATE_DATA_SEEDS, GATE_L2, GATE_LR, GATE_STEPS, LEARNING_RATE,
    NOVELTY_THRESHOLD, RANDOM_ROUTER_SEED, RESONANCE_MULTIPLIER,
    SCHOOL_WEIGHT, STAGE_UPDATES, TOPIC_NAMES, TOPICS, TRAIN_SAMPLE_SEEDS,
    TopicRoleTask, balanced_topic_dataset, clone_cpu, expert_probabilities,
    fit_gate, make_core, own_token_statistics, score_routers, state_sha256,
    tensor_sha256, train_stage, visible_features,
)

REQUIRED_IMPLEMENTATION = {
    "neuropixel/research/growth.py",
    "scripts/research_growth.py",
    "docs/research/06_protocol_amendment.md",
}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_protocol():
    frozen = read_json(ROOT / "docs/research/protocol_frozen.json")
    for relative, expected in frozen["files"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen protocol changed: {relative}")
    return read_json(ROOT / "docs/research/protocol.json")

def planned_growth():
    return {
        "init_seeds": [20, 21],
        "topics": [list(x) for x in TOPICS],
        "topic_names": list(TOPIC_NAMES),
        "policies": ["fixed_snapshot", "adaptive_novelty", "adaptive_resonance"],
        "stages_per_policy": 3,
        "stage_updates": STAGE_UPDATES,
        "batch_size": BATCH_SIZE,
        "steps": 16,
        "school_weight": SCHOOL_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "train_sample_seeds": list(TRAIN_SAMPLE_SEEDS),
        "firing_seeds": list(FIRING_SEEDS),
        "decision_probe_seeds": list(DECISION_PROBE_SEEDS),
        "gate_data_seeds": list(GATE_DATA_SEEDS),
        "final_data_seeds": list(FINAL_DATA_SEEDS),
        "novelty_threshold": NOVELTY_THRESHOLD,
        "novelty_token_probability_threshold": 0.5,
        "resonance_multiplier": RESONANCE_MULTIPLIER,
        "gate": {"features": 40, "steps": GATE_STEPS, "learning_rate": GATE_LR,
                 "l2": GATE_L2, "training_examples_per_topic": 512},
        "routers": ["legacy_reconstruction", "uniform_mixture", "random", "learned_gate"],
        "random_router_seed": RANDOM_ROUTER_SEED,
        "final_examples_per_topic": 1024,
        "max_experts": 3,
        "planned_training_trajectories": 6,
        "planned_stage_trainings": 18,
        "planned_gate_fits": 8,
    }

def load_plan(path):
    plan = read_json(path)
    if plan.get("item") != 6 or plan.get("status") != "frozen":
        raise ValueError("item-6 execution plan must be frozen")
    if plan.get("growth") != planned_growth():
        raise ValueError("frozen growth inventory differs from the approved specification")
    implementation = plan.get("implementation_sha256", {})
    if not REQUIRED_IMPLEMENTATION <= set(implementation):
        raise ValueError("growth source and amendment must be hashed by the plan")
    for relative, expected in implementation.items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen item-6 implementation changed: {relative}")
    return plan, implementation

def _topic_records(tasks):
    result = []
    for task in tasks:
        splits = {}
        for split, triples in (("train", task.train_triples),
                               ("validation", task.validation_triples),
                               ("test", task.test_triples)):
            values = [list(map(int, t)) for t in triples]
            raw = json.dumps(values, separators=(",", ":")).encode()
            splits[split] = {"n_triples": len(values),
                             "sha256": hashlib.sha256(raw).hexdigest(),
                             "triples": values}
        result.append({"topic": task.topic, "name": task.topic_name, "splits": splits})
    return result

def _checker(device, minimum_ram_gib):
    return lambda: resource_sample(device, minimum_ram_gib)

def _save_expert(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    if set(restored) != set(model.state_dict()):
        raise OSError("growth checkpoint keys changed during persistence")
    for name, value in model.state_dict().items():
        if not torch.equal(restored[name], value.detach().cpu()):
            raise OSError(f"growth checkpoint tensor changed: {name}")
    return {"path": str(path), "sha256": file_sha256(path),
            "state_sha256": state_sha256(model), "bytes": path.stat().st_size}

def _run_fixed(seed, tasks, output, device, minimum_ram_gib):
    model = make_core(seed).to(device)
    snapshots, stages = [], []
    for stage, task in enumerate(tasks):
        started = time.perf_counter()
        log = train_stage(model, task, stage, device, _checker(device, minimum_ram_gib))
        log["seconds"] = time.perf_counter() - started
        snapshot = clone_cpu(model)
        record = _save_expert(output / f"expert_stage{stage}.pt", snapshot)
        snapshots.append(snapshot)
        stages.append({"stage": stage, "topic": task.topic,
                       "action": "update_single_trajectory",
                       "training": log, "expert": record})
    return snapshots, stages

def _probe(task, seed):
    return balanced_topic_dataset(task, "train", 512, seed)

def _run_adaptive(seed, tasks, output, device, minimum_ram_gib, mode):
    if mode not in {"novelty", "resonance"}:
        raise ValueError("unknown adaptive growth mode")
    bank, stages, tau = [], [], None
    for stage, task in enumerate(tasks):
        probe = _probe(task, DECISION_PROBE_SEEDS[stage])
        scores_raw, novelty_raw = [], []
        for expert in bank:
            resonance, novelty = own_token_statistics(expert, probe[0], device)
            scores_raw.append(float(resonance.mean()))
            novelty_raw.append(float(novelty))
        if not bank:
            selected, create = None, True
        elif mode == "novelty":
            rounded = [round(x, 4) for x in novelty_raw]
            selected = min(range(len(bank)), key=lambda i: (rounded[i], i))
            create = rounded[selected] > NOVELTY_THRESHOLD
        else:
            rounded = [round(x, 4) for x in scores_raw]
            selected = max(range(len(bank)), key=lambda i: (rounded[i], -i))
            create = rounded[selected] < tau
        if create:
            model = make_core(seed) if selected is None else clone_cpu(bank[selected])
            parent = selected
            bank.append(model)
            selected_after = len(bank) - 1
            action = "create"
        else:
            selected_after = selected
            model = bank[selected_after]
            parent = selected_after
            action = "revisit"
        started = time.perf_counter()
        training = train_stage(model, task, stage, device, _checker(device, minimum_ram_gib))
        training["seconds"] = time.perf_counter() - started
        bank[selected_after] = clone_cpu(model)
        if mode == "resonance" and stage == 0:
            resonance, _ = own_token_statistics(bank[0], probe[0], device)
            tau = RESONANCE_MULTIPLIER * float(resonance.mean())
        expert_record = _save_expert(output / f"stage{stage}_expert{selected_after}.pt",
                                     bank[selected_after])
        stages.append({
            "stage": stage, "topic": task.topic, "mode": mode,
            "probe_sha256": tensor_sha256(*probe),
            "raw_resonance": scores_raw,
            "rounded_resonance": [round(x, 4) for x in scores_raw],
            "raw_novelty": novelty_raw,
            "rounded_novelty": [round(x, 4) for x in novelty_raw],
            "threshold": NOVELTY_THRESHOLD if mode == "novelty" else tau,
            "selected_before_training": selected,
            "selected_after_training": selected_after,
            "parent": parent, "action": action,
            "expert_count_after": len(bank),
            "training": training, "expert": expert_record,
        })
        if len(bank) > 3:
            raise RuntimeError("adaptive policy exceeded the declared three-expert ceiling")
        gc.collect()
    return [clone_cpu(x) for x in bank], stages

def _concat(datasets):
    return tuple(torch.cat([dataset[i] for dataset in datasets], 0) for i in range(3))

def _fit_bank_gate(bank, gate_dataset, role_ids, device):
    canvas, target, _ = gate_dataset
    probs = expert_probabilities(bank, canvas, device)
    features = visible_features(canvas, role_ids)
    return fit_gate(features, probs, target)

def _score_single(model, dataset, device):
    canvas, target, roles = dataset
    probs = expert_probabilities([model], canvas, device)[0]
    pred = probs.argmax(-1)
    return classification_metrics(pred, target.numpy(), roles.numpy(), intervals=False), pred

def _topic_slices():
    return [slice(i * 1024, (i + 1) * 1024) for i in range(3)]

def _per_topic(predictions, dataset):
    _, target, roles = dataset
    target_np, roles_np = target.numpy(), roles.numpy()
    return {
        TOPIC_NAMES[topic]: {
            router: classification_metrics(pred[section], target_np[section],
                                           roles_np[section], intervals=False)
            for router, pred in predictions.items()
        }
        for topic, section in enumerate(_topic_slices())
    }

def run_growth(output, plan_path, device, minimum_ram_gib):
    plan, implementation = load_plan(plan_path)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "growth_records.json").exists() or (output / "evaluation_gate.json").exists():
        raise RuntimeError("growth final-phase evidence already exists; explicit recovery is required")
    tasks = [TopicRoleTask(i, seed=0) for i in range(3)]
    topics = _topic_records(tasks)
    save_json(output / "topic_splits.json", topics)

    decision_records, gate_parts = [], []
    for stage, task in enumerate(tasks):
        probe = _probe(task, DECISION_PROBE_SEEDS[stage])
        decision_records.append({"topic": stage, "seed": DECISION_PROBE_SEEDS[stage],
                                 "content_sha256": tensor_sha256(*probe), "n": len(probe[0])})
        gate_data = balanced_topic_dataset(task, "train", 512, GATE_DATA_SEEDS[stage])
        gate_parts.append(gate_data)
        save_arrays(output / "datasets" / f"gate_train_topic{stage}.npz",
                    canvas=gate_data[0].numpy(), target=gate_data[1].numpy(),
                    roles=gate_data[2].numpy())
    gate_dataset = _concat(gate_parts)
    train_only = {"decision_probes": decision_records,
                  "gate_dataset_sha256": tensor_sha256(*gate_dataset), "gates": {}}

    all_seed_data, all_banks, gate_records = {}, {}, {}
    total_started = time.perf_counter()
    for seed in (20, 21):
        seed_dir = output / f"seed{seed}"
        fixed, fixed_stages = _run_fixed(seed, tasks, seed_dir / "fixed", device, minimum_ram_gib)
        novelty, novelty_stages = _run_adaptive(seed, tasks, seed_dir / "novelty",
                                                device, minimum_ram_gib, "novelty")
        resonance, resonance_stages = _run_adaptive(seed, tasks, seed_dir / "resonance",
                                                    device, minimum_ram_gib, "resonance")
        first_hashes = [fixed_stages[0]["expert"]["state_sha256"],
                        novelty_stages[0]["expert"]["state_sha256"],
                        resonance_stages[0]["expert"]["state_sha256"]]
        if len(set(first_hashes)) != 1:
            raise RuntimeError("paired policies diverged during their identical first stage")
        banks = {
            "fixed": fixed,
            "duplicate": [clone_cpu(fixed[0]), clone_cpu(fixed[1]), clone_cpu(fixed[1])],
            "novelty": novelty,
            "resonance": resonance,
        }
        all_banks[seed] = banks
        all_seed_data[str(seed)] = {
            "fixed_stages": fixed_stages, "novelty_stages": novelty_stages,
            "resonance_stages": resonance_stages,
            "first_stage_identity_control": first_hashes,
            "banks": {
                name: {"expert_count": len(bank),
                       "state_sha256": [state_sha256(x) for x in bank],
                       "nominal_core_parameters": len(bank) * 29824,
                       "unique_state_count": len(set(state_sha256(x) for x in bank))}
                for name, bank in banks.items()
            },
        }
        for name, bank in banks.items():
            fitted = _fit_bank_gate(bank, gate_dataset, tasks[0].role_ids, device)
            gate_file = seed_dir / f"gate_{name}.npz"
            artifact = save_arrays(gate_file, weight=fitted["weight"])
            meta = {k: v for k, v in fitted.items() if k != "weight"}
            meta.update(artifact)
            gate_records[f"{seed}:{name}"] = meta
            train_only["gates"][f"{seed}:{name}"] = meta
            resource_sample(device, minimum_ram_gib)

    if len(all_seed_data) != 2 or len(gate_records) != 8:
        raise RuntimeError("six trajectories and eight gate fits are required before final evaluation")
    gate = {"at_utc": utc_now(), "complete_training_trajectories": 6,
            "complete_stage_trainings": 18, "complete_gate_fits": 8,
            "topic_splits_sha256": file_sha256(output / "topic_splits.json"),
            "train_only": train_only, "implementation_sha256": implementation}
    save_json(output / "evaluation_gate.json", gate)

    final_parts, final_records = [], []
    for topic, task in enumerate(tasks):
        dataset = balanced_topic_dataset(task, "test", 1024, FINAL_DATA_SEEDS[topic])
        final_parts.append(dataset)
        artifact = save_arrays(output / "datasets" / f"final_topic{topic}.npz",
                               canvas=dataset[0].numpy(), target=dataset[1].numpy(),
                               roles=dataset[2].numpy())
        artifact.update(topic=topic, seed=FINAL_DATA_SEEDS[topic],
                        content_sha256=tensor_sha256(*dataset), n=1024)
        final_records.append(artifact)
    final_dataset = _concat(final_parts)
    results = {}
    for seed in (20, 21):
        banks = all_banks[seed]
        seed_results = {}
        single_metrics, single_pred = _score_single(banks["fixed"][-1], final_dataset, device)
        target_np, roles_np = final_dataset[1].numpy(), final_dataset[2].numpy()
        single_topics = {
            TOPIC_NAMES[t]: classification_metrics(single_pred[s], target_np[s],
                                                   roles_np[s], intervals=False)
            for t, s in enumerate(_topic_slices())
        }
        seed_results["single_final_expert"] = {
            "metrics": single_metrics, "per_topic": single_topics,
            "state_sha256": state_sha256(banks["fixed"][-1])}
        for name, bank in banks.items():
            with np.load(output / f"seed{seed}" / f"gate_{name}.npz", allow_pickle=False) as stored:
                gate_weight = stored["weight"]
            routed = score_routers(bank, *final_dataset, tasks[0].role_ids,
                                   gate_weight, device, random_seed=RANDOM_ROUTER_SEED)
            per_topic = _per_topic(routed["predictions"], final_dataset)
            prediction_artifact = save_arrays(
                output / f"seed{seed}" / f"predictions_{name}.npz",
                target=final_dataset[1].numpy(), roles=final_dataset[2].numpy(),
                **{f"prediction_{k}": v for k, v in routed["predictions"].items()},
                **{f"index_{k}": v for k, v in routed["indices"].items()})
            clean = {k: v for k, v in routed.items()
                     if k not in {"_expert_probs", "predictions", "indices"}}
            clean.update(per_topic=per_topic, predictions=prediction_artifact,
                         expert_count=len(bank),
                         state_sha256=[state_sha256(x) for x in bank],
                         gate_parameters=int(gate_weight.size),
                         maximum_expert_plus_gate_parameters=len(bank) * 29824 + int(gate_weight.size))
            seed_results[name] = clean
        seed_results["preallocated_equivalence_control"] = {
            "same_expert_state_hashes": seed_results["fixed"]["state_sha256"],
            "identical_to_fixed_bank_by_construction": True,
            "interpretation": "allocation timing control only; not an independent training replicate"}
        results[str(seed)] = seed_results
        resource_sample(device, minimum_ram_gib)

    summary = {
        "schema_version": 1, "item": 6, "panel": "growth", "status": "completed",
        "plan_growth": plan["growth"], "environment": environment_record(device),
        "topic_splits": topics, "training": all_seed_data, "gate_fits": gate_records,
        "evaluation_gate": gate, "final_datasets": final_records, "results": results,
        "total_wall_seconds": time.perf_counter() - total_started,
        "interpretation_scope": "exploratory two-seed, one-split panel; snapshots and router variants are not independent training replications"}
    save_json(output / "growth_records.json", summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "docs/research/06_execution_plan.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--minimum-ram-gib", type=float, default=8.0)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(planned_growth(), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for execution")
    if args.minimum_ram_gib < 8 or not 1 <= args.threads <= 4:
        raise ValueError("item-2 resource policy requires RAM >=8 GiB and one to four CPU threads")
    import psutil
    if psutil.virtual_memory().available / 2**30 < args.minimum_ram_gib:
        raise RuntimeError("available RAM is below the declared floor; no growth training started")
    load_plan(args.plan)
    device = configure_runtime(args.device, args.threads)
    print(json.dumps({"event": "preflight",
                      "resources": resource_sample(device, args.minimum_ram_gib),
                      "environment": environment_record(device)}), flush=True)
    run_growth(args.output, args.plan, device, args.minimum_ram_gib)

if __name__ == "__main__":
    main()
