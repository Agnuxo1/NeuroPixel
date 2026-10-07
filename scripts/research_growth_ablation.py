"""Run the frozen item-6 growth and routing panel with its own final-data gate.

The --describe path uses only standard-library imports. The operational worker
must admit RAM/device resources before execution. Original model/data sources
remain unchanged; every new stage, heuristic decision, checkpoint and gate is
recorded. No H1 decision or cross-panel selection is made here.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


operations = _load("item6_growth_operations", ROOT / "neuropixel/research/growth_ablation.py")
core = _load("item6_growth_core_contract", ROOT / "scripts/research_ablations.py")
REQUIRED_IMPLEMENTATION = {
    "neuropixel/research/growth_ablation.py", "scripts/research_growth_ablation.py",
    "tests/test_research_growth_ablation.py", "docs/research/06_protocol_amendment.md",
}


def load_execution_plan(path):
    """Require the same frozen scientific anchor plus the exact growth inventory."""
    protocol, plan, controller, historical = core.load_execution_plan(path)
    if plan.get("growth") != operations.growth_inventory():
        raise ValueError("the frozen growth inventory differs from the reviewed implementation")
    if not REQUIRED_IMPLEMENTATION <= set(plan["implementation_sha256"]):
        raise ValueError("growth code, tests and amendment must all be hashed before execution")
    return protocol, plan, controller, historical



def verify_source_integrity(expected_source, controller):
    """Reject changes to source provenance, implementation bytes or the frozen plan."""
    if core.source_record() != expected_source:
        raise RuntimeError("scientific source or Git HEAD changed during execution")
    if any(core.sha256(core.relative_source(path)) != digest for path, digest in controller.items()):
        raise RuntimeError("the frozen new implementation or execution plan changed")


def enter_final_phase(trajectories, banks, preallocation, authorize, create_final_datasets):
    """Require all development work before authorization and any final-data callback.

    This small standard-library gate is independently testable without importing
    Torch or constructing data. Authorization verifies source/artifact hashes and
    writes the gate; only a successful authorization may construct final examples.
    """
    specification = operations.growth_inventory()
    expected = {row["trajectory_id"]: row for row in specification["trajectories"]}
    if len(trajectories) != len(expected):
        raise RuntimeError("all six completed trajectories are required before final data")
    seen = set()
    for row in trajectories:
        config = row.get("config", {})
        identity = config.get("trajectory_id")
        if (identity not in expected or identity in seen or config != expected[identity]
                or row.get("status") != "completed" or row.get("final_test_accessed") is not False):
            raise RuntimeError("trajectory identities, recipes or completion status differ")
        seen.add(identity)
        stages = row.get("stages", [])
        if (len(stages) != 3 or [stage.get("stage") for stage in stages] != [0, 1, 2]
                or any(stage.get("optimizer_updates") != config["stage_updates"]
                       or stage.get("final_test_accessed") is not False for stage in stages)
                or row.get("optimizer_updates") != 3 * config["stage_updates"]):
            raise RuntimeError("all eighteen complete training stages are required before final data")
    kinds = ("snapshots", "duplicate_slots", "single_final",
             "adaptive_novelty", "adaptive_resonance")
    expected_banks = {f"{kind}_i{seed}": (kind, seed)
                      for seed in operations.SEEDS for kind in kinds}
    if (len(banks) != len(expected_banks)
            or {bank.get("bank_id") for bank in banks} != set(expected_banks)):
        raise RuntimeError("the complete expert-bank inventory is required before final data")
    fits = 0
    for bank in banks:
        kind, seed = expected_banks[bank["bank_id"]]
        if bank.get("kind") != kind or bank.get("init_seed") != seed:
            raise RuntimeError("bank identity or seed differs")
        requires_fit = kind != "single_final"
        if bank.get("fit_gate") is not requires_fit:
            raise RuntimeError("bank gate requirements differ")
        if requires_fit:
            fitted = bank.get("gate", {})
            if (fitted.get("updates") != 128 or fitted.get("final_test_accessed") is not False
                    or not isinstance(fitted.get("weights"), dict)):
                raise RuntimeError("all eight fitted gates are required before final data")
            fits += 1
    if fits != 8:
        raise RuntimeError("all eight fitted gates are required before final data")
    if (len(preallocation) != 2
            or {row.get("init_seed") for row in preallocation} != set(operations.SEEDS)
            or any(row.get("status") != "verified" for row in preallocation)):
        raise RuntimeError("both construction-equivalence checks are required before final data")
    authorization = authorize()
    return authorization, create_final_datasets()


def run_growth(output, plan_path, device, minimum_ram_gib):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from neuropixel.model import NeuroPixel
    from neuropixel.research.data import frozen_dataset
    from neuropixel.research.experiment import (
        dataset_artifact, environment_record, resource_sample, save_arrays,
        save_json, utc_now,
    )

    protocol, plan, controller, historical = load_execution_plan(plan_path)
    specification = plan["growth"]
    source = core.source_record()
    if source["git_commit"] == "unavailable" or source["sha256"] != historical:
        raise RuntimeError("the current source does not match the recovered frozen scientific anchor")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("growth output is not empty; preserve it and use explicit recovery")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    events = []

    def event(name, **fields):
        row = {"event": name, "at_utc": utc_now(), **fields}
        events.append(row)
        save_json(output / "events.json", events)
        return row

    def check_source():
        verify_source_integrity(source, controller)

    def portable(record):
        result = dict(record)
        result["path"] = Path(result["path"]).resolve().relative_to(output).as_posix()
        return result

    def save_npz(relative, **arrays):
        return portable(save_arrays(output / relative, **arrays))

    def digest_arrays(named):
        digest = hashlib.sha256()
        for name in sorted(named):
            array = np.ascontiguousarray(named[name])
            header = json.dumps({"name": name, "shape": list(array.shape), "dtype": str(array.dtype)},
                                sort_keys=True).encode()
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(array.tobytes())
        return digest.hexdigest()

    def state_digest(state):
        return digest_arrays({name: value.detach().cpu().numpy() for name, value in state.items()})

    def checkpoint(model, relative):
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        if not all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in state.values()):
            raise FloatingPointError("checkpoint tensors are not finite float32")
        stream = io.BytesIO()
        torch.save(state, stream)
        path = output / relative
        # Use the same atomic persistence primitive as the frozen JSON writer.
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(stream.getvalue())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if path.read_bytes() != stream.getvalue():
            raise OSError("checkpoint changed during atomic persistence")
        return {"path": relative, "sha256": core.sha256(path), "bytes": path.stat().st_size,
                "tensor_sha256": state_digest(state), "parameter_count": sum(value.numel() for value in state.values())}

    def new_model(init_seed):
        # Initial weights have a private CPU seed; copying or loading never advances
        # the data or firing streams used in any training stage.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.default_generator.manual_seed(init_seed)
            model = NeuroPixel(35, (7, 7), c_id=16, c=48, hidden=128, steps=16, fire_rate=0.5)
        if sum(parameter.numel() for parameter in model.parameters()) != operations.EXPERT_PARAMETERS:
            raise RuntimeError("the original expert parameter count changed")
        return model.to(device)

    def read_model(record, init_seed):
        path = output / record["path"]
        if core.sha256(path) != record["sha256"]:
            raise OSError("expert checkpoint changed")
        model = new_model(init_seed)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        if state_digest(model.state_dict()) != record["tensor_sha256"]:
            raise OSError("restored expert tensors differ")
        return model.eval()

    def expert_outputs(model, canvas, batch_size=64, retain_own=False):
        """Compute logits and scanner values in the same forward, without targets."""
        model.eval()
        logp, scanner, occupied_hits, occupied_count = [], [], 0, 0
        own_values = [] if retain_own else None
        with torch.no_grad():
            for start in range(0, len(canvas), batch_size):
                resource_sample(device, minimum_ram_gib)
                batch = canvas[start:start + batch_size].to(device)
                result = model(batch)
                logits = result["logits"]
                lens_probabilities = model.lens_logits(result["state"]).softmax(-1)
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(lens_probabilities).all()):
                    raise FloatingPointError("expert outputs became nonfinite")
                own = lens_probabilities.gather(-1, batch.unsqueeze(-1)).squeeze(-1)
                occupied = batch != 0
                if not bool(occupied.flatten(1).any(1).all()):
                    raise ValueError("scanner requires at least one visible token per example")
                scores = (own * occupied.float()).sum((1, 2)) / occupied.float().sum((1, 2))
                scanner.append(scores.detach())
                if retain_own:
                    own_values.append(own.cpu().numpy())
                logp.append(F.log_softmax(logits, dim=-1).cpu().numpy())
                occupied_hits += int(((own < 0.5) & occupied).sum().item())
                occupied_count += int(occupied.sum().item())
            all_scanner = torch.cat(scanner)
            # Retain the original float32 reduction/threshold statistics.
            mean_resonance = float(all_scanner.mean().item())
            novelty = float(np.float32(occupied_hits) / np.float32(occupied_count))
            outputs = {"log_probabilities": np.concatenate(logp),
                       "scanner": all_scanner.cpu().numpy(),
                       "mean_resonance": mean_resonance, "novelty": novelty}
            if retain_own:
                outputs["own_token_probability"] = np.concatenate(own_values)
            return outputs

    def train_stage(model, task, config, stage):
        """Use the same fresh optimizer and independent streams in every policy."""
        resource_sample(device, minimum_ram_gib)
        settings = specification["stages"][stage]
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        generator = torch.Generator(device="cpu").manual_seed(settings["train_sample_seed"])
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        curve, data_digest = [], hashlib.sha256()
        stage_started = time.perf_counter()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.default_generator.manual_seed(settings["firing_seed"])
            for index in cuda_devices:
                torch.cuda.default_generators[index].manual_seed(settings["firing_seed"])
            for update in range(1, config["stage_updates"] + 1):
                if update == 1 or update % 128 == 0:
                    resource_sample(device, minimum_ram_gib)
                    check_source()
                canvas, target = task.sample(config["batch_size"], "train", generator, "cpu")
                data_digest.update(canvas.numpy().tobytes())
                data_digest.update(target.numpy().tobytes())
                canvas, target = canvas.to(device), target.to(device)
                result = model(canvas, lens_every=4)
                answer_loss = F.cross_entropy(result["logits"], target)
                if "lens" not in result or result["lens"].shape[1] != 4:
                    raise RuntimeError("the original four school observations are required")
                count = result["lens"].shape[1]
                occupied = (canvas != 0).unsqueeze(1).expand(-1, count, -1, -1)
                labels = canvas.unsqueeze(1).expand(-1, count, -1, -1)
                school_loss = F.cross_entropy(result["lens"][occupied], labels[occupied])
                loss = answer_loss + config["school_weight"] * school_loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("training objective became nonfinite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                if update % 128 == 0:
                    row = {"update": update, "answer_cross_entropy": float(answer_loss.detach().item()),
                           "school_cross_entropy": float(school_loss.detach().item()),
                           "objective": float(loss.detach().item()), "gradient_norm_before_clip": float(norm)}
                    curve.append(row)
                    print(json.dumps({"event": "growth_update", "trajectory": config["trajectory_id"],
                                      "stage": stage, **row}), flush=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return {"curve": curve, "optimizer_updates": config["stage_updates"],
                "training_seconds": time.perf_counter() - stage_started,
                "train_stream_sha256": data_digest.hexdigest(),
                "train_sample_seed": settings["train_sample_seed"], "firing_seed": settings["firing_seed"]}

    manifest = {"schema_version": 1, "item": 6, "panel": "growth", "specification": specification,
                "source": source, "controller_source": controller, "environment": environment_record(device)}
    save_json(output / "run_manifest.json", manifest)
    event("growth_started")
    try:
        tasks = [operations.make_topic_task(index) for index in range(3)]
        partitions = [operations.partition_manifest(task) for task in tasks]
        save_json(output / "partitions.json", partitions)
        probes, probe_records, fit_data, fit_records = [], [], [], []
        for stage, task in enumerate(tasks):
            settings = specification["stages"][stage]
            generator = torch.Generator(device="cpu").manual_seed(settings["diagnostic_seed"])
            canvas, _unused_target = task.sample(settings["diagnostic_n"], "train", generator, "cpu")
            del _unused_target
            probes.append(canvas)
            record = save_npz(f"datasets/topic{stage}/diagnostic_train.npz", canvas=canvas.numpy())
            query_tokens = canvas[:, task.query_pos[0], task.query_pos[1]].numpy()
            role_counts = {name: int(np.count_nonzero(query_tokens == token))
                           for name, token in zip(operations.ROLES, task.role_ids.tolist())}
            if sum(role_counts.values()) != settings["diagnostic_n"]:
                raise RuntimeError("diagnostic query roles do not cover the saved examples")
            record.update(topic=stage, split="train", n=settings["diagnostic_n"],
                          sampling_seed=settings["diagnostic_seed"], targets_retained=False,
                          role_balance=specification["diagnostic_role_balance"],
                          query_role_counts=role_counts)
            probe_records.append(record)
            data, descriptor = dataset_artifact(task, "train", settings["gate_fit_n"], settings["gate_fit_seed"],
                                                output / "datasets" / f"topic{stage}")
            fit_data.append(data)
            fit_records.append({**portable(descriptor), "topic": stage})
        event("development_datasets_saved", diagnostic=probe_records, gate_fit=fit_records)

        trajectories, first_stage_hash, stage_stream_hash = [], {}, {}
        for config in specification["trajectories"]:
            resource_sample(device, minimum_ram_gib)
            bank, final_references, stages, tau = [], [], [], None
            trajectory_id = config["trajectory_id"]
            for stage, task in enumerate(tasks):
                check_source()
                before = [state_digest(model.state_dict()) for model in bank]
                diagnostics = [expert_outputs(model, probes[stage], batch_size=512, retain_own=True)
                               for model in bank]
                diagnostic_outputs = []
                for index, measured in enumerate(diagnostics):
                    prior = final_references[index]
                    if before[index] != prior["tensor_sha256"]:
                        raise RuntimeError("diagnostic weights differ from the recorded prior checkpoint")
                    diagnostic = save_npz(
                        f"trajectories/{trajectory_id}/stage{stage}/diagnostic_expert{index}.npz",
                        own_token_probability=measured["own_token_probability"], scanner=measured["scanner"])
                    diagnostic.update(expert_index=index, checkpoint_before_intervention=dict(prior),
                                      input_dataset=probe_records[stage],
                                      mean_resonance=measured["mean_resonance"], novelty=measured["novelty"],
                                      recount_atol=specification["diagnostic_recount_atol"], recount_rtol=0.0)
                    diagnostic_outputs.append(diagnostic)
                novelty = [row["novelty"] for row in diagnostics]
                resonance = [row["mean_resonance"] for row in diagnostics]
                decision = operations.growth_decision(config["policy"], stage, novelty, resonance, tau)
                if decision["action"] == "create":
                    model = new_model(config["init_seed"]) if decision["parent"] is None else copy.deepcopy(bank[decision["parent"]])
                    bank.append(model)
                    final_references.append(None)
                else:
                    model = bank[decision["target_index"]]
                training = train_stage(model, task, config, stage)
                reference = checkpoint(model, f"trajectories/{trajectory_id}/stage{stage}/weights.pt")
                final_references[decision["target_index"]] = reference
                # Unselected experts must stay unchanged; creation copies the parent
                # before optimization, while revisiting updates exactly one old slot.
                for index, fingerprint in enumerate(before):
                    if decision["action"] == "create" or index != decision["target_index"]:
                        if state_digest(bank[index].state_dict()) != fingerprint:
                            raise RuntimeError("an unselected retained expert was modified")
                if stage in stage_stream_hash and training["train_stream_sha256"] != stage_stream_hash[stage]:
                    raise RuntimeError("policies or initializations consumed different training examples")
                stage_stream_hash[stage] = training["train_stream_sha256"]
                post_stage_a_mean_resonance, post_stage_a_diagnostic = None, None
                if stage == 0:
                    if config["init_seed"] in first_stage_hash and first_stage_hash[config["init_seed"]] != reference["tensor_sha256"]:
                        raise RuntimeError("the shared stage-A recipe produced different initial policy checkpoints")
                    first_stage_hash[config["init_seed"]] = reference["tensor_sha256"]
                    if config["policy"] == "adaptive_resonance":
                        measured = expert_outputs(model, probes[0], batch_size=512, retain_own=True)
                        post_stage_a_mean_resonance = measured["mean_resonance"]
                        tau = 0.75 * post_stage_a_mean_resonance
                        post_stage_a_diagnostic = save_npz(
                            f"trajectories/{trajectory_id}/stage0/post_stage_a_diagnostic.npz",
                            own_token_probability=measured["own_token_probability"], scanner=measured["scanner"])
                        post_stage_a_diagnostic.update(
                            checkpoint_after_stage=dict(reference), input_dataset=probe_records[0],
                            mean_resonance=post_stage_a_mean_resonance, novelty=measured["novelty"],
                            recount_atol=specification["diagnostic_recount_atol"], recount_rtol=0.0)
                stage_record = {"topic": stage, "stage": stage, "decision": decision,
                                "raw_novelty": novelty, "raw_resonance": resonance,
                                "diagnostic_expert_outputs": diagnostic_outputs,
                                "post_stage_a_diagnostic": post_stage_a_diagnostic,
                                "tau_after_stage": tau, "post_stage_a_mean_resonance": post_stage_a_mean_resonance,
                                "experts_after_stage": len(bank),
                                "diagnostic_dataset": probe_records[stage], "checkpoint": reference,
                                "final_test_accessed": False, **training}
                save_json(output / "trajectories" / trajectory_id / f"stage{stage}" / "training.json", stage_record)
                stages.append(stage_record)
                event("growth_stage_completed", trajectory=trajectory_id, stage=stage,
                      decision=decision, experts=len(bank), checkpoint=reference)
            record = {"status": "completed", "config": config, "stages": stages,
                      "final_experts": final_references, "expert_count": len(bank),
                      "optimizer_updates": sum(row["optimizer_updates"] for row in stages),
                      "training_seconds": sum(row["training_seconds"] for row in stages),
                      "final_test_accessed": False, "source": source, "controller_source": controller}
            save_json(output / "trajectories" / trajectory_id / "training.json", record)
            trajectories.append(record)
            del bank, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(trajectories) != 6 or sum(len(row["stages"]) for row in trajectories) != 18:
            raise RuntimeError("the complete growth training inventory is required")

        by_key = {(row["config"]["policy"], row["config"]["init_seed"]): row for row in trajectories}
        banks, preallocation = [], []
        for seed in operations.SEEDS:
            fixed = by_key[("fixed_sequential", seed)]
            refs = [row["checkpoint"] for row in fixed["stages"]]
            banks.extend([
                {"bank_id": f"snapshots_i{seed}", "init_seed": seed, "kind": "snapshots", "experts": refs, "fit_gate": True},
                {"bank_id": f"duplicate_slots_i{seed}", "init_seed": seed, "kind": "duplicate_slots", "experts": [refs[0], refs[1], refs[1]], "fit_gate": True},
                {"bank_id": f"single_final_i{seed}", "init_seed": seed, "kind": "single_final", "experts": [refs[2]], "fit_gate": False},
            ])
            # Reconstructing the same ordered weights in slots checks functional
            # allocation equivalence. It is not another trajectory or a timing claim.
            copied = []
            for reference in refs:
                slot = read_model(reference, seed)
                copied.append(state_digest(slot.state_dict()))
                del slot
            if copied != [reference["tensor_sha256"] for reference in refs]:
                raise RuntimeError("preallocated slots did not restore the same ordered expert tensors")
            preallocation.append({"init_seed": seed, "status": "verified",
                                  "kind": "construction_equivalence_not_replication",
                                  "ordered_tensor_sha256": copied,
                                  "same_physical_storage_bytes_claimed": False})
            for policy in ("adaptive_novelty", "adaptive_resonance"):
                trajectory = by_key[(policy, seed)]
                banks.append({"bank_id": f"{policy}_i{seed}", "init_seed": seed, "kind": policy,
                              "experts": trajectory["final_experts"], "fit_gate": True})

        def combined(data):
            canvas = torch.cat([row[0] for row in data])
            target = torch.cat([row[1] for row in data]).numpy()
            roles = torch.cat([row[2] for row in data]).numpy()
            topic = np.concatenate([np.full(len(row[0]), index, dtype=np.int64) for index, row in enumerate(data)])
            features = operations.input_features(canvas.numpy(), tasks[0].query_pos, tasks[0].role_ids.numpy())
            return canvas, target, roles, topic, features

        cache = {}

        def bank_outputs(bank, canvas, phase):
            values, scores = [], []
            for reference in bank["experts"]:
                key = phase, reference["tensor_sha256"]
                if key not in cache:
                    resource_sample(device, minimum_ram_gib)
                    model = read_model(reference, bank["init_seed"])
                    cache[key] = expert_outputs(model, canvas)
                    del model
                values.append(cache[key]["log_probabilities"])
                scores.append(cache[key]["scanner"])
            return np.stack(values), np.stack(scores)

        fit_canvas, fit_target, fit_roles, fit_topic, fit_features = combined(fit_data)
        gates = {}
        for bank in banks:
            if not bank["fit_gate"]:
                continue
            check_source()
            logp, scanner = bank_outputs(bank, fit_canvas, "gate_fit")
            fitting_predictions = save_npz(f"banks/{bank['bank_id']}/gate_fit_experts.npz",
                                          log_probabilities=logp, scanner=scanner, target=fit_target,
                                          role=fit_roles, topic=fit_topic)
            fitted = operations.fit_gate(fit_features, logp, fit_target)
            weights_record = save_npz(f"banks/{bank['bank_id']}/gate_weights.npz", weights=fitted["weights"])
            gates[bank["bank_id"]] = fitted["weights"]
            bank["gate"] = {key: value for key, value in fitted.items() if key != "weights"}
            bank["gate"].update(weights=weights_record, expert_outputs=fitting_predictions,
                                data=fit_records, final_test_accessed=False)
            save_json(output / "banks" / bank["bank_id"] / "gate.json", bank["gate"])
            event("growth_gate_fitted", bank=bank["bank_id"], weights=weights_record)
        def authorize_final():
            save_json(output / "training_bundle.json", {"trajectories": trajectories, "banks": banks,
                      "preallocation_checks": preallocation, "source": source, "controller_source": controller})
            check_source()
            artifact_hashes = {
                path.relative_to(output).as_posix(): core.sha256(path)
                for path in output.rglob("*") if path.is_file() and path.name != "events.json"
            }
            # Verify every saved checkpoint reference, including duplicated logical slots.
            for bank in banks:
                for reference in bank["experts"]:
                    if core.sha256(output / reference["path"]) != reference["sha256"]:
                        raise OSError("expert checkpoint changed before final authorization")
            gate = {"at_utc": utc_now(), "completed_trajectories": 6, "completed_training_stages": 18,
                    "completed_gate_fits": 8, "source": source, "controller_source": controller,
                    "development_artifacts_sha256": artifact_hashes}
            save_json(output / "evaluation_gate.json", gate)
            event("growth_final_evaluation_authorized", **gate)
            return artifact_hashes

        def create_final_datasets():
            # First test-example construction, after the complete development gate.
            resource_sample(device, minimum_ram_gib)
            test_data, test_records = [], []
            for stage, task in enumerate(tasks):
                settings = specification["stages"][stage]
                data, descriptor = dataset_artifact(task, "test", settings["test_n"], settings["test_seed"],
                                                    output / "datasets" / f"topic{stage}")
                test_data.append(data)
                test_records.append({**portable(descriptor), "topic": stage})
            event("growth_final_datasets_opened", datasets=test_records)
            return test_data, test_records

        artifact_hashes, (test_data, test_records) = enter_final_phase(
            trajectories, banks, preallocation, authorize_final, create_final_datasets)
        test_canvas, target, roles, topic, features = combined(test_data)
        evaluations, bank_records = [], []
        for bank in banks:
            check_source()
            resource_sample(device, minimum_ram_gib)
            logp, scanner = bank_outputs(bank, test_canvas, "test")
            outputs_record = save_npz(f"banks/{bank['bank_id']}/final_experts.npz",
                                     log_probabilities=logp, scanner=scanner, target=target, role=roles, topic=topic)
            hard_oracle = operations.metrics_from_correct(np.any(logp.argmax(-1) == target[None, :], axis=0), roles, topic)
            parameters = len(bank["experts"]) * operations.EXPERT_PARAMETERS
            unique = {row["sha256"]: row for row in bank["experts"]}
            bank_record = {**bank, "final_expert_outputs": outputs_record,
                           "logical_expert_slots": len(bank["experts"]),
                           "nominal_expert_parameters": parameters,
                           "unique_checkpoint_files": len(unique),
                           "unique_checkpoint_file_bytes": sum(row["bytes"] for row in unique.values()),
                           "unique_checkpoint_tensor_parameters": len(unique) * operations.EXPERT_PARAMETERS,
                           "hard_selection_oracle": hard_oracle,
                           "oracle_scope": specification["hard_selection_oracle_scope"]}
            policies = ["uniform"] if bank["kind"] == "single_final" else operations.ROUTERS
            for policy in policies:
                routed = operations.route_experts(logp, policy, scanner=scanner, features=features,
                                                  weights=gates.get(bank["bank_id"]), random_seed=62013)
                metrics = operations.score_routing(routed, target, roles, topic)
                mixed = routed["log_probabilities"]
                predictions = save_npz(f"evaluations/{bank['bank_id']}_{policy}/predictions.npz",
                                       prediction=routed["prediction"], target=target, role=roles, topic=topic,
                                       confidence=np.exp(mixed).max(axis=1),
                                       nll=-mixed[np.arange(len(target)), target],
                                       expert_pick=routed["expert_pick"], routing_weights=routed["routing_weights"])
                row = {"status": "completed", "bank_id": bank["bank_id"], "kind": bank["kind"],
                       "init_seed": bank["init_seed"], "router": policy, "metrics": metrics,
                       "predictions": predictions, "expert_slots": len(bank["experts"]),
                       "nominal_expert_parameters": parameters,
                       "gate_fitted_parameter_count": bank["gate"]["parameter_count"] if policy == "learned" else 0,
                       "unique_expert_evaluations_required_per_input": len(unique),
                       "counterfactual_policy_outputs_from_same_expert_tensor": True,
                       "routing_frequency_source": "expert_pick" if policy in ("random", "scanner") else "routing_weights",
                       "expert_pick_is_hard_selection": policy in ("random", "scanner"),
                       "source": source, "controller_source": controller, "evaluated_at_utc": utc_now()}
                save_json(output / "evaluations" / f"{bank['bank_id']}_{policy}" / "evaluation.json", row)
                evaluations.append(row)
                event("growth_final_saved", bank=bank["bank_id"], router=policy)
                print(json.dumps({"event": "growth_final", "bank": bank["bank_id"], "router": policy,
                                  "binding": metrics["macro_agent_patient_accuracy"]}), flush=True)
            bank_records.append(bank_record)
        if len(evaluations) != 34:
            raise RuntimeError("the final routing inventory is incomplete")
        check_source()
        for relative, digest in artifact_hashes.items():
            if core.sha256(output / relative) != digest:
                raise OSError(f"a development artifact changed after final authorization: {relative}")
        event("growth_completed")
        result = {"schema_version": 1, "item": 6, "panel": "growth", "status": "completed",
                  "source": source, "controller_source": controller, "environment": environment_record(device),
                  "specification": specification, "partitions": partitions,
                  "diagnostic_datasets": probe_records, "gate_fit_datasets": fit_records,
                  "final_datasets": test_records, "trajectories": trajectories, "banks": bank_records,
                  "evaluations": evaluations, "preallocation_checks": preallocation,
                  "complete_training_stages": 18, "complete_gate_fits": 8, "complete_routing_evaluations": 34,
                  "total_optimizer_updates": sum(row["optimizer_updates"] for row in trajectories),
                  "total_training_seconds": sum(row["training_seconds"] for row in trajectories),
                  "total_wall_seconds": time.perf_counter() - started, "events": events,
                  "h1_eligible": False, "statistical_analysis": "pending separate recount of saved arrays",
                  "inference_latency_seconds": None, "physical_energy_joules": None,
                  "limitations": [
                      "Historical growth duplicates model weights; it does not enlarge a spatial canvas.",
                      "Adaptive K and parent lineage are outcomes. Adaptive/fixed differences include capacity and allocation history.",
                      "All stages use 512 updates and fixed LR with optimizer resets; historical shorter reviews and schedules are not reproduced.",
                      "Snapshot retention is preservation of trained models, not evidence of persistent activation memory.",
                      "Duplicated slots control nominal storage slots, not independent functional capacity or specialization.",
                      "Learned routing adds supervised fitting and useful degrees of freedom; dummy/frozen weights would not equalize those.",
                      "Hard-selection oracle scores bound hard selectors only, not probability mixtures.",
                      "Preallocation checks establish functional construction equivalence, not independent training replications.",
                      "Routing counterfactuals share cached expert outputs. Physical latency and energy are not measured.",
                      "Two seeds and three toy noun groups provide exploratory diagnostics, not a broad continual-learning benchmark.",
                  ]}
        save_json(output / "growth_records.json", result)
        return result
    except Exception as error:
        event("growth_failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
                  "source": source, "controller_source": controller,
                  "final_gate_exists": (output / "evaluation_gate.json").exists(),
                  "at_utc": utc_now(), "artifacts_preserved": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--plan", type=Path, default=ROOT / "docs/research/06_execution_plan.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--minimum-ram-gib", type=float, default=8.0)
    args = parser.parse_args()
    if args.describe:
        core.load_protocol()
        print(json.dumps(operations.growth_inventory(), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for execution")
    if (not math.isfinite(args.minimum_ram_gib) or args.minimum_ram_gib < 8
            or not 1 <= args.threads <= 4):
        raise ValueError("one to four threads and the unchanged 8 GiB RAM floor are required")
    load_execution_plan(args.plan)
    import psutil
    if psutil.virtual_memory().available / 2**30 < args.minimum_ram_gib:
        raise RuntimeError("available RAM is below the declared floor; Torch was not imported")
    from neuropixel.research.experiment import configure_runtime, environment_record, resource_sample
    device = configure_runtime(args.device, args.threads)
    print(json.dumps({"event": "growth_preflight", "resources": resource_sample(device, args.minimum_ram_gib),
                      "environment": environment_record(device)}), flush=True)
    run_growth(args.output, args.plan, device, args.minimum_ram_gib)


if __name__ == "__main__":
    main()
