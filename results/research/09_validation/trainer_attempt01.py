"""Frozen, bounded item-9 development, training and consume-once evaluation.

Execute only through the reviewed source plan and CPU supervisor. Historical
experiment controllers and results are not imported or reinterpreted here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from neuropixel.model import NeuroPixel, n_params
from neuropixel.research.models import RelativeTransformer
from neuropixel.research.complex_binding_metrics import point_metrics
from neuropixel.research import complex_binding_data as data
from neuropixel.research.complex_binding_protocol import (
    FAMILIES, artifact, canonical_bytes, claim_final_access,
    create_final_inventory, expected_primary, read_json, select_pilot,
    sha256, utc_now, verify_artifact, write_new_json,
)


class ResourceStop(RuntimeError):
    pass


def resource_sample(label):
    row = {"at_utc": utc_now(), "phase": label,
           "available_ram_gib": psutil.virtual_memory().available / 2**30}
    if row["available_ram_gib"] < 8:
        raise ResourceStop(f"RAM floor violated: {row['available_ram_gib']}")
    return row


def configure_runtime(recipe):
    runtime = recipe["runtime"]
    if platform.python_version() != runtime["python"]:
        raise ValueError("Python version differs from frozen CPU recipe")
    versions = {name: importlib.metadata.version(name) for name in runtime["versions"]}
    if versions != runtime["versions"] or torch.version.cuda is not None:
        raise ValueError("runtime versions differ or Torch is not CPU-only")
    torch.set_num_threads(runtime["intra_threads"])
    if torch.get_num_interop_threads() != runtime["inter_threads"]:
        torch.set_num_interop_threads(runtime["inter_threads"])
    torch.use_deterministic_algorithms(True)
    return {"python": platform.python_version(), "versions": versions,
            "platform": platform.platform(), "device": "cpu",
            "torch_cuda_version": torch.version.cuda,
            "intra_threads": torch.get_num_threads(),
            "inter_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


@contextmanager
def new_binary(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="_pending_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_arrays(path, **values):
    values = {name: np.asarray(value) for name, value in values.items()}
    with new_binary(path) as stream:
        np.savez_compressed(stream, **values)
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(values):
            raise OSError("array names changed on disk")
        for name, value in values.items():
            if not np.array_equal(stored[name], value):
                raise OSError(f"array persistence mismatch: {name}")


def save_gzip_json(path, value):
    raw = canonical_bytes(value)
    with new_binary(path) as stream:
        stream.write(gzip.compress(raw, compresslevel=6, mtime=0))
    if gzip.decompress(Path(path).read_bytes()) != raw:
        raise OSError("compressed JSON persistence mismatch")


def load_gzip_json(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def tensor_content_digest(canvas, target):
    digest = hashlib.sha256()
    for name, value in (("canvas", canvas), ("target", target)):
        a = np.ascontiguousarray(value, dtype="<i8")
        header = canonical_bytes({"name": name, "shape": list(a.shape), "dtype": "<i8"})
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(a.tobytes())
    return digest.hexdigest()


def model_for(family, recipe):
    spec = dict(recipe["models"][family])
    spec.pop("class")
    common = {"vocab": data.VOCAB_SIZE, "out_pos": tuple(recipe["data"]["out_pos"])}
    if family == "neuropixel":
        model = NeuroPixel(**common, **spec)
        if type(model) is not NeuroPixel or torch.count_nonzero(model.dictionary()[0]).item():
            raise ValueError("current core and its corrected PAD policy are required")
    elif family == "relative_transformer":
        model = RelativeTransformer(**common, h=data.HEIGHT, w=data.WIDTH, **spec)
    else:
        raise ValueError("unknown declared model family")
    return model


def render_panel(scenarios, conditions):
    records = []
    for ci, condition in enumerate(conditions):
        for gi, scenario in enumerate(scenarios):
            for event in range(2):
                for role in range(4):
                    row = data.render_scenario(scenario, event, role, condition)
                    row.update(condition_index=ci, group_index=gi)
                    records.append(row)
    return records


def panel_arrays(records):
    return {"canvas": np.asarray([r["canvas"] for r in records], dtype=np.int64),
            "target": np.asarray([r["target"] for r in records], dtype=np.int64),
            "base_target": np.asarray([r["base_target"] for r in records], dtype=np.int64),
            "role": np.asarray([r["query_role"] for r in records], dtype=np.int64),
            "query_event": np.asarray([r["query_event"] for r in records], dtype=np.int64),
            "base_query_event": np.asarray([r["base_query_event"] for r in records], dtype=np.int64),
            "group_index": np.asarray([r["group_index"] for r in records], dtype=np.int64),
            "condition_index": np.asarray([r["condition_index"] for r in records], dtype=np.int64),
            "changed_gold": np.asarray([r["changed_gold"] for r in records], dtype=np.bool_),
            "group_id": np.asarray([r["group_id"] for r in records]),
            "record_id": np.asarray([r["record_id"] for r in records]),
            "pair_id": np.asarray([r["pair_id"] for r in records])}


def predict(model, arrays, batch_size):
    was_training = model.training
    logits = []
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(arrays["target"]), batch_size):
                resource_sample("prediction_batch")
                batch = torch.from_numpy(arrays["canvas"][start:start + batch_size])
                value = model(batch)["logits"]
                if value.shape != (len(batch), data.VOCAB_SIZE) or not torch.isfinite(value).all():
                    raise ValueError("nonfinite or malformed evaluation logits")
                logits.append(value.detach().cpu())
        values = torch.cat(logits)
        target = torch.from_numpy(arrays["target"])
        nll = -F.log_softmax(values.to(torch.float64), dim=-1).gather(1, target[:, None]).squeeze(1)
        return {"logits": values.numpy(), "pred": values.argmax(-1).numpy(), "nll": nll.numpy()}
    finally:
        model.train(was_training)


def dataset_support(scenarios):
    by_role = {str(role): Counter() for role in range(4)}
    by_event_role = {f"{e}:{r}": Counter() for e in range(2) for r in range(4)}
    for scenario in scenarios:
        data.validate_scenario(scenario)
        for fact in scenario["facts"]:
            by_role[str(fact["role"])][str(fact["filler"])] += 1
            by_event_role[f"{fact['event']}:{fact['role']}"][str(fact["filler"])] += 1
    return {"per_role": {k: dict(sorted(v.items(), key=lambda x: int(x[0]))) for k, v in by_role.items()},
            "per_event_role": {k: dict(sorted(v.items(), key=lambda x: int(x[0]))) for k, v in by_event_role.items()}}


def prepare_development(output, recipe, expected_hashes=None):
    """Generate train/development only; never request final groups here."""
    cfg = recipe["data"]
    folder = output / "development_data"
    folder.mkdir()
    fixture_ledger = data.fixture_exposure_ledger()
    excluded = set(fixture_ledger["excluded_groups"])
    options = {"excluded_groups": excluded}
    write_new_json(folder / "fixture_exposure.json", fixture_ledger)
    train = data.generate_scenarios("train", cfg["train_bags"], cfg["train_seed"], **options)
    validation = data.generate_scenarios("validation", cfg["validation_bags"], cfg["validation_seed"], **options)
    rng = random.Random(cfg["probe_layout_seed"])
    probe = [data.randomize_scenario(scenario, rng) for scenario in train[:cfg["probe_bags"]]]
    rng = random.Random(cfg["memorization_layout_seed"])
    memorization = [data.randomize_scenario(scenario, rng) for scenario in train[:cfg["memorization_bags"]]]
    groups = {"train": {s["group_id"] for s in train}, "validation": {s["group_id"] for s in validation}}
    if groups["train"] & groups["validation"] or (groups["train"] | groups["validation"]) & excluded:
        raise ValueError("development group overlap or fixture exposure")
    scenarios = {"train": train, "validation": validation, "probe": probe, "memorization": memorization}
    files, arrays, identities = {}, {}, {}
    for name, value in scenarios.items():
        file = folder / f"{name}_scenarios.json.gz"
        save_gzip_json(file, value)
        files[file.name] = artifact(file, output)
        identities[file.name] = sha256(file)
        if name != "train":
            rows = render_panel(value, ("base",))
            arrays[name] = panel_arrays(rows)
            file = folder / f"{name}.npz"
            save_arrays(file, **arrays[name])
            files[file.name] = artifact(file, output)
            # Content identity does not depend on ZIP container timestamps.
            identities[file.name] = tensor_content_digest(arrays[name]["canvas"], arrays[name]["target"])
    if expected_hashes is not None and identities != expected_hashes:
        raise ValueError("regenerated development data differs from the frozen pilot identities")
    majority = {}
    support = dataset_support(train)["per_role"]
    for role in range(4):
        counts = {int(k): v for k, v in support[str(role)].items()}
        majority[str(role)] = min(counts, key=lambda token: (-counts[token], token))
    manifest = {"schema_version": 1, "item": 9, "identities": identities, "files": files,
                "counts": {name: len(value) for name, value in scenarios.items()},
                "support": {name: dataset_support(value) for name, value in scenarios.items()},
                "excluded_fixture_groups": sorted(excluded), "train_only_role_majority": majority,
                "final_performance_data_generated": False}
    write_new_json(folder / "manifest.json", manifest)
    return train, arrays, manifest


def load_development(output):
    folder = output / "development_data"
    manifest = read_json(folder / "manifest.json")
    for record in manifest["files"].values():
        verify_artifact(record, output)
    train = load_gzip_json(folder / "train_scenarios.json.gz")
    arrays = {}
    for name in ("probe", "validation", "memorization"):
        with np.load(folder / f"{name}.npz", allow_pickle=False) as src:
            arrays[name] = {key: src[key] for key in src.files}
    return train, arrays, manifest


def training_batch(train, rng, batch_size):
    rows = []
    chosen = []
    for _ in range(batch_size):
        i = rng.randrange(len(train))
        scenario = data.randomize_scenario(train[i], rng)
        row = data.render_scenario(scenario, rng.randrange(2), rng.randrange(4))
        rows.append(row)
        chosen.append(i)
    canvas = np.asarray([row["canvas"] for row in rows], dtype=np.int64)
    target = np.asarray([row["target"] for row in rows], dtype=np.int64)
    return canvas, target, chosen


def run_training(output, run_id, phase, family, seed, rate, budget, recipe,
                 context, train, panels, *, memorize=False):
    directory = output / "runs" / run_id
    directory.mkdir(parents=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"run already has evidence: {run_id}")
    cfg = recipe["training"]
    started, samples = time.monotonic(), [resource_sample("run_admission")]
    torch.manual_seed(seed)
    model = model_for(family, recipe)
    configuration = {"family": family, "model": recipe["models"][family],
                     "vocab": data.VOCAB_SIZE, "height": data.HEIGHT, "width": data.WIDTH,
                     "out_pos": list(data.OUT_POS), "trainable_parameters": n_params(model),
                     "initialization": seed, "learning_rate": rate,
                     "optimizer": "AdamW", "weight_decay": cfg["weight_decay"],
                     "gradient_clip": cfg["gradient_clip"], "batch_size": cfg["batch_size"],
                     "loss": cfg["loss"], "scheduled_updates": budget, "memorization": memorize}
    write_new_json(directory / "configuration.json", configuration)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=cfg["weight_decay"])
    rng = random.Random(cfg["minibatch_seed_offset"] + seed)
    torch.manual_seed(cfg["update_rng_seed_offset"] + seed)
    aggregate_digest, checks = hashlib.sha256(), []
    completed, streak, criterion = 0, 0, False
    write_new_json(directory / "started.json", {"at_utc": utc_now(), "context": context,
                                               "configuration": configuration})
    model.train()
    try:
        with (directory / "training.jsonl").open("x", buffering=1) as stream:
            for step in range(1, budget + 1):
                if step == 1 or step % 32 == 0:
                    samples.append(resource_sample(f"step_{step}"))
                if memorize:
                    a = panels["memorization"]
                    canvas, target = a["canvas"], a["target"]
                    chosen = list(range(len(canvas)))
                else:
                    canvas, target, chosen = training_batch(train, rng, cfg["batch_size"])
                digest = tensor_content_digest(canvas, target)
                aggregate_digest.update(bytes.fromhex(digest))
                optimizer.zero_grad(set_to_none=True)
                values = model(torch.from_numpy(canvas))["logits"]
                loss = F.cross_entropy(values, torch.from_numpy(target))
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"],
                                                          error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise FloatingPointError("nonfinite model parameter after update")
                completed = step
                line = {"step": step, "loss": float(loss.detach()), "gradient_norm_before_clip": float(grad_norm),
                        "input_sha256": digest, "training_group_indices": chosen,
                        "elapsed_seconds": time.monotonic() - started}
                stream.write(json.dumps(line, sort_keys=True, allow_nan=False) + "\n")
                if step % cfg["training_progress_every"] == 0 or step == budget:
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(json.dumps({"run": run_id, **line}, sort_keys=True), flush=True)
                if memorize and step % cfg["memorization_check_every"] == 0:
                    values = predict(model, panels["memorization"], cfg["evaluation_batch_size"])
                    a = panels["memorization"]
                    score = point_metrics(values["pred"], a["target"], a["role"], values["nll"], a["group_index"])
                    reached = (score["global"] >= cfg["memorization_min_accuracy"]
                               and score["cross_entropy"] <= cfg["memorization_max_ce"])
                    streak = streak + 1 if reached else 0
                    checks.append({"step": step, "metrics": score, "consecutive": streak})
                    if streak >= cfg["memorization_consecutive_checks"]:
                        criterion = True
                        break
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint = directory / "checkpoint.pt"
        with new_binary(checkpoint) as stream:
            torch.save({"state_dict": model.state_dict(), "configuration": configuration}, stream)
        endpoints, artifacts = {}, {"checkpoint": artifact(checkpoint, output),
                                   "training_log": artifact(directory / "training.jsonl", output)}
        targets = ("memorization",) if memorize else ("probe", "validation")
        for name in targets:
            a = panels[name]
            values = predict(model, a, cfg["evaluation_batch_size"])
            endpoints[name] = point_metrics(values["pred"], a["target"], a["role"], values["nll"], a["group_index"])
            path = directory / f"{name}_predictions.npz"
            save_arrays(path, **values, target=a["target"], role=a["role"], group_index=a["group_index"],
                        record_id=a["record_id"])
            artifacts[f"{name}_predictions"] = artifact(path, output)
        samples.append(resource_sample("run_completion"))
        result = {"schema_version": 1, "item": 9, "run_id": run_id, "phase": phase,
                  "status": "completed", "family": family, "initialization": seed,
                  "learning_rate": rate, "updates_completed": completed, "context": context,
                  "configuration": configuration, "artifacts": artifacts,
                  "training_input_sequence_sha256": aggregate_digest.hexdigest(),
                  "wall_seconds": time.monotonic() - started, "resource_samples": samples,
                  "memorization_checks": checks, "memorization_criterion_reached": criterion if memorize else None,
                  **endpoints}
        if not memorize:
            result["probe_below_95_percent_binding"] = endpoints["probe"]["binding"] < cfg["training_probe_binding_flag_below"]
        write_new_json(directory / "run.json", result)
        return result
    except Exception as error:
        failure = {"schema_version": 1, "item": 9, "run_id": run_id, "phase": phase,
                   "status": "failed", "family": family, "initialization": seed,
                   "learning_rate": rate, "updates_completed": completed, "context": context,
                   "type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "wall_seconds": time.monotonic() - started,
                   "resource_samples": samples}
        write_new_json(directory / "failure.json", failure)
        if isinstance(error, ResourceStop):
            raise
        return failure
    finally:
        del optimizer, model
        gc.collect()


def controls_for(records, output, majority):
    controls = {name: [] for name in data.CONTROLS}
    controls["train_role_majority"] = []
    for row in records:
        for name in data.CONTROLS:
            weights = data.control_weights(row["canvas"], name)
            if abs(sum(weights.values()) - 1) > 1e-14:
                raise ValueError("control probabilities do not sum to one")
            controls[name].append(weights.get(row["target"], 0.0))
        controls["train_role_majority"].append(float(majority[str(row["query_role"])] == row["target"]))
    if set(controls["symbolic"]) != {1.0}:
        raise ValueError("input-only exact control failed")
    save_arrays(output, **{name: np.asarray(values, dtype=np.float64) for name, values in controls.items()})


def preflight(output, recipe, plan, context):
    train, panels, manifest = prepare_development(output, recipe)
    context["development_identities"] = manifest["identities"]
    write_new_json(output / "context.json", context)
    rows = render_panel(load_gzip_json(output / "development_data" / "validation_scenarios.json.gz"), ("base",))
    controls_for(rows, output / "development_controls.npz", manifest["train_only_role_majority"])
    cfg, memorization, pilots = recipe["training"], [], []
    for family in FAMILIES:
        memorization.append(run_training(output, f"memorization_{family}", "memorization", family,
                                         cfg["memorization_initialization"], cfg["memorization_lr"],
                                         cfg["memorization_max_updates"], recipe, context, train, panels, memorize=True))
    for family in FAMILIES:
        for rate in cfg["pilot_learning_rates"]:
            name = f"pilot_{family}_lr{rate:g}".replace(".", "p")
            pilots.append(run_training(output, name, "pilot", family, cfg["pilot_initialization"],
                                       rate, cfg["pilot_updates"], recipe, context, train, panels))
    write_new_json(output / "preflight_runs.json", {"memorization": memorization, "pilots": pilots})
    if any(row["status"] != "completed" for row in memorization + pilots):
        raise RuntimeError("a declared preflight run failed; all eligible runs and failures are retained")
    rates = select_pilot(pilots, recipe)
    selection = {"schema_version": 1, "item": 9, "selected_at_utc": utc_now(),
                 "context": context, "selected_learning_rates": rates,
                 "rule": cfg["lr_selection"], "pilot_runs": [row["run_id"] for row in pilots],
                 "pilot_records": artifact(output / "preflight_runs.json", output),
                 "development_identities": manifest["identities"],
                 "final_performance_data_generated": False,
                 "memorization_flags": {row["family"]: not row["memorization_criterion_reached"] for row in memorization}}
    write_new_json(output / "pilot_selection.json", selection)
    return {"status": "completed", "selected_learning_rates": rates,
            "memorization_completed": 2, "pilot_completed": 4,
            "final_performance_data_generated": False}


def train_primary(output, recipe, plan, context):
    evidence = plan["pilot_evidence"]
    rates = evidence["selected_learning_rates"]
    train, panels, manifest = prepare_development(output, recipe, evidence["development_identities"])
    context["development_identities"] = manifest["identities"]
    write_new_json(output / "context.json", context)
    records = []
    for spec in expected_primary(recipe, rates):
        records.append(run_training(output, spec["run_id"], "primary", spec["family"],
                                    spec["initialization"], spec["learning_rate"], spec["updates"],
                                    recipe, context, train, panels))
    success = all(row["status"] == "completed" for row in records)
    write_new_json(output / "primary_runs.json", {"status": "completed" if success else "incomplete",
                                                  "runs": [row["run_id"] for row in records],
                                                  "statuses": {row["run_id"]: row["status"] for row in records}})
    if not success:
        raise RuntimeError("incomplete primary panel; no final access is permitted")
    create_final_inventory(output, recipe, rates, context)
    return {"status": "completed", "primary_runs": len(records), "final_performance_data_generated": False}


def final_evaluation(output, recipe, plan, initial_context):
    context = read_json(output / "context.json")
    if any(context[key] != value for key, value in initial_context.items()):
        raise ValueError("final phase source/plan context differs from training")
    _, _, manifest = load_development(output)
    if manifest["identities"] != context["development_identities"]:
        raise ValueError("development identity changed before final access")
    gate = claim_final_access(output, recipe, plan["pilot_evidence"]["selected_learning_rates"], context)
    # The consume-once receipt above is durable before this first final request.
    folder = output / "final_data"
    folder.mkdir()
    cfg = recipe["data"]
    excluded = set(manifest["excluded_fixture_groups"])
    options = {"excluded_groups": excluded}
    scenarios = data.generate_scenarios("final", cfg["final_bags"], cfg["final_seed"], **options)
    train_ids = {s["group_id"] for s in load_gzip_json(output / "development_data" / "train_scenarios.json.gz")}
    val_ids = {s["group_id"] for s in load_gzip_json(output / "development_data" / "validation_scenarios.json.gz")}
    final_ids = {s["group_id"] for s in scenarios}
    if len(final_ids) != cfg["final_bags"] or final_ids & (train_ids | val_ids | excluded):
        raise ValueError("final groups are duplicate, exposed fixtures, or overlap development")
    save_gzip_json(folder / "scenarios.json.gz", scenarios)
    records = render_panel(scenarios, cfg["final_conditions"])
    arrays = panel_arrays(records)
    save_gzip_json(folder / "records.json.gz", records)
    save_arrays(folder / "examples.npz", **arrays)
    controls_for(records, folder / "controls.npz", manifest["train_only_role_majority"])
    counts = Counter(bytes(row.astype(np.uint8)) for row in arrays["canvas"].reshape(len(records), -1))
    write_new_json(folder / "manifest.json", {"schema_version": 1, "item": 9,
                     "created_after_access_sha256": sha256(output / "final_access.json"),
                     "groups": len(scenarios), "conditions": cfg["final_conditions"],
                     "nominal_rows": len(records), "unique_canvases": len(counts),
                     "duplicate_multiplicity_counts": dict(sorted(Counter(counts.values()).items())),
                     "support": dataset_support(scenarios),
                     "input_target_digest": tensor_content_digest(arrays["canvas"], arrays["target"]),
                     "files": [artifact(folder / name, output) for name in
                               ("scenarios.json.gz", "records.json.gz", "examples.npz", "controls.npz")],
                     "shared_tokens_allowed": True, "scenario_resampling_unit": True})
    cfg_train, summaries = recipe["training"], []
    for entry in gate["entries"]:
        path = verify_artifact(entry["checkpoint"], output)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = model_for(entry["family"], recipe)
        expected_config = read_json(verify_artifact(entry["summary"], output))["configuration"]
        if checkpoint["configuration"] != expected_config:
            raise ValueError("checkpoint metadata differs from its gated run summary")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        started = time.monotonic()
        values = predict(model, arrays, cfg_train["evaluation_batch_size"])
        path = output / "final_predictions" / f"{entry['run_id']}.npz"
        save_arrays(path, **values, target=arrays["target"], role=arrays["role"],
                    group_index=arrays["group_index"], condition_index=arrays["condition_index"],
                    record_id=arrays["record_id"])
        condition_metrics = {}
        for ci, name in enumerate(cfg["final_conditions"]):
            pick = arrays["condition_index"] == ci
            condition_metrics[name] = point_metrics(values["pred"][pick], arrays["target"][pick], arrays["role"][pick],
                                                   values["nll"][pick], arrays["group_index"][pick])
        summary = {"run_id": entry["run_id"], "family": entry["family"],
                   "initialization": entry["initialization"], "condition_metrics": condition_metrics,
                   "predictions": artifact(path, output), "checkpoint": entry["checkpoint"],
                   "evaluation_wall_seconds": time.monotonic() - started}
        write_new_json(output / "final_predictions" / f"{entry['run_id']}.json", summary)
        summaries.append(summary)
        print(json.dumps({"final_completed": entry["run_id"], "base": condition_metrics["base"]}), flush=True)
        del model, checkpoint, values
        gc.collect()
    write_new_json(output / "final_summary.json", {"schema_version": 1, "item": 9,
                   "status": "completed", "context": context, "runs": summaries,
                   "final_inventory_sha256": sha256(output / "final_inventory.json"),
                   "final_access_sha256": sha256(output / "final_access.json"),
                   "secondary_intervals": "Independent post-run analysis from these saved arrays; no new model selection."})
    return {"status": "completed", "checkpoints_evaluated": len(summaries), "nominal_rows_per_checkpoint": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "train", "final"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path, output = args.plan.resolve(), args.output.resolve()
    plan = read_json(plan_path)
    expected_phase = "preflight" if args.phase == "preflight" else "study"
    if (plan.get("item") != 9 or plan.get("status") != "frozen" or plan.get("phase") != expected_phase
            or plan.get("recipe_path") != "docs/research/09_experiment_recipe.json"):
        raise ValueError("item, phase, status or recipe path differs")
    recipe_path = ROOT / plan["recipe_path"]
    if sha256(recipe_path) != plan["recipe_sha256"]:
        raise ValueError("recipe does not match frozen plan")
    for name, expected in plan["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"frozen implementation differs: {name}")
    recipe = read_json(recipe_path)
    if tuple(recipe["data"]["final_conditions"]) != data.CONDITIONS:
        raise ValueError("condition recipe differs from generator")
    runtime = configure_runtime(recipe)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=30).strip()
    context = {"source_commit": commit, "plan_sha256": sha256(plan_path),
               "recipe_sha256": sha256(recipe_path)}
    if args.phase != "final":
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("training output is absent")
    write_new_json(output / f"{args.phase}_started.json", {"at_utc": utc_now(), "context": context,
                                                          "runtime": runtime, "resource": resource_sample("phase_admission")})
    started = time.monotonic()
    try:
        result = {"preflight": preflight, "train": train_primary, "final": final_evaluation}[args.phase](
            output, recipe, plan, context)
        write_new_json(output / f"{args.phase}_completed.json", {**result, "at_utc": utc_now(),
                                                               "wall_seconds": time.monotonic() - started,
                                                               "resource": resource_sample("phase_completion")})
        return 0
    except Exception as error:
        write_new_json(output / f"{args.phase}_failed.json", {"at_utc": utc_now(), "type": type(error).__name__,
                                                             "message": str(error), "traceback": traceback.format_exc(),
                                                             "wall_seconds": time.monotonic() - started})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
