"""Bounded learned stream-memory study for item 11.

This controller trains fresh models on a known 48-element cue census. It does
not claim novel content, native across-call state, indefinite memory, or a
replacement for historical H1. Final interventions are caller-owned state
operations in the explicitly named research stream helper. All output arrays
remain outside the ten frozen training subtrees.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILIES = ("neuropixel", "gru")
POSITIONS = tuple((r, c) for r in range(3) for c in range(3) if (r, c) != (2, 2))
NORMAL_DELAYS = (0, 2, 4, 8, 16, 32, 64)
RECIPE = {
    "schema_version": 1,
    "task": {"vocab": 8, "pad": 0, "query": 1, "answer_tokens": [2, 3, 4, 5, 6, 7],
             "height": 3, "width": 3, "out_pos": [2, 2], "query_pos": [2, 1],
             "cue_positions": [list(p) for p in POSITIONS],
             "census_order": "target token outer; row-major cue position inner",
             "census_n": 48, "fact_updates": 3, "gap_updates": 3, "query_updates": 10},
    "models": {
        "neuropixel": {"c_id": 16, "c": 48, "hidden": 128, "fire_rate": 0.5,
                      "grounded": False, "retina": False, "parameters": 29392,
                      "state_values_per_example": 432, "state_payload_bytes_float32": 1728},
        "gru": {"e": 8, "d": 48, "hid": 70, "parameters": 29336,
                "state_values_per_example": 70, "state_payload_bytes_float32": 280}},
    "optimization": {"optimizer": "AdamW", "weight_decay": 0.0001,
                     "gradient_clip": 1.0, "batch_size": 32,
                     "schedule": "constant", "loss": "answer_cross_entropy_only"},
    "randomness": {"generator": "PCG64", "index_seed_offset": 100000,
                   "firing_seed_offset": 110000, "train_delays": [0, 1, 2],
                   "delay_unit": "one independently sampled delay per minibatch",
                   "sample_unit": "uniform replacement from 48 target-position pairs"},
    "preflight": {"seed": 69, "updates": 1024, "learning_rates": [0.001, 0.003],
                  "development_delays": [3, 4], "competence_delay": 2,
                  "competence_threshold": 0.95,
                  "selection": ["mean_dev_accuracy_desc", "mean_dev_ce_asc", "lr_asc"],
                  "admission": "both selected pilots pass the full 48-row delay-2 census"},
    "main": {"seeds": [70, 71, 72, 73, 74], "updates": 2048},
    "evaluation": {
        "normal_delays": list(NORMAL_DELAYS), "intervention_delays": [2, 8],
        "interventions": ["cue_erased", "zero", "swap"],
        "state_boundary": "immediately before query, after all cue and gap frames",
        "swap": "state[recipient] = state[donor]; next cyclic target at same cue position",
        "interference_delay": 8, "interference_cases": ["blank_repeated", "same_position", "next_position"],
        "distractors": "all 6 target x 8 cue-position x 6 independent distractor labels",
        "distractor_frame": "replace first blank frame; unchanged frame/update inventory",
        "interference_strata": ["all", "same_label", "different_label"],
        "case_count_per_checkpoint": 16, "primary_delay": 8,
        "primary": "five paired NeuroPixel-minus-GRU normal-delay8 accuracy differences",
        "near_competence_threshold": 0.95, "retention_threshold": 0.95},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_json(path, value):
    with Path(path).open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()


def relative_path(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    require(path.is_relative_to(root), "artifact escapes declared root")
    return path.relative_to(root).as_posix()


def descriptor(root, path):
    return {"path": relative_path(root, path), "bytes": Path(path).stat().st_size, "sha256": sha(path)}


def verify_reference(root, reference):
    require(isinstance(reference, dict) and {"path", "bytes", "sha256"} <= set(reference),
            "incomplete artifact reference")
    path = (Path(root) / reference["path"]).resolve()
    relative_path(root, path)
    require(path.is_file() and path.stat().st_size == reference["bytes"]
            and sha(path) == reference["sha256"], "artifact identity differs: " + reference["path"])
    return path


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "refusing artifact overwrite")
    pending = path.with_name("_pending_" + path.name)
    with pending.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, timeout=30,
                                   stdin=subprocess.DEVNULL).decode("utf-8").strip()


def source_record(plan_path, plan):
    require(not git("status", "--porcelain", "--untracked-files=no"), "tracked source is dirty")
    path = relative_path(ROOT, plan_path)
    committed = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT,
                                        timeout=30, stdin=subprocess.DEVNULL)
    require(committed == Path(plan_path).read_bytes(), "plan differs from committed HEAD")
    bindings = plan["implementation_sha256"]
    require(isinstance(bindings, dict) and bindings, "empty implementation map")
    for name, expected in bindings.items():
        file = (ROOT / name).resolve()
        relative_path(ROOT, file)
        require(file.is_file() and sha(file) == expected, "scientific source drift: " + name)
    return {"source_commit": git("rev-parse", "HEAD"), "plan_sha256": sha(plan_path),
            "implementation_sha256": bindings, "recorded_at_utc": now(), "tracked_source_clean": True}


def stable_source(source):
    return {key: source[key] for key in
            ("source_commit", "plan_sha256", "implementation_sha256", "tracked_source_clean")}


def admit():
    import psutil
    available = psutil.virtual_memory().available / 1024**3
    require(available >= 8.0, "available RAM is below 8 GiB")
    return available


def configure_runtime(plan):
    expected = plan["runtime"]
    require(platform.python_version() == expected["python"], "Python runtime differs")
    actual = {name: importlib.metadata.version(name) for name in expected["packages"]}
    require(actual == expected["packages"], "package runtime differs")
    require(expected["threads"] == 2 and expected["interop_threads"] == 1,
            "unexpected thread policy")
    admit()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1
            and torch.version.cuda is None, "actual CPU runtime differs")
    return torch, np, {"python": platform.python_version(), "packages": actual,
                      "threads": torch.get_num_threads(), "interop_threads": 1,
                      "cuda_version": torch.version.cuda, "available_ram_gib": admit()}


def validate_recipe(recipe):
    require(recipe == RECIPE and digest(recipe) == digest(RECIPE),
            "recipe differs from the prospectively declared item-11 design")


def census(delay, condition="normal"):
    """Return known task rows; no held-out-content claim is attached."""
    require(type(delay) is int and delay >= 0, "invalid blank-frame delay")
    require(condition in ("normal", "cue_erased", "zero", "swap",
                          "blank_repeated", "same_position", "next_position"), "unknown condition")
    interference = condition in ("blank_repeated", "same_position", "next_position")
    require(not interference or delay == 8, "interference is declared only at delay8")
    rows = []
    for target in range(2, 8):
        for position, (row, col) in enumerate(POSITIONS):
            for distractor in (range(2, 8) if interference else (-1,)):
                dpos = position if condition == "same_position" else ((position + 1) % 8)
                active = condition in ("same_position", "next_position")
                rows.append({"row_id": f"{condition}_d{delay}_t{target}_p{position}_x{distractor}",
                             "target": target, "cue_position_index": position,
                             "cue_position": [row, col], "delay": delay, "condition": condition,
                             "donor_target": 2 + (target - 2 + 1) % 6 if condition == "swap" else -1,
                             "distractor_target": distractor,
                             "distractor_position": list(POSITIONS[dpos]) if active else [-1, -1],
                             "distractor_present": active,
                             "same_label": distractor == target if interference else None})
    return rows


def evaluation_cases():
    cases = [{"case_id": f"normal_d{d}", "condition": "normal", "delay": d, "n": 48}
             for d in NORMAL_DELAYS]
    cases += [{"case_id": f"{kind}_d{d}", "condition": kind, "delay": d, "n": 48}
              for d in (2, 8) for kind in ("cue_erased", "zero", "swap")]
    cases += [{"case_id": f"{kind}_d8", "condition": kind, "delay": 8, "n": 288}
              for kind in ("blank_repeated", "same_position", "next_position")]
    require(len(cases) == 16, "evaluation inventory differs")
    return cases


def frames_for_rows(rows, torch):
    require(bool(rows), "empty rows")
    delay, condition = rows[0]["delay"], rows[0]["condition"]
    require(all(r["delay"] == delay and r["condition"] == condition for r in rows),
            "one prediction batch must have a common schedule/condition")
    frames = [torch.zeros(len(rows), 3, 3, dtype=torch.long) for _ in range(delay + 2)]
    for i, record in enumerate(rows):
        r, c = record["cue_position"]
        if condition != "cue_erased":
            frames[0][i, r, c] = record["target"]
        if record["distractor_present"]:
            dr, dc = record["distractor_position"]
            frames[1][i, dr, dc] = record["distractor_target"]
    frames[-1][:, 2, 1] = 1
    steps = [3] + [3] * delay + [10]
    intervention = None
    if condition == "zero":
        intervention = {"before_frame": len(frames) - 1, "kind": "zero"}
    elif condition == "swap":
        lookup = {(r["target"], r["cue_position_index"]): i for i, r in enumerate(rows)}
        order = [lookup[(r["donor_target"], r["cue_position_index"])] for r in rows]
        require(sorted(order) == list(range(len(rows))), "swap donor mapping is not a bijection")
        intervention = {"before_frame": len(frames) - 1, "kind": "swap", "permutation": order}
    return frames, steps, intervention


def make_model(family, torch):
    from neuropixel.model import NeuroPixel
    from neuropixel.phase3 import FrameGRU
    if family == "neuropixel":
        model = NeuroPixel(8, (2, 2), c_id=16, c=48, hidden=128, fire_rate=0.5)
    else:
        require(family == "gru", "unknown family")
        model = FrameGRU(8, 3, 3, e=8, d=48, hid=70)
    count = sum(p.numel() for p in model.parameters())
    require(count == RECIPE["models"][family]["parameters"], "actual parameter count differs")
    return model


def stream(model, family, frames, steps, intervention=None, capture=False):
    from neuropixel.research.stream_memory import strict_np_stream, strict_gru_stream
    if family == "neuropixel":
        return strict_np_stream(model, frames, steps, intervention=intervention, capture_states=capture)
    return strict_gru_stream(model, frames, intervention=intervention, capture_states=capture)


def metrics(prediction, target, losses, *, mask=None, donor=None, distractor=None):
    selected = list(range(len(target))) if mask is None else [i for i, value in enumerate(mask) if value]
    n = len(selected)
    correct = sum(int(prediction[i] == target[i]) for i in selected)
    result = {"n": n, "correct": correct, "accuracy": correct / n if n else None,
              "cross_entropy": sum(float(losses[i]) for i in selected) / n if n else None,
              "cross_entropy_n": n}
    if donor is not None:
        result["donor_correct"] = sum(int(prediction[i] == donor[i]) for i in selected)
        result["donor_accuracy"] = result["donor_correct"] / n if n else None
    if distractor is not None:
        result["distractor_correct"] = sum(int(prediction[i] == distractor[i]) for i in selected)
        result["distractor_accuracy"] = result["distractor_correct"] / n if n else None
    return result


def predict(model, family, rows, torch, np):
    """Evaluate the whole declared case; a swap must not be split into batches."""
    admit()
    was_training = model.training
    model.eval()
    frames, steps, intervention = frames_for_rows(rows, torch)
    try:
        with torch.inference_mode():
            result = stream(model, family, frames, steps, intervention, capture=True)
            value = result["logits"]
            target = torch.tensor([r["target"] for r in rows], dtype=torch.long)
            require(value.dtype == torch.float32 and tuple(value.shape) == (len(rows), 8)
                    and bool(torch.isfinite(value).all()), "malformed/nonfinite logits")
            losses = -torch.log_softmax(value.to(torch.float64), -1).gather(1, target[:, None]).squeeze(1)
            snapshots = result["frame_states"]
            before = snapshots[:, -2].cpu().numpy().astype(np.float32, copy=True)
            after = result["state"] if family == "neuropixel" else result["state"][0]
            require(bool(torch.isfinite(after).all()) and np.isfinite(before).all(), "nonfinite memory state")
            arrays = {
                "logits": value.cpu().numpy().copy(), "prediction": value.argmax(-1).cpu().numpy(),
                "target": target.cpu().numpy(), "nll": losses.cpu().numpy(),
                "row_id": np.asarray([r["row_id"] for r in rows], dtype="U80"),
                "frames": torch.stack(frames, 1).cpu().numpy(),
                "steps": np.asarray(steps, dtype=np.int64),
                "cue_position": np.asarray([r["cue_position"] for r in rows], dtype=np.int64),
                "donor_target": np.asarray([r["donor_target"] for r in rows], dtype=np.int64),
                "distractor_target": np.asarray([r["distractor_target"] for r in rows], dtype=np.int64),
                "distractor_position": np.asarray([r["distractor_position"] for r in rows], dtype=np.int64),
                "pre_query_state": before, "post_query_state": after.cpu().numpy().astype(np.float32, copy=True)}
            if intervention is not None:
                for key in ("intervention_before", "intervention_after"):
                    snapshot = result[key]
                    require(bool(torch.isfinite(snapshot).all()), "nonfinite intervention boundary")
                    # Preserve native state shape, including GRU's leading layer dimension.
                    arrays[key] = snapshot.cpu().numpy().astype(np.float32, copy=True)
                if intervention["kind"] == "swap":
                    arrays["donor_row_index"] = np.asarray(intervention["permutation"], dtype=np.int64)
    finally:
        model.train(was_training)
    donor = arrays["donor_target"] if rows[0]["condition"] == "swap" else None
    interference = rows[0]["condition"] in ("blank_repeated", "same_position", "next_position")
    distractor = arrays["distractor_target"] if interference else None
    scored = metrics(arrays["prediction"], arrays["target"], arrays["nll"],
                     donor=donor, distractor=distractor)
    scored["per_target"] = {str(t): metrics(arrays["prediction"], arrays["target"], arrays["nll"],
                                           mask=arrays["target"] == t) for t in range(2, 8)}
    scored["per_position"] = {str(p): metrics(arrays["prediction"], arrays["target"], arrays["nll"],
                            mask=[r["cue_position_index"] == p for r in rows]) for p in range(8)}
    if interference:
        scored["strata"] = {name: metrics(arrays["prediction"], arrays["target"], arrays["nll"],
                                          mask=mask, distractor=distractor)
                           for name, mask in (("same_label", arrays["target"] == distractor),
                                              ("different_label", arrays["target"] != distractor))}
    return arrays, scored


def save_prediction(output, directory, case_id, rows, model, family, torch, np):
    arrays, scored = predict(model, family, rows, torch, np)
    path = directory / (case_id + ".npz")
    atomic_write(path, lambda stream_: np.savez_compressed(stream_, **arrays))
    metadata = {"schema_version": 1, "case_id": case_id, "rows": rows,
                "rows_sha256": digest(rows), "predictions": descriptor(output, path),
                "metrics": scored, "frame_count": rows[0]["delay"] + 2,
                "np_updates": 13 + 3 * rows[0]["delay"],
                "gru_sequence_steps": rows[0]["delay"] + 2,
                "state_boundary": "saved pre_query_state is after last gap and BEFORE intervention/query"}
    meta_path = directory / (case_id + ".json")
    save_json(meta_path, metadata)
    return {"case_id": case_id, "metrics": scored, "predictions": descriptor(output, path),
            "metadata": descriptor(output, meta_path)}


def run_configuration(family, seed, lr, updates, kind):
    require(family in FAMILIES and kind in ("pilot", "train"), "unknown run configuration")
    run_id = (f"pilot_{family}_lr{int(round(lr * 1000)):04d}_s{seed}" if kind == "pilot"
              else f"train_{family}_s{seed}")
    return {"run_id": run_id, "family": family, "seed": seed, "learning_rate": lr,
            "updates": updates, "kind": kind, "batch_size": 32,
            "optimizer": "AdamW", "weight_decay": 0.0001, "gradient_clip": 1.0,
            "schedule": "constant", "loss": "answer_cross_entropy_only",
            "model": RECIPE["models"][family], "task": RECIPE["task"],
            "index_seed": 100000 + seed, "initialization_seed": seed, "firing_seed": 110000 + seed,
            "recipe_sha256": digest(RECIPE)}


def preflight_configs():
    return [run_configuration(f, 69, lr, 1024, "pilot")
            for f in FAMILIES for lr in (0.001, 0.003)]


def primary_configs(selection):
    return [run_configuration(f, seed, selection["families"][f]["learning_rate"], 2048, "train")
            for seed in range(70, 75) for f in FAMILIES]


def training_stream(config, np):
    generator = np.random.Generator(np.random.PCG64(config["index_seed"]))
    pairs = generator.integers(0, 48, size=(config["updates"], 32), dtype=np.int64)
    delays = generator.integers(0, 3, size=(config["updates"],), dtype=np.int64)
    return pairs, delays


def train_one(output, config, source, runtime, torch, np):
    all_time = time.perf_counter()
    directory = output / "study" / "runs" / config["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / "config.json"
    save_json(config_path, config)
    pairs, delays = training_stream(config, np)
    pair_path, delay_path = directory / "train_indices.npy", directory / "training_delays.npy"
    atomic_write(pair_path, lambda stream_: np.save(stream_, pairs, allow_pickle=False))
    atomic_write(delay_path, lambda stream_: np.save(stream_, delays, allow_pickle=False))
    torch.manual_seed(config["initialization_seed"])
    model = make_model(config["family"], torch)
    torch.manual_seed(config["firing_seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.0001)
    model.train()
    history_path = directory / "training_log.jsonl"
    losses, times, samples = [], [], []
    loop_time = time.perf_counter()
    for update in range(config["updates"]):
        if update % 128 == 0:
            samples.append({"update": update, "at_utc": now(), "available_ram_gib": admit()})
        begin = time.perf_counter()
        d = int(delays[update])
        base = census(d)
        rows = [base[int(i)] for i in pairs[update]]
        frames, steps, intervention = frames_for_rows(rows, torch)
        out = stream(model, config["family"], frames, steps, intervention)
        target = torch.tensor([r["target"] for r in rows], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(out["logits"], target)
        require(bool(torch.isfinite(loss)), "nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        elapsed = time.perf_counter() - begin
        losses.append(float(loss.detach()))
        times.append(elapsed)
        if (update + 1) % 128 == 0:
            entry = {"update": update + 1,
                     "mean_training_loss_since_last_log": statistics.mean(losses[-128:]),
                     "last_answer_loss": losses[-1], "last_unclipped_gradient_norm": float(norm),
                     "mean_update_seconds_since_last_log": statistics.mean(times[-128:]),
                     "elapsed_training_seconds": time.perf_counter() - loop_time}
            append_json(history_path, entry)
    training_seconds = time.perf_counter() - loop_time
    model.eval()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite checkpoint")
    checkpoint = directory / "checkpoint.pt"
    atomic_write(checkpoint, lambda stream_: torch.save(model.state_dict(), stream_))
    checkpoint_sha256_before_development = sha(checkpoint)
    # Constructor and explicit nonpersistent buffers are bound separately.
    buffers = {name: tensor.detach().cpu().tolist() for name, tensor in model.named_buffers()}
    buffer_path = directory / "nonpersistent_buffers.json"
    save_json(buffer_path, buffers)
    development = {}
    for d in (2, 3, 4):
        key = f"normal_d{d}"
        development[key] = save_prediction(output, directory, key, census(d), model,
                                           config["family"], torch, np)
    require(sha(checkpoint) == checkpoint_sha256_before_development, "checkpoint changed during development evaluation")
    result = {"schema_version": 1, "item": 11, "run_id": config["run_id"], "status": "completed",
              "config": config, "config_sha256": digest(config), "source": source, "runtime": runtime,
              "completed_updates": config["updates"], "parameter_count": sum(p.numel() for p in model.parameters()),
              "config_artifact": descriptor(output, config_path), "checkpoint": descriptor(output, checkpoint),
              "nonpersistent_buffers": descriptor(output, buffer_path),
              "minibatch_indices": descriptor(output, pair_path), "training_delays": descriptor(output, delay_path),
              "training_log": descriptor(output, history_path), "development": development,
              "training_seconds": training_seconds, "mean_update_seconds": statistics.mean(times),
              "median_update_seconds": statistics.median(times),
              "elapsed_total_seconds": time.perf_counter() - all_time,
              "state_values_per_example": RECIPE["models"][config["family"]]["state_values_per_example"],
              "state_payload_bytes_float32": RECIPE["models"][config["family"]]["state_payload_bytes_float32"],
              "cost_scope": "State payload excludes allocator/autograd overhead and is not information capacity or peak RSS; parameter matching does not match state size or compute.",
              "resource_samples": samples, "at_utc": now()}
    save_json(directory / "run.json", result)
    return result, descriptor(output, directory / "run.json")


def selection_from(runs):
    expected = preflight_configs()
    require(len(runs) == 4 and [r["run_id"] for r in runs] == [c["run_id"] for c in expected],
            "preflight needs exactly the four ordered pilots")
    families = {}
    for run, config in zip(runs, expected):
        require(run["status"] == "completed" and run["config"] == config
                and run["completed_updates"] == 1024, "incomplete or changed pilot")
    for family in FAMILIES:
        candidates = []
        for run in runs:
            if run["config"]["family"] != family:
                continue
            mean_accuracy = statistics.mean(run["development"][f"normal_d{d}"]["metrics"]["accuracy"] for d in (3, 4))
            mean_ce = statistics.mean(run["development"][f"normal_d{d}"]["metrics"]["cross_entropy"] for d in (3, 4))
            require(math.isfinite(mean_accuracy) and math.isfinite(mean_ce), "nonfinite selection value")
            candidates.append({"run_id": run["run_id"], "learning_rate": run["config"]["learning_rate"],
                               "mean_dev_accuracy": mean_accuracy, "mean_dev_cross_entropy": mean_ce,
                               "near_accuracy": run["development"]["normal_d2"]["metrics"]["accuracy"]})
        selected = sorted(candidates, key=lambda r: (-r["mean_dev_accuracy"],
                                                    r["mean_dev_cross_entropy"], r["learning_rate"]))[0]
        families[family] = {**selected, "near_competence_pass": selected["near_accuracy"] >= 0.95,
                            "candidates": candidates}
    return {"schema_version": 1, "item": 11, "recipe_sha256": digest(RECIPE),
            "rule": RECIPE["preflight"]["selection"], "families": families,
            "admission_passed": all(r["near_competence_pass"] for r in families.values())}


def preflight(output, source, runtime, torch, np):
    runs, refs = [], []
    for config in preflight_configs():
        run, reference = train_one(output, config, source, runtime, torch, np)
        runs.append(run)
        refs.append(reference)
    selection = selection_from(runs)
    save_json(output / "selection.json", selection)
    summary = {"schema_version": 1, "item": 11, "status": "completed",
               "all_runs_completed": True, "admission_passed": selection["admission_passed"],
               "ordered_run_ids": [r["run_id"] for r in runs], "runs": refs,
               "selection": descriptor(output, output / "selection.json"),
               "recipe_sha256": digest(RECIPE), "source": source, "completed_at_utc": now()}
    save_json(output / "preflight_summary.json", summary)
    require(selection["admission_passed"], "completed preflight does not pass near-delay competence admission")


def recount_development(root, run, np):
    for d in (2, 3, 4):
        entry = run["development"][f"normal_d{d}"]
        meta = read_json(verify_reference(root, entry["metadata"]))
        expected_rows = census(d)
        require(meta["rows"] == expected_rows and meta["rows_sha256"] == digest(expected_rows),
                "development rows differ")
        path = verify_reference(root, entry["predictions"])
        with np.load(path, allow_pickle=False) as data:
            logits, target = data["logits"], data["target"]
            require(logits.dtype == np.float32 and logits.shape == (48, 8)
                    and np.isfinite(logits).all(), "malformed development logits")
            require(np.array_equal(target, np.asarray([r["target"] for r in expected_rows], dtype=np.int64))
                    and np.array_equal(data["prediction"], logits.argmax(-1)), "development target/prediction drift")
            value = logits.astype(np.float64)
            maximum = value.max(1)
            losses = np.log(np.exp(value - maximum[:, None]).sum(1)) + maximum - value[np.arange(48), target]
            require(np.allclose(losses, data["nll"], atol=1e-10, rtol=1e-12), "development NLL drift")
            found = metrics(data["prediction"], target, losses)
            saved = entry["metrics"]
            require(found["n"] == saved["n"] == 48 and found["correct"] == saved["correct"]
                    and found["accuracy"] == saved["accuracy"]
                    and math.isclose(found["cross_entropy"], saved["cross_entropy"], abs_tol=1e-10, rel_tol=1e-12),
                    "development selection metrics differ")
            require(meta["metrics"] == saved and meta["predictions"] == entry["predictions"],
                    "development metadata differs")


def load_selection(plan, output, source, np):
    from research_item11_inputs import recover_input
    spec = plan["inputs"]["preflight"]
    directory = Path(recover_input(spec, output, "preflight")).resolve()
    for name in ("preflight_summary.json", "selection.json"):
        require(name in spec["files_sha256"] and sha(directory / name) == spec["files_sha256"][name],
                "pinned preflight receipt drift")
    summary, saved = read_json(directory / "preflight_summary.json"), read_json(directory / "selection.json")
    require(summary["status"] == "completed" and summary["all_runs_completed"]
            and summary["recipe_sha256"] == digest(RECIPE), "preflight completion/recipe differs")
    configs = preflight_configs()
    require(summary["ordered_run_ids"] == [c["run_id"] for c in configs] and len(summary["runs"]) == 4,
            "preflight inventory differs")
    paths = ("neuropixel/model.py", "neuropixel/phase3.py",
             "neuropixel/research/stream_memory.py", "scripts/research_memory_study.py")
    require(all(summary["source"]["implementation_sha256"][p] == source["implementation_sha256"][p] for p in paths),
            "preflight scientific implementation differs")
    runs = []
    for ref, config in zip(summary["runs"], configs):
        run = read_json(verify_reference(directory, ref))
        require(run["status"] == "completed" and run["config"] == config
                and run["config_sha256"] == digest(config) and run["completed_updates"] == 1024,
                "pilot configuration/budget differs")
        for key in ("checkpoint", "config_artifact", "nonpersistent_buffers",
                    "minibatch_indices", "training_delays", "training_log"):
            path = verify_reference(directory, run[key])
            if key == "config_artifact":
                require(read_json(path) == config, "saved pilot config differs")
        recount_development(directory, run, np)
        runs.append(run)
    selection = selection_from(runs)
    require(selection == saved and selection["admission_passed"] == summary["admission_passed"]
            and selection["admission_passed"], "preflight selection/admission differs or failed")
    return selection, {"archive_commit": spec["archive_commit"],
                       "selection_sha256": sha(directory / "selection.json"),
                       "summary_sha256": sha(directory / "preflight_summary.json"),
                       "input_spec_sha256": digest(spec)}


def train_primary(output, source, runtime, selection, selection_identity, torch, np):
    inventory = []
    for config in primary_configs(selection):
        run, ref = train_one(output, config, source, runtime, torch, np)
        inventory.append({"run_id": run["run_id"], "config": config, "config_sha256": digest(config),
                          "checkpoint": run["checkpoint"], "summary": ref})
    save_json(output / "training_manifest.json", {
        "schema_version": 1, "item": 11, "status": "completed", "run_key": output.name,
        "source_commit": source["source_commit"], "source": source,
        "recipe_sha256": digest(RECIPE), "selection": selection, "selection_identity": selection_identity,
        "ordered_run_ids": [r["run_id"] for r in inventory], "runs": inventory, "created_at_utc": now()})


def authenticate_gate(output, source, selection, identity):
    manifest_path, gate_path = output / "training_manifest.json", output / "pre_final_archive_receipt.json"
    manifest, gate = read_json(manifest_path), read_json(gate_path)
    require(manifest["schema_version"] == gate["schema_version"] == 1
            and manifest["item"] == gate["item"] == 11 and manifest["status"] == "completed",
            "incorrect manifest/gate")
    require(manifest["run_key"] == gate["run_key"] == output.name
            and manifest["source_commit"] == gate["source_commit"] == source["source_commit"]
            and stable_source(manifest["source"]) == stable_source(source)
            and gate["training_manifest_sha256"] == sha(manifest_path), "gate identity differs")
    require(manifest["recipe_sha256"] == digest(RECIPE) and manifest["selection"] == selection
            and manifest["selection_identity"] == identity, "training recipe/selection differs")
    commit = gate["training_archive_commit"]
    require(isinstance(commit, str) and len(commit) == 40 and all(x in "0123456789abcdef" for x in commit),
            "invalid remote training archive commit")
    expected = primary_configs(selection)
    require(manifest["ordered_run_ids"] == [c["run_id"] for c in expected]
            and len(manifest["runs"]) == 10, "incomplete final checkpoint inventory")
    for item, config in zip(manifest["runs"], expected):
        require(item["run_id"] == config["run_id"] and item["config"] == config
                and item["config_sha256"] == digest(config), "manifest configuration differs")
        run = read_json(verify_reference(output, item["summary"]))
        require(run["status"] == "completed" and run["config"] == config
                and run["config_sha256"] == digest(config) and run["completed_updates"] == 2048
                and run["checkpoint"] == item["checkpoint"]
                and stable_source(run["source"]) == stable_source(source), "completed run differs")
        for key in ("checkpoint", "config_artifact", "nonpersistent_buffers",
                    "minibatch_indices", "training_delays", "training_log"):
            path = verify_reference(output, run[key])
            if key == "config_artifact":
                require(read_json(path) == config, "saved main config differs")
        for entry in run["development"].values():
            verify_reference(output, entry["predictions"])
            verify_reference(output, entry["metadata"])
    require(datetime.fromisoformat(gate["at_utc"]) >= datetime.fromisoformat(manifest["created_at_utc"]),
            "gate predates training manifest")
    return manifest, gate


def final_evaluation(output, source, selection, identity, torch, np):
    manifest, gate = authenticate_gate(output, source, selection, identity)
    access = {"schema_version": 1, "item": 11, "source": source, "access_at_utc": now(),
              "status": "consumed", "training_manifest_sha256": sha(output / "training_manifest.json"),
              "gate_sha256": sha(output / "pre_final_archive_receipt.json"),
              "training_archive_commit": gate["training_archive_commit"],
              "scope": "Access to final model-output inventory; task census is known by design."}
    save_json(output / "final_access.json", access)
    final_dir = output / "final_predictions"
    final_dir.mkdir(exist_ok=False)
    cases = evaluation_cases()
    save_json(output / "final_inventory.json", {"schema_version": 1, "item": 11, "cases": cases,
                                               "source": source, "recipe_sha256": digest(RECIPE)})
    outputs = []
    for item in manifest["runs"]:
        config = item["config"]
        run = read_json(verify_reference(output, item["summary"]))
        checkpoint = verify_reference(output, item["checkpoint"])
        model = make_model(config["family"], torch)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        buffers = read_json(verify_reference(output, run["nonpersistent_buffers"]))
        actual = {name: tensor.detach().cpu().tolist() for name, tensor in model.named_buffers()}
        require(actual == buffers, "constructor/nonpersistent-buffer metadata differs")
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite loaded checkpoint")
        model.eval()
        directory = final_dir / config["run_id"]
        directory.mkdir(exist_ok=False)
        one = []
        start = time.perf_counter()
        for case in cases:
            rows = census(case["delay"], case["condition"])
            require(len(rows) == case["n"], "final row inventory differs")
            one.append(save_prediction(output, directory, case["case_id"], rows, model,
                                       config["family"], torch, np))
        require(sha(checkpoint) == item["checkpoint"]["sha256"], "checkpoint changed during final evaluation")
        outputs.append({"run_id": config["run_id"], "config": config, "checkpoint": item["checkpoint"],
                        "cases": one, "evaluation_seconds": time.perf_counter() - start})
    save_json(output / "final_metrics.json", {"schema_version": 1, "item": 11, "status": "completed",
              "source": source, "recipe_sha256": digest(RECIPE), "runs": outputs, "cases": cases,
              "training_manifest_sha256": sha(output / "training_manifest.json"),
              "final_access_sha256": sha(output / "final_access.json"), "completed_at_utc": now(),
              "scope": "Deterministic known census. No statistical inference or retuning occurs in this controller."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "train", "final"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({"recipe": RECIPE, "preflight": preflight_configs(),
                          "evaluation_cases": evaluation_cases()}, sort_keys=True, indent=2))
        return 0
    require(args.phase and args.plan and args.output, "phase/plan/output are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.plan)
    require(plan["schema_version"] == 1 and plan["item"] == 11 and plan["status"] == "frozen"
            and plan.get("freeze_utc"), "not a frozen item-11 plan")
    require(plan["phase"] == ("preflight" if args.phase == "preflight" else "study"), "plan phase differs")
    validate_recipe(plan["recipe"])
    source = source_record(args.plan.resolve(), plan)
    save_json(output / (args.phase + "_started.json"),
              {"phase": args.phase, "started_at_utc": now(), "source": source})
    start = time.perf_counter()
    try:
        torch, np, runtime = configure_runtime(plan)
        save_json(output / ("study_" + args.phase + "_environment.json"), runtime)
        if args.phase == "preflight":
            preflight(output, source, runtime, torch, np)
        else:
            selection, identity = load_selection(plan, output, source, np)
            if args.phase == "train":
                train_primary(output, source, runtime, selection, identity, torch, np)
            else:
                final_evaluation(output, source, selection, identity, torch, np)
        after = source_record(args.plan.resolve(), plan)
        require(stable_source(after) == stable_source(source), "source changed during phase")
        save_json(output / ("study_" + args.phase + "_status.json"), {
            "phase": args.phase, "status": "completed", "completed_at_utc": now(),
            "wall_seconds": time.perf_counter() - start, "source": source, "source_after": after,
            "available_ram_gib": admit()})
        return 0
    except Exception as error:
        save_json(output / ("study_" + args.phase + "_status.json"), {
            "phase": args.phase, "status": "failed", "completed_at_utc": now(),
            "wall_seconds": time.perf_counter() - start, "source": source,
            "error": {"type": type(error).__name__, "message": str(error)}})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
