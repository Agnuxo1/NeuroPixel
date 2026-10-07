"""Independent saved-array audit for item 11 learned stream memory.

No Torch, model imports, checkpoint deserialization, fitting, or new evaluation
data. The task census and saved training RNG streams are independently rebuilt;
all performance comes exclusively from archived logits. Git archive custody and
G-to-F ancestry belong to the separately frozen operational wrapper.
"""
from __future__ import annotations

import argparse
import csv
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

FAMILIES = ("neuropixel", "gru")
POSITIONS = tuple((i // 3, i % 3) for i in range(8))
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
ANALYSIS_RUNTIME = {"python": "3.12.14", "numpy": "2.3.5", "scipy": "1.17.0",
                    "psutil": "7.2.2", "threads": 1}
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class AuditFailure(ValueError):
    pass


class Audit:
    def __init__(self):
        self.checks = 0
        self.issues = []
        self.inputs = {}
        self.case_rows = []
        self.costs = []
        self.runtime = {}
        self.resource_samples = []

    def check(self, ok, message):
        self.checks += 1
        if not bool(ok):
            self.issues.append(message)
            raise AuditFailure(message)

    def path(self, root, name):
        base = Path(root).resolve()
        relative = Path(name)
        self.check(not relative.is_absolute() and ".." not in relative.parts, "unsafe relative path")
        result = (base / relative).resolve()
        self.check(result.is_relative_to(base) and result.is_file(), "missing/escaped artifact: " + str(name))
        return result

    def track(self, path):
        path = Path(path).resolve()
        value = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        previous = self.inputs.get(str(path))
        self.check(previous is None or previous == value, "input changed during audit")
        self.inputs[str(path)] = value
        return path

    def read(self, path):
        self.track(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def ref(self, root, reference):
        self.check(isinstance(reference, dict) and set(reference) == {"path", "bytes", "sha256"},
                   "artifact reference schema differs")
        path = self.track(self.path(root, reference["path"]))
        identity = self.inputs[str(path)]
        self.check(identity == {k: reference[k] for k in ("bytes", "sha256")},
                   "artifact hash/size differs: " + reference["path"])
        return path

    def close(self, left, right, label):
        if isinstance(left, dict):
            self.check(isinstance(right, dict) and set(left) == set(right), label + ": keys differ")
            for key in left:
                self.close(left[key], right[key], label + "/" + str(key))
        elif isinstance(left, list):
            self.check(isinstance(right, list) and len(left) == len(right), label + ": length differs")
            for i, (a, b) in enumerate(zip(left, right)):
                self.close(a, b, label + "/" + str(i))
        elif type(left) is float:
            self.check(type(right) in (int, float) and math.isfinite(right)
                       and math.isclose(left, right, abs_tol=1e-10, rel_tol=1e-12),
                       label + ": floating value differs")
        else:
            self.check(type(left) is type(right) and left == right, label + ": value/type differs")

    def array(self, actual, expected, label):
        self.check(actual.dtype == expected.dtype and actual.shape == expected.shape
                   and np.array_equal(actual, expected), label + ": array differs")

    def ram(self, label):
        import psutil
        amount = psutil.virtual_memory().available / 1024**3
        self.check(math.isfinite(amount) and amount >= 8, "available RAM below 8 GiB")
        self.resource_samples.append({"stage": label, "at_utc": utc(), "available_ram_gib": amount})

    def unchanged(self):
        for name, expected in list(self.inputs.items()):
            path = Path(name)
            self.check(path.is_file() and path.stat().st_size == expected["bytes"]
                       and file_hash(path) == expected["sha256"], "input changed at closing: " + name)


def source_record(a, root, plan_path, expected_phase):
    root = Path(root).resolve()
    plan_path = Path(plan_path).resolve()
    a.check(plan_path.is_relative_to(root), "scientific plan not under source root")
    plan = a.read(plan_path)
    a.check(plan["schema_version"] == 1 and plan["item"] == 11 and plan["status"] == "frozen"
            and bool(plan["freeze_utc"]) and plan["phase"] == expected_phase, "scientific plan is not frozen/matched")
    a.close(RECIPE, plan["recipe"], "frozen recipe")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                    timeout=30, stdin=subprocess.DEVNULL).decode().strip()
    tracked = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                     timeout=30, stdin=subprocess.DEVNULL)
    a.check(not tracked, "tracked scientific source is dirty")
    committed = subprocess.check_output(["git", "-C", str(root), "show",
                                         "HEAD:" + plan_path.relative_to(root).as_posix()],
                                        timeout=30, stdin=subprocess.DEVNULL)
    a.check(committed == plan_path.read_bytes(), "scientific plan differs from its commit")
    bindings = plan["implementation_sha256"]
    a.check(isinstance(bindings, dict) and bool(bindings), "empty scientific source inventory")
    for name, expected in bindings.items():
        path = a.track(a.path(root, name))
        a.check(a.inputs[str(path)]["sha256"] == expected, "scientific source hash differs: " + name)
    expected = {"source_commit": commit, "plan_sha256": file_hash(plan_path),
                "implementation_sha256": bindings, "tracked_source_clean": True}
    return plan, expected


def check_source(a, recorded, expected):
    a.check(set(recorded) == set(expected) | {"recorded_at_utc"}, "source record schema differs")
    for key, value in expected.items():
        a.close(value, recorded[key], "source/" + key)
    timestamp(a, recorded["recorded_at_utc"])


def timestamp(a, value):
    result = datetime.fromisoformat(value)
    a.check(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "timestamp is not UTC")
    return result


def row_inventory(delay, condition):
    interference = condition in ("blank_repeated", "same_position", "next_position")
    rows = []
    for label in range(2, 8):
        for index, position in enumerate(POSITIONS):
            for distractor in (range(2, 8) if interference else (-1,)):
                active = condition in ("same_position", "next_position")
                next_index = index if condition == "same_position" else (index + 1) % 8
                rows.append({"row_id": "%s_d%d_t%d_p%d_x%d" % (condition, delay, label, index, distractor),
                             "target": label, "cue_position_index": index, "cue_position": list(position),
                             "delay": delay, "condition": condition,
                             "donor_target": 2 + (label - 1) % 6 if condition == "swap" else -1,
                             "distractor_target": distractor,
                             "distractor_position": list(POSITIONS[next_index]) if active else [-1, -1],
                             "distractor_present": active,
                             "same_label": label == distractor if interference else None})
    return rows


def cases():
    result = [{"case_id": "normal_d" + str(d), "condition": "normal", "delay": d, "n": 48}
              for d in NORMAL_DELAYS]
    result.extend({"case_id": kind + "_d" + str(d), "condition": kind, "delay": d, "n": 48}
                  for d in (2, 8) for kind in ("cue_erased", "zero", "swap"))
    result.extend({"case_id": kind + "_d8", "condition": kind, "delay": 8, "n": 288}
                  for kind in ("blank_repeated", "same_position", "next_position"))
    return result


def expected_arrays(rows):
    n, delay, condition = len(rows), rows[0]["delay"], rows[0]["condition"]
    frames = np.zeros((n, delay + 2, 3, 3), dtype=np.int64)
    for i, r in enumerate(rows):
        if condition != "cue_erased":
            frames[i, 0, *r["cue_position"]] = r["target"]
        if r["distractor_present"]:
            frames[i, 1, *r["distractor_position"]] = r["distractor_target"]
    frames[:, -1, 2, 1] = 1
    result = {"frames": frames, "steps": np.array([3] * (delay + 1) + [10], dtype=np.int64),
              "row_id": np.array([r["row_id"] for r in rows], dtype="U80")}
    for key in ("target", "cue_position", "donor_target", "distractor_target", "distractor_position"):
        result[key] = np.array([r[key] for r in rows], dtype=np.int64)
    if condition == "swap":
        result["donor_row_index"] = (np.arange(n, dtype=np.int64) + 8) % 48
    return result


def score(pred, target, losses, mask=None, donor=None, distractor=None):
    mask = np.ones(len(target), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    good = int(np.count_nonzero((pred == target) & mask))
    result = {"n": n, "correct": good, "accuracy": good / n if n else None,
              "cross_entropy": float(losses[mask].sum(dtype=np.float64) / n) if n else None,
              "cross_entropy_n": n}
    for label, alternative in (("donor", donor), ("distractor", distractor)):
        if alternative is not None:
            count = int(np.count_nonzero((pred == alternative) & mask))
            result[label + "_correct"] = count
            result[label + "_accuracy"] = count / n if n else None
    return result


def all_scores(arrays, rows, losses):
    pred, target = arrays["prediction"], arrays["target"]
    kind = rows[0]["condition"]
    distractor = arrays["distractor_target"] if kind in ("blank_repeated", "same_position", "next_position") else None
    donor = arrays["donor_target"] if kind == "swap" else None
    result = score(pred, target, losses, donor=donor, distractor=distractor)
    result["per_target"] = {str(label): score(pred, target, losses, target == label) for label in range(2, 8)}
    position = np.array([r["cue_position_index"] for r in rows])
    result["per_position"] = {str(p): score(pred, target, losses, position == p) for p in range(8)}
    if distractor is not None:
        result["strata"] = {key: score(pred, target, losses, mask, distractor=distractor)
                            for key, mask in (("same_label", target == distractor),
                                              ("different_label", target != distractor))}
    return result


def prediction_case(a, root, entry, family, run_id, delay, condition, phase):
    rows = row_inventory(delay, condition)
    expected = expected_arrays(rows)
    case_id = condition + "_d" + str(delay)
    a.check(entry["case_id"] == case_id, "case identity differs")
    metadata = a.read(a.ref(root, entry["metadata"]))
    a.check(metadata["schema_version"] == 1 and metadata["case_id"] == case_id, "case metadata schema/ID differs")
    a.close(rows, metadata["rows"], "census rows")
    a.check(metadata["rows_sha256"] == digest(rows), "census row hash differs")
    a.check(metadata["predictions"] == entry["predictions"], "prediction references disagree")
    a.check(metadata["frame_count"] == delay + 2 and metadata["np_updates"] == 13 + 3 * delay
            and metadata["gru_sequence_steps"] == delay + 2, "frame/update cost metadata differs")
    a.check(metadata["state_boundary"] ==
            "saved pre_query_state is after last gap and BEFORE intervention/query", "state boundary metadata differs")
    path = a.ref(root, entry["predictions"])
    required = set(expected) | {"logits", "prediction", "nll", "pre_query_state", "post_query_state"}
    if condition in ("zero", "swap"):
        required |= {"intervention_before", "intervention_after"}
    with np.load(path, allow_pickle=False) as archive:
        a.check(set(archive.files) == required and len(archive.files) == len(required), "NPZ fields differ")
        arrays = {k: archive[k] for k in archive.files}
    n = len(rows)
    for key, value in expected.items():
        a.array(arrays[key], value, case_id + "/" + key)
    logits = arrays["logits"]
    a.check(logits.dtype == np.float32 and logits.shape == (n, 8) and np.isfinite(logits).all(),
            "invalid logits")
    a.check(np.all(logits[:, 0] == -10000), "PAD output convention differs")
    a.array(arrays["prediction"], logits.argmax(1).astype(np.int64), "argmax predictions")
    target = expected["target"]
    x = logits.astype(np.float64)
    high = x.max(axis=1)
    losses = high + np.log(np.exp(x - high[:, None]).sum(axis=1)) - x[np.arange(n), target]
    a.check(arrays["nll"].dtype == np.float64 and arrays["nll"].shape == (n,)
            and np.isfinite(arrays["nll"]).all()
            and np.allclose(losses, arrays["nll"], atol=1e-10, rtol=1e-12),
            "saved NLL differs from all-eight-token log-softmax")
    state_shape = (n, 48, 3, 3) if family == "neuropixel" else (n, 70)
    for key in ("pre_query_state", "post_query_state"):
        value = arrays[key]
        a.check(value.dtype == np.float32 and value.shape == state_shape and np.isfinite(value).all(),
                "invalid captured state: " + key)
    if condition in ("zero", "swap"):
        before, after = arrays["intervention_before"], arrays["intervention_after"]
        native_shape = state_shape if family == "neuropixel" else (1,) + state_shape
        for value in (before, after):
            a.check(value.dtype == np.float32 and value.shape == native_shape
                    and np.isfinite(value).all(), "invalid intervention tensor")
        expected_before = arrays["pre_query_state"] if family == "neuropixel" else arrays["pre_query_state"][None]
        a.array(before, expected_before, "actual before-intervention snapshot")
        if condition == "zero":
            wanted = np.zeros_like(before)
        else:
            order = expected["donor_row_index"]
            wanted = np.take(before, order, axis=0 if family == "neuropixel" else 1)
            a.array(np.sort(order), np.arange(48, dtype=np.int64), "swap bijection")
            a.check(np.all(target[order] == expected["donor_target"])
                    and np.all(target[order] != target), "swap donor labels differ")
        a.array(after, wanted, "actual after-intervention snapshot")
    metrics = all_scores(arrays, rows, losses)
    a.close(metrics, metadata["metrics"], "metadata metrics")
    a.close(metrics, entry["metrics"], "entry metrics")
    row = {"phase": phase, "run_id": run_id, "family": family, "case_id": case_id,
           "condition": condition, "delay": delay, "stratum": "all"}
    scalar = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    a.case_rows.append({**row, **scalar})
    for grouping in ("per_target", "per_position", "strata"):
        for name, value in metrics.get(grouping, {}).items():
            a.case_rows.append({**row, "stratum": grouping + ":" + name, **value})
    a.costs.append({"phase": phase, "run_id": run_id, "case_id": case_id, "n": n,
                    "unique_input_histories": 1 if condition == "cue_erased" else 48 if condition == "blank_repeated" else n,
                    "note_unique_histories": ("cue_erased has one visible history; zero retains distinct histories before erasure"
                                              if condition in ("cue_erased", "zero") else "known deterministic census"),
                    "frame_examples": n * (delay + 2),
                    "np_forward_state_cell_update_positions": n * (13 + 3 * delay) * 9 if family == "neuropixel" else None,
                    "gru_recurrent_example_steps": n * (delay + 2) if family == "gru" else None})
    return {"metrics": metrics, "arrays": arrays}


def configuration(family, seed, lr, kind):
    updates = 1024 if kind == "pilot" else 2048
    run_id = ("pilot_%s_lr%04d_s%d" % (family, int(round(lr * 1000)), seed) if kind == "pilot"
              else "train_%s_s%d" % (family, seed))
    return {"run_id": run_id, "family": family, "seed": seed, "learning_rate": lr,
            "updates": updates, "kind": kind, "batch_size": 32, "optimizer": "AdamW",
            "weight_decay": 0.0001, "gradient_clip": 1.0, "schedule": "constant",
            "loss": "answer_cross_entropy_only", "model": RECIPE["models"][family], "task": RECIPE["task"],
            "index_seed": 100000 + seed, "initialization_seed": seed, "firing_seed": 110000 + seed,
            "recipe_sha256": digest(RECIPE)}


def training_run(a, root, reference, config, expected_source, plan):
    run = a.read(a.ref(root, reference))
    a.check(run["schema_version"] == 1 and run["item"] == 11 and run["status"] == "completed"
            and run["run_id"] == config["run_id"], "run completion/identity differs")
    a.close(config, run["config"], "run config")
    a.check(run["config_sha256"] == digest(config) and run["completed_updates"] == config["updates"],
            "run config digest/budget differs")
    check_source(a, run["source"], expected_source)
    a.close(config, a.read(a.ref(root, run["config_artifact"])), "config artifact")
    a.ref(root, run["checkpoint"])  # Deliberately never deserialize neural weights.
    buffers = a.read(a.ref(root, run["nonpersistent_buffers"]))
    a.check(isinstance(buffers, dict), "nonpersistent buffers must be explicitly recorded")
    if config["family"] == "neuropixel":
        a.check(set(buffers) == {"g_rgb", "g_mask"} and buffers["g_rgb"] == [[0.0] * 3 for _ in range(8)]
                and buffers["g_mask"] == [[False] for _ in range(8)], "fresh ungrounded buffer metadata differs")
    else:
        a.check(buffers == {}, "unexpected GRU buffers")
    for field in ("parameter_count", "state_values_per_example", "state_payload_bytes_float32"):
        key = "parameters" if field == "parameter_count" else field
        a.check(run[field] == config["model"][key], "parameter/state payload metadata differs")
    runtime = run["runtime"]
    a.check(runtime["python"] == plan["runtime"]["python"] and runtime["packages"] == plan["runtime"]["packages"]
            and runtime["threads"] == plan["runtime"]["threads"] == 2
            and runtime["interop_threads"] == plan["runtime"]["interop_threads"] == 1
            and runtime["cuda_version"] is None, "training runtime differs")
    a.check(math.isfinite(runtime["available_ram_gib"]) and runtime["available_ram_gib"] >= 8,
            "invalid recorded training RAM")
    paths = {key: a.ref(root, run[key]) for key in ("minibatch_indices", "training_delays", "training_log")}
    pair = np.load(paths["minibatch_indices"], allow_pickle=False)
    delays = np.load(paths["training_delays"], allow_pickle=False)
    generator = np.random.Generator(np.random.PCG64(100000 + config["seed"]))
    expected_pair = generator.integers(0, 48, (config["updates"], 32), dtype=np.int64)
    expected_delays = generator.integers(0, 3, (config["updates"],), dtype=np.int64)
    a.array(pair, expected_pair, "saved private training indices")
    a.array(delays, expected_delays, "saved per-minibatch delay stream")
    log = [json.loads(line) for line in paths["training_log"].read_text(encoding="utf-8").splitlines()]
    a.check([v["update"] for v in log] == list(range(128, config["updates"] + 1, 128)), "training log schedule differs")
    previous = 0.0
    for value in log:
        expected_keys = {"update", "mean_training_loss_since_last_log", "last_answer_loss",
                         "last_unclipped_gradient_norm", "mean_update_seconds_since_last_log",
                         "elapsed_training_seconds"}
        a.check(set(value) == expected_keys, "training log fields differ")
        a.check(all(type(value[k]) in (int, float) and math.isfinite(value[k]) and value[k] >= 0
                    for k in expected_keys - {"update"}), "invalid training log scalar")
        a.check(value["elapsed_training_seconds"] >= previous, "training elapsed times decreased")
        previous = value["elapsed_training_seconds"]
    for name in ("training_seconds", "mean_update_seconds", "median_update_seconds", "elapsed_total_seconds"):
        a.check(type(run[name]) in (int, float) and math.isfinite(run[name]) and run[name] > 0, "invalid time: " + name)
    a.check(previous <= run["training_seconds"] <= run["elapsed_total_seconds"], "training wall-time bounds differ")
    a.close(float(statistics.mean(v["mean_update_seconds_since_last_log"] for v in log)),
            run["mean_update_seconds"], "mean step timing from equal windows")
    a.check(run["mean_update_seconds"] * config["updates"] <= run["training_seconds"] + 1e-8,
            "summed update timing exceeds training wall time")
    samples = run["resource_samples"]
    a.check([r["update"] for r in samples] == list(range(0, config["updates"], 128)), "RAM sampling schedule differs")
    stamp = None
    for value in samples:
        t = timestamp(a, value["at_utc"])
        a.check(stamp is None or t >= stamp, "resource timestamps decreased")
        a.check(math.isfinite(value["available_ram_gib"]) and value["available_ram_gib"] >= 8, "recorded training RAM below floor")
        stamp = t
    a.check(timestamp(a, run["at_utc"]) >= stamp, "run completion predates resource observation")
    a.check(set(run["development"]) == {"normal_d2", "normal_d3", "normal_d4"}, "development case inventory differs")
    dev = {}
    for delay in (2, 3, 4):
        name = "normal_d" + str(delay)
        dev[name] = prediction_case(a, root, run["development"][name], config["family"], config["run_id"],
                                    delay, "normal", config["kind"] + "_development")
    total_delay = int(delays.sum())
    cost = {"run_id": config["run_id"], "family": config["family"], "updates": config["updates"],
            "batch_size": 32, "sampled_example_presentations": config["updates"] * 32,
            "sum_minibatch_delays": total_delay,
            "frame_examples": 32 * (2 * config["updates"] + total_delay),
            "np_forward_state_cell_update_positions": 32 * 9 * (13 * config["updates"] + 3 * total_delay)
                                                     if config["family"] == "neuropixel" else None,
            "gru_recurrent_example_steps": 32 * (2 * config["updates"] + total_delay)
                                           if config["family"] == "gru" else None,
            "parameters": run["parameter_count"], "state_values_per_example": run["state_values_per_example"],
            "state_payload_bytes_float32": run["state_payload_bytes_float32"],
            "training_seconds": run["training_seconds"], "elapsed_total_seconds": run["elapsed_total_seconds"],
            "minimum_sampled_available_ram_gib": min(r["available_ram_gib"] for r in samples),
            "index_array_sha256": run["minibatch_indices"]["sha256"], "delay_array_sha256": run["training_delays"]["sha256"],
            "training_log": log}
    return {"run": run, "development": dev, "cost": cost}


def selection(a, audited):
    families = {}
    for family in FAMILIES:
        candidates = []
        for record in audited:
            config = record["run"]["config"]
            if config["family"] != family:
                continue
            values = record["development"]
            candidates.append({"run_id": config["run_id"], "learning_rate": config["learning_rate"],
                               "mean_dev_accuracy": statistics.mean(values["normal_d" + str(d)]["metrics"]["accuracy"] for d in (3, 4)),
                               "mean_dev_cross_entropy": statistics.mean(values["normal_d" + str(d)]["metrics"]["cross_entropy"] for d in (3, 4)),
                               "near_accuracy": values["normal_d2"]["metrics"]["accuracy"]})
        a.check(len(candidates) == 2, "selection requires both rates for each family")
        winner = min(candidates, key=lambda x: (-x["mean_dev_accuracy"], x["mean_dev_cross_entropy"], x["learning_rate"]))
        families[family] = {**winner, "near_competence_pass": winner["near_accuracy"] >= 0.95,
                            "candidates": candidates}
    return {"schema_version": 1, "item": 11, "recipe_sha256": digest(RECIPE),
            "rule": RECIPE["preflight"]["selection"], "families": families,
            "admission_passed": all(r["near_competence_pass"] for r in families.values())}


def phase_receipts(a, root, phase, expected_source, plan, allow_negative=False):
    started = a.read(a.path(root, phase + "_started.json"))
    status = a.read(a.path(root, "study_" + phase + "_status.json"))
    env = a.read(a.path(root, "study_" + phase + "_environment.json"))
    a.check(started["phase"] == status["phase"] == phase, "phase receipt identity differs")
    check_source(a, started["source"], expected_source)
    check_source(a, status["source"], expected_source)
    a.check(status["status"] == "completed" or
            (allow_negative and status["status"] == "failed" and status.get("error", {}).get("message") ==
             "completed preflight does not pass near-delay competence admission"),
            "scientific phase did not complete")
    if status["status"] == "completed":
        check_source(a, status["source_after"], expected_source)
        a.check(math.isfinite(status["available_ram_gib"]) and status["available_ram_gib"] >= 8, "phase RAM invalid")
    a.check(env["python"] == plan["runtime"]["python"] and env["packages"] == plan["runtime"]["packages"]
            and env["threads"] == 2 and env["interop_threads"] == 1 and env["cuda_version"] is None,
            "phase environment differs")
    a.check(math.isfinite(env["available_ram_gib"]) and env["available_ram_gib"] >= 8, "environment RAM invalid")
    start, end = timestamp(a, started["started_at_utc"]), timestamp(a, status["completed_at_utc"])
    a.check(end >= start and math.isfinite(status["wall_seconds"]) and status["wall_seconds"] > 0,
            "phase timestamps/time invalid")
    return {"started": started, "status": status, "environment": env}


def audit_preflight(a, root, plan_path, source_root):
    plan, source = source_record(a, source_root, plan_path, "preflight")
    summary = a.read(a.path(root, "preflight_summary.json"))
    a.check(summary["schema_version"] == 1 and summary["item"] == 11 and summary["status"] == "completed"
            and summary["all_runs_completed"] is True and summary["recipe_sha256"] == digest(RECIPE),
            "preflight summary differs")
    check_source(a, summary["source"], source)
    configs = [configuration(f, 69, lr, "pilot") for f in FAMILIES for lr in (0.001, 0.003)]
    a.check(summary["ordered_run_ids"] == [c["run_id"] for c in configs] and len(summary["runs"]) == 4,
            "four ordered pilots are required")
    records = [training_run(a, root, ref, c, source, plan) for ref, c in zip(summary["runs"], configs)]
    computed = selection(a, records)
    saved = a.read(a.ref(root, summary["selection"]))
    a.check(Path(summary["selection"]["path"]).as_posix() == "selection.json", "selection path differs")
    a.close(computed, saved, "independent pilot selection")
    a.check(computed["admission_passed"] is summary["admission_passed"], "admission field differs")
    fingerprints = {(r["run"]["minibatch_indices"]["sha256"], r["run"]["training_delays"]["sha256"]) for r in records}
    a.check(len(fingerprints) == 1, "pilot data streams are not identical across rates/families")
    finish = timestamp(a, summary["completed_at_utc"])
    a.check(all(timestamp(a, r["run"]["at_utc"]) <= finish for r in records), "summary predates pilot completion")
    receipts = phase_receipts(a, root, "preflight", source, plan, allow_negative=not computed["admission_passed"])
    a.check(timestamp(a, receipts["started"]["started_at_utc"]) <= min(timestamp(a, r["run"]["resource_samples"][0]["at_utc"]) for r in records)
            and timestamp(a, receipts["status"]["completed_at_utc"]) >= finish, "preflight chronology differs")
    for name in ("final_access.json", "final_inventory.json", "final_metrics.json", "final_predictions"):
        a.check(not (Path(root) / name).exists(), "final artifacts exist in preflight input")
    return {"selection": computed, "records": records, "source": source, "plan": plan,
            "summary": summary, "receipts": receipts}


def interval(values, with_ci=False):
    """Summaries across five realizations; only the primary difference gets a CI."""
    if len(values) != 5 or not all(math.isfinite(v) for v in values):
        raise AuditFailure("all five finite training realizations are required")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    result = {"values": [float(x) for x in values], "n": 5, "mean": float(mean),
              "sample_sd": float(sd), "minimum": float(min(values)), "maximum": float(max(values))}
    if with_ci:
        critical = float(student_t.ppf(0.975, 4))
        half = critical * sd / math.sqrt(5)
        result.update({"df": 4, "t_critical": critical,
                       "ci95_t": [float(mean - half), float(mean + half)],
                       "scope": "Primary paired training-realization interval; no clipping or event-level bootstrap."})
    else:
        result["scope"] = "Descriptive five-realization values/mean/SD; no CI, p-value, or decision rule."
    return result

def cross_path_diagnostics(records):
    result = []
    for config, audited in records:
        for delay in (2, 8):
            normal = audited["normal_d" + str(delay)]["arrays"]
            for kind in ("zero", "swap"):
                alternative = audited[kind + "_d" + str(delay)]["arrays"]
                left, right = normal["pre_query_state"], alternative["pre_query_state"]
                result.append({"run_id": config["run_id"], "comparison": kind + "_prefix_d" + str(delay),
                               "max_abs_difference": float(np.max(np.abs(left.astype(np.float64) - right))),
                               "allclose_atol1e-6_rtol1e-5": bool(np.allclose(left, right, atol=1e-6, rtol=1e-5)),
                               "scope": "Cross-call float32 diagnostic; actual intervention tensor identities are checked exactly."})
            donor = audited["swap_d" + str(delay)]["arrays"]
            idx = donor["donor_row_index"]
            left, right = normal["logits"][idx], donor["logits"]
            result.append({"run_id": config["run_id"], "comparison": "swap_donor_logits_d" + str(delay),
                           "max_abs_difference": float(np.max(np.abs(left.astype(np.float64) - right))),
                           "allclose_atol1e-6_rtol1e-5": bool(np.allclose(left, right, atol=1e-6, rtol=1e-5)),
                           "argmax_agreement": float(np.mean(normal["prediction"][idx] == donor["prediction"])),
                           "scope": "Donor consistency diagnostic; split GRU calls may differ by floating-point rounding."})
        normal, repeated = audited["normal_d8"]["arrays"], audited["blank_repeated_d8"]["arrays"]
        left, right = np.repeat(normal["logits"], 6, axis=0), repeated["logits"]
        result.append({"run_id": config["run_id"], "comparison": "blank_repeated_vs_normal_d8",
                       "max_abs_difference": float(np.max(np.abs(left.astype(np.float64) - right))),
                       "allclose_atol1e-6_rtol1e-5": bool(np.allclose(left, right, atol=1e-6, rtol=1e-5)),
                       "argmax_agreement": float(np.mean(np.repeat(normal["prediction"], 6) == repeated["prediction"])),
                       "scope": "Batch 48 versus 288; interference contrasts use the batch-288 blank control."})
    return result


def scientific_summaries(finals):
    by_key = {(c["family"], c["seed"]): cases_ for c, cases_ in finals}
    def values(family, case_id, metric="accuracy", stratum="all"):
        out = []
        for seed in range(70, 75):
            score_ = by_key[family, seed][case_id]["metrics"]
            if stratum != "all":
                score_ = score_["strata"][stratum]
            out.append(float(score_[metric]))
        return out
    primary_np = values("neuropixel", "normal_d8")
    primary_gru = values("gru", "normal_d8")
    primary = interval([x - y for x, y in zip(primary_np, primary_gru)], with_ci=True)
    primary.update({"estimand": "NeuroPixel minus GRU accuracy on normal delay8",
                    "seeds": list(range(70, 75)), "neuropixel": interval(primary_np), "gru": interval(primary_gru),
                    "historical_H1": "unchanged; this is the separate item-11 estimand"})
    cases_summary, contrasts = {}, []
    for case in cases():
        cid = case["case_id"]
        strata = ("all", "same_label", "different_label") if case["n"] == 288 else ("all",)
        cases_summary[cid] = {}
        for stratum in strata:
            cases_summary[cid][stratum] = {f: interval(values(f, cid, stratum=stratum)) for f in FAMILIES}
            delta = [x - y for x, y in zip(values("neuropixel", cid, stratum=stratum),
                                         values("gru", cid, stratum=stratum))]
            contrasts.append({"contrast": "neuropixel_minus_gru", "case_id": cid, "stratum": stratum,
                              "metric": "accuracy", **interval(delta)})
        if case["condition"] == "swap":
            cases_summary[cid]["donor_accuracy"] = {f: interval(values(f, cid, "donor_accuracy")) for f in FAMILIES}
        if case["n"] == 288:
            cases_summary[cid]["distractor_accuracy"] = {
                stratum: {f: interval(values(f, cid, "distractor_accuracy", stratum)) for f in FAMILIES}
                for stratum in strata}
    for family in FAMILIES:
        for delay in NORMAL_DELAYS:
            if delay != 2:
                delta = [x - y for x, y in zip(values(family, "normal_d" + str(delay)),
                                              values(family, "normal_d2"))]
                contrasts.append({"contrast": family + "_normal_minus_near2", "case_id": "normal_d" + str(delay),
                                  "stratum": "all", "metric": "accuracy", **interval(delta)})
        for delay in (2, 8):
            for kind in ("cue_erased", "zero", "swap"):
                cid = kind + "_d" + str(delay)
                delta = [x - y for x, y in zip(values(family, cid), values(family, "normal_d" + str(delay)))]
                contrasts.append({"contrast": family + "_intervention_minus_normal", "case_id": cid,
                                  "stratum": "all", "metric": "origin_accuracy", **interval(delta)})
        for kind in ("same_position", "next_position"):
            for stratum in ("all", "same_label", "different_label"):
                cid = kind + "_d8"
                delta = [x - y for x, y in zip(values(family, cid, stratum=stratum),
                                              values(family, "blank_repeated_d8", stratum=stratum))]
                contrasts.append({"contrast": family + "_distractor_minus_matched_blank", "case_id": cid,
                                  "stratum": stratum, "metric": "origin_accuracy", **interval(delta)})
    screens = {}
    for family in FAMILIES:
        near, retained = values(family, "normal_d2"), values(family, "normal_d8")
        screens[family] = {"seeds": list(range(70, 75)), "near_delay2_accuracy": near,
                           "near_delay2_pass_by_seed": [v >= 0.95 for v in near],
                           "near_delay2_all_five_pass": all(v >= 0.95 for v in near),
                           "retention_delay8_accuracy": retained,
                           "retention_delay8_pass_by_seed": [v >= 0.95 for v in retained],
                           "retention_delay8_all_five_pass": all(v >= 0.95 for v in retained),
                           "threshold": 0.95}
    return {"primary": primary, "all_cases": cases_summary, "contrasts": contrasts, "screens": screens,
            "scope": "48 known target-position pairs; 288 interference rows include six repeated blank histories per pair. No held-out-content claim.",
            "analytical_no_memory_reference": {"balanced_six_label_accuracy": 1 / 6,
                                                "scope": "Mathematical marginal reference, not a newly executed model."},
            "state_payload": RECIPE["models"],
            "limits": ["Five realizations vary initialization, sampled training stream, and NP firing; this is not isolated initialization variance.",
                       "Census rows, cells, frames, delays, and repeated histories are not independent training replicates.",
                       "The primary t interval with n=5 requires strong small-sample assumptions; secondary contrasts do not receive intervals or decisions.",
                       "Post-query accuracy does not measure information capacity or indefinite persistence; zero-state damage alone is not content-specific proof.",
                       "Parameters are approximately matched; state sizes, local updates, frame encodings, and compute are not matched.",
                       "Only raw state tensor payloads are counted; no inference about peak RSS, information capacity, FLOPs, or energy.",
                       "The auditor reconstructs saved contracts and scores; it does not replay neural dynamics or independently establish physical execution from arrays alone."]}


def audit_study(a, root, plan_path, source_root, prior_root, prior_plan, prior_source):
    prior = audit_preflight(a, prior_root, prior_plan, prior_source)
    a.check(prior["selection"]["admission_passed"] is True, "main study requires positive preflight admission")
    plan, source = source_record(a, source_root, plan_path, "study")
    for name in ("neuropixel/model.py", "neuropixel/phase3.py",
                 "neuropixel/research/stream_memory.py", "scripts/research_memory_study.py"):
        a.check(source["implementation_sha256"][name] == prior["source"]["implementation_sha256"][name],
                "preflight/study scientific implementation differs")
    spec = plan["inputs"]["preflight"]
    for name, expected in spec["files_sha256"].items():
        path = a.track(a.path(prior_root, name))
        a.check(a.inputs[str(path)]["sha256"] == expected, "pinned preflight input differs: " + name)
    for name in ("selection.json", "preflight_summary.json"):
        a.check(name in spec["files_sha256"], "preflight selection/completion is not pinned")
    identity = {"archive_commit": spec["archive_commit"],
                "selection_sha256": file_hash(Path(prior_root) / "selection.json"),
                "summary_sha256": file_hash(Path(prior_root) / "preflight_summary.json"),
                "input_spec_sha256": digest(spec)}
    manifest_path = a.path(root, "training_manifest.json")
    manifest = a.read(manifest_path)
    selected = prior["selection"]
    configs = [configuration(f, seed, selected["families"][f]["learning_rate"], "train")
               for seed in range(70, 75) for f in FAMILIES]
    a.check(manifest["schema_version"] == 1 and manifest["item"] == 11 and manifest["status"] == "completed"
            and manifest["run_key"] == Path(root).name and manifest["source_commit"] == source["source_commit"]
            and manifest["recipe_sha256"] == digest(RECIPE), "training manifest identity differs")
    check_source(a, manifest["source"], source)
    a.close(selected, manifest["selection"], "manifest preflight selection")
    a.close(identity, manifest["selection_identity"], "manifest selection input identity")
    a.check(manifest["ordered_run_ids"] == [c["run_id"] for c in configs] and len(manifest["runs"]) == 10,
            "exact ten ordered main runs required")
    trained = []
    for item, config in zip(manifest["runs"], configs):
        a.check(item["run_id"] == config["run_id"] and item["config_sha256"] == digest(config), "manifest run identity differs")
        a.close(config, item["config"], "manifest configuration")
        record = training_run(a, root, item["summary"], config, source, plan)
        a.check(record["run"]["checkpoint"] == item["checkpoint"], "manifest checkpoint differs")
        trained.append(record)
    for offset in range(0, 10, 2):
        left, right = trained[offset]["cost"], trained[offset + 1]["cost"]
        a.check(left["index_array_sha256"] == right["index_array_sha256"]
                and left["delay_array_sha256"] == right["delay_array_sha256"], "paired main data streams differ")
    manifest_time = timestamp(a, manifest["created_at_utc"])
    a.check(all(timestamp(a, r["run"]["at_utc"]) <= manifest_time for r in trained), "manifest predates run completion")
    train_receipts = phase_receipts(a, root, "train", source, plan)
    a.check(timestamp(a, train_receipts["started"]["started_at_utc"]) <=
            min(timestamp(a, r["run"]["resource_samples"][0]["at_utc"]) for r in trained),
            "main run started before train phase")
    gate_path = a.path(root, "pre_final_archive_receipt.json")
    gate = a.read(gate_path)
    a.check(gate["schema_version"] == 1 and gate["item"] == 11 and gate["run_key"] == Path(root).name
            and gate["source_commit"] == source["source_commit"]
            and gate["training_manifest_sha256"] == file_hash(manifest_path), "local gate identity differs")
    archive = gate["training_archive_commit"]
    a.check(isinstance(archive, str) and len(archive) == 40 and all(c in "0123456789abcdef" for c in archive), "invalid gate commit")
    access_path = a.path(root, "final_access.json")
    access = a.read(access_path)
    a.check(access["schema_version"] == 1 and access["item"] == 11 and access["status"] == "consumed"
            and access["training_manifest_sha256"] == file_hash(manifest_path)
            and access["gate_sha256"] == file_hash(gate_path)
            and access["training_archive_commit"] == archive, "final consumed-access binding differs")
    check_source(a, access["source"], source)
    gate_time, access_time = timestamp(a, gate["at_utc"]), timestamp(a, access["access_at_utc"])
    a.check(manifest_time <= timestamp(a, train_receipts["status"]["completed_at_utc"]) <= gate_time <= access_time,
            "train/gate/final-access order differs")
    inventory = a.read(a.path(root, "final_inventory.json"))
    a.check(inventory["schema_version"] == 1 and inventory["item"] == 11
            and inventory["recipe_sha256"] == digest(RECIPE), "final inventory identity differs")
    check_source(a, inventory["source"], source)
    a.close(cases(), inventory["cases"], "declared sixteen-case inventory")
    metrics = a.read(a.path(root, "final_metrics.json"))
    a.check(metrics["schema_version"] == 1 and metrics["item"] == 11 and metrics["status"] == "completed"
            and metrics["recipe_sha256"] == digest(RECIPE)
            and metrics["training_manifest_sha256"] == file_hash(manifest_path)
            and metrics["final_access_sha256"] == file_hash(access_path), "final metrics bindings differ")
    check_source(a, metrics["source"], source)
    a.close(cases(), metrics["cases"], "completed final inventory")
    a.check(len(metrics["runs"]) == 10, "ten final checkpoint outputs are required")
    finals, evaluation_times = [], []
    for item, trained_one, config in zip(metrics["runs"], trained, configs):
        a.check(item["run_id"] == config["run_id"] and item["checkpoint"] == trained_one["run"]["checkpoint"],
                "final checkpoint identity differs")
        a.close(config, item["config"], "final config")
        a.check([x["case_id"] for x in item["cases"]] == [x["case_id"] for x in cases()], "all sixteen final cases required")
        parsed = {}
        for entry, expected in zip(item["cases"], cases()):
            parsed[expected["case_id"]] = prediction_case(a, root, entry, config["family"], config["run_id"],
                                                         expected["delay"], expected["condition"], "final")
        a.check(sum(p["metrics"]["n"] for p in parsed.values()) == 1488, "final checkpoint denominator differs")
        a.check(math.isfinite(item["evaluation_seconds"]) and item["evaluation_seconds"] > 0, "invalid final evaluation time")
        evaluation_times.append({"run_id": config["run_id"], "evaluation_seconds": item["evaluation_seconds"]})
        finals.append((config, parsed))
    final_receipts = phase_receipts(a, root, "final", source, plan)
    finish = timestamp(a, metrics["completed_at_utc"])
    a.check(gate_time <= timestamp(a, final_receipts["started"]["started_at_utc"]) <= access_time
            <= finish <= timestamp(a, final_receipts["status"]["completed_at_utc"]), "final phase chronology differs")
    scientific = scientific_summaries(finals)
    scientific["cross_path_diagnostics"] = cross_path_diagnostics(finals)
    scientific["cross_path_diagnostics_note"] = (
        "Tolerance booleans are diagnostic, not proof of equal neural trajectories or additional statistical decision rules. "
        "Within-call intervention arrays, their original boundary, and zero/permutation operations must match exactly.")
    return {"preflight": public_preflight(prior), "training": [public_run(r) for r in trained],
            "analysis": scientific, "evaluation_times": evaluation_times,
            "gate": {"training_archive_commit": archive, "manifest_sha256": file_hash(manifest_path),
                     "gate_sha256": file_hash(gate_path), "access_sha256": file_hash(access_path),
                     "local_order_verified": True, "remote_ancestry_scope": "Independent operational wrapper required."},
            "phase_receipts": {"train": train_receipts, "final": final_receipts},
            "counts": {"main_runs": 10, "cases_per_checkpoint": 16, "nominal_final_rows_per_checkpoint": 1488,
                       "nominal_final_rows_total": 14880, "unique_cue_census": 48,
                       "interference_rows_per_case": 288, "same_label_rows": 48, "different_label_rows": 240,
                       "training_realizations_per_family": 5}}


def public_run(record):
    return {"run_id": record["run"]["run_id"], "config": record["run"]["config"],
            "checkpoint": record["run"]["checkpoint"], "source": record["run"]["source"],
            "development": {key: value["metrics"] for key, value in record["development"].items()},
            "cost": record["cost"]}


def public_preflight(record):
    return {"selection": record["selection"], "source": record["source"],
            "runs": [public_run(r) for r in record["records"]], "receipts": record["receipts"],
            "all_four_runs_completed": True, "admission_passed": record["selection"]["admission_passed"],
            "scope": "Development selection only; negative complete admission is a valid audited scientific result."}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":"), allow_nan=False)
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "study"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preflight-plan", type=Path)
    parser.add_argument("--preflight-source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "study" and not all((args.preflight, args.preflight_plan, args.preflight_source_root)):
        parser.error("study requires all three preflight arguments")
    output = args.output_dir.resolve()
    for root in (args.input, args.source_root, args.preflight, args.preflight_source_root):
        if root is not None and (output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output)):
            parser.error("output must be separate from all scientific/archive input roots")
    output.mkdir(parents=True, exist_ok=False)
    audit, result = Audit(), {}
    start = time.perf_counter()
    started = utc()
    error = None
    try:
        audit.check(platform.python_version() == ANALYSIS_RUNTIME["python"], "analysis Python differs")
        audit.check(all(os.environ.get(name) == "1" for name in THREAD_KEYS), "analysis thread env differs")
        audit.check("torch" not in sys.modules, "Torch was imported before saved-array audit")
        actual = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "psutil")}
        audit.check(actual == {name: ANALYSIS_RUNTIME[name] for name in actual}, "analysis packages differ")
        global np, student_t
        import numpy as np
        from scipy.stats import t as student_t
        audit.runtime = {**ANALYSIS_RUNTIME, "actual_packages": actual,
                         "thread_environment": {k: os.environ.get(k) for k in THREAD_KEYS}}
        audit.track(Path(__file__))
        audit.ram("start")
        if args.phase == "preflight":
            result = {"preflight": public_preflight(audit_preflight(audit, args.input, args.plan, args.source_root))}
        else:
            result = audit_study(audit, args.input, args.plan, args.source_root,
                                 args.preflight, args.preflight_plan, args.preflight_source_root)
        audit.check("torch" not in sys.modules, "Torch was imported during saved-array audit")
        audit.ram("closing")
        audit.unchanged()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        if not audit.issues:
            audit.issues.append(str(exc))
    status = "verified" if error is None else "failed"
    csv_paths = []
    if audit.case_rows:
        path = output / "recounted_metrics.csv"
        write_csv(path, audit.case_rows)
        csv_paths.append(path)
    if result.get("analysis", {}).get("contrasts"):
        path = output / "secondary_paired_descriptions.csv"
        write_csv(path, result["analysis"]["contrasts"])
        csv_paths.append(path)
    report = {"schema_version": 1, "item": 11, "phase": args.phase, "status": status,
              "checks": audit.checks, "issues": audit.issues, "error": error,
              "started_at_utc": started, "completed_at_utc": utc(), "audit_seconds": time.perf_counter() - start,
              "auditor_sha256": file_hash(Path(__file__)), "runtime": audit.runtime,
              "resource_samples": audit.resource_samples, "input_sha256": audit.inputs,
              "results": result, "evaluation_costs": audit.costs,
              "csv_artifacts": [{"path": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in csv_paths],
              "independence": "Recount independent of controller scoring; same research project, not external replication.",
              "limitations": ["No Torch, checkpoint deserialization, model replay, or new model outputs.",
                              "File identities and source contracts do not by themselves establish physical execution.",
                              "Full Git archive sets, source ZIPs, and G-to-F custody require the separately frozen wrapper.",
                              "Training streams are reconstructed with recorded PCG64/integer calls and compared exactly; this does not assert arbitrary cross-version sampling compatibility.",
                              "Only the primary paired normal-delay8 difference receives a t interval; secondary values are descriptive.",
                              "48 rows form a known census, not independent held-out episodic contents; no event-level bootstrap."]}
    save_json(output / "scientific_audit.json", report)
    print(json.dumps({"status": status, "checks": audit.checks, "issues": len(audit.issues),
                      "report": str(output / "scientific_audit.json")}, sort_keys=True))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
