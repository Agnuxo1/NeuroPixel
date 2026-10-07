"""Independent item15B audit of saved learned-state traces, without Torch.

This program does not replay a learned forward pass. It verifies source/data
identities, archived compatibility outputs, visible-input gold, state-derived
metrics, directional perturbations and a scale-aware readout consistency bound.
Checkpoint deserialization belongs to the separately authenticated binding
component. Twenty pairs from five already selected checkpoints and four known
validation queries are correlated diagnostics, not new training replications.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (40, 41, 42, 43, 44)
ATOL, RTOL = 1e-9, 1e-12
READOUT_FACTOR, READOUT_FLOOR = 128, 1e-5
CAP = 1e12
WEIGHT_SHAPES = {
    "embed.weight": (37, 16), "seed.weight": (48, 16, 1, 1), "seed.bias": (48,),
    "perceive.weight": (96, 1, 3, 3), "f1.weight": (128, 160, 1, 1),
    "f1.bias": (128,), "f2.weight": (48, 128, 1, 1), "f2.bias": (48,),
    "read.weight": (16, 48), "read.bias": (16,), "g_rgb": (37, 3), "g_mask": (37, 1),
}

OLD_SOURCE = "83fe135e301f76bc0c74e30c66bb18e067ca5959"
OLD_ARCHIVE = "b64d0e2ba222df76caf72bc9870c7602873e7236"
OLD_PATH = "results/research/09_cloud_runs/37593731891-1-study"
OLD_MANIFEST_SHA = "49cda016746d6f020e22da2e10e09f8706b1f17f15491baf438d6952484f78e6"
MODEL_SHA = "677f3aab045fc6e0ae73ccc19e7093c1b8211e939dcadbe36db8a793130cd345"
RECIPE = {
    "panel": "retrospective_learned", "seeds": list(SEEDS), "row_indices": [0, 1, 2, 3],
    "vocab": 37, "height": 10, "width": 8, "out_pos": [9, 7],
    "c_id": 16, "state_channels": 48, "hidden": 128, "parameters": 29856,
    "dtype": "float32", "mode": "eval", "fire_rate": 0.5,
    "gate_batch_size": 64, "gate_steps": 16, "gate_absolute_tolerance": 1e-4,
    "gate_relative_tolerance": 1e-5, "gate_predictions_exact": True,
    "all_five_gates_before_any_extension": True,
    "continuation_steps": 256, "native_steps": 257, "initialization_hook_after_step": 1,
    "perturbation_relative_rms": 1e-4, "perturbation_rms_floor": 1.0,
    "direction": "(-1)**(channel+row+column)",
    "perturbed_initial": "float32(float64(base)+nominal_amplitude*direction)",
    "gain_denominator": "RMS(float64(perturbed_initial)-float64(base))",
    "future_input": "held selected canvas; no reseeding after initialization hook",
    "maximum_absolute_state": 1e12, "component_limit_seconds": 240,
    "threads": 1, "interop_threads": 1, "optimization_steps": 0,
    "trained_task_competence_assay": False,
    "readout_error_factor": 128,
    "metric_absolute_tolerance": 1e-9, "metric_relative_tolerance": 1e-12,
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def rooted(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError("unsafe relative path: " + name)
    path = root / path
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("missing or indirect file: " + name)
    return path


def ram():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024 / 2**30


def iso(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Audit:
    def __init__(self):
        self.checks, self.issues, self.results, self.errors = 0, [], {}, {}
    def check(self, value, label):
        self.checks += 1
        if not bool(value):
            self.issues.append(label)
        return bool(value)
    def need(self, value, label):
        if not self.check(value, label):
            raise ValueError(label)
    def same(self, observed, expected, label):
        if isinstance(expected, dict):
            self.need(isinstance(observed, dict) and set(observed) == set(expected), label + ": keys")
            for key in expected:
                self.same(observed[key], expected[key], label + "." + key)
        elif isinstance(expected, list):
            self.need(isinstance(observed, list) and len(observed) == len(expected), label + ": length")
            for i, value in enumerate(expected):
                self.same(observed[i], value, label + "[" + str(i) + "]")
        elif isinstance(expected, float):
            self.check(type(observed) in (int, float) and math.isfinite(observed)
                       and abs(observed - expected) <= ATOL + RTOL * abs(expected), label + ": value")
        else:
            self.check(type(observed) is type(expected) and observed == expected, label + ": value")
    def array(self, observed, expected, label, atol=ATOL, rtol=RTOL):
        self.need(observed.shape == expected.shape, label + ": shape")
        self.check(np.array_equal(np.isnan(observed), np.isnan(expected)), label + ": NaN mask")
        self.check(np.array_equal(np.isposinf(observed), np.isposinf(expected))
                   and np.array_equal(np.isneginf(observed), np.isneginf(expected)), label + ": infinity mask")
        mask = np.isfinite(expected)
        self.need(np.isfinite(observed[mask]).all(), label + ": finite expected positions")
        delta = np.abs(observed[mask] - expected[mask])
        self.errors[label] = float(delta.max()) if delta.size else 0.0
        self.check(np.all(delta <= atol + rtol * np.abs(expected[mask])), label + ": tolerance")
    def exact(self, observed, expected, label):
        self.check(np.array_equal(observed, expected, equal_nan=True), label + ": exact")


def arrays(path, audit):
    with np.load(path, allow_pickle=False) as stored:
        audit.need(len(stored.files) == len(set(stored.files)), str(path) + ": duplicate NPZ names")
        return {key: stored[key].copy() for key in stored.files}


def schema(values, expected, audit, label, descriptors=None):
    audit.need(set(values) == set(expected), label + ": array inventory")
    for key, (shape, dtype) in expected.items():
        a = values[key]
        dtype_ok = a.dtype.kind == "U" if dtype == "unicode" else str(a.dtype) == dtype
        shape_ok = a.shape == shape if shape is not None else a.ndim == 1 and a.size > 0
        audit.need(dtype_ok and shape_ok and not a.dtype.hasobject, label + "." + key + ": schema")
        if descriptors is not None:
            audit.same(descriptors[key], {"shape": list(a.shape), "dtype": str(a.dtype)}, label + "." + key)


def weight_schema():
    result = {}
    for key, shape in WEIGHT_SHAPES.items():
        for prefix in ("weight", "weight_after"):
            result[prefix + "__" + key] = (shape, "bool" if key == "g_mask" else "float32")
    for key in ("rng_before", "rng_after"):
        result[key] = (None, "uint8")
    return result


def weights(values, audit, label, original=None):
    for key in WEIGHT_SHAPES:
        before, after = values["weight__" + key], values["weight_after__" + key]
        audit.need(np.isfinite(before).all(), label + "." + key + ": finite")
        audit.exact(before, after, label + "." + key + ": unchanged")
        if original is not None:
            audit.exact(before, original["weight__" + key], label + "." + key + ": same checkpoint")
    audit.check(not values["weight__g_mask"].any() and not values["weight__g_rgb"].any(),
                label + ": original ungrounded buffers")
    audit.exact(values["rng_before"], values["rng_after"], label + ": RNG unchanged")


def visible_gold(canvas):
    """Parse ordered visible triples without labels, metadata or project code."""
    if canvas.shape != (10, 8) or canvas.dtype.kind not in "iu":
        raise ValueError("invalid visible canvas")
    query = canvas[9]
    if np.flatnonzero(query).tolist() != [5, 6] or query[5] not in (35, 36) or not 1 <= query[6] <= 4:
        raise ValueError("invalid query/output placement")
    facts, nouns, verbs, places = {}, [], [], []
    for row in canvas[:9]:
        occupied = np.flatnonzero(row).tolist()
        if not occupied:
            continue
        if len(occupied) != 3 or occupied != list(range(occupied[0], occupied[0] + 3)):
            raise ValueError("nonliteral fact placement")
        event, role, filler = [int(row[i]) for i in occupied]
        if event not in (35, 36) or role not in (1, 2, 3, 4) or (event, role) in facts:
            raise ValueError("duplicate or invalid event/role")
        target_list = nouns if role in (1, 3) else verbs if role == 2 else places
        low, high = (5, 16) if role in (1, 3) else (17, 26) if role == 2 else (27, 34)
        if not low <= filler <= high:
            raise ValueError("wrong filler category")
        target_list.append(filler)
        facts[event, role] = filler
    if len(facts) != 8 or len(set(nouns)) != 4 or len(set(verbs)) != 2 or len(set(places)) != 2:
        raise ValueError("missing facts or nonunique category fillers")
    return facts[int(query[5]), int(query[6])]


def input_rows(values, reference, count, audit, label):
    for key in ("canvas", "target", "role", "group_index", "query_event", "condition_index", "record_id"):
        audit.check(np.array_equal(values[key], reference[key][:count]), label + ": original " + key)
    for i in range(count):
        audit.check(int(values["target"][i]) == visible_gold(values["canvas"][i]), label + ": visible gold " + str(i))
    audit.check(np.array_equal(values["row_indices"], np.arange(4, dtype=np.int64)), label + ": fixed four rows")
    audit.check(np.array_equal(values["role"][:4], np.arange(4))
                and np.all(values["group_index"][:4] == 0)
                and np.all(values["query_event"][:4] == 0)
                and np.all(values["condition_index"][:4] == 0), label + ": four roles, one known event/group")


def readout(state, stored_logits, finite, values, audit, label):
    """Check two affine readout products with a prospective float32 bound."""
    d = values["weight__embed.weight"].astype(np.float64).copy()
    d[:, :3] = np.where(values["weight__g_mask"], values["weight__g_rgb"], d[:, :3])
    d[0] = 0
    w = values["weight__read.weight"].astype(np.float64)
    bias = values["weight__read.bias"].astype(np.float64)
    x = state[..., :, 9, 7].astype(np.float64)
    x, observed = x[finite], stored_logits[finite].astype(np.float64)
    if not len(x):
        return
    hidden = x @ w.T + bias
    expected = hidden @ d.T
    expected[:, 0] = -10000.0
    absolute_products = (np.abs(x) @ np.abs(w).T + np.abs(bias)) @ np.abs(d).T
    bound = READOUT_FACTOR * np.finfo(np.float32).eps * absolute_products + READOUT_FLOOR
    error = np.abs(observed - expected)
    audit.need(np.isfinite(expected).all(), label + ": float64 readout finite")
    audit.check(np.all(error[:, 1:] <= bound[:, 1:]), label + ": float32 scale-aware readout")
    audit.check(np.all(observed[:, 0] == -10000.0), label + ": PAD readout")
    audit.errors[label + ": maximum readout absolute error"] = float(error.max())
    audit.errors[label + ": maximum readout error/bound"] = float((error[:, 1:] / bound[:, 1:]).max())


def metric_arrays(values, audit, label):
    """One pair at a time avoids a full float64 copy of all state trajectories."""
    state, observed = values["states"], values["observed"]
    finite = np.isfinite(state).all(axis=(-3, -2, -1)) & observed
    logit_finite = np.isfinite(values["logits"]).all(axis=-1) & observed
    audit.exact(values["state_finite"], finite, label + ": state finite flags")
    audit.exact(values["logits_finite"], logit_finite, label + ": logit finite flags")
    metrics = {name: np.full((4, 2, 257), np.nan, np.float64)
               for name in ("token_nll", "state_l2", "state_rms", "state_maxabs", "step_l2")}
    prediction = np.full((4, 2, 257), -1, dtype=np.int64)
    pair_rms = np.full((4, 257), np.nan, np.float64)
    pair_gain = pair_rms.copy()
    pair_valid = finite[:, 0] & finite[:, 1]
    for row in range(4):
        audit.need(ram() >= 8, label + ": RAM during metrics")
        for branch in range(2):
            s = state[row, branch].astype(np.float64)
            good = finite[row, branch]
            norms = np.sqrt(np.sum(s[good] * s[good], axis=(-3, -2, -1)))
            metrics["state_l2"][row, branch, good] = norms
            metrics["state_rms"][row, branch, good] = norms / math.sqrt(48 * 10 * 8)
            metrics["state_maxabs"][row, branch, good] = np.max(np.abs(s[good]), axis=(-3, -2, -1))
            step_good = good[1:] & good[:-1]
            change = s[1:][step_good] - s[:-1][step_good]
            metrics["step_l2"][row, branch, 1:][step_good] = np.sqrt(np.sum(change * change, axis=(-3, -2, -1)))
            lg = logit_finite[row, branch]
            z = values["logits"][row, branch, lg].astype(np.float64)
            if len(z):
                prediction[row, branch, lg] = z.argmax(axis=1)
                z -= z.max(axis=1, keepdims=True)
                metrics["token_nll"][row, branch, lg] = np.logaddexp.reduce(z, axis=1) - z[:, int(values["target"][row])]
        good_pair = pair_valid[row]
        delta = state[row, 1, good_pair].astype(np.float64) - state[row, 0, good_pair].astype(np.float64)
        pair_rms[row, good_pair] = np.sqrt(np.mean(delta * delta, axis=(-3, -2, -1)))
        pair_gain[row, good_pair] = pair_rms[row, good_pair] / float(values["actual_delta_rms"][row])
    for key, expected in metrics.items():
        audit.array(values[key], expected, label + "." + key)
    audit.exact(values["prediction"], prediction, label + ": prediction")
    audit.exact(values["pair_valid"], pair_valid, label + ": common finite observed horizon")
    audit.array(values["pair_rms"], pair_rms, label + ": pair RMS")
    audit.array(values["pair_gain"], pair_gain, label + ": pair gain")
    readout(state, values["logits"], finite & logit_finite, values, audit, label + ": readout")
    return metrics, prediction, finite, logit_finite, pair_valid, pair_rms, pair_gain

def common_schema(count):
    result = {"row_indices": ((4,), "int64"), "canvas": ((count, 10, 8), "int64")}
    for key in ("target", "role", "group_index", "query_event", "condition_index"):
        result[key] = ((count,), "int64")
    result["record_id"] = ((count,), "unicode")
    result.update(weight_schema())
    return result


def gate_schema():
    result = common_schema(64)
    for key in ("saved_logits", "replayed_logits"):
        result[key] = ((64, 37), "float32")
    for key in ("saved_pred", "replayed_pred"):
        result[key] = ((64,), "int64")
    result["initial_state"] = ((4, 48, 10, 8), "float32")
    return result


def trajectory_schema():
    result = common_schema(4)
    result.update({
        "states": ((4, 2, 257, 48, 10, 8), "float32"),
        "logits": ((4, 2, 257, 37), "float32"),
        "prediction": ((4, 2, 257), "int64"),
        "initial_base": ((4, 48, 10, 8), "float32"),
        "initial_perturbed": ((4, 48, 10, 8), "float32"),
        "actual_delta": ((4, 48, 10, 8), "float64"),
        "direction": ((48, 10, 8), "int8"), "times": ((257,), "int64"),
        "native_final_state": ((4, 2, 48, 10, 8), "float32"),
        "native_final_logits": ((4, 2, 37), "float32"), "completed": ((4, 2), "bool"),
        "pair_valid": ((4, 257), "bool"), "pair_rms": ((4, 257), "float64"),
        "pair_gain": ((4, 257), "float64"),
    })
    for key in ("observed", "state_finite", "logits_finite"):
        result[key] = ((4, 2, 257), "bool")
    for key in ("token_nll", "state_l2", "state_rms", "state_maxabs", "step_l2"):
        result[key] = ((4, 2, 257), "float64")
    for key in ("actual_delta_l2", "actual_delta_rms", "nominal_amplitude"):
        result[key] = ((4,), "float64")
    return result


def gate_recount(values, historical, reference, audit, label):
    schema(values, gate_schema(), audit, label)
    input_rows(values, reference, 64, audit, label)
    weights(values, audit, label)
    for key, oldkey in (("saved_logits", "logits"), ("saved_pred", "pred")):
        audit.exact(values[key], historical[oldkey][:64], label + ": archived " + key)
    for key in ("target", "role", "group_index", "record_id"):
        audit.check(np.array_equal(historical[key][:64], reference[key][:64]), label + ": old prediction " + key)
    audit.need(np.isfinite(values["saved_logits"]).all(), label + ": finite archived logits")
    audit.exact(values["saved_pred"], values["saved_logits"].argmax(axis=1), label + ": saved argmax")
    replay_finite = np.isfinite(values["replayed_logits"]).all(axis=1)
    expected_pred = np.full(64, -1, dtype=np.int64)
    expected_pred[replay_finite] = values["replayed_logits"][replay_finite].argmax(axis=1)
    audit.exact(values["replayed_pred"], expected_pred, label + ": replay argmax/undefined")
    difference = np.abs(values["replayed_logits"].astype(np.float64) - values["saved_logits"].astype(np.float64))
    tolerance = 1e-4 + 1e-5 * np.abs(values["saved_logits"].astype(np.float64))
    finite = bool(replay_finite.all() and np.isfinite(values["initial_state"]).all())
    close = bool(replay_finite.all() and np.all(difference <= tolerance))
    pred_equal = bool(np.array_equal(values["saved_pred"], values["replayed_pred"]))
    initial_finite = np.isfinite(values["initial_state"]).all(axis=(1, 2, 3)) & replay_finite[:4]
    readout(values["initial_state"], values["replayed_logits"][:4], initial_finite,
            values, audit, label + ": T16 state readout")
    return {"passed": finite and close and pred_equal,
            "finite_logits_and_selected_initial_states": finite,
            "logits_match": close, "predictions_match": pred_equal,
            "prediction_mismatches": int(np.count_nonzero(values["replayed_pred"] != values["saved_pred"])),
            "maximum_absolute_logit_error": float(difference.max()) if np.isfinite(difference).all() else None,
            "maximum_tolerance_ratio": float((difference / tolerance).max()) if np.isfinite(difference).all() else None,
            "weights_and_buffers_unchanged": True, "rng_unchanged": True}


def perturbation(values, gate, audit, label):
    audit.exact(values["initial_base"], gate["initial_state"], label + ": initial base from gate")
    direction = np.empty((48, 10, 8), dtype=np.int8)
    for c in range(48):
        for r in range(10):
            for col in range(8):
                direction[c, r, col] = 1 if (c + r + col) % 2 == 0 else -1
    audit.exact(values["direction"], direction, label + ": literal checkerboard")
    base = values["initial_base"].astype(np.float64)
    rms = np.sqrt(np.mean(base * base, axis=(1, 2, 3)))
    amplitude = 1e-4 * np.maximum(1.0, rms)
    perturbed = (base + amplitude[:, None, None, None] * direction).astype(np.float32)
    delta = perturbed.astype(np.float64) - base
    delta_l2 = np.sqrt(np.sum(delta * delta, axis=(1, 2, 3)))
    delta_rms = delta_l2 / math.sqrt(48 * 10 * 8)
    audit.array(values["nominal_amplitude"], amplitude, label + ": nominal amplitude")
    audit.exact(values["initial_perturbed"], perturbed, label + ": perturbed float32 cast")
    audit.exact(values["actual_delta"], delta, label + ": exact represented delta")
    audit.array(values["actual_delta_l2"], delta_l2, label + ": delta L2")
    audit.array(values["actual_delta_rms"], delta_rms, label + ": delta RMS")
    audit.need(np.isfinite(delta_rms).all() and np.all(delta_rms > 0), label + ": positive actual perturbation")
    audit.exact(values["states"][:, 0, 0], values["initial_base"], label + ": base t0")
    audit.exact(values["states"][:, 1, 0], values["initial_perturbed"], label + ": perturbed t0")
    audit.exact(values["times"], np.arange(257, dtype=np.int64), label + ": continuation times")


def null_number(value):
    return float(value) if math.isfinite(float(value)) else None


def censor_recount(values, metrics, finite, logit_finite, audit, label):
    """Return finite-prefix endpoints, preserving the first stopping observation."""
    rows = []
    for row in range(4):
        for branch in range(2):
            observed = values["observed"][row, branch]
            count = int(observed.sum())
            audit.need(1 <= count <= 257 and np.array_equal(observed, np.arange(257) < count),
                       label + ": nonempty contiguous prefix")
            last = count - 1
            audit.check(np.isnan(values["states"][row, branch, count:]).all()
                        and np.isnan(values["logits"][row, branch, count:]).all(),
                        label + ": unobserved raw arrays NaN")
            bad_state = np.flatnonzero(observed & ~finite[row, branch])
            cap = np.flatnonzero(observed & finite[row, branch]
                                 & (metrics["state_maxabs"][row, branch] > CAP))
            audit.check(len(bad_state) <= 1 and (not len(bad_state) or bad_state[0] == last),
                        label + ": first nonfinite state retained at end")
            audit.check(len(cap) <= 1 and (not len(cap) or cap[0] == last),
                        label + ": first finite cap crossing retained at end")
            if len(bad_state):
                reason = "nonfinite_state"
            elif len(cap):
                reason = "max_abs_state"
            else:
                reason = "completed"
            complete = reason == "completed"
            audit.check(not complete or count == 257, label + ": no unexplained short trajectory")
            audit.check(bool(values["completed"][row, branch]) == complete, label + ": completed flag")
            if complete:
                audit.exact(values["native_final_state"][row, branch], values["states"][row, branch, 256],
                            label + ": native final state")
                audit.exact(values["native_final_logits"][row, branch], values["logits"][row, branch, 256],
                            label + ": native final logits")
            else:
                audit.check(np.isnan(values["native_final_state"][row, branch]).all()
                            and np.isnan(values["native_final_logits"][row, branch]).all(),
                            label + ": no invented native return for censored trajectory")
            rows.append({"row_index": row, "branch": "base" if branch == 0 else "perturbed", "observed_count": count,
                         "last_time": last, "corresponding_total_update": 16 + last, "stop_reason": reason,
                         "last_state_l2": null_number(metrics["state_l2"][row, branch, last]),
                         "last_state_rms": null_number(metrics["state_rms"][row, branch, last]),
                         "last_token_nll": null_number(metrics["token_nll"][row, branch, last])})
    return rows

def expected_config(seed):
    return {"family": "neuropixel", "model": {
        "class": "neuropixel.model.NeuroPixel", "c_id": 16, "c": 48, "hidden": 128,
        "steps": 16, "fire_rate": 0.5, "grounded": None, "retina": False},
        "vocab": 37, "height": 10, "width": 8, "out_pos": [9, 7],
        "trainable_parameters": 29856, "initialization": seed, "learning_rate": 0.001,
        "batch_size": 32, "scheduled_updates": 4096, "optimizer": "AdamW",
        "weight_decay": 0.0001, "gradient_clip": 1,
        "loss": "answer_cross_entropy_only", "memorization": False}


def original_inputs(root, spec, audit):
    audit.need(set(spec) == {"archive_commit", "source_commit", "path", "files_sha256", "files_bytes"},
               "historical input spec keys")
    audit.need((spec["archive_commit"], spec["source_commit"], spec["path"]) ==
               (OLD_ARCHIVE, OLD_SOURCE, OLD_PATH), "historical identities")
    names = {"archive_manifest.json", "study/development_data/validation.npz"}
    for seed in SEEDS:
        names.update(f"study/runs/neuropixel_seed{seed}/{name}" for name in
                     ("checkpoint.pt", "configuration.json", "run.json", "validation_predictions.npz"))
    audit.need(set(spec["files_sha256"]) == names and set(spec["files_bytes"]) == names, "22 original files")
    audit.need(spec["files_sha256"]["archive_manifest.json"] == OLD_MANIFEST_SHA, "old manifest anchor")
    files = {name: rooted(root, name) for name in sorted(names)}
    for name, path in files.items():
        audit.need(path.stat().st_size == spec["files_bytes"][name]
                   and sha(path) == spec["files_sha256"][name], "original bytes: " + name)
    manifest = read_json(files["archive_manifest.json"])
    audit.need(manifest["source_commit"] == OLD_SOURCE and manifest["final"] is True
               and manifest["attempt_status"] == "completed", "original archive completed")
    for name in names - {"archive_manifest.json"}:
        audit.same(manifest["files"][name], {"bytes": spec["files_bytes"][name],
                   "sha256": spec["files_sha256"][name]}, "old archive descriptor." + name)
    for seed in SEEDS:
        prefix = f"study/runs/neuropixel_seed{seed}/"
        cfg, run = read_json(files[prefix + "configuration.json"]), read_json(files[prefix + "run.json"])
        audit.need(cfg == expected_config(seed) and run["configuration"] == cfg, "original configuration")
        audit.need(run["run_id"] == f"neuropixel_seed{seed}" and run["family"] == "neuropixel"
                   and run["initialization"] == seed and run["status"] == "completed"
                   and run["updates_completed"] == 4096
                   and run["context"]["source_commit"] == OLD_SOURCE, "original run identity")
        for key, filename in (("checkpoint", "checkpoint.pt"), ("validation_predictions", "validation_predictions.npz")):
            audit.need(run["artifacts"][key]["sha256"] == spec["files_sha256"][prefix + filename],
                       "old run artifact linkage")
    return files


def native_descriptor(root, record, expected_name, audit):
    audit.need(record["path"] == expected_name, "native descriptor identity: " + expected_name)
    path = rooted(root, expected_name)
    audit.need(record["bytes"] == path.stat().st_size and record["sha256"] == sha(path),
               "native descriptor bytes: " + expected_name)
    return path


def payload(root, record, expected_name, audit):
    audit.need(record["array_file"] == expected_name, "NPZ payload identity")
    path = rooted(root, expected_name)
    audit.need(record["array_bytes"] == path.stat().st_size and record["array_sha256"] == sha(path),
               "NPZ payload hash/bytes")
    return arrays(path, audit)


def saved_tree(root, audit):
    result = {}
    for path in sorted(root.rglob("*")):
        audit.need(not path.is_symlink(), "native symlink")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha(path)
    return result


def validate_binding(args, plan, commit, native_before, audit):
    path = rooted(args.binding, "checkpoint_binding.json")
    record = read_json(path)
    audit.need(record["schema_version"] == 1 and record["item"] == 15
               and record["status"] == "verified" and record["issues"] == []
               and type(record["checks"]) is int and record["checks"] > 0, "checkpoint binding result")
    audit.need(record["source_commit"] == commit and record["plan_sha256"] == sha(args.plan)
               and record["original_source_commit"] == OLD_SOURCE
               and record["original_archive_commit"] == OLD_ARCHIVE, "checkpoint binding source")
    audit.same(record["seeds"], list(SEEDS), "binding seeds")
    audit.same(record["input_sha256"], plan["inputs"]["learned"]["files_sha256"], "binding old inputs")
    expected = {k: v for k, v in native_before.items() if k.endswith((".json", ".npz"))}
    audit.same(record["native_sha256"], expected, "binding native files")
    for key in ("checkpoint_weights_exact", "configuration_exact", "constructor_buffers_zero"):
        audit.need(record[key] is True, "checkpoint binding assertion: " + key)
    return sha(path)


def event_audit(events, report, gate_records, seed_records, audit):
    audit.need(all(isinstance(e, dict) and "event" in e and "at_utc" in e for e in events), "event schema")
    stamps = [iso(e["at_utc"]) for e in events]
    audit.need(stamps == sorted(stamps), "monotone event timestamps")
    cursor = 0
    for seed, record in zip(SEEDS, gate_records):
        audit.need(events[cursor]["event"] == "gate_started" and events[cursor]["seed"] == seed, "gate start order")
        audit.need(events[cursor + 1]["event"] == "gate_completed"
                   and events[cursor + 1]["seed"] == seed
                   and events[cursor + 1]["passed"] == record["passed"]
                   and events[cursor + 1]["gate_sha256"] == report["gates"][cursor // 2]["sha256"],
                   "gate completion event")
        audit.need(stamps[cursor] <= iso(record["completed_at_utc"]) <= stamps[cursor + 1], "gate chronology")
        cursor += 2
    e = events[cursor]
    audit.need(e["event"] == "all_gates_completed"
               and e["all_five_passed"] == report["compatibility_gate_passed"]
               and e["summary_sha256"] == report["gates_summary"]["sha256"], "durable five-gate event")
    gate_time = stamps[cursor]
    cursor += 1
    for seed, record in zip(SEEDS, seed_records):
        for row in record["trajectories"]:
            audit.need(cursor + 1 < len(events), "missing trajectory events")
            start, finish = events[cursor], events[cursor + 1]
            branch = 0 if row["branch"] == "base" else 1
            audit.need(start["event"] == "trajectory_started" and start["seed"] == seed
                       and start["row"] == row["row_index"] and start["branch"] == branch,
                       "trajectory start identity")
            audit.need(finish["event"] == "trajectory_completed" and finish["seed"] == seed
                       and all(finish[k] == v for k, v in row.items()), "trajectory completion identity")
            audit.need(stamps[cursor] >= gate_time, "all five gates precede trajectory")
            cursor += 2
    audit.need(cursor == len(events), "no unexpected events")
    return gate_time


def run(args, audit):
    global np
    audit.need(ram() >= 8, "RAM admission below 8 GiB")
    audit.results["available_ram_gib_at_admission"] = ram()
    import numpy as np
    plan = read_json(args.plan)
    audit.need(plan.get("item") == 15 and plan.get("status") == "frozen"
               and plan["evidence_recipe"]["learned"] == RECIPE, "frozen learned recipe")
    runtime = plan["runtime"]
    audit.need(platform.python_version() == runtime["python"]
               and np.__version__ == runtime["packages"]["numpy"], "auditor runtime versions")
    bindings = plan["implementation_sha256"]
    audit.need(isinstance(bindings, dict) and bool(bindings), "source bindings present")
    source_before = {name: sha(rooted(ROOT, name)) for name in bindings}
    audit.same(source_before, bindings, "current source hashes")
    audit.need(source_before["neuropixel/model.py"] == MODEL_SHA, "unchanged original model source")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True, timeout=15).stdout.strip()
    audit.need(commit == os.environ.get("GITHUB_SHA"), "current HEAD/environment identity")
    plan_sha = sha(args.plan)
    files = original_inputs(args.inputs, plan["inputs"]["learned"], audit)
    native_before = saved_tree(args.native, audit)
    report = read_json(rooted(args.native, "report.json"))
    audit.need(report["schema_version"] == 1 and report["item"] == 15
               and report["panel"] == "learned_retrospective" and report["status"] == "verified",
               "native completed scientific result")
    audit.need(report["source_commit"] == commit and report["plan_sha256"] == plan_sha
               and report["recipe"] == RECIPE, "native current source/recipe")
    audit.same(report["historical"], {"source_commit": OLD_SOURCE, "archive_commit": OLD_ARCHIVE,
               "path": OLD_PATH, "model_sha256": MODEL_SHA}, "native historical provenance")
    audit.same(report["inputs_sha256"], plan["inputs"]["learned"]["files_sha256"], "native old hashes")
    audit.same(report["runtime"], {"python": runtime["python"], "torch": runtime["packages"]["torch"],
               "numpy": runtime["packages"]["numpy"], "threads": 1, "interop_threads": 1,
               "dtype": "float32"}, "native runtime")
    binding_sha = validate_binding(args, plan, commit, native_before, audit)
    validation = arrays(files["study/development_data/validation.npz"], audit)
    audit.need(validation["canvas"].shape == (1024, 10, 8)
               and validation["canvas"].dtype == np.int64
               and np.all((validation["canvas"] >= 0) & (validation["canvas"] < 37)), "original validation canvases")
    for key in ("target", "role", "group_index", "query_event", "condition_index"):
        audit.need(validation[key].shape == (1024,) and validation[key].dtype == np.int64, "validation metadata schema")
    audit.need(validation["record_id"].shape == (1024,) and validation["record_id"].dtype.kind == "U",
               "validation record identifiers")
    audit.need(np.array_equal(validation["role"], np.arange(1024) % 4)
               and np.array_equal(validation["query_event"], (np.arange(1024) // 4) % 2)
               and np.array_equal(validation["group_index"], np.arange(1024) // 8)
               and np.all(validation["condition_index"] == 0), "fixed BASE validation order")
    audit.need(len(set(validation["record_id"].tolist())) == 1024, "unique original record identities")
    audit.need(len(report["gates"]) == 5 and [x["seed"] for x in report["gates"]] == list(SEEDS),
               "five gates ordered exactly")
    gates, gate_records, gate_metrics = [], [], []
    expected_files = {"report.json", "gates_summary.json", "events.jsonl"}
    for seed, descriptor in zip(SEEDS, report["gates"]):
        audit.need(ram() >= 8, "RAM before gate read")
        prefix = f"gates/seed{seed}"
        record = read_json(native_descriptor(args.native, descriptor, prefix + ".json", audit))
        audit.need(record["schema_version"] == 1 and record["item"] == 15
                   and record["panel"] == "learned_retrospective" and record["seed"] == seed, "gate record identity")
        values = payload(args.native, record, prefix + ".npz", audit)
        schema(values, gate_schema(), audit, prefix, record["arrays"])
        historical = arrays(files[f"study/runs/neuropixel_seed{seed}/validation_predictions.npz"], audit)
        expected_old_keys = {"logits", "pred", "nll", "target", "role", "group_index", "record_id"}
        audit.need(set(historical) == expected_old_keys and historical["logits"].shape == (1024, 37)
                   and historical["logits"].dtype == np.float32, "archived prediction schema")
        score = gate_recount(values, historical, validation, audit, prefix)
        for key, value in score.items():
            audit.same(record[key], value, prefix + "." + key)
        audit.need(descriptor["passed"] is score["passed"], "global gate flag")
        gates.append(values)
        gate_records.append(record)
        gate_metrics.append({"seed": seed, **score})
        expected_files.update((prefix + ".json", prefix + ".npz"))
    passed = all(row["passed"] for row in gate_metrics)
    audit.need(report["compatibility_gate_passed"] is passed
               and report["extended_panel_executed"] is passed, "conditional extension decision")
    summary = read_json(native_descriptor(args.native, report["gates_summary"], "gates_summary.json", audit))
    audit.need(summary["schema_version"] == 1 and summary["item"] == 15
               and summary["source_commit"] == commit and summary["plan_sha256"] == plan_sha
               and summary["gates"] == report["gates"] and summary["all_five_passed"] is passed,
               "durable all-gate summary")
    audit.need(len(report["seed_reports"]) == (5 if passed else 0), "conditional seed inventory")
    seed_records, seed_results = [], []
    if passed:
        audit.need([x["seed"] for x in report["seed_reports"]] == list(SEEDS), "all seeds extended without selection")
        for seed, descriptor, gate in zip(SEEDS, report["seed_reports"], gates):
            audit.need(ram() >= 8, "RAM before full state read")
            prefix = f"seeds/seed{seed}"
            record = read_json(native_descriptor(args.native, descriptor, prefix + ".json", audit))
            audit.need(record["schema_version"] == 1 and record["item"] == 15
                       and record["panel"] == "learned_retrospective" and record["seed"] == seed
                       and record["status"] == "completed", "seed completed record")
            values = payload(args.native, record, prefix + ".npz", audit)
            schema(values, trajectory_schema(), audit, prefix, record["arrays"])
            input_rows(values, validation, 4, audit, prefix)
            weights(values, audit, prefix, gate)
            perturbation(values, gate, audit, prefix)
            metrics, prediction, finite, logit_finite, valid, distance, gain = metric_arrays(values, audit, prefix)
            trajectories = censor_recount(values, metrics, finite, logit_finite, audit, prefix)
            audit.need(len(record["trajectories"]) == 8, "all eight trajectory records")
            for stored, computed in zip(record["trajectories"], trajectories):
                required = {key: computed[key] for key in ("row_index", "branch", "observed_count", "last_time",
                                                          "corresponding_total_update", "stop_reason")}
                audit.same(stored, required, prefix + ": trajectory summary")
            complete_n = int(values["completed"].sum())
            audit.need(descriptor["completed_trajectories"] == complete_n
                       and descriptor["guarded_trajectories"] == 8 - complete_n, "complete/censored counts")
            audit.need(record["weights_and_buffers_unchanged"] is True and record["rng_unchanged"] is True,
                       "seed invariant flags")
            pairs = []
            for row in range(4):
                valid_times = np.flatnonzero(valid[row])
                audit.need(len(valid_times) > 0, "nonempty common initial pair")
                peak_i = int(valid_times[np.argmax(gain[row, valid_times])])
                last = int(valid_times[-1])
                pairs.append({"row_index": row, "role": int(values["role"][row]),
                    "target": int(values["target"][row]), "group_index": int(values["group_index"][row]),
                    "actual_delta_rms": float(values["actual_delta_rms"][row]),
                    "nominal_amplitude": float(values["nominal_amplitude"][row]),
                    "common_finite_states": len(valid_times), "last_shared_time": last,
                    "peak_observed_gain": float(gain[row, peak_i]), "first_peak_time": peak_i,
                    "last_shared_gain": float(gain[row, last]), "last_shared_rms": float(distance[row, last]),
                    "gain_by_time": [null_number(x) for x in gain[row]]})
            seed_results.append({"seed": seed, "completed_trajectories": complete_n,
                                 "guarded_trajectories": 8 - complete_n,
                                 "trajectories": trajectories, "pairs": pairs})
            seed_records.append(record)
            expected_files.update((prefix + ".json", prefix + ".npz"))
            del values, metrics, prediction, finite, logit_finite, valid, distance, gain
    else:
        audit.need(not (args.native / "seeds").exists(), "negative gate has no continuation directory")
    audit.need(set(native_before) == expected_files, "exact native output file inventory")
    events_path = native_descriptor(args.native, report["events"], "events.jsonl", audit)
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    gate_time = event_audit(events, report, gate_records, seed_records, audit)
    audit.need(max(iso(r["completed_at_utc"]) for r in gate_records)
               <= iso(summary["completed_at_utc"]) <= gate_time, "durable summary chronology")
    audit.need(iso(report["started_at_utc"]) <= iso(events[0]["at_utc"])
               and iso(events[-1]["at_utc"]) <= iso(report["completed_at_utc"]), "native event bounds")
    names = ["all_five_compatibility_evaluations_preserved", "compatibility_policy_respected",
             "conditional_trajectory_inventory", "historical_inputs_unchanged", "bound_source_unchanged"]
    audit.same(report["checks"], [{"name": name, "passed": True} for name in names], "native checks")
    audit.need(report["optimization_steps"] == 0 and report["unique_checkpoints"] == 5
               and report["checkpoints_loaded"] == (10 if passed else 5), "no training/unique checkpoint inventory")
    audit.need(report["final_resource_observation"]["available_ram_gib"] >= 8, "recorded native RAM boundary")
    audit.same(saved_tree(args.native, audit), native_before, "native output bytes unchanged")
    audit.same({name: sha(path) for name, path in files.items()}, plan["inputs"]["learned"]["files_sha256"],
               "original input bytes unchanged")
    audit.same({name: sha(rooted(ROOT, name)) for name in bindings}, source_before, "source bytes unchanged")
    audit.need(sha(args.plan) == plan_sha, "plan bytes unchanged")
    audit.need(sha(args.binding / "checkpoint_binding.json") == binding_sha, "binding receipt unchanged")
    audit.need(ram() >= 8, "RAM completion below 8 GiB")
    audit.results.update(source_commit=commit, plan_sha256=plan_sha,
        original_source_commit=OLD_SOURCE, original_archive_commit=OLD_ARCHIVE,
        implementation_sha256=source_before, input_sha256=plan["inputs"]["learned"]["files_sha256"],
        native_sha256=native_before, binding_receipt_sha256=binding_sha,
        auditor_sha256=sha(Path(__file__)), compatibility_gate_passed=passed,
        extended_panel_executed=passed, gates=gate_metrics, seeds=seed_results,
        trajectories=40 if passed else 0, pairs=20 if passed else 0,
        observed_states=sum(r["observed_count"] for s in seed_results for r in s["trajectories"]),
        maximum_absolute_errors=audit.errors, metric_tolerance={"atol": ATOL, "rtol": RTOL},
        readout_tolerance={"factor": READOUT_FACTOR, "epsilon": float(np.finfo(np.float32).eps),
            "absolute_floor": READOUT_FLOOR,
            "scale": "(abs(output_state)@abs(read_weight).T+abs(read_bias))@abs(effective_dictionary).T"},
        available_ram_gib_at_finish=ram())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "inputs", "native", "binding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit, started = Audit(), time.monotonic()
    try:
        run(args, audit)
    except BaseException as error:
        audit.issues.append(type(error).__name__ + ": " + str(error))
        audit.results["failure_traceback"] = traceback.format_exc()
    result = {"schema_version": 1, "item": 15, "panel": "learned_retrospective",
        "status": "failed" if audit.issues else "verified", "checks": audit.checks,
        "issues": audit.issues, "results": audit.results,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started,
        "runtime": {"python": platform.python_version(), "numpy": globals().get("np").__version__
                    if "np" in globals() else None, "numerical_threads": 1},
        "limits": [
            "No Torch, producer/model imports, checkpoint deserialization, or learned forward replay.",
            "Checkpoint contents are authenticated by the separately hashed Torch binding receipt.",
            "Archived T16 output compatibility does not establish historical hidden-state identity.",
            "All continuations retain current input and use dense eval dynamics; no stored-memory claim.",
            "Twenty directional pairs share five checkpoints and four queries from one known validation scene.",
            "Finite directional gain is not an operator norm, global Lipschitz bound, or infinite-time certificate.",
            "Censored trajectories retain their first guarded state; pair metrics use only their common finite observed horizon.",
            "Observed NaN/Inf logits remain distinct from unobserved slots; no undefined value is counted as a prediction.",
            "Token likelihood/prediction diagnostics are not a new task-competence or generalization assay.",
            "Resource measurements are admission/boundary observations under the external supervisor."
        ]}
    with (args.output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "checks": audit.checks, "issues": len(audit.issues)}), flush=True)
    return int(bool(audit.issues))


if __name__ == "__main__":
    raise SystemExit(main())
