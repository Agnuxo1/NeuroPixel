"""Item15B: separately frozen retrospective native dynamics of five checkpoints.

Compatibility with all five saved T16 validation outputs is measured and saved
before any extension. A negative compatibility gate is a valid verified result.
State guard crossings are censored scientific observations, not clipped states.
No training, optimizer, checkpoint selection or useful-memory claim is made.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import research_item9_worker as ops

OLD_SOURCE = "83fe135e301f76bc0c74e30c66bb18e067ca5959"
OLD_ARCHIVE = "b64d0e2ba222df76caf72bc9870c7602873e7236"
OLD_PATH = "results/research/09_cloud_runs/37593731891-1-study"
OLD_MANIFEST_SHA = "49cda016746d6f020e22da2e10e09f8706b1f17f15491baf438d6952484f78e6"
MODEL_SHA = "677f3aab045fc6e0ae73ccc19e7093c1b8211e939dcadbe36db8a793130cd345"
SEEDS = (40, 41, 42, 43, 44)
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
    "readout_error_factor": 128, "metric_absolute_tolerance": 1e-9,
    "metric_relative_tolerance": 1e-12,
}


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def safe(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == relative,
            "unsafe relative artifact path")
    path = root / rel
    require(path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()), "missing or indirect artifact")
    return path


def descriptor(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def expected_config(seed):
    return {"family": "neuropixel", "model": {
                "class": "neuropixel.model.NeuroPixel", "c_id": 16, "c": 48,
                "hidden": 128, "steps": 16, "fire_rate": 0.5, "grounded": None, "retina": False},
            "vocab": 37, "height": 10, "width": 8, "out_pos": [9, 7],
            "trainable_parameters": 29856, "initialization": seed,
            "learning_rate": 0.001, "batch_size": 32, "scheduled_updates": 4096,
            "optimizer": "AdamW", "weight_decay": 0.0001, "gradient_clip": 1,
            "loss": "answer_cross_entropy_only", "memorization": False}


def verify_inputs(root, spec):
    require(set(spec) == {"archive_commit", "source_commit", "path", "files_sha256", "files_bytes"},
            "input specification keys differ")
    require((spec["archive_commit"], spec["source_commit"], spec["path"]) ==
            (OLD_ARCHIVE, OLD_SOURCE, OLD_PATH), "historical input identity differs")
    expected = {"archive_manifest.json", "study/development_data/validation.npz"}
    for seed in SEEDS:
        expected.update(f"study/runs/neuropixel_seed{seed}/{name}" for name in
                        ("checkpoint.pt", "validation_predictions.npz", "configuration.json", "run.json"))
    require(set(spec["files_sha256"]) == expected and set(spec["files_bytes"]) == expected,
            "exact22input inventory differs")
    require(spec["files_sha256"]["archive_manifest.json"] == OLD_MANIFEST_SHA,
            "historical manifest anchor differs")
    files = {name: safe(root, name) for name in sorted(expected)}
    for name, path in files.items():
        require(path.stat().st_size == spec["files_bytes"][name]
                and sha(path) == spec["files_sha256"][name], "input bytes differ: " + name)
    manifest = read_json(files["archive_manifest.json"])
    require(manifest["source_commit"] == OLD_SOURCE and manifest["final"] is True
            and manifest["attempt_status"] == "completed", "historical archive not complete")
    for name in expected - {"archive_manifest.json"}:
        require(manifest["files"][name] == {"bytes": spec["files_bytes"][name],
                                          "sha256": spec["files_sha256"][name]},
                "historical descriptor differs: " + name)
    return files


def load_arrays(path):
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def save_arrays(path, arrays, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return {**descriptor(path, root),
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       for name, value in arrays.items()}}


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in
            list(model.named_parameters()) + list(model.named_buffers())}


def add_snapshot(arrays, values, prefix):
    for name, value in values.items():
        arrays[prefix + name] = value.numpy().copy()


def restore(files, seed):
    from neuropixel.model import NeuroPixel
    prefix = f"study/runs/neuropixel_seed{seed}/"
    cfg = expected_config(seed)
    require(read_json(files[prefix + "configuration.json"]) == cfg, "configuration differs")
    run = read_json(files[prefix + "run.json"])
    require(run["status"] == "completed" and run["updates_completed"] == 4096
            and run["configuration"] == cfg and run["run_id"] == f"neuropixel_seed{seed}"
            and run["family"] == "neuropixel" and run["initialization"] == seed
            and run["context"]["source_commit"] == OLD_SOURCE, "run metadata differs")
    require(run["artifacts"]["checkpoint"]["sha256"] == sha(files[prefix + "checkpoint.pt"])
            and run["artifacts"]["validation_predictions"]["sha256"] ==
                sha(files[prefix + "validation_predictions.npz"]), "run artifact linkage differs")
    loaded = torch.load(files[prefix + "checkpoint.pt"], map_location="cpu", weights_only=True)
    require(isinstance(loaded, dict) and set(loaded) == {"state_dict", "configuration"}
            and loaded["configuration"] == cfg, "checkpoint metadata differs")
    with torch.random.fork_rng(devices=[]):
        with torch.device("cpu"):
            model = NeuroPixel(37, (9, 7), c_id=16, c=48, hidden=128,
                               steps=16, fire_rate=0.5, grounded=None, retina=False)
    reference = model.state_dict()
    require(set(loaded["state_dict"]) == set(reference), "checkpoint tensor inventory differs")
    for name, value in loaded["state_dict"].items():
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                and value.device.type == "cpu" and value.shape == reference[name].shape
                and torch.isfinite(value).all(), "checkpoint tensor invalid: " + name)
    model.load_state_dict(loaded["state_dict"], strict=True)
    model.eval()
    require(type(model) is NeuroPixel and sum(p.numel() for p in model.parameters()) == 29856,
            "native architecture differs")
    require(not torch.count_nonzero(model.g_rgb) and not model.g_mask.any()
            and not torch.count_nonzero(model.dictionary()[0]), "default grounding/PAD boundary differs")
    return model


def metadata_rows(validation, index):
    return {name: validation[name][index].copy() for name in
            ("canvas", "target", "role", "group_index", "query_event", "condition_index", "record_id")}


def verify_validation(validation):
    require(validation["canvas"].shape == (1024, 10, 8)
            and validation["canvas"].dtype == np.int64
            and ((validation["canvas"] >= 0) & (validation["canvas"] < 37)).all(), "validation canvas differs")
    for name in ("target", "role", "group_index", "query_event", "condition_index"):
        require(validation[name].shape == (1024,) and validation[name].dtype == np.int64,
                "validation index/label schema differs")
    require(((validation["target"] > 0) & (validation["target"] < 37)).all()
            and validation["record_id"].shape == (1024,)
            and validation["record_id"].dtype.kind in "US", "validation target/ID schema differs")
    require(np.array_equal(validation["role"][:4], [0, 1, 2, 3])
            and np.array_equal(validation["query_event"][:4], [0, 0, 0, 0])
            and np.array_equal(validation["group_index"][:4], [0, 0, 0, 0])
            and np.array_equal(validation["condition_index"][:4], [0, 0, 0, 0]),
            "fixed four-row interpretation differs")


def run_gate(seed, model, validation, saved, output):
    for name in ("target", "role", "group_index", "record_id"):
        require(np.array_equal(saved[name], validation[name]), "saved prediction alignment differs")
    require(saved["logits"].shape == (1024, 37) and saved["logits"].dtype == np.float32
            and np.isfinite(saved["logits"]).all(), "saved logits invalid")
    require(saved["pred"].dtype == np.int64 and np.array_equal(saved["pred"], saved["logits"].argmax(-1)),
            "saved predictions differ from saved logits")
    arrays = metadata_rows(validation, slice(0, 64))
    arrays["row_indices"] = np.arange(4, dtype=np.int64)
    arrays["saved_logits"] = saved["logits"][:64].copy()
    arrays["saved_pred"] = saved["pred"][:64].copy()
    before = snapshot(model)
    add_snapshot(arrays, before, "weight__")
    arrays["rng_before"] = torch.get_rng_state().numpy().copy()
    with torch.no_grad():
        result = model(torch.from_numpy(validation["canvas"][:64]), steps=16)
    arrays["replayed_logits"] = result["logits"].numpy().copy()
    arrays["initial_state"] = result["state"][:4].numpy().copy()
    finite_rows = np.isfinite(arrays["replayed_logits"]).all(axis=-1)
    arrays["replayed_pred"] = np.full(64, -1, dtype=np.int64)
    arrays["replayed_pred"][finite_rows] = arrays["replayed_logits"][finite_rows].argmax(-1)
    after = snapshot(model)
    add_snapshot(arrays, after, "weight_after__")
    arrays["rng_after"] = torch.get_rng_state().numpy().copy()
    invariant = all(torch.equal(value, after[name]) for name, value in before.items())
    rng_same = np.array_equal(arrays["rng_before"], arrays["rng_after"])
    delta = np.abs(arrays["replayed_logits"].astype(np.float64) - arrays["saved_logits"].astype(np.float64))
    allowed = 1e-4 + 1e-5 * np.abs(arrays["saved_logits"].astype(np.float64))
    finite = bool(finite_rows.all() and np.isfinite(arrays["initial_state"]).all())
    logits_match = bool(finite_rows.all() and np.all(delta <= allowed))
    pred_match = bool(np.array_equal(arrays["replayed_pred"], arrays["saved_pred"]))
    path = output / "gates" / f"seed{seed}.npz"
    desc = save_arrays(path, arrays, output)
    rec = {"schema_version": 1, "item": 15, "panel": "learned_retrospective", "seed": seed,
           "passed": finite and logits_match and pred_match,
           "finite_logits_and_selected_initial_states": finite, "logits_match": logits_match,
           "predictions_match": pred_match, "prediction_mismatches":
                int(np.count_nonzero(arrays["replayed_pred"] != arrays["saved_pred"])),
           "maximum_absolute_logit_error": float(delta.max()) if np.isfinite(delta).all() else None,
           "maximum_tolerance_ratio": float((delta / allowed).max()) if np.isfinite(delta).all() else None,
           "weights_and_buffers_unchanged": invariant, "rng_unchanged": rng_same,
           "array_file": desc["path"], "array_sha256": desc["sha256"], "array_bytes": desc["bytes"],
           "arrays": desc["arrays"], "completed_at_utc": ops.now()}
    record_path = output / "gates" / f"seed{seed}.json"
    ops.save(record_path, rec, exclusive=True)
    require(invariant and rng_same, "model or RNG mutated during compatibility gate")
    return rec, descriptor(record_path, output)


class TrajectoryGuard(Exception):
    def __init__(self, reason, time_index):
        super().__init__(reason)
        self.reason, self.time_index = reason, time_index


def allocate_seed(validation, initial):
    a = metadata_rows(validation, slice(0, 4))
    a.update(row_indices=np.arange(4, dtype=np.int64), times=np.arange(257, dtype=np.int64),
             initial_base=initial.copy())
    direction = np.fromfunction(lambda c, r, col: 1 - 2 * ((c + r + col) % 2), (48, 10, 8), dtype=int).astype(np.int8)
    base64 = initial.astype(np.float64)
    rms = np.sqrt(np.mean(base64 * base64, axis=(1, 2, 3)))
    amplitude = 1e-4 * np.maximum(1.0, rms)
    perturbed = (base64 + amplitude[:, None, None, None] * direction).astype(np.float32)
    delta = perturbed.astype(np.float64) - base64
    a.update(direction=direction, nominal_amplitude=amplitude,
             initial_perturbed=perturbed, actual_delta=delta,
             actual_delta_l2=np.sqrt(np.sum(delta * delta, axis=(1, 2, 3))),
             actual_delta_rms=np.sqrt(np.mean(delta * delta, axis=(1, 2, 3))))
    for name, shape in {
        "states": (4, 2, 257, 48, 10, 8), "logits": (4, 2, 257, 37),
        "native_final_state": (4, 2, 48, 10, 8), "native_final_logits": (4, 2, 37)
    }.items():
        a[name] = np.full(shape, np.nan, dtype=np.float32)
    for name in ("token_nll", "state_l2", "state_rms", "state_maxabs", "step_l2"):
        a[name] = np.full((4, 2, 257), np.nan, dtype=np.float64)
    for name in ("pair_rms", "pair_gain"):
        a[name] = np.full((4, 257), np.nan, dtype=np.float64)
    for name in ("observed", "state_finite", "logits_finite"):
        a[name] = np.zeros((4, 2, 257), dtype=np.bool_)
    a["pair_valid"] = np.zeros((4, 257), dtype=np.bool_)
    a["prediction"] = np.full((4, 2, 257), -1, dtype=np.int64)
    a["completed"] = np.zeros((4, 2), dtype=np.bool_)
    return a


def evaluate_seed(seed, model, validation, initial, output, deadline, event):
    arrays = allocate_seed(validation, initial)
    before = snapshot(model)
    add_snapshot(arrays, before, "weight__")
    arrays["rng_before"] = torch.get_rng_state().numpy().copy()
    rows, failed = [], None
    try:
        require(np.isfinite(arrays["actual_delta"]).all()
                and np.isfinite(arrays["actual_delta_rms"]).all()
                and (arrays["actual_delta_rms"] > 0).all(), "perturbation denominator not representable")
        dictionary = model.dictionary().detach()
        with torch.no_grad():
            for row in range(4):
                for branch in range(2):
                    ops.admit(deadline, minimum_seconds=1)
                    event("trajectory_started", seed=seed, row=row, branch=branch)
                    init = arrays["initial_base" if branch == 0 else "initial_perturbed"][row]
                    injected = torch.from_numpy(init.copy()).unsqueeze(0)
                    stop_reason, last = "completed", -1
                    def capture(step, state):
                        nonlocal last
                        if step == 1:
                            state = injected.clone()
                        t = step - 1
                        raw = state[0].detach().numpy().copy()
                        arrays["states"][row, branch, t] = raw
                        arrays["observed"][row, branch, t] = True
                        last = t
                        finite = bool(np.isfinite(raw).all())
                        arrays["state_finite"][row, branch, t] = finite
                        if finite:
                            value = raw.astype(np.float64)
                            arrays["state_l2"][row, branch, t] = np.sqrt(np.sum(value * value))
                            arrays["state_rms"][row, branch, t] = np.sqrt(np.mean(value * value))
                            arrays["state_maxabs"][row, branch, t] = np.max(np.abs(value))
                            if t and arrays["state_finite"][row, branch, t - 1]:
                                difference = value - arrays["states"][row, branch, t - 1].astype(np.float64)
                                arrays["step_l2"][row, branch, t] = np.sqrt(np.sum(difference * difference))
                        readout = (model.read(state[:, :, 9, 7]) @ dictionary.T)
                        readout[:, 0] = -1e4
                        scores = readout[0].numpy().copy()
                        arrays["logits"][row, branch, t] = scores
                        valid_logits = bool(np.isfinite(scores).all())
                        arrays["logits_finite"][row, branch, t] = valid_logits
                        if valid_logits:
                            arrays["prediction"][row, branch, t] = int(scores.argmax())
                            centered = scores.astype(np.float64) - float(scores.max())
                            arrays["token_nll"][row, branch, t] = (
                                np.log(np.exp(centered).sum()) - centered[int(arrays["target"][row])])
                        if not finite:
                            raise TrajectoryGuard("nonfinite_state", t)
                        if arrays["state_maxabs"][row, branch, t] > 1e12:
                            raise TrajectoryGuard("max_abs_state", t)
                        ops.remaining(deadline)
                        return state
                    try:
                        result = model(torch.from_numpy(arrays["canvas"][row:row+1]), steps=257, hook=capture)
                        arrays["completed"][row, branch] = True
                        arrays["native_final_state"][row, branch] = result["state"][0].numpy()
                        arrays["native_final_logits"][row, branch] = result["logits"][0].numpy()
                    except TrajectoryGuard as guard:
                        stop_reason = guard.reason
                        require(last == guard.time_index, "guard observation not retained")
                    rec = {"row_index": row, "branch": "base" if branch == 0 else "perturbed",
                           "observed_count": last + 1, "last_time": last,
                           "corresponding_total_update": 16 + last, "stop_reason": stop_reason}
                    rows.append(rec)
                    event("trajectory_completed", seed=seed, **rec)
    except BaseException as error:
        failed = error
    finally:
        for row in range(4):
            shared = arrays["observed"][row].all(axis=0) & arrays["state_finite"][row].all(axis=0)
            if np.isfinite(arrays["actual_delta_rms"][row]) and arrays["actual_delta_rms"][row] > 0:
                difference = arrays["states"][row, 1, shared].astype(np.float64) - arrays["states"][row, 0, shared].astype(np.float64)
                distance = np.sqrt(np.mean(difference * difference, axis=(1, 2, 3)))
                arrays["pair_valid"][row, shared] = True
                arrays["pair_rms"][row, shared] = distance
                arrays["pair_gain"][row, shared] = distance / arrays["actual_delta_rms"][row]
        after = snapshot(model)
        add_snapshot(arrays, after, "weight_after__")
        arrays["rng_after"] = torch.get_rng_state().numpy().copy()
        unchanged = all(torch.equal(value, after[name]) for name, value in before.items())
        rng_same = np.array_equal(arrays["rng_before"], arrays["rng_after"])
        desc = save_arrays(output / "seeds" / f"seed{seed}.npz", arrays, output)
        rec = {"schema_version": 1, "item": 15, "panel": "learned_retrospective", "seed": seed,
               "status": "failed" if failed is not None else "completed",
               "array_file": desc["path"], "array_sha256": desc["sha256"], "array_bytes": desc["bytes"],
               "arrays": desc["arrays"], "trajectories": rows,
               "weights_and_buffers_unchanged": unchanged, "rng_unchanged": rng_same,
               "completed_at_utc": ops.now(),
               "partial_policy": "Observed prefixes and first guarded state retained. Unobserved slots are NaN/false/-1, not measured outcomes."}
        if failed is not None:
            rec["error"] = {"type": type(failed).__name__, "message": str(failed)}
        path = output / "seeds" / f"seed{seed}.json"
        ops.save(path, rec, exclusive=True)
    if failed is not None:
        raise failed
    require(unchanged and rng_same, "model or RNG changed during extension")
    require(len(rows) == 8, "trajectory inventory incomplete")
    for row in range(4):
        for branch in range(2):
            if arrays["completed"][row, branch]:
                require(np.array_equal(arrays["states"][row, branch, -1], arrays["native_final_state"][row, branch])
                        and np.array_equal(arrays["logits"][row, branch, -1], arrays["native_final_logits"][row, branch], equal_nan=True),
                        "final native state/readout differs from captured observation")
    return rec, descriptor(path, output)


def execute(args):
    global np, torch
    started = time.monotonic()
    deadline = started + 240
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "item": 15, "panel": "learned_retrospective",
              "status": "running", "source_commit": os.environ.get("GITHUB_SHA"),
              "plan_sha256": sha(args.plan), "recipe": RECIPE, "started_at_utc": ops.now(),
              "gates": [], "seed_reports": [], "checks": [],
              "compatibility_gate_passed": None, "extended_panel_executed": False}
    events_path = args.output / "events.jsonl"
    with events_path.open("x", encoding="utf-8") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    def event(name, **fields):
        record = {"event": name, "at_utc": ops.now(), **fields}
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    def check(name, value):
        report["checks"].append({"name": name, "passed": bool(value)})
        require(value, name)
    try:
        plan = read_json(args.plan)
        require(plan["item"] == 15 and plan["status"] == "frozen"
                and plan["evidence_recipe"]["learned"] == RECIPE, "frozen learned recipe differs")
        ops.admit(deadline, minimum_seconds=10)
        require(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "thread environment differs")
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.version.cuda is None and str(torch.__version__) == "2.6.0+cpu",
                "pinned CPU Torch required")
        bindings = plan["implementation_sha256"]
        source_before = {name: sha(safe(ROOT, name)) for name in bindings}
        require(source_before == bindings and sha(ROOT / "neuropixel/model.py") == MODEL_SHA,
                "bound source or historical native model differs")
        files = verify_inputs(args.inputs, plan["inputs"]["learned"])
        report["historical"] = {"source_commit": OLD_SOURCE, "archive_commit": OLD_ARCHIVE,
                                "path": OLD_PATH, "model_sha256": MODEL_SHA}
        report["inputs_sha256"] = plan["inputs"]["learned"]["files_sha256"]
        report["runtime"] = {"python": sys.version.split()[0], "torch": str(torch.__version__),
                             "numpy": np.__version__, "threads": torch.get_num_threads(),
                             "interop_threads": torch.get_num_interop_threads(), "dtype": "float32"}
        validation = load_arrays(files["study/development_data/validation.npz"])
        verify_validation(validation)
        initial_states = {}
        all_passed = True
        for seed in SEEDS:
            ops.admit(deadline, minimum_seconds=1)
            event("gate_started", seed=seed)
            model = restore(files, seed)
            saved = load_arrays(files[f"study/runs/neuropixel_seed{seed}/validation_predictions.npz"])
            gate, desc = run_gate(seed, model, validation, saved, args.output)
            gate_arrays = load_arrays(args.output / gate["array_file"])
            initial_states[seed] = gate_arrays["initial_state"].copy()
            report["gates"].append({"seed": seed, **desc, "passed": gate["passed"]})
            all_passed = all_passed and gate["passed"]
            event("gate_completed", seed=seed, passed=gate["passed"], gate_sha256=desc["sha256"])
            del model, saved, gate_arrays
        summary = {"schema_version": 1, "item": 15, "source_commit": report["source_commit"],
                   "plan_sha256": report["plan_sha256"], "gates": report["gates"],
                   "all_five_passed": bool(all_passed), "completed_at_utc": ops.now(),
                   "policy": "No extended trajectory may precede this durable five-gate summary."}
        summary_path = args.output / "gates_summary.json"
        ops.save(summary_path, summary, exclusive=True)
        report["gates_summary"] = descriptor(summary_path, args.output)
        report["compatibility_gate_passed"] = bool(all_passed)
        event("all_gates_completed", all_five_passed=bool(all_passed),
              summary_sha256=report["gates_summary"]["sha256"])
        if all_passed:
            report["extended_panel_executed"] = True
            for seed in SEEDS:
                ops.admit(deadline, minimum_seconds=1)
                model = restore(files, seed)
                rec, desc = evaluate_seed(seed, model, validation, initial_states[seed], args.output, deadline, event)
                report["seed_reports"].append({"seed": seed, **desc,
                    "completed_trajectories": sum(r["stop_reason"] == "completed" for r in rec["trajectories"]),
                    "guarded_trajectories": sum(r["stop_reason"] != "completed" for r in rec["trajectories"])})
                del model
        # A negative compatibility gate still completes and can be independently audited.
        check("all_five_compatibility_evaluations_preserved", len(report["gates"]) == 5)
        check("compatibility_policy_respected",
              (all_passed and len(report["seed_reports"]) == 5)
              or (not all_passed and not report["seed_reports"] and not (args.output / "seeds").exists()))
        check("conditional_trajectory_inventory",
              not all_passed or sum(r["completed_trajectories"] + r["guarded_trajectories"]
                                    for r in report["seed_reports"]) == 40)
        check("historical_inputs_unchanged",
              all(sha(path) == plan["inputs"]["learned"]["files_sha256"][name] for name, path in files.items()))
        check("bound_source_unchanged", {name: sha(safe(ROOT, name)) for name in bindings} == source_before)
        report.update(status="verified", optimization_steps=0, checkpoints_loaded=5 if not all_passed else 10,
            unique_checkpoints=5, interpretation="Retrospective finite-time directional state dynamics of five fixed checkpoints and four queries from one validation scene. No useful-memory, new generalization, global Lipschitz, operator-norm or infinite-time claim.")
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        event("operational_failure", error_type=type(error).__name__, message=str(error))
    report.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic() - started,
                  final_resource_observation=ops.resource_sample("item15_learned_finish"),
                  events=descriptor(events_path, args.output))
    if report["final_resource_observation"]["available_ram_gib"] < 8:
        report.update(status="failed", final_ram_floor_violation=True)
    ops.save(args.output / "report.json", report, exclusive=True)
    print(json.dumps({"status": report["status"], "compatibility_gate_passed": report["compatibility_gate_passed"],
                      "extended_panel_executed": report["extended_panel_executed"]}), flush=True)
    return int(report["status"] != "verified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
