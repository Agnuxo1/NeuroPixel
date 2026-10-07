"""Independent saved-array recount of item-11 untrained memory-path diagnostics.

No Torch import, checkpoint deserialization, model inference or producer import.
The operational wrapper separately authenticates complete Git/ZIP inventories.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import numpy as np

RECIPE = {"active_f2_seed": 110002, "active_f2_std": 0.01, "cue_pos": [0, 0],
          "cue_steps": 3, "delays": [0, 4, 16, 32], "gap_steps": 3,
          "init_seed": 110001, "out_pos": [2, 2], "query_pos": [2, 1],
          "query_steps": 10, "shape": [3, 3], "vocabulary": 8}
ATOL, RTOL = 1e-6, 1e-5

def now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def safe(root, name):
    value = PurePosixPath(name)
    if not isinstance(name, str) or not name or value.is_absolute() or ".." in value.parts or value.as_posix() != name or name == ".":
        raise ValueError("unsafe artifact path")
    result = (root / name).resolve()
    result.relative_to(root.resolve())
    if not result.is_file() or result.is_symlink():
        raise ValueError("artifact missing or irregular")
    return result

def tensor_digest(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        h.update(canonical({"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}))
        h.update(array.tobytes(order="C"))
    return h.hexdigest()

class Recount:
    def __init__(self, output):
        self.output = output
        self.report = {"schema_version": 1, "item": 11, "phase": "probe",
                       "status": "running", "started_at_utc": now(),
                       "checks": [], "issues": [], "files": {}, "observations": [],
                       "scope": "Saved arrays and exact source bindings; no model execution, Torch seed reconstruction, checkpoint deserialization or external custody."}
    def check(self, ok, name, detail=None):
        row = {"check": name, "passed": bool(ok)}
        if detail is not None:
            row["detail"] = detail
        self.report["checks"].append(row)
        if not ok:
            raise ValueError(name)
    def equal(self, a, b, name):
        self.check(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b), name)
    def close(self, a, b, name):
        error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        self.check(a.shape == b.shape and np.allclose(a, b, atol=ATOL, rtol=RTOL), name,
                   {"max_abs_difference": error, "atol": ATOL, "rtol": RTOL})
    def save(self):
        self.report["check_count"] = len(self.report["checks"])
        self.report["failed_checks"] = sum(not r["passed"] for r in self.report["checks"])
        temporary = self.output / "_pending_probe_audit.json"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, self.output / "probe_audit.json")

def recount(root, plan_path, source_root, audit):
    source_plan = read_json(plan_path)
    producer = read_json(root / "probe/probe_report.json")
    state = read_json(root / "memory_probe_status.json")
    audit.check(source_plan["schema_version"] == 1 and source_plan["item"] == 11
                and source_plan["phase"] == "probe" and source_plan["status"] == "frozen", "frozen_probe_plan")
    audit.check(source_plan["probe_recipe"] == RECIPE == producer["recipe"], "prospective_recipe_identity")
    audit.check(source_plan["probe_recipe_sha256"] == hashlib.sha256(canonical(RECIPE)).hexdigest(),
                "canonical_recipe_hash")
    audit.check(producer["status"] == "verified" and producer["training_performed"] is False
                and state["status"] == "completed", "completed_untrained_probe")
    audit.check(state["report_sha256"] == sha(root / "probe/probe_report.json"), "producer_report_receipt")
    audit.check(producer["source"]["plan_sha256"] == sha(plan_path)
                == state["source"]["plan_sha256"], "source_plan_hash")
    audit.check(producer["source"]["implementation_sha256"] == source_plan["implementation_sha256"],
                "producer_source_bindings")
    audit.check(producer["source"]["source_commit"] == state["source"]["source_commit"]
                == state["source_after"]["source_commit"], "reported_source_stability")
    for name, digest in source_plan["implementation_sha256"].items():
        audit.check(sha(safe(source_root, name)) == digest, "exact_source:" + name)
    audit.check(producer["tolerance"] == {"atol": ATOL, "rtol": RTOL}, "fixed_tolerances")
    audit.check(len(producer["checks"]) == len({r["check"] for r in producer["checks"]}) == 47
                and all(r["passed"] is True for r in producer["checks"])
                and state["checks"] == 47, "producer_check_inventory")
    audit.check(len(producer["conditions"]) == 10, "producer_condition_inventory")
    values = {}
    for name, reference in producer["files"].items():
        path = safe(root, reference["path"])
        audit.check(path.stat().st_size == reference["bytes"] and sha(path) == reference["sha256"],
                    "saved_file_identity:" + name)
        audit.report["files"][name] = reference
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            audit.check(set(arrays) == set(reference["arrays"]), "array_inventory:" + name)
            for key, array in arrays.items():
                meta = reference["arrays"][key]
                audit.check(list(array.shape) == meta["shape"] and str(array.dtype) == meta["dtype"]
                            and array.dtype != object, "array_shape_dtype:" + name + ":" + key)
                if array.dtype.kind in "fciub":
                    audit.check(bool(np.isfinite(array).all()), "finite:" + name + ":" + key)
            values[name] = arrays
    audit.check(len(producer["files"]) == 18 and len(values) == 17, "full_probe_file_inventory")
    parameters, buffers = values["diagnostic_parameters"], values["diagnostic_buffers"]
    audit.check(sum(a.size for a in parameters.values()) == producer["parameter_count"] == 29392,
                "actual_saved_parameter_count")
    audit.check(np.count_nonzero(parameters["f2.weight"]) > 0, "active_final_update_projection")
    hashes = {"parameters": tensor_digest(parameters), "buffers": tensor_digest(buffers)}
    audit.check(hashes == producer["state_hashes_before"] == producer["state_hashes_after"],
                "saved_parameters_buffers_match_reported_before_after_hashes")
    initial, final_rng = values["initial_rng"], values["final_rng"]
    audit.equal(initial["torch_global"], final_rng["torch_global"], "saved_global_RNG_unchanged")
    audit.check(initial["torch_global"].dtype == np.uint8
                and initial["private_f2_before"].dtype == initial["private_f2_after"].dtype == np.uint8
                and not np.array_equal(initial["private_f2_before"], initial["private_f2_after"]),
                "private_initializer_consumed_separate_stream")
    inputs = values["inputs"]
    cue = np.zeros((6, 3, 3), dtype=np.int64)
    cue[:, 0, 0] = np.arange(2, 8, dtype=np.int64)
    blank = np.zeros_like(cue)
    query = np.zeros_like(cue)
    query[:, 2, 1] = 1
    donor = np.asarray([1, 2, 3, 4, 5, 0], dtype=np.int64)
    for name, expected in (("cue", cue), ("blank", blank), ("query", query),
                           ("cue_token_ids", np.arange(2, 8, dtype=np.int64)),
                           ("donor_permutation", donor), ("distractor", cue[donor])):
        audit.equal(inputs[name], expected, "literal_input:" + name)
    ref = values["default_query_reference"]
    audit.check(ref["state"].shape == (6, 48, 3, 3)
                and ref["logits"].shape == (6, 8) and ref["update_states"].shape == (6, 11, 48, 3, 3),
                "query_reference_shapes")
    audit.equal(ref["state"], ref["update_states"][:, -1], "query_trace_endpoint")
    audit.equal(ref["state"], np.repeat(ref["state"][:1], 6, axis=0), "identical_query_batch_states")
    audit.equal(ref["logits"], np.repeat(ref["logits"][:1], 6, axis=0), "identical_query_batch_logits")
    histories = {"empty": [], "cue": [(cue, 3)], "cue_delay4": [(cue, 3)] + [(blank, 3)]*4,
                 "cue_delay16": [(cue, 3)] + [(blank, 3)]*16,
                 "cue_distractor_delay4": [(cue, 3), (cue[donor], 3)] + [(blank, 3)]*4}
    for name, history in histories.items():
        saved = values["separate_calls_" + name]
        frames = np.stack([r[0] for r in history], axis=1) if history else np.empty((6, 0, 3, 3), dtype=np.int64)
        audit.equal(saved["history_frames"], frames, "literal_history_frames:" + name)
        audit.equal(saved["history_steps"], np.asarray([r[1] for r in history], dtype=np.int64),
                    "literal_history_updates:" + name)
        audit.check(saved["history_states"].shape == (6, len(history), 48, 3, 3), "history_activity_shape:" + name)
        audit.equal(saved["state"], ref["state"], "separate_call_state_reset:" + name)
        audit.equal(saved["logits"], ref["logits"], "separate_call_logit_reset:" + name)
    own = values["returned_state_ownership"]
    audit.check(own["finite_sentinel"].shape == () and float(own["finite_sentinel"]) == 3.25
                and bool((own["mutated_returned_state"] == 3.25).all()), "actual_returned_state_mutation")
    for key, baseline in (("original_returned_state", "state"), ("fresh_state", "state"),
                          ("original_returned_logits", "logits"), ("fresh_logits", "logits")):
        audit.equal(own[key], ref[baseline], "returned_state_ownership:" + key)
    for delay in RECIPE["delays"]:
        label = "stream_delay" + str(delay)
        saved = values[label]
        frames = np.stack([cue] + [blank]*delay + [query], axis=1)
        steps = np.asarray([3] + [3]*delay + [10], dtype=np.int64)
        audit.equal(saved["frames"], frames, label + ":literal_frames")
        audit.equal(saved["steps"], steps, label + ":literal_updates")
        audit.check(saved["frame_states"].shape == (6, delay+2, 48, 3, 3), label + ":frame_state_shape")
        audit.equal(saved["whole_state"], saved["frame_states"][:, -1], label + ":whole_endpoint")
        audit.equal(saved["prefix_state"], saved["frame_states"][:, -2], label + ":prefix_endpoint")
        for suffix in ("state", "logits"):
            audit.equal(saved["native_" + suffix], saved["whole_" + suffix], label + ":native_" + suffix)
            audit.equal(saved["continued_" + suffix], saved["whole_" + suffix], label + ":continued_" + suffix)
            audit.equal(saved["zero_" + suffix], ref[suffix], label + ":zero_query_" + suffix)
            audit.close(saved["swap_" + suffix], saved["whole_" + suffix][donor], label + ":swap_donor_" + suffix)
        audit.equal(saved["zero_before"], saved["prefix_state"], label + ":zero_before")
        audit.equal(saved["zero_after"], np.zeros_like(saved["prefix_state"]), label + ":zero_after")
        audit.equal(saved["swap_before"], saved["prefix_state"], label + ":swap_before")
        audit.equal(saved["swap_after"], saved["prefix_state"][donor], label + ":swap_after")
        audit.equal(saved["external_hook_injected_state"], saved["prefix_state"], label + ":actual_injected_prefix")
        cue_effect = float(np.max(np.abs(saved["whole_state"] - saved["no_cue_state"])))
        hook_effect = float(np.max(np.abs(saved["external_hook_state"] - ref["state"])))
        condition = next(r for r in producer["conditions"] if r["id"] == label)
        audit.check(condition["local_updates"] == int(steps.sum()) == 13+3*delay
                    and condition["blank_frames"] == delay
                    and condition["cue_vs_no_cue_state_max_abs"] == cue_effect
                    and condition["external_hook_vs_default_state_max_abs"] == hook_effect,
                    label + ":recounted_observations")
        if delay == 0:
            audit.check(cue_effect > 1e-10 and hook_effect > 1e-10, "nontrivial_activity_positive_controls")
        audit.report["observations"].append({"blank_frames": delay, "local_updates": int(steps.sum()),
             "cue_vs_no_cue_state_max_abs": cue_effect, "explicit_hook_vs_default_state_max_abs": hook_effect,
             "whole_state_max_abs": float(np.max(np.abs(saved["whole_state"]))),
             "no_learned_accuracy_claim": True})
    ground = values["grounding_serialization"]
    expected_rgb = np.zeros((8, 3), dtype=np.float32)
    expected_rgb[2], expected_rgb[3] = [0.2, 0.4, 0.6], [0.8, 0.1, 0.3]
    expected_mask = np.zeros(8, dtype=np.bool_)
    expected_mask[2:4] = True
    audit.equal(ground["g_rgb"], expected_rgb, "declared_grounding_RGB")
    audit.equal(ground["g_mask"], expected_mask, "declared_grounding_mask")
    audit.equal(ground["grounded_dictionary"], ground["complete_dictionary"], "grounded_dictionary_restoration")
    audit.check(not np.array_equal(ground["grounded_dictionary"], ground["parameter_only_dictionary"]),
                "parameter_only_dictionary_incomplete")
    for name in ("state", "logits"):
        audit.equal(ground["grounded_" + name], ground["complete_" + name], "grounded_eval_restoration:" + name)
    audit.report["source"] = producer["source"]
    audit.report["recipe"] = RECIPE
    audit.report["parameter_count"] = 29392
    audit.report["state_payload"] = {"values_per_example": 432, "float32_bytes_per_example": 1728,
                                   "scope": "Allocated array payload, not capacity, peak memory or energy."}
    audit.report["limits"] = producer["limits"] + [
        "Saved hashes link the reported before/after parameter state to saved parameters; no independent live trajectory was observed.",
        "The .pt checkpoint is hashed only; its deserialization witness is produced by the original executed test.",
        "Finite seeded activity checks do not measure learned retrieval or independent experimental replication."]
    audit.check(state["source"]["implementation_sha256"] == state["source_after"]["implementation_sha256"]
                and state["source"]["plan_sha256"] == state["source_after"]["plan_sha256"], "end_source_bindings_stable")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = Recount(output)
    start = time.perf_counter()
    try:
        import psutil
        available = psutil.virtual_memory().available / 1024**3
        audit.check(available >= 8, "admission_RAM_floor", available)
        recount(args.input.resolve(), args.plan.resolve(), args.source_root.resolve(), audit)
        audit.report["status"] = "verified"
    except Exception as error:
        audit.report["status"] = "failed"
        audit.report["issues"].append({"type": type(error).__name__, "message": str(error)})
    audit.report.update(completed_at_utc=now(), wall_seconds=time.perf_counter()-start)
    audit.save()
    print(json.dumps({"status": audit.report["status"], "checks": len(audit.report["checks"]),
                      "issues": len(audit.report["issues"])}), flush=True)
    return int(audit.report["status"] != "verified")

if __name__ == "__main__":
    raise SystemExit(main())
