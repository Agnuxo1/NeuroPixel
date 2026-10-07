"""Item-11 finite activity/API diagnostics; no optimizer or training.

Run only from a frozen plan in the declared CPU environment. Default forward
calls, the existing within-sequence np_stream, and explicitly caller-enabled
continuation/injection are distinct conditions. No accuracy competence claim
is derived from these seeded, untrained diagnostic parameters.
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
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RECIPE = {
    "vocabulary": 8, "shape": [3, 3], "cue_pos": [0, 0],
    "query_pos": [2, 1], "out_pos": [2, 2], "cue_steps": 3,
    "gap_steps": 3, "query_steps": 10, "delays": [0, 4, 16, 32],
    "init_seed": 110001, "active_f2_seed": 110002, "active_f2_std": 0.01,
}
ATOL, RTOL = 1e-6, 1e-5


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, timeout=30,
                                   stdin=subprocess.DEVNULL).decode().strip()


def source_record(path, plan):
    require(not git("status", "--porcelain", "--untracked-files=no"), "tracked source is dirty")
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT,
                                       timeout=30, stdin=subprocess.DEVNULL)
    require(committed == path.read_bytes(), "plan differs from committed HEAD")
    for name, expected in plan["implementation_sha256"].items():
        candidate = (ROOT / name).resolve()
        candidate.relative_to(ROOT)
        require(candidate.is_file() and digest(candidate) == expected, "bound source differs: " + name)
    return {"source_commit": git("rev-parse", "HEAD"), "plan_sha256": digest(path),
            "implementation_sha256": plan["implementation_sha256"],
            "tracked_source_clean": True, "recorded_at_utc": now()}


def runtime(plan):
    require(platform.python_version() == plan["runtime"]["python"], "Python version differs")
    actual = {name: importlib.metadata.version(name) for name in plan["runtime"]["packages"]}
    require(actual == plan["runtime"]["packages"], "package versions differ")
    require(plan["runtime"]["threads"] == 1 and plan["runtime"]["interop_threads"] == 1,
            "probe requires one numerical and one interop thread")
    import psutil
    available = psutil.virtual_memory().available / 1024**3
    require(available >= 8, "less than 8 GiB RAM available")
    import torch
    import numpy as np
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    require(torch.version.cuda is None, "CPU-only Torch required")
    return torch, np, {"python": platform.python_version(), "packages": actual,
                       "threads": torch.get_num_threads(),
                       "interop_threads": torch.get_num_interop_threads(),
                       "cuda_version": torch.version.cuda, "available_ram_gib": available}


def run_probe(output, torch, np, source):
    from neuropixel.model import NeuroPixel
    from neuropixel.phase3 import np_stream
    from neuropixel.research.stream_memory import strict_np_stream
    directory = output / "probe"
    directory.mkdir(exist_ok=False)
    files, checks, conditions = {}, [], []

    def tensor_digest(values):
        h = hashlib.sha256()
        for name, tensor in sorted(values.items()):
            array = tensor.detach().cpu().contiguous().numpy()
            h.update(canonical({"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}))
            h.update(array.tobytes(order="C"))
        return h.hexdigest()

    def inventory(model):
        return {"parameters": tensor_digest(dict(model.named_parameters())),
                "buffers": tensor_digest(dict(model.named_buffers()))}

    def arrays(name, values):
        path = directory / (name + ".npz")
        converted = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                     for k, v in values.items()}
        require(all(v.dtype != object for v in converted.values()), "object array is forbidden")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **converted)
            stream.flush()
            os.fsync(stream.fileno())
        row = {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
               "sha256": digest(path),
               "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in converted.items()}}
        files[name] = row
        return row

    def maximum(a, b):
        value = float((a - b).abs().max().item())
        return value if math.isfinite(value) else None

    def check(name, ok, **observed):
        checks.append({"check": name, "passed": bool(ok), **observed})

    def close(a, b):
        return bool(torch.allclose(a, b, rtol=RTOL, atol=ATOL))

    torch.manual_seed(RECIPE["init_seed"])
    model = NeuroPixel(8, (2, 2), c_id=16, c=48, hidden=128, steps=16,
                       fire_rate=0.5, retina=False).cpu().eval()
    private = torch.Generator(device="cpu").manual_seed(RECIPE["active_f2_seed"])
    f2_rng_before = private.get_state().clone()
    with torch.no_grad():
        model.f2.weight.normal_(0.0, RECIPE["active_f2_std"], generator=private)
        model.f2.bias.normal_(0.0, RECIPE["active_f2_std"], generator=private)
    before = inventory(model)
    global_rng_before = torch.get_rng_state().clone()
    parameter_count = sum(p.numel() for p in model.parameters())
    arrays("diagnostic_parameters", dict(model.named_parameters()))
    arrays("diagnostic_buffers", dict(model.named_buffers()))
    arrays("initial_rng", {"torch_global": global_rng_before, "private_f2_before": f2_rng_before,
                          "private_f2_after": private.get_state()})
    check("active_parameter_count", parameter_count == 29392, parameter_count=parameter_count)
    check("nonzero_update_projection", bool(model.f2.weight.abs().sum() > 0))
    labels = torch.arange(2, 8, dtype=torch.long)
    cue = torch.zeros(6, 3, 3, dtype=torch.long)
    cue[:, 0, 0] = labels
    blank = torch.zeros_like(cue)
    query = torch.zeros_like(cue)
    query[:, 2, 1] = 1
    permutation = [1, 2, 3, 4, 5, 0]
    order = torch.tensor(permutation, dtype=torch.long)
    distractor = cue.index_select(0, order)
    arrays("inputs", {"cue": cue, "blank": blank, "query": query,
                      "cue_token_ids": labels, "donor_permutation": order, "distractor": distractor})

    with torch.inference_mode():
        reference = model(query, steps=10, trace=True)
        arrays("default_query_reference", {"state": reference["state"], "logits": reference["logits"],
                                           "update_states": reference["frames"]})
        histories = {
            "empty": [],
            "cue": [(cue, 3)],
            "cue_delay4": [(cue, 3)] + [(blank, 3)] * 4,
            "cue_delay16": [(cue, 3)] + [(blank, 3)] * 16,
            "cue_distractor_delay4": [(cue, 3), (distractor, 3)] + [(blank, 3)] * 4,
        }
        for name, history in histories.items():
            states = []
            for frame, count in history:
                states.append(model(frame, steps=count)["state"].detach().clone())
            answer = model(query, steps=10)
            saved = {"state": answer["state"], "logits": answer["logits"],
                     "history_states": torch.stack(states, 1) if states else
                        torch.empty(6, 0, 48, 3, 3),
                     "history_frames": torch.stack([v[0] for v in history], 1) if history else
                        torch.empty(6, 0, 3, 3, dtype=torch.long),
                     "history_steps": np.asarray([v[1] for v in history], dtype=np.int64)}
            arrays("separate_calls_" + name, saved)
            equal = torch.equal(answer["state"], reference["state"]) and torch.equal(answer["logits"], reference["logits"])
            check("separate_calls_" + name + "_exact_current_query_identity", equal,
                  max_state_difference=maximum(answer["state"], reference["state"]),
                  max_logit_difference=maximum(answer["logits"], reference["logits"]))
            conditions.append({"id": "separate_calls_" + name,
                               "mechanism": "original forward; prior returned states discarded",
                               "history_call_count": len(history)})

        # Returned activity is caller-owned data, not implicit model persistence.
        returned = model(query, steps=10)
        original_returned_state = returned["state"].detach().clone()
        original_returned_logits = returned["logits"].detach().clone()
        ownership_rng = torch.get_rng_state().clone()
        returned["state"].fill_(3.25)
        fresh = model(query, steps=10)
        arrays("returned_state_ownership", {
            "original_returned_state": original_returned_state,
            "original_returned_logits": original_returned_logits,
            "mutated_returned_state": returned["state"],
            "fresh_state": fresh["state"], "fresh_logits": fresh["logits"],
            "finite_sentinel": np.asarray(3.25, dtype=np.float32)})
        check("returned_state_mutation_does_not_persist",
              torch.equal(fresh["state"], reference["state"])
              and torch.equal(fresh["logits"], reference["logits"])
              and bool((returned["state"] == 3.25).all()),
              max_fresh_state_difference=maximum(fresh["state"], reference["state"]),
              max_fresh_logit_difference=maximum(fresh["logits"], reference["logits"]))
        check("returned_state_mutation_preserves_parameters_buffers_and_rng",
              inventory(model) == before and torch.equal(ownership_rng, torch.get_rng_state()))
        conditions.append({"id": "returned_state_ownership",
                           "mechanism": "ordinary returned state mutated by caller; next call initializes anew",
                           "finite_sentinel": 3.25})

        for delay in RECIPE["delays"]:
            frames = [cue] + [blank] * delay + [query]
            steps = [3] + [3] * delay + [10]
            native = np_stream(model, frames, steps)
            whole = strict_np_stream(model, frames, steps, capture_states=True)
            prefix = strict_np_stream(model, frames[:-1], steps[:-1], capture_states=True)
            continued = strict_np_stream(model, [query], [10], initial_state=prefix["state"])
            zero = strict_np_stream(model, frames, steps, capture_states=True,
                                    intervention={"before_frame": len(frames)-1, "kind": "zero"})
            swap = strict_np_stream(model, frames, steps, capture_states=True,
                                    intervention={"before_frame": len(frames)-1, "kind": "swap",
                                                  "permutation": permutation})
            no_cue = strict_np_stream(model, [blank] + [blank]*delay + [query], steps)
            # Explicit caller mechanism: add retained prefix state AFTER query update1.
            # This is deliberately not claimed to be the original forward's memory.
            carried = prefix["state"].detach().clone()
            injected = model(query, steps=10,
                             hook=lambda t, state, carry=carried: state + carry if t == 1 else state)
            values = {"frames": torch.stack(frames, 1), "steps": np.asarray(steps, dtype=np.int64),
                      "native_state": native["state"], "native_logits": native["logits"],
                      "whole_state": whole["state"], "whole_logits": whole["logits"],
                      "frame_states": whole["frame_states"], "prefix_state": prefix["state"],
                      "continued_state": continued["state"], "continued_logits": continued["logits"],
                      "zero_state": zero["state"], "zero_logits": zero["logits"],
                      "zero_before": zero["intervention_before"], "zero_after": zero["intervention_after"],
                      "swap_state": swap["state"], "swap_logits": swap["logits"],
                      "swap_before": swap["intervention_before"], "swap_after": swap["intervention_after"],
                      "no_cue_state": no_cue["state"], "no_cue_logits": no_cue["logits"],
                      "external_hook_state": injected["state"], "external_hook_logits": injected["logits"],
                      "external_hook_injected_state": carried}
            arrays("stream_delay" + str(delay), values)
            check("delay%d_native_helper_exact" % delay,
                  torch.equal(native["state"], whole["state"]) and torch.equal(native["logits"], whole["logits"]))
            check("delay%d_chunk_continuation_exact" % delay,
                  torch.equal(continued["state"], whole["state"]) and torch.equal(continued["logits"], whole["logits"]))
            check("delay%d_zero_boundary" % delay,
                  torch.equal(zero["intervention_before"], prefix["state"])
                  and torch.count_nonzero(zero["intervention_after"]).item() == 0)
            check("delay%d_zero_matches_query_only" % delay,
                  torch.equal(zero["state"], reference["state"]) and torch.equal(zero["logits"], reference["logits"]))
            check("delay%d_swap_boundary" % delay,
                  torch.equal(swap["intervention_before"], prefix["state"])
                  and torch.equal(swap["intervention_after"], prefix["state"].index_select(0, order)))
            check("delay%d_swap_output_follows_donor" % delay,
                  close(swap["state"], whole["state"].index_select(0, order))
                  and close(swap["logits"], whole["logits"].index_select(0, order)))
            finite = all(bool(torch.isfinite(v).all()) for v in values.values() if isinstance(v, torch.Tensor))
            check("delay%d_all_saved_values_finite" % delay, finite)
            signal = maximum(whole["state"], no_cue["state"])
            injection_signal = maximum(injected["state"], reference["state"])
            if delay == 0:
                check("positive_control_retained_cue_changes_activity", signal is not None and signal > 1e-10,
                      maximum_absolute_state_difference=signal)
                check("positive_control_external_hook_changes_activity",
                      injection_signal is not None and injection_signal > 1e-10,
                      maximum_absolute_state_difference=injection_signal)
            conditions.append({"id": "stream_delay%d" % delay, "blank_frames": delay,
                               "local_updates": sum(steps),
                               "mechanism": "existing within-sequence state; same query after distinct cue histories",
                               "cue_vs_no_cue_state_max_abs": signal,
                               "external_hook_vs_default_state_max_abs": injection_signal})

        check("eval_global_rng_unchanged", torch.equal(global_rng_before, torch.get_rng_state()))
        check("diagnostic_parameters_and_buffers_unchanged", inventory(model) == before)
        arrays("final_rng", {"torch_global": torch.get_rng_state()})

        # Independent clones expose the incomplete state_dict grounding contract.
        # The diagnostic model above and its parameter/buffer hashes stay fixed.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(110003)
            rgb = torch.zeros(8, 3)
            mask = torch.zeros(8, dtype=torch.bool)
            rgb[2], rgb[3] = torch.tensor([0.2, 0.4, 0.6]), torch.tensor([0.8, 0.1, 0.3])
            mask[2:4] = True
            grounded = NeuroPixel(8, (2, 2), grounded=(rgb, mask)).eval()
            grounded.load_state_dict(model.state_dict(), strict=True)
            checkpoint = directory / "grounded_parameters_only.pt"
            with checkpoint.open("xb") as stream:
                torch.save(grounded.state_dict(), stream)
                stream.flush()
                os.fsync(stream.fileno())
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            missing = NeuroPixel(8, (2, 2)).eval()
            missing.load_state_dict(weights, strict=True)
            complete = NeuroPixel(8, (2, 2), grounded=(rgb.clone(), mask.clone())).eval()
            complete.load_state_dict(weights, strict=True)
            check("grounding_absent_from_state_dict", "g_rgb" not in weights and "g_mask" not in weights)
            check("parameter_only_load_passes_but_dictionary_differs",
                  not torch.equal(grounded.dictionary(), missing.dictionary()))
            check("explicit_grounding_restores_dictionary",
                  torch.equal(grounded.dictionary(), complete.dictionary()))
            g = grounded(query, steps=10)
            restored = complete(query, steps=10)
            check("explicit_grounding_restores_eval",
                  torch.equal(g["state"], restored["state"]) and torch.equal(g["logits"], restored["logits"]))
            arrays("grounding_serialization", {
                "grounded_dictionary": grounded.dictionary(), "parameter_only_dictionary": missing.dictionary(),
                "complete_dictionary": complete.dictionary(), "g_rgb": rgb, "g_mask": mask,
                "grounded_state": g["state"], "grounded_logits": g["logits"],
                "complete_state": restored["state"], "complete_logits": restored["logits"]})
            files["grounded_parameters_only"] = {"path": checkpoint.relative_to(output).as_posix(),
                "bytes": checkpoint.stat().st_size, "sha256": digest(checkpoint)}
        check("original_model_unchanged_after_serialization_witness", inventory(model) == before)
        check("global_rng_restored_after_serialization_witness", torch.equal(global_rng_before, torch.get_rng_state()))

    report = {"schema_version": 1, "item": 11, "phase": "probe",
        "status": "verified" if all(c["passed"] for c in checks) else "failed",
        "source": source, "recipe": RECIPE, "created_at_utc": now(),
        "parameter_count": parameter_count, "state_hashes_before": before,
        "state_hashes_after": inventory(model), "tolerance": {"atol": ATOL, "rtol": RTOL},
        "conditions": conditions, "checks": checks, "files": files,
        "training_performed": False,
        "limits": [
            "This is an untrained activity/API counterfactual, not a learned-memory accuracy test.",
            "Default forward resets; native np_stream retains state only within its own sequence call.",
            "Explicit initial_state, boundary interventions and core-hook injection are caller-enabled mechanisms.",
            "Zeroing tests necessity of retained state for cue dependence; swapping tests donor-specific dependence.",
            "Finite sampled delays do not establish indefinite retention, consolidation or robust interference resistance.",
            "state_dict excludes grounded buffers and constructor/runtime configuration; explicit reconstruction is required."
        ]}
    save_json(directory / "probe_report.json", report)
    require(report["status"] == "verified", "a retained probe check failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("probe",), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"schema_version": 1, "item": 11, "phase": "probe", "started_at_utc": now()}
    try:
        plan = json.loads(args.plan.read_bytes())
        require(plan["schema_version"] == 1 and plan["item"] == 11
                and plan["phase"] == "probe" and plan["status"] == "frozen" and plan["freeze_utc"],
                "not a frozen item-11 probe plan")
        require(plan["probe_recipe"] == RECIPE, "probe recipe differs")
        source = source_record(args.plan.resolve(), plan)
        status["source"] = source
        torch, np, environment = runtime(plan)
        save_json(output / "memory_probe_environment.json", environment)
        report = run_probe(output, torch, np, source)
        after = source_record(args.plan.resolve(), plan)
        require(all(source[k] == after[k] for k in ("source_commit", "plan_sha256", "implementation_sha256")),
                "source changed during probe")
        status.update(status="completed", source_after=after,
                      report_sha256=digest(output / "probe/probe_report.json"),
                      checks=len(report["checks"]))
    except Exception as error:
        status.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    status.update(completed_at_utc=now(), wall_seconds=time.perf_counter() - started)
    save_json(output / "memory_probe_status.json", status)
    print(json.dumps({"status": status["status"], "phase": "probe"}), flush=True)
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
