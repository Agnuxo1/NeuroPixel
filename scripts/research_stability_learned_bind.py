"""Independently bind saved item15B parameter arrays to the five original checkpoints.

This fresh process imports Torch solely for weights_only deserialization, performs
no inference and does not import the native producer or its restoration helper.
NumPy's separate dynamics audit consumes this receipt and rehashes its inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_item9_worker as ops

OLD_SOURCE = "83fe135e301f76bc0c74e30c66bb18e067ca5959"
OLD_ARCHIVE = "b64d0e2ba222df76caf72bc9870c7602873e7236"
OLD_PATH = "results/research/09_cloud_runs/37593731891-1-study"
SEEDS = [40, 41, 42, 43, 44]
PARAMETER_SHAPES = {
    "embed.weight": (37, 16), "seed.weight": (48, 16, 1, 1), "seed.bias": (48,),
    "perceive.weight": (96, 1, 3, 3), "f1.weight": (128, 160, 1, 1),
    "f1.bias": (128,), "f2.weight": (48, 128, 1, 1), "f2.bias": (48,),
    "read.weight": (16, 48), "read.bias": (16,),
}
BUFFER_SHAPES = {"g_rgb": (37, 3), "g_mask": (37, 1)}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)

def safe(root, relative):
    rel = Path(relative)
    if (rel.is_absolute() or ".." in rel.parts or rel.as_posix() != relative):
        raise ValueError("unsafe relative path")
    path = root / rel
    if not (path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve())):
        raise ValueError("missing or indirect input: " + relative)
    return path

def configuration(seed):
    return {"family": "neuropixel", "model": {
            "class": "neuropixel.model.NeuroPixel", "c_id": 16, "c": 48,
            "hidden": 128, "steps": 16, "fire_rate": 0.5, "grounded": None, "retina": False},
        "vocab": 37, "height": 10, "width": 8, "out_pos": [9, 7],
        "trainable_parameters": 29856, "initialization": seed,
        "learning_rate": 0.001, "batch_size": 32, "scheduled_updates": 4096,
        "optimizer": "AdamW", "weight_decay": 0.0001, "gradient_clip": 1,
        "loss": "answer_cross_entropy_only", "memorization": False}

def execute(args):
    started = time.monotonic()
    deadline = started + 60
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "item": 15, "status": "running",
        "source_commit": os.environ.get("GITHUB_SHA"), "plan_sha256": sha(args.plan),
        "original_source_commit": OLD_SOURCE, "original_archive_commit": OLD_ARCHIVE,
        "seeds": SEEDS, "input_sha256": {}, "native_sha256": {},
        "checkpoint_weights_exact": False, "configuration_exact": False,
        "constructor_buffers_zero": False, "checks": 0, "issues": [], "per_seed": [],
        "started_at_utc": ops.now(), "inference_calls": 0, "optimization_steps": 0}
    def check(condition, label):
        receipt["checks"] += 1
        if not bool(condition):
            receipt["issues"].append(label)
            raise ValueError(label)
    try:
        p = read_json(args.plan)
        check(p["item"] == 15 and p["phase"] == "preflight" and p["status"] == "frozen",
              "frozen plan identity")
        check(p["evidence_recipe"]["component_limits_seconds"]["checkpoint_binding"] == 60,
              "component bound")
        check(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "thread environment")
        check(receipt["source_commit"] is not None, "current source identity")
        bindings = p["implementation_sha256"]
        check({name: sha(safe(ROOT, name)) for name in bindings} == bindings, "source bindings")
        spec = p["inputs"]["learned"]
        check((spec["source_commit"], spec["archive_commit"], spec["path"]) ==
              (OLD_SOURCE, OLD_ARCHIVE, OLD_PATH), "original identity")
        check(set(spec) == {"source_commit", "archive_commit", "path", "files_sha256", "files_bytes"},
              "input specification keys")
        expected_inputs = {"archive_manifest.json", "study/development_data/validation.npz"}
        for seed in SEEDS:
            expected_inputs.update(f"study/runs/neuropixel_seed{seed}/{name}" for name in
                                  ("checkpoint.pt", "validation_predictions.npz", "configuration.json", "run.json"))
        check(set(spec["files_sha256"]) == expected_inputs and
              set(spec["files_bytes"]) == expected_inputs, "exact22 original files")
        for name in sorted(expected_inputs):
            path = safe(args.inputs, name)
            check(sha(path) == spec["files_sha256"][name]
                  and path.stat().st_size == spec["files_bytes"][name], "original file: " + name)
        receipt["input_sha256"] = dict(spec["files_sha256"])
        manifest = read_json(safe(args.inputs, "archive_manifest.json"))
        check(spec["files_sha256"]["archive_manifest.json"] ==
              "49cda016746d6f020e22da2e10e09f8706b1f17f15491baf438d6952484f78e6",
              "original manifest anchor")
        check(manifest["source_commit"] == OLD_SOURCE and manifest["final"] is True
              and manifest["attempt_status"] == "completed", "original completed archive")
        for name in expected_inputs - {"archive_manifest.json"}:
            check(manifest["files"][name] == {"bytes": spec["files_bytes"][name],
                  "sha256": spec["files_sha256"][name]}, "original manifest membership: " + name)
        report = read_json(safe(args.native, "report.json"))
        summary = read_json(safe(args.native, "gates_summary.json"))
        check(report["item"] == 15 and report["status"] == "verified"
              and report["source_commit"] == receipt["source_commit"]
              and report["plan_sha256"] == receipt["plan_sha256"], "native report identity")
        check(report["inputs_sha256"] == spec["files_sha256"], "native original inputs")
        check(type(report["compatibility_gate_passed"]) is bool
              and report["extended_panel_executed"] is report["compatibility_gate_passed"],
              "conditional extension policy")
        extended = report["extended_panel_executed"]
        check(summary["item"] == 15 and summary["source_commit"] == receipt["source_commit"]
              and summary["plan_sha256"] == receipt["plan_sha256"]
              and summary["all_five_passed"] is extended
              and [r["seed"] for r in summary["gates"]] == SEEDS, "durable all-five summary")
        check([r["seed"] for r in report["gates"]] == SEEDS
              and [r["seed"] for r in report["seed_reports"]] == (SEEDS if extended else []),
              "native seed inventory")
        expected_native = {"report.json", "gates_summary.json"}
        for seed in SEEDS:
            for kind in (["gates", "seeds"] if extended else ["gates"]):
                expected_native.update(f"{kind}/seed{seed}.{suffix}" for suffix in ("json", "npz"))
        actual_native = {x.relative_to(args.native).as_posix() for x in args.native.rglob("*")
                         if x.is_file() and x.suffix in (".json", ".npz")}
        check(actual_native == expected_native, "complete native JSON/NPZ inventory")
        native_before = {name: sha(safe(args.native, name)) for name in sorted(expected_native)}
        receipt["native_sha256"] = native_before
        check(report["gates_summary"] == {"path": "gates_summary.json",
              "bytes": safe(args.native, "gates_summary.json").stat().st_size,
              "sha256": native_before["gates_summary.json"]}, "summary file descriptor")
        ops.admit(deadline, minimum_seconds=10)
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        check(torch.version.cuda is None and str(torch.__version__) == "2.6.0+cpu",
              "pinned CPU Torch")
        receipt["runtime"] = {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": np.__version__, "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads()}
        tensor_names = set(PARAMETER_SHAPES) | set(BUFFER_SHAPES)
        for seed in SEEDS:
            ops.admit(deadline, minimum_seconds=1)
            prefix = f"study/runs/neuropixel_seed{seed}/"
            cfg = configuration(seed)
            check(read_json(safe(args.inputs, prefix + "configuration.json")) == cfg,
                  f"configuration seed{seed}")
            run = read_json(safe(args.inputs, prefix + "run.json"))
            check(run["configuration"] == cfg and run["status"] == "completed"
                  and run["updates_completed"] == 4096 and run["run_id"] == f"neuropixel_seed{seed}"
                  and run["family"] == "neuropixel" and run["initialization"] == seed
                  and run["context"]["source_commit"] == OLD_SOURCE, f"run seed{seed}")
            for artifact, name in (("checkpoint", "checkpoint.pt"),
                                   ("validation_predictions", "validation_predictions.npz")):
                check(run["artifacts"][artifact]["sha256"] == spec["files_sha256"][prefix + name],
                      f"run linkage seed{seed} {name}")
            checkpoint = torch.load(safe(args.inputs, prefix + "checkpoint.pt"),
                                    map_location="cpu", weights_only=True)
            check(isinstance(checkpoint, dict) and set(checkpoint) == {"configuration", "state_dict"}
                  and checkpoint["configuration"] == cfg, f"checkpoint metadata seed{seed}")
            weights = checkpoint["state_dict"]
            check(isinstance(weights, dict) and set(weights) == set(PARAMETER_SHAPES),
                  f"checkpoint parameter inventory seed{seed}")
            for name, shape in PARAMETER_SHAPES.items():
                value = weights[name]
                check(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                      and value.dtype == torch.float32 and value.device.type == "cpu"
                      and bool(torch.isfinite(value).all()), f"checkpoint tensor seed{seed} {name}")
            check(sum(value.numel() for value in weights.values()) == 29856, f"parameter count seed{seed}")
            arrays_checked = []
            for kind in (["gates", "seeds"] if extended else ["gates"]):
                relative = f"{kind}/seed{seed}.json"
                rec = read_json(safe(args.native, relative))
                npz_name = f"{kind}/seed{seed}.npz"
                check(rec["item"] == 15 and rec["seed"] == seed and rec["array_file"] == npz_name,
                      f"array descriptor seed{seed} {kind}")
                check(rec["array_sha256"] == native_before[npz_name]
                      and rec["array_bytes"] == safe(args.native, npz_name).stat().st_size,
                      f"array bytes seed{seed} {kind}")
                with np.load(safe(args.native, npz_name), allow_pickle=False) as arrays:
                    check(set(arrays.files) == set(rec["arrays"]), f"array schema seed{seed} {kind}")
                    check({name[len("weight__"):] for name in arrays.files if name.startswith("weight__")}
                          == tensor_names and
                          {name[len("weight_after__"):] for name in arrays.files if name.startswith("weight_after__")}
                          == tensor_names, f"saved tensor inventory seed{seed} {kind}")
                    for tensor_prefix in ("weight__", "weight_after__"):
                        for name, shape in PARAMETER_SHAPES.items():
                            array = arrays[tensor_prefix + name]
                            check(array.dtype == np.float32 and array.shape == shape
                                  and np.array_equal(array, weights[name].numpy()),
                                  f"exact checkpoint tensor seed{seed} {kind} {tensor_prefix}{name}")
                        for name, shape in BUFFER_SHAPES.items():
                            array = arrays[tensor_prefix + name]
                            dtype = np.float32 if name == "g_rgb" else np.bool_
                            check(array.dtype == dtype and array.shape == shape
                                  and not np.count_nonzero(array),
                                  f"default zero buffer seed{seed} {kind} {tensor_prefix}{name}")
                arrays_checked.append(npz_name)
            receipt["per_seed"].append({"seed": seed, "checkpoint_sha256": spec["files_sha256"][prefix+"checkpoint.pt"],
                "parameter_count": 29856, "parameter_tensors": 10, "default_buffer_tensors": 2,
                "native_arrays_checked": arrays_checked, "configuration_exact": True,
                "checkpoint_weights_exact": True, "constructor_buffers_zero": True})
            del checkpoint, weights
        check({name: sha(safe(args.native, name)) for name in expected_native} == native_before,
              "native files unchanged")
        check({name: sha(safe(args.inputs, name)) for name in expected_inputs} ==
              spec["files_sha256"], "original files unchanged")
        check({name: sha(safe(ROOT, name)) for name in bindings} == bindings, "source unchanged")
        ops.remaining(deadline)
        receipt.update(status="verified", checkpoint_weights_exact=True, configuration_exact=True,
                       constructor_buffers_zero=True, extended_panel_executed=extended,
                       checkpoint_deserializations=5, native_array_files_checked=10 if extended else 5)
    except BaseException as error:
        if not receipt["issues"]:
            receipt["issues"].append(type(error).__name__ + ": " + str(error))
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    receipt.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic()-started,
                   final_resource_observation=ops.resource_sample("learned_checkpoint_binding_finish"))
    if receipt["final_resource_observation"]["available_ram_gib"] < 8:
        receipt["issues"].append("final RAM floor violation")
        receipt.update(status="failed", final_ram_floor_violation=True)
    ops.save(args.output / "checkpoint_binding.json", receipt, exclusive=True)
    print(json.dumps({"status": receipt["status"], "checks": receipt["checks"],
                      "issues": receipt["issues"]}), flush=True)
    return int(receipt["status"] != "verified")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
