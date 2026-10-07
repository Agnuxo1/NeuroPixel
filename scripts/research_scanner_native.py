"""Item16: bounded native scanner counterexamples and learned projection ranks.

Constructed states/weights are diagnostic controls, not learned explanations.
Learned checkpoints supply projection matrices only: no learned forward occurs.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import research_item9_worker as ops

OLD_SOURCE = "771975b4833bfcf20e06f848253912c699b2152e"
OLD_ARCHIVE = "205507b71e2f7abb00ce97aa1dbec8321fc0af26"
OLD_PATH = "results/research/15_cloud_runs/37654405152-1-preflight"
MODEL_SHA = "677f3aab045fc6e0ae73ccc19e7093c1b8211e939dcadbe36db8a793130cd345"
SEEDS = (40, 41, 42, 43, 44)
RECIPE = {
    "panel": "scanner_causal_projection",
    "constructed": {
        "vocab": 4,
        "c_id": 3,
        "state_channels": 4,
        "hidden": 8,
        "height": 1,
        "width": 1,
        "out_pos": [
            0,
            0
        ],
        "dtype": "float64",
        "mode": "eval",
        "rules": [
            "hidden_to_visible",
            "hidden_unused"
        ],
        "states": [
            [
                1,
                0,
                0,
                0
            ],
            [
                1,
                0,
                0,
                2
            ]
        ],
        "steps": 2,
        "hook_initialization_after_step": 1,
        "lens_every": 1,
        "pad_nonpad_logits": [
            -1,
            -20000
        ],
        "compression_probabilities": [
            [
                0.6,
                0.3,
                0.1
            ],
            [
                0.6,
                0.1,
                0.3
            ]
        ]
    },
    "learned": {
        "seeds": [
            40,
            41,
            42,
            43,
            44
        ],
        "read_shape": [
            16,
            48
        ],
        "dictionary_shape": [
            37,
            16
        ],
        "nonpad_projection_shape": [
            36,
            48
        ],
        "matrix_dtype": "float64",
        "rank_threshold": "max(matrix.shape)*float64_epsilon*sigma_max",
        "forward_calls": 0
    },
    "comparison_absolute_tolerance": 1e-12,
    "projector_absolute_tolerance": 1e-10,
    "component_limit_seconds": 120,
    "threads": 1,
    "interop_threads": 1,
    "optimization_steps": 0,
    "native_check_count": 10
}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def safe(root, name):
    rel = Path(name)
    require(not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == name, "unsafe path")
    path = root / rel
    require(path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()), "missing or indirect file: " + name)
    return path


def descriptor(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def save_arrays(path, values, root):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    return {**descriptor(path, root),
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       for key, value in values.items()}}


def snapshot(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in
            list(model.named_parameters()) + list(model.named_buffers())}


def fixture_model(coupled):
    from neuropixel.model import NeuroPixel
    with torch.random.fork_rng(devices=[]):
        with torch.device("cpu"):
            m = NeuroPixel(4, (0, 0), c_id=3, c=4, hidden=8, steps=2, fire_rate=0.5).double().eval()
    with torch.no_grad():
        for value in m.parameters():
            value.zero_()
        m.embed.weight[1:] = torch.eye(3, dtype=torch.float64)
        m.seed.weight[:3, :, 0, 0] = torch.eye(3, dtype=torch.float64)
        m.read.weight[:, :3] = torch.eye(3, dtype=torch.float64)
        m.f1.weight[0, 3, 0, 0] = 1
        if coupled:
            m.f2.weight[1, 0, 0, 0] = 1
    return m


def masked(raw):
    logits = raw.clone()
    logits[..., 0] = -1e4
    return logits


def constructed(values, event):
    from neuropixel.scanner import lens
    init = torch.tensor(RECIPE["constructed"]["states"], dtype=torch.float64).reshape(2, 4, 1, 1)
    canvas = torch.zeros((2, 1, 1), dtype=torch.long)
    values["causal_initial"] = init.numpy().copy()
    values["causal_canvas"] = canvas.numpy().copy()
    values["causal_rule_observed"] = np.zeros(2, dtype=np.bool_)
    for key, shape in {
        "causal_frames": (2, 2, 3, 4, 1, 1),
        "causal_lens_pre": (2, 2, 2, 1, 1, 4),
        "causal_lens_post": (2, 2, 3, 1, 1, 4),
        "causal_masked_logits": (2, 2, 3, 4),
        "causal_probs": (2, 2, 3, 4),
        "causal_conf": (2, 2, 3, 1, 1),
        "causal_final_logits": (2, 2, 4)
    }.items():
        values[key] = np.full(shape, np.nan, dtype=np.float64)
    values["causal_word"] = np.full((2, 2, 3, 1, 1), -1, dtype=np.int64)
    models = []
    for rule, coupled in enumerate((True, False)):
        model = fixture_model(coupled)
        models.append(model)
        for key, value in snapshot(model).items():
            values[f"rule{rule}_before__" + key] = value
        event("constructed_rule_started", rule=rule)
        with torch.no_grad():
            out = model(canvas, trace=True, lens_every=1, steps=2,
                        hook=lambda t, state: init.clone() if t == 1 else state)
            frames = out["frames"]
            flat = frames.reshape(6, 4, 1, 1)
            raw = model.lens_logits(flat).reshape(2, 3, 1, 1, 4)
            logits = masked(raw)[:, :, 0, 0]
            word, confidence = lens(model, flat)
        values["causal_frames"][rule] = frames.numpy()
        values["causal_lens_pre"][rule] = out["lens"].numpy()
        values["causal_lens_post"][rule] = raw.numpy()
        values["causal_masked_logits"][rule] = logits.numpy()
        values["causal_probs"][rule] = logits.softmax(-1).numpy()
        values["causal_word"][rule] = word.reshape(2, 3, 1, 1).numpy()
        values["causal_conf"][rule] = confidence.reshape(2, 3, 1, 1).numpy()
        values["causal_final_logits"][rule] = out["logits"].numpy()
        values["causal_rule_observed"][rule] = True
        event("constructed_rule_completed", rule=rule)
    model = models[1]
    pad = torch.zeros((2, 4, 1, 1), dtype=torch.float64)
    pad[0, :3] = -1
    pad[1, :3] = -20000
    compression = torch.zeros_like(pad)
    compression[:, :3, 0, 0] = torch.tensor(
        RECIPE["constructed"]["compression_probabilities"], dtype=torch.float64).log()
    with torch.no_grad():
        for name, state in (("pad", pad), ("compression", compression)):
            raw = model.lens_logits(state)[:, 0, 0]
            logits = masked(raw)
            word, confidence = lens(model, state)
            values[name + "_states"] = state.numpy().copy()
            values[name + "_raw_logits"] = raw.numpy().copy()
            values[name + "_masked_logits"] = logits.numpy().copy()
            values[name + "_probabilities"] = logits.softmax(-1).numpy().copy()
            values[name + "_scanner_word"] = word.numpy().copy()
            values[name + "_scanner_conf"] = confidence.numpy().copy()
        old_conf, old_word = model.lens_logits(pad).softmax(-1).max(-1)
        values["pad_legacy_word"] = old_word.numpy().copy()
        values["pad_legacy_conf"] = old_conf.numpy().copy()
        result = model(canvas, steps=1, hook=lambda t, state: pad.clone())
        values["pad_native_final_logits"] = result["logits"].numpy().copy()
    for rule, model in enumerate(models):
        for key, value in snapshot(model).items():
            values[f"rule{rule}_after__" + key] = value
    event("constructed_compression_and_pad_completed")
    return {"rules": RECIPE["constructed"]["rules"],
            "constructed_parameter_count_each": sum(p.numel() for p in models[0].parameters()),
            "native_forward_calls": 3, "learned_forward_calls": 0,
            "immediate_labels": values["causal_word"][:, :, 1, 0, 0].tolist(),
            "future_labels": values["causal_word"][:, :, 2, 0, 0].tolist(),
            "pad_legacy_labels": values["pad_legacy_word"][:, 0, 0].tolist(),
            "pad_scanner_labels": values["pad_scanner_word"][:, 0, 0].tolist(),
            "compression_labels": values["compression_scanner_word"][:, 0, 0].tolist(),
            "compression_confidence": values["compression_scanner_conf"][:, 0, 0].tolist()}


def verify_inputs(root, spec):
    require(set(spec) == {"archive_commit", "source_commit", "path", "files_sha256", "files_bytes",
                         "scope", "original_item9_source", "original_item9_archive", "provenance"},
            "input specification keys")
    require((spec["archive_commit"], spec["source_commit"], spec["path"]) ==
            (OLD_ARCHIVE, OLD_SOURCE, OLD_PATH), "historical identities")
    names = {"archive_manifest.json", "binding/checkpoint_binding.json"}
    for seed in SEEDS:
        names.update(f"native/gates/seed{seed}.{ext}" for ext in ("json", "npz"))
    require(set(spec["files_sha256"]) == names and set(spec["files_bytes"]) == names, "12 original files")
    files = {name: safe(root, name) for name in sorted(names)}
    for name, path in files.items():
        require(sha(path) == spec["files_sha256"][name]
                and path.stat().st_size == spec["files_bytes"][name], "input bytes differ: " + name)
    manifest = read_json(files["archive_manifest.json"])
    require(manifest["source_commit"] == OLD_SOURCE and manifest["final"] is True
            and manifest["attempt_status"] == "completed", "old archive incomplete")
    for name in names - {"archive_manifest.json"}:
        require(manifest["files"][name] == {"bytes": spec["files_bytes"][name],
                "sha256": spec["files_sha256"][name]}, "old manifest membership")
    binding = read_json(files["binding/checkpoint_binding.json"])
    require(binding["item"] == 15 and binding["status"] == "verified" and not binding["issues"]
            and binding["source_commit"] == OLD_SOURCE and binding["seeds"] == list(SEEDS)
            and binding["checkpoint_weights_exact"] is True
            and binding["configuration_exact"] is True and binding["constructor_buffers_zero"] is True,
            "old checkpoint binding not verified")
    require(spec["original_item9_source"] == binding["original_source_commit"] ==
            "83fe135e301f76bc0c74e30c66bb18e067ca5959"
            and spec["original_item9_archive"] == binding["original_archive_commit"] ==
            "b64d0e2ba222df76caf72bc9870c7602873e7236", "original item9 lineage")
    for seed in SEEDS:
        for ext in ("json", "npz"):
            short = f"gates/seed{seed}.{ext}"
            require(binding["native_sha256"][short] == spec["files_sha256"]["native/" + short],
                    "old binding linkage")
    return files


def decompose(matrix):
    # The independent auditor uses NumPy SVD; this producer uses Torch float64.
    _, singular_tensor, vh_tensor = torch.linalg.svd(torch.from_numpy(matrix), full_matrices=True)
    singular, vh = singular_tensor.numpy().copy(), vh_tensor.numpy().copy()
    threshold = max(matrix.shape) * np.finfo(np.float64).eps * float(singular[0])
    rank = int(np.count_nonzero(singular > threshold))
    null = vh[rank:]
    projector = null.T @ null
    summary = {"rows": matrix.shape[0], "columns": matrix.shape[1],
               "threshold": threshold, "rank": rank, "nullity": matrix.shape[1] - rank,
               "minimum_computed_singular_threshold_margin": float(np.min(np.abs(singular - threshold))),
               "kernel_residual_maxabs": float(np.max(np.abs(matrix @ projector))),
               "projector_symmetry_maxabs": float(np.max(np.abs(projector - projector.T))),
               "projector_idempotence_maxabs": float(np.max(np.abs(projector @ projector - projector))),
               "projector_trace": float(np.trace(projector)),
               "matrix_maxabs": float(np.max(np.abs(matrix)))}
    return singular, projector, summary


def learned_ranks(files, values, event):
    shapes = {"source_read": ((5, 16, 48), np.float32),
              "source_embed": ((5, 37, 16), np.float32),
              "source_g_rgb": ((5, 37, 3), np.float32),
              "source_g_mask": ((5, 37, 1), np.bool_),
              "effective_dictionary": ((5, 37, 16), np.float64),
              "nonpad_projection": ((5, 36, 48), np.float64),
              "read_singular_values": ((5, 16), np.float64),
              "nonpad_singular_values": ((5, 36), np.float64),
              "read_null_projector": ((5, 48, 48), np.float64),
              "nonpad_null_projector": ((5, 48, 48), np.float64)}
    for key, (shape, dtype) in shapes.items():
        values[key] = np.zeros(shape, dtype=dtype) if dtype == np.bool_ else np.full(shape, np.nan, dtype=dtype)
    for name in ("read_rank", "read_nullity", "nonpad_rank", "nonpad_nullity"):
        values[name] = np.full(5, -1, dtype=np.int64)
    for name in ("read_threshold", "nonpad_threshold"):
        values[name] = np.full(5, np.nan, dtype=np.float64)
    values["seeds"] = np.asarray(SEEDS, dtype=np.int64)
    values["observed"] = np.zeros(5, dtype=np.bool_)
    rows = []
    mapping = {"source_read": "read.weight", "source_embed": "embed.weight",
               "source_g_rgb": "g_rgb", "source_g_mask": "g_mask"}
    for i, seed in enumerate(SEEDS):
        event("learned_matrix_started", seed=seed)
        rec = read_json(files[f"native/gates/seed{seed}.json"])
        npz = files[f"native/gates/seed{seed}.npz"]
        require(rec["item"] == 15 and rec["seed"] == seed and rec["passed"] is True
                and rec["array_sha256"] == sha(npz) and rec["array_bytes"] == npz.stat().st_size,
                "old gate metadata")
        with np.load(npz, allow_pickle=False) as original:
            for name, key in mapping.items():
                before, after = original["weight__" + key], original["weight_after__" + key]
                require(before.shape == shapes[name][0][1:] and before.dtype == shapes[name][1]
                        and np.isfinite(before).all() and np.array_equal(before, after), "old projection tensor")
                values[name][i] = before
        require(not values["source_g_rgb"][i].any() and not values["source_g_mask"][i].any(),
                "expected original zero grounding")
        d = values["source_embed"][i].astype(np.float64)
        d[:, :3] = np.where(values["source_g_mask"][i], values["source_g_rgb"][i], d[:, :3])
        d[0] = 0
        matrix = values["source_read"][i].astype(np.float64)
        projected = d[1:] @ matrix
        values["effective_dictionary"][i], values["nonpad_projection"][i] = d, projected
        summaries = {}
        for name, value in (("read", matrix), ("nonpad", projected)):
            singular, projector, summary = decompose(value)
            values[name + "_singular_values"][i] = singular
            values[name + "_null_projector"][i] = projector
            for metric in ("rank", "nullity", "threshold"):
                values[name + "_" + metric][i] = summary[metric]
            summaries[name] = summary
        values["observed"][i] = True
        rows.append({"seed": seed, **summaries})
        event("learned_matrix_completed", seed=seed)
    return rows


def execute(args):
    global np, torch
    started = time.monotonic()
    deadline = started + 120
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "item": 16, "status": "running", "recipe": RECIPE,
              "source_commit": os.environ.get("GITHUB_SHA"), "plan_sha256": sha(args.plan),
              "started_at_utc": ops.now(), "checks": [], "artifacts": {}}
    constructed_arrays, rank_arrays = {}, {}
    event_path = args.output / "events.jsonl"
    event_path.touch(exist_ok=False)
    def event(name, **fields):
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": name, "at_utc": ops.now(), **fields},
                                    sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    def check(name, value):
        report["checks"].append({"name": name, "passed": bool(value)})
        require(value, name)
    error = None
    try:
        plan = read_json(args.plan)
        require(plan["item"] == 16 and plan["phase"] == "probe" and plan["status"] == "frozen"
                and plan["evidence_recipe"]["scanner"] == RECIPE, "frozen recipe")
        ops.admit(deadline, minimum_seconds=10)
        require(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "thread environment")
        source_before = {k: sha(safe(ROOT, k)) for k in plan["implementation_sha256"]}
        require(source_before == plan["implementation_sha256"]
                and sha(ROOT / "neuropixel/model.py") == MODEL_SHA, "source bindings")
        files = verify_inputs(args.inputs, plan["inputs"]["scanner"])
        report["input_sha256"] = plan["inputs"]["scanner"]["files_sha256"]
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.version.cuda is None and str(torch.__version__) == "2.6.0+cpu", "CPU Torch")
        report["runtime"] = {"python": sys.version.split()[0], "torch": str(torch.__version__),
                             "numpy": np.__version__, "threads": 1, "interop_threads": 1}
        constructed_arrays["rng_before"] = torch.get_rng_state().numpy().copy()
        report["constructed"] = constructed(constructed_arrays, event)
        report["learned_ranks"] = learned_ranks(files, rank_arrays, event)
        constructed_arrays["rng_after"] = torch.get_rng_state().numpy().copy()
        report["optimization_steps"] = 0
        report["learned_forward_calls"] = 0
        ops.remaining(deadline)
    except BaseException as caught:
        error = caught
    # Preserve all accumulated raw arrays before assertions or exception reports.
    if constructed_arrays:
        report["artifacts"]["constructed"] = save_arrays(args.output / "constructed.npz", constructed_arrays, args.output)
    if rank_arrays:
        report["artifacts"]["learned_ranks"] = save_arrays(args.output / "learned_ranks.npz", rank_arrays, args.output)
    if error is None:
        try:
            v, r = constructed_arrays, rank_arrays
            check("source_and_inputs_unchanged",
                  {k: sha(safe(ROOT, k)) for k in source_before} == source_before
                  and all(sha(path) == plan["inputs"]["scanner"]["files_sha256"][k] for k, path in files.items()))
            parameter_same = all(np.array_equal(value, v[key.replace("_before__", "_after__")])
                                 for key, value in v.items() if "_before__" in key)
            check("parameter_buffer_and_rng_invariants", parameter_same
                  and np.array_equal(v["rng_before"], v["rng_after"]))
            check("identical_immediate_full_projections",
                  np.array_equal(v["causal_lens_post"][:, 0, 1], v["causal_lens_post"][:, 1, 1])
                  and np.array_equal(v["causal_probs"][:, 0, 1], v["causal_probs"][:, 1, 1]))
            check("hidden_to_visible_causal_witness",
                  np.array_equal(v["causal_word"][0, :, 2, 0, 0], [1, 2])
                  and not np.array_equal(v["causal_lens_post"][0, 0, 2], v["causal_lens_post"][0, 1, 2]))
            check("unused_hidden_negative_control",
                  np.array_equal(v["causal_lens_post"][1, 0, 2], v["causal_lens_post"][1, 1, 2]))
            check("native_preupdate_lens_and_posthook_frames",
                  np.array_equal(v["causal_lens_pre"], v["causal_lens_post"][:, :, :2])
                  and np.array_equal(v["causal_frames"][:, :, 1], np.broadcast_to(v["causal_initial"], (2, 2, 4, 1, 1)))
                  and np.array_equal(v["causal_final_logits"], v["causal_masked_logits"][:, :, 2]))
            check("finite_pad_policy_and_legacy_difference",
                  np.array_equal(v["pad_legacy_word"][:, 0, 0], [0, 0])
                  and np.array_equal(v["pad_scanner_word"][:, 0, 0], [1, 0])
                  and np.array_equal(v["pad_native_final_logits"], v["pad_masked_logits"]))
            check("top_label_confidence_compression",
                  np.array_equal(v["compression_scanner_word"][:, 0, 0], [1, 1])
                  and abs(float(v["compression_scanner_conf"][0, 0, 0]
                                - v["compression_scanner_conf"][1, 0, 0])) <= 1e-12
                  and not np.array_equal(v["compression_raw_logits"][0], v["compression_raw_logits"][1]))
            check("learned_rank_dimension_bounds", r["observed"].all()
                  and ((r["read_rank"] >= 0) & (r["read_rank"] <= 16)).all()
                  and ((r["nonpad_rank"] >= 0) & (r["nonpad_rank"] <= r["read_rank"])).all()
                  and (r["read_nullity"] >= 32).all() and (r["nonpad_nullity"] >= 32).all())
            check("learned_kernel_projector_witnesses",
                  all(s["kernel_residual_maxabs"] <= 1e-10 * max(1, s["matrix_maxabs"])
                      and s["projector_symmetry_maxabs"] <= 1e-10
                      and s["projector_idempotence_maxabs"] <= 1e-10
                      and abs(s["projector_trace"] - s["nullity"]) <= 1e-10
                      for row in report["learned_ranks"] for s in (row["read"], row["nonpad"])))
            require(len(report["checks"]) == 10, "native check inventory")
            report["status"] = "verified"
        except BaseException as caught:
            error = caught
    if error is not None:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        event("failure", error_type=type(error).__name__, message=str(error))
    report.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic() - started,
                  final_resource_observation=ops.resource_sample("scanner_native_finish"),
                  events=descriptor(event_path, args.output),
                  limits=["Assigned-weight counterexamples, not learned causal semantics.",
                          "Learned projection ranks only; no learned forward, intervention or optimization.",
                          "A rank bound or null direction is not proof that any particular hidden direction affects future output.",
                          "Top-label/confidence compression differs from full probability-vector identifiability.",
                          "Finite PAD suppression is shared policy, not an unconditional exclusion guarantee."])
    if report["final_resource_observation"]["available_ram_gib"] < 8:
        report.update(status="failed", final_ram_floor_violation=True)
    ops.save(args.output / "report.json", report, exclusive=True)
    print(json.dumps({"status": report["status"], "checks": len(report["checks"])}), flush=True)
    return int(report["status"] != "verified")


def main():
    p = argparse.ArgumentParser()
    for name in ("plan", "inputs", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    return execute(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
