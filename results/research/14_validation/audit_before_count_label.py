"""Independent item-14 saved-array audit of constructed state-repair controls.

No Torch, model, producer, helper, checkpoint loading, training, or RNG is used.
Weights are verified against literal constructions; local averaging uses explicit
zero-padded spatial loops. These finite witnesses establish no learned competence
or long-time stability. The input rule can reconstruct from a retained cue.
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
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
MODELS = ("input_copier", "state_holder", "spatial_average")
LESIONS = ("none", "center", "ring", "all")
SOURCES = ("held", "removed", "changed")
RECIPE = {"models": list(MODELS), "lesions": list(LESIONS),
          "sources": list(SOURCES), "cues": [2, 3, 4, 5, 6, 7],
          "vocab": 8, "c_id": 8, "state_channels": 8, "hidden": 16,
          "height": 3, "width": 3, "out_pos": [1, 1],
          "warmup_steps": 2, "recovery_steps": 8,
          "dtype": "float64", "mode": "eval", "optimization_steps": 0,
          "trained_task_competence_assay": False}
TOLERANCE = 1e-12


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def ram():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024 / 2**30


def rooted(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError("unsafe relative path: " + name)
    path = root / path
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("missing or indirect path: " + name)
    return path


class Audit:
    def __init__(self):
        self.checks, self.issues, self.results = 0, [], {}
    def check(self, condition, label):
        self.checks += 1
        if not bool(condition):
            self.issues.append(label)
        return bool(condition)
    def need(self, condition, label):
        if not self.check(condition, label):
            raise ValueError(label)
    def same(self, observed, expected, label):
        if isinstance(expected, dict):
            self.need(isinstance(observed, dict) and set(observed) == set(expected), label + ": keys")
            for key in expected:
                self.same(observed[key], expected[key], label + "." + key)
        elif isinstance(expected, list):
            self.need(isinstance(observed, list) and len(observed) == len(expected), label + ": length")
            for index, value in enumerate(expected):
                self.same(observed[index], value, label + "[" + str(index) + "]")
        elif isinstance(expected, float):
            self.check(type(observed) in (int, float) and math.isfinite(observed)
                       and abs(observed - expected) <= TOLERANCE, label + ": numeric")
        else:
            self.check(type(observed) is type(expected) and observed == expected, label + ": value")


def expected_weights(kind):
    shapes = {"embed.weight": (8, 8), "seed.weight": (8, 8, 1, 1), "seed.bias": (8,),
              "perceive.weight": (16, 1, 3, 3), "f1.weight": (16, 32, 1, 1),
              "f1.bias": (16,), "f2.weight": (8, 16, 1, 1), "f2.bias": (8,),
              "read.weight": (8, 8), "read.bias": (8,), "g_rgb": (8, 3)}
    weights = {key: np.zeros(shape, dtype=np.float64) for key, shape in shapes.items()}
    weights["g_mask"] = np.zeros((8, 1), dtype=np.bool_)
    for channel in range(8):
        weights["embed.weight"][channel, channel] = 1.0
        weights["seed.weight"][channel, channel, 0, 0] = 1.0
        weights["read.weight"][channel, channel] = 1.0
        if kind != "state_holder":
            weights["f1.weight"][channel, channel, 0, 0] = 1.0
            weights["f2.weight"][channel, channel, 0, 0] = -1.0
            weights["f2.weight"][channel, 8 + channel, 0, 0] = 1.0
            if kind == "input_copier":
                weights["f1.weight"][8 + channel, 24 + channel, 0, 0] = 1.0
            else:
                for y in range(3):
                    for x in range(3):
                        weights["perceive.weight"][2 * channel, 0, y, x] = 1.0 / 9.0
                weights["f1.weight"][8 + channel, 8 + 2 * channel, 0, 0] = 1.0
    return weights


def identity(canvas):
    result = np.zeros((6, 8, 3, 3), dtype=np.float64)
    for batch in range(6):
        for y in range(3):
            for x in range(3):
                token = int(canvas[batch, y, x])
                if token:
                    result[batch, token, y, x] = 1.0
    return result


def update(state, identities, kind):
    """Algebraic rules after verifying every native parameter and buffer."""
    if kind == "input_copier":
        return identities.copy()
    if kind == "state_holder":
        return state.copy()
    answer = np.zeros_like(state)
    for y in range(3):
        for x in range(3):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < 3 and 0 <= xx < 3:
                        answer[:, :, y, x] += state[:, :, yy, xx] / 9.0
    return answer


def readout(state):
    result = state[:, :, 1, 1].copy()
    result[:, 0] = -10000.0
    return result


def audit(args, result):
    global np
    result.need(ram() >= 8, "RAM admission below 8 GiB")
    result.results["available_ram_gib_at_admission"] = ram()
    import numpy as np
    plan = read_json(args.plan)
    result.need(plan.get("item") == 14 and plan.get("status") == "frozen", "plan is not frozen item14")
    result.same(plan["evidence_recipe"]["native"], RECIPE, "native recipe")
    bindings = plan["implementation_sha256"]
    result.need(isinstance(bindings, dict) and bool(bindings), "missing implementation bindings")
    before_source = {name: sha(rooted(ROOT, name)) for name in bindings}
    result.same(before_source, bindings, "source file hashes")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True, timeout=15).stdout.strip()
    result.need(commit == os.environ.get("GITHUB_SHA"), "actual source HEAD differs from GITHUB_SHA")
    files = {"report.json": rooted(args.native, "report.json"),
             "traces.npz": rooted(args.native, "traces.npz")}
    before_inputs = {name: sha(path) for name, path in files.items()}
    plan_sha = sha(args.plan)
    report = read_json(files["report.json"])
    result.need(report["item"] == 14 and report["schema_version"] == 1
                and report["status"] == "verified", "native producer not verified")
    result.need(report["source_commit"] == commit and report["plan_sha256"] == plan_sha,
                "native source or plan differs")
    result.same(report["recipe"], RECIPE, "reported recipe")
    result.need(report["array_file"] == "traces.npz"
                and report["array_sha256"] == before_inputs["traces.npz"], "trace descriptor differs")
    result.need(report["optimization_steps"] == 0 and report["checkpoints_loaded"] == 0,
                "constructed-only scope differs")
    result.need(report["final_resource_observation"]["available_ram_gib"] >= 8,
                "recorded native RAM floor failed")
    schemas = {
        "cues": ((6,), "int64"), "canvas": ((6, 3, 3), "int64"),
        "future": ((3, 6, 3, 3), "int64"), "keep": ((4, 1, 3, 3), "bool"),
        "prefix_states": ((3, 6, 3, 8, 3, 3), "float64"),
        "prefix_logits": ((3, 6, 3, 8), "float64"),
        "pre_damage": ((3, 4, 3, 6, 8, 3, 3), "float64"),
        "post_damage": ((3, 4, 3, 6, 8, 3, 3), "float64"),
        "recovery_states": ((3, 4, 3, 6, 9, 8, 3, 3), "float64"),
        "recovery_logits": ((3, 4, 3, 6, 9, 8), "float64"),
        "native_held_states": ((3, 4, 6, 9, 8, 3, 3), "float64"),
        "native_held_logits": ((3, 4, 6, 8), "float64"),
        "predictions": ((3, 4, 3, 6, 9), "int64"),
        "correct": ((3, 4, 3, 6, 9), "bool"),
        "nll": ((3, 4, 3, 6, 9), "float64"),
        "case_index": ((216, 4), "int64")}
    weights = [expected_weights(kind) for kind in MODELS]
    for mi, mapping in enumerate(weights):
        for name, value in mapping.items():
            for prefix in ("weight", "weight_after"):
                schemas[f"{prefix}__{mi}__{name}"] = (value.shape, str(value.dtype))
    result.need(set(report["arrays"]) == set(schemas), "reported array inventory differs")
    with np.load(files["traces.npz"], allow_pickle=False) as loaded:
        result.need(set(loaded.files) == set(schemas), "NPZ array inventory differs")
        arrays = {key: loaded[key].copy() for key in schemas}
    for key, (shape, dtype) in schemas.items():
        value = arrays[key]
        result.need(value.shape == shape and str(value.dtype) == dtype, "array schema: " + key)
        result.need(np.isfinite(value).all(), "nonfinite array: " + key)
        result.same(report["arrays"][key], {"shape": list(shape), "dtype": dtype}, "descriptor." + key)
    for mi, mapping in enumerate(weights):
        for name, expected in mapping.items():
            key, after = f"weight__{mi}__{name}", f"weight_after__{mi}__{name}"
            result.check(np.array_equal(arrays[key], expected), "literal weights: " + key)
            result.check(np.array_equal(arrays[after], arrays[key]), "unchanged weights: " + key)
    cues = np.array([2, 3, 4, 5, 6, 7], dtype=np.int64)
    canvas = np.broadcast_to(cues[:, None, None], (6, 3, 3)).copy()
    changed = np.broadcast_to(np.array([3, 4, 5, 6, 7, 2])[:, None, None], (6, 3, 3)).copy()
    future = np.stack([canvas, np.zeros_like(canvas), changed])
    keep = np.ones((4, 1, 3, 3), dtype=np.bool_)
    keep[1, 0, 1, 1] = False
    keep[2] = False
    keep[2, 0, 1, 1] = True
    keep[3] = False
    cases = np.array([(m, l, s, c) for m in range(3) for l in range(4)
                      for s in range(3) for c in range(6)], dtype=np.int64)
    for key, expected in (("cues", cues), ("canvas", canvas), ("future", future),
                          ("keep", keep), ("case_index", cases)):
        result.check(np.array_equal(arrays[key], expected), "literal input/order: " + key)
    expected = {key: np.empty(shape, dtype=np.float64) for key, (shape, dtype) in schemas.items()
                if key in ("prefix_states", "prefix_logits", "pre_damage", "post_damage",
                           "recovery_states", "recovery_logits")}
    ids = identity(canvas)
    for mi, kind in enumerate(MODELS):
        result.need(ram() >= 8, "RAM floor during reconstruction")
        prefix = [ids.copy()]
        for step in range(2):
            prefix.append(update(prefix[-1], ids, kind))
        expected["prefix_states"][mi] = np.stack(prefix, axis=1)
        expected["prefix_logits"][mi] = np.stack([readout(state) for state in prefix], axis=1)
        for li in range(4):
            for si in range(3):
                start = prefix[-1].copy()
                damaged = start * keep[li]
                expected["pre_damage"][mi, li, si] = start
                expected["post_damage"][mi, li, si] = damaged
                frames = [damaged]
                for step in range(8):
                    frames.append(update(frames[-1], identity(future[si]), kind))
                expected["recovery_states"][mi, li, si] = np.stack(frames, axis=1)
                expected["recovery_logits"][mi, li, si] = np.stack([readout(state) for state in frames], axis=1)
    errors = {}
    for key, value in expected.items():
        errors[key] = float(np.max(np.abs(arrays[key] - value)))
        result.check(errors[key] <= TOLERANCE, "independent dynamics/readout: " + key)
    native = []
    for mi, kind in enumerate(MODELS):
        for li, lesion in enumerate(LESIONS):
            state_error = float(np.max(np.abs(arrays["native_held_states"][mi, li]
                                             - arrays["recovery_states"][mi, li, 0])))
            logit_error = float(np.max(np.abs(arrays["native_held_logits"][mi, li]
                                             - arrays["recovery_logits"][mi, li, 0, :, -1])))
            result.check(state_error == 0 and logit_error == 0, "native equality: " + kind + "/" + lesion)
            native.append({"model": kind, "lesion": lesion,
                           "maximum_state_error": state_error, "maximum_logit_error": logit_error})
    result.same(report["native_equivalence"], native, "native equivalence records")
    result.same(report["models_unchanged"], [{"model": m, "unchanged": True} for m in MODELS],
                "weight invariance records")
    logits = arrays["recovery_logits"]
    prediction = np.argmax(logits, axis=-1)
    result.check(np.array_equal(prediction, arrays["predictions"]), "saved argmax")
    result.check(np.array_equal(prediction, np.argmax(expected["recovery_logits"], axis=-1)),
                 "independent rule argmax")
    gold = np.broadcast_to(cues.reshape(1, 1, 1, 6, 1), prediction.shape)
    correct = prediction == gold
    result.check(np.array_equal(correct, arrays["correct"]), "saved correctness")
    centered = logits - logits.max(axis=-1, keepdims=True)
    nll = np.logaddexp.reduce(centered, axis=-1) - np.take_along_axis(
        centered, gold[..., None], axis=-1)[..., 0]
    nll_error = float(np.max(np.abs(nll - arrays["nll"])))
    result.check(nll_error <= TOLERANCE and (arrays["nll"] >= 0).all(), "independent stable NLL")
    rows = []
    for mi, kind in enumerate(MODELS):
        before = arrays["prefix_logits"][mi, :, 2].argmax(-1) == cues
        for li, lesion in enumerate(LESIONS):
            for si, source in enumerate(SOURCES):
                immediate = correct[mi, li, si, :, 0]
                lost = before & ~immediate
                for step in range(9):
                    later = correct[mi, li, si, :, step]
                    delta = arrays["recovery_states"][mi, li, si, :, step] - arrays["recovery_states"][mi, 0, si, :, step]
                    rows.append({"model": kind, "lesion": lesion, "source": source, "recovery_step": step,
                        "n": 6, "pre_correct": int(np.count_nonzero(before)),
                        "immediate_correct": int(np.count_nonzero(immediate)),
                        "late_correct": int(np.count_nonzero(later)),
                        "matched_sham_correct": int(np.count_nonzero(correct[mi, 0, si, :, step])),
                        "initially_correct_then_lost": int(np.count_nonzero(lost)),
                        "lost_then_recovered": int(np.count_nonzero(lost & later)),
                        "survived_and_still_correct": int(np.count_nonzero(before & immediate & later)),
                        "late_matches_changed_cue": int(np.count_nonzero(prediction[mi, li, si, :, step] == changed[:, 0, 0])),
                        "mean_nll_original_cue": float(nll[mi, li, si, :, step].sum() / 6),
                        "rms_state_difference_from_time_matched_sham": float(math.sqrt(np.square(delta).sum() / delta.size))})
    result.same(report["rows"], rows, "all summary rows")
    names = ["complete_inventory", "held_source_exact_native_equivalence",
             "source_copier_full_erasure_rebuilds", "source_copier_reconstruction_is_source_dependent",
             "state_holder_survives_but_cannot_repair_center",
             "redundant_spatial_state_restores_categorical_readout",
             "total_erasure_with_common_future_is_cue_indistinguishable", "all_states_finite_and_models_unchanged"]
    checks = report["checks"]
    result.need([row["name"] for row in checks] == names, "native check inventory")
    for row in checks:
        result.check(row["passed"] is True, "recorded native check: " + row["name"])
    result.same({k: checks[0][k] for k in ("trajectories", "summary_rows", "endpoints")},
                {"trajectories": 216, "summary_rows": 324, "endpoints": 1944}, "population counts")
    result.check(correct[0, 3, 0, :, 1:].all() and not correct[0, 3, 0, :, 0].any(), "copier recovery")
    result.check(not correct[0, 3, 1].any() and
                 (prediction[0, 3, 2, :, 1:] == changed[:, 0, 0, None]).all(), "copier depends on source")
    result.check(correct[1, 0, 1].all() and not correct[1, 1, 1].any(), "holder survival versus loss")
    result.check(not correct[2, 1, 1, :, 0].any() and correct[2, 1, 1, :, 1:].all(), "spatial redundancy witness")
    indistinguishable = all(np.array_equal(logits[mi, 3, 1, 0], logits[mi, 3, 1, ci])
                            for mi in range(3) for ci in range(6))
    result.check(indistinguishable, "complete erasure removes cue distinction")
    result.same(checks[6]["accuracy_on_balanced_cue_set"], int(correct[:, 3, 1].sum()), "erasure correct count")
    result.same(checks[6]["chance_upper_bound_for_identical_predictor"], 1 / 6, "balanced maximum")
    result.need({key: sha(path) for key, path in files.items()} == before_inputs, "native inputs changed")
    result.need(sha(args.plan) == plan_sha, "plan changed")
    result.need({name: sha(rooted(ROOT, name)) for name in bindings} == before_source, "source changed")
    result.results.update(rows=rows, source_commit=commit, plan_sha256=plan_sha,
        implementation_sha256=bindings, input_sha256=before_inputs, arrays_checked=len(schemas),
        trajectories=216, endpoints=1944, summary_rows=324, native_equivalence=native,
        maximum_absolute_errors=errors, maximum_nll_error=nll_error,
        absolute_tolerance=TOLERANCE, relative_tolerance=0,
        auditor_sha256=sha(Path(__file__)), available_ram_gib_at_finish=ram())
    result.need(result.results["available_ram_gib_at_finish"] >= 8, "RAM floor at finish")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result, started = Audit(), time.monotonic()
    try:
        audit(args, result)
    except BaseException as error:
        result.issues.append({"exception": type(error).__name__, "message": str(error)})
    record = {"schema_version": 1, "item": 14, "status": "verified" if not result.issues else "failed",
              "checks": result.checks, "issues": result.issues, "results": result.results,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "wall_seconds": time.monotonic() - started,
              "runtime": {"python": platform.python_version(),
                          "numpy": sys.modules["numpy"].__version__ if "numpy" in sys.modules else None,
                          "numerical_threads": 1},
              "limits": ["Finite constructed rules and balanced six-cue witnesses; no learned task assay.",
                         "Recovery is separated from intact survival and input-driven reconstruction.",
                         "No claim about training, generalization, long-time stability, or external replication."]}
    with (args.output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": record["status"], "checks": result.checks,
                      "issues": len(result.issues)}), flush=True)
    return int(bool(result.issues))


if __name__ == "__main__":
    raise SystemExit(main())
