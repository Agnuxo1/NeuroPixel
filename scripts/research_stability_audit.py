"""Independent item15 saved-array audit; no Torch, model, producer or RNG calls.

Reconstructs seven literal affine dynamics on the visited nonnegative orthant.
Fourteen constructed trajectories and 3598 correlated states are not training
replications. Native Jacobians are compared with literal matrices at positive
ones; no global ReLU, learned-model, or infinite-time empirical claim follows.
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
CASES = ("identity", "decay", "expansion", "hidden_drift", "two_cycle",
         "nonnormal_transient", "forced_contraction")
SELECTED = (0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256)
ATOL, RTOL = 1e-10, 1e-12
RECIPE = {
    "cases": list(CASES), "vocab": 4, "c_id": 4, "state_channels": 4,
    "hidden": 8, "height": 1, "width": 1, "out_pos": [0, 0],
    "dtype": "float64", "mode": "eval", "construction_seed": 150001,
    "native_steps": 257, "initialization_hook_after_step": 1,
    "continuation_steps": 256, "epsilon": 1 / 16,
    "initial_base": "e2 except forced_contraction zero",
    "initial_perturbation": "epsilon*e3", "future_input": "all PAD",
    "readout": "read.weight[2,2]=1; all other read parameters zero",
    "selected_times": list(SELECTED), "jacobian_point": [1, 1, 1, 1],
    "jacobian_difference_step": 2**-20,
    "jacobian_absolute_tolerance": 1e-9, "jacobian_relative_tolerance": 0,
    "jacobian_autograd_absolute_tolerance": 1e-12,
    "orbit_absolute_tolerance": ATOL, "orbit_relative_tolerance": RTOL,
    "maximum_absolute_state": 1e12, "component_limit_seconds": 120,
    "optimization_steps": 0, "trained_task_competence_assay": False,
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
        raise ValueError("unsafe path: " + name)
    path = root / path
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("missing or indirect file: " + name)
    return path


def ram():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024 / 2**30


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
        delta = np.abs(observed - expected)
        self.errors[label] = float(delta.max()) if delta.size else 0.0
        self.check(np.isfinite(observed).all() and np.isfinite(expected).all()
                   and np.all(delta <= atol + rtol * np.abs(expected)), label + ": tolerance")


def literal_cases():
    matrices = np.zeros((7, 4, 4), dtype=np.float64)
    bias = np.zeros((7, 4), dtype=np.float64)
    initial = np.zeros((7, 2, 4), dtype=np.float64)
    for k in range(7):
        diagonal = 0.5 if k in (1, 4, 5, 6) else 1.05 if k == 2 else 1.0
        for channel in range(4):
            matrices[k, channel, channel] = diagonal
        if k != 6:
            initial[k, :, 2] = 1.0
        initial[k, 1, 3] = 1 / 16
    bias[3, 3], bias[6, 2] = 1.0, 1.0
    matrices[4, 2, 2], matrices[4, 3, 3] = 0.0, 0.0
    matrices[4, 2, 3], matrices[4, 3, 2] = 1.0, 1.0
    matrices[5, 2, 2], matrices[5, 3, 3], matrices[5, 2, 3] = 0.9, 0.9, 4.0
    return matrices, bias, initial


def literal_weights(matrix, bias):
    shapes = {"embed.weight": (4, 4), "seed.weight": (4, 4, 1, 1),
        "seed.bias": (4,), "perceive.weight": (8, 1, 3, 3),
        "f1.weight": (8, 16, 1, 1), "f1.bias": (8,),
        "f2.weight": (4, 8, 1, 1), "f2.bias": (4,),
        "read.weight": (4, 4), "read.bias": (4,), "g_rgb": (4, 3)}
    weights = {key: np.zeros(shape, dtype=np.float64) for key, shape in shapes.items()}
    weights["g_mask"] = np.zeros((4, 1), dtype=np.bool_)
    for i in range(4):
        weights["embed.weight"][i, i] = 1.0
        weights["seed.weight"][i, i, 0, 0] = 1.0
        weights["f1.weight"][i, i, 0, 0] = 1.0
        weights["f2.bias"][i] = bias[i]
        for j in range(4):
            weights["f2.weight"][i, j, 0, 0] = matrix[i, j] - float(i == j)
    weights["read.weight"][2, 2] = 1.0
    return weights


def affine(vector, matrix, bias):
    """Four explicit dot products, independent of the producer einsum."""
    return np.array([math.fsum(float(matrix[i, j]) * float(vector[j]) for j in range(4))
                     + float(bias[i]) for i in range(4)], dtype=np.float64)


def lengths(values):
    return np.sqrt(np.sum(values * values, axis=-1))


def run(args, audit):
    global np
    audit.need(ram() >= 8, "RAM admission below 8 GiB")
    audit.results["available_ram_gib_at_admission"] = ram()
    import numpy as np
    plan = read_json(args.plan)
    audit.need(plan.get("item") == 15 and plan.get("status") == "frozen", "plan not frozen item15")
    audit.need(plan["evidence_recipe"]["native"] == RECIPE, "native recipe differs")
    bindings = plan["implementation_sha256"]
    audit.need(isinstance(bindings, dict) and bool(bindings), "empty source binding map")
    source_before = {name: sha(rooted(ROOT, name)) for name in bindings}
    audit.same(source_before, bindings, "source hashes")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True, timeout=15).stdout.strip()
    audit.need(commit == os.environ.get("GITHUB_SHA"), "source HEAD differs from environment")
    files = {name: rooted(args.native, name) for name in ("report.json", "traces.npz")}
    input_before = {name: sha(path) for name, path in files.items()}
    plan_sha = sha(args.plan)
    report = read_json(files["report.json"])
    audit.need(report["item"] == 15 and report["schema_version"] == 1
               and report["status"] == "verified", "native result not verified")
    audit.need(report["source_commit"] == commit and report["plan_sha256"] == plan_sha,
               "native source or plan differs")
    audit.need(report["recipe"] == RECIPE, "reported recipe differs")
    audit.need(report["array_file"] == "traces.npz" and
               report["array_sha256"] == input_before["traces.npz"], "trace hash differs")
    audit.need(report["optimization_steps"] == 0 and report["checkpoints_loaded"] == 0,
               "constructed-only scope differs")
    audit.need(report["final_resource_observation"]["available_ram_gib"] >= 8,
               "recorded native resource floor")
    A, b, initial = literal_cases()
    schemas = {
        "A": ((7, 4, 4), "float64"), "b": ((7, 4), "float64"), "initial": ((7, 2, 4), "float64"),
        "canvas": ((2, 1, 1), "int64"), "times": ((257,), "int64"), "step_times": ((256,), "int64"),
        "selected_times": ((11,), "int64"), "jacobian_point": ((4,), "float64"),
        "jacobian_h": ((), "float64"), "native_frames": ((7, 2, 258, 4, 1, 1), "float64"),
        "states": ((7, 2, 257, 4), "float64"), "pre_clamp": ((7, 2, 4), "float64"),
        "post_clamp": ((7, 2, 4), "float64"), "native_final_state": ((7, 2, 4), "float64"),
        "native_final_logits": ((7, 2, 4), "float64"), "logits": ((7, 2, 257, 4), "float64"),
        "fd_outputs": ((7, 4, 2, 4), "float64"), "jacobian_fd": ((7, 4, 4), "float64"),
        "jacobian_autograd": ((7, 4, 4), "float64"), "eigenvalues_real": ((7, 4), "float64"),
        "eigenvalues_imag": ((7, 4), "float64"), "singular_values": ((7, 4), "float64"),
        "spectral_radius": ((7,), "float64"), "operator_2_norm": ((7,), "float64"),
        "state_l2": ((7, 2, 257), "float64"), "step_difference_l2": ((7, 2, 256), "float64"),
        "fixedpoint_residual": ((7, 2, 257, 4), "float64"),
        "fixedpoint_residual_l2": ((7, 2, 257), "float64"),
        "pair_l2": ((7, 257), "float64"), "pair_gain": ((7, 257), "float64"),
        "prediction": ((7, 2, 257), "int64"), "token2_nll": ((7, 2, 257), "float64")}
    weights = [literal_weights(A[k], b[k]) for k in range(7)]
    for k, mapping in enumerate(weights):
        for name, value in mapping.items():
            for prefix in ("weight", "weight_after"):
                schemas[f"{prefix}__{k}__{name}"] = (value.shape, str(value.dtype))
        for prefix in ("rng_before", "rng_after"):
            schemas[f"{prefix}__{k}"] = (None, "uint8")
    audit.need(len(schemas) == 214 and set(report["arrays"]) == set(schemas), "array inventory differs")
    with np.load(files["traces.npz"], allow_pickle=False) as stored:
        audit.need(set(stored.files) == set(schemas), "NPZ keys differ")
        arrays = {key: stored[key].copy() for key in schemas}
    for key, (shape, dtype) in schemas.items():
        value = arrays[key]
        audit.need(str(value.dtype) == dtype and (value.shape == shape if shape is not None
                   else value.ndim == 1 and value.size > 0), "array schema: " + key)
        audit.need(np.isfinite(value).all(), "nonfinite array: " + key)
        audit.same(report["arrays"][key], {"shape": list(value.shape), "dtype": dtype}, "descriptor." + key)
    for k, mapping in enumerate(weights):
        for name, value in mapping.items():
            key = f"weight__{k}__{name}"
            audit.check(np.array_equal(arrays[key], value), "literal native weight: " + key)
            audit.check(np.array_equal(arrays[key], arrays[f"weight_after__{k}__{name}"]),
                        "unchanged native weight: " + key)
        audit.check(np.array_equal(arrays[f"rng_before__{k}"], arrays[f"rng_after__{k}"]),
                    "unchanged recorded RNG: " + CASES[k])
    literal = {"A": A, "b": b, "initial": initial, "canvas": np.zeros((2, 1, 1), dtype=np.int64),
        "times": np.arange(257, dtype=np.int64), "step_times": np.arange(1, 257, dtype=np.int64),
        "selected_times": np.asarray(SELECTED, dtype=np.int64),
        "jacobian_point": np.ones(4), "jacobian_h": np.asarray(2**-20, dtype=np.float64)}
    for key, value in literal.items():
        audit.check(np.array_equal(arrays[key], value), "literal case input: " + key)
    state = arrays["states"]
    audit.check(np.array_equal(arrays["native_frames"][:, :, 0], np.zeros((7, 2, 4, 1, 1))),
                "all-PAD native initial seed")
    audit.check(np.array_equal(arrays["pre_clamp"], np.broadcast_to(b[:, None], (7, 2, 4))),
                "native first update before clamp")
    audit.check(np.array_equal(arrays["post_clamp"], initial) and np.array_equal(state[:, :, 0], initial),
                "injected initial pair")
    audit.check(np.array_equal(state, arrays["native_frames"][:, :, 1:, :, 0, 0]), "frame/time alignment")
    audit.check(np.array_equal(state[:, :, -1], arrays["native_final_state"]), "native final state")
    audit.need((state >= 0).all() and np.max(np.abs(state)) <= 1e12, "visited nonnegative bounded census")
    expected = np.empty_like(state)
    expected[:, :, 0] = initial
    residual = np.empty_like(state)
    fd = np.empty((7, 4, 2, 4), dtype=np.float64)
    eigen_real, eigen_imag, singular = [], [], []
    for k in range(7):
        audit.need(ram() >= 8, "RAM floor during reconstruction")
        for batch in range(2):
            for t in range(256):
                expected[k, batch, t + 1] = affine(expected[k, batch, t], A[k], b[k])
            for t in range(257):
                residual[k, batch, t] = affine(state[k, batch, t], A[k], b[k]) - state[k, batch, t]
        for column in range(4):
            for sign_index, sign in enumerate((-1, 1)):
                point = np.ones(4)
                point[column] += sign * 2**-20
                fd[k, column, sign_index] = affine(point, A[k], b[k])
        eigen = np.linalg.eigvals(A[k])
        ordered = sorted((float(value.real), float(value.imag)) for value in eigen)
        eigen_real.append([value[0] for value in ordered])
        eigen_imag.append([value[1] for value in ordered])
        singular.append(np.linalg.svd(A[k], compute_uv=False).tolist())
    audit.array(state, expected, "independent affine orbit")
    audit.array(arrays["fixedpoint_residual"], residual, "analytic residual at saved states")
    audit.array(np.diff(state, axis=2), residual[:, :, :-1], "observed step versus previous residual")
    audit.array(arrays["fd_outputs"], fd, "independent native FD endpoints")
    fd_recount = np.empty((7, 4, 4), dtype=np.float64)
    for k in range(7):
        for column in range(4):
            fd_recount[k, :, column] = (arrays["fd_outputs"][k, column, 1]
                                       - arrays["fd_outputs"][k, column, 0]) / (2 * 2**-20)
    audit.array(arrays["jacobian_fd"], fd_recount, "saved FD formula", atol=1e-9, rtol=0)
    audit.array(arrays["jacobian_fd"], A, "FD versus literal Jacobian", atol=1e-9, rtol=0)
    audit.array(arrays["jacobian_autograd"], A, "autograd versus literal Jacobian", atol=1e-12, rtol=0)
    logits = np.zeros_like(state)
    logits[..., 0] = -10000.0
    logits[..., 2] = state[..., 2]
    audit.array(arrays["logits"], logits, "readout from verified native table", atol=0, rtol=0)
    audit.check(np.array_equal(arrays["native_final_logits"], arrays["logits"][:, :, -1]), "native final logits")
    centered = logits - logits.max(axis=-1, keepdims=True)
    nll = np.logaddexp.reduce(centered, axis=-1) - centered[..., 2]
    prediction = np.argmax(logits, axis=-1)
    audit.check(np.array_equal(arrays["prediction"], prediction), "all diagnostic argmax IDs")
    audit.need((arrays["token2_nll"] >= 0).all(), "negative NLL")
    pair_l2 = lengths(state[:, 1] - state[:, 0])
    recomputed = {"state_l2": lengths(state), "step_difference_l2": lengths(np.diff(state, axis=2)),
        "fixedpoint_residual_l2": lengths(residual), "pair_l2": pair_l2,
        "pair_gain": pair_l2 / (1 / 16), "token2_nll": nll,
        "eigenvalues_real": np.asarray(eigen_real), "eigenvalues_imag": np.asarray(eigen_imag),
        "singular_values": np.asarray(singular),
        "spectral_radius": np.max(np.hypot(eigen_real, eigen_imag), axis=1),
        "operator_2_norm": np.asarray(singular)[:, 0]}
    for key, value in recomputed.items():
        audit.array(arrays[key], value, "metric." + key)
    rows, summaries = [], []
    for k, case in enumerate(CASES):
        gain = recomputed["pair_gain"][k]
        summaries.append({"case": case, "spectral_radius": float(recomputed["spectral_radius"][k]),
            "operator_2_norm": float(recomputed["operator_2_norm"][k]),
            "peak_pair_gain": float(max(gain)), "first_peak_pair_time": int(np.argmax(gain)),
            "final_pair_gain": float(gain[-1]),
            "maximum_state_l2_by_trajectory": [float(max(row)) for row in recomputed["state_l2"][k]],
            "final_state_l2_by_trajectory": recomputed["state_l2"][k, :, -1].tolist(),
            "final_residual_l2_by_trajectory": recomputed["fixedpoint_residual_l2"][k, :, -1].tolist(),
            "maximum_jacobian_absolute_error": float(np.max(np.abs(arrays["jacobian_fd"][k] - A[k]))),
            "maximum_autograd_absolute_error": float(np.max(np.abs(arrays["jacobian_autograd"][k] - A[k])))})
        for batch, trajectory in enumerate(("base", "perturbed")):
            for t in SELECTED:
                rows.append({"case": case, "trajectory": trajectory, "time": t,
                    "state_l2": float(recomputed["state_l2"][k, batch, t]),
                    "step_difference_l2": None if t == 0 else float(recomputed["step_difference_l2"][k, batch, t - 1]),
                    "fixedpoint_residual_l2": float(recomputed["fixedpoint_residual_l2"][k, batch, t]),
                    "pair_gain": float(gain[t]), "prediction": int(prediction[k, batch, t]),
                    "token2_nll": float(nll[k, batch, t])})
    audit.same(report["rows"], rows, "all selected rows")
    audit.same(report["case_summaries"], summaries, "all case summaries")
    audit.same(report["models_unchanged"], [True] * 7, "reported weight invariance")
    check_names = ["complete_inventory", "native_initialization_and_alignment", "native_final_state_and_logits",
        "literal_model_parameters_unchanged", "runtime_rng_unchanged", "finite_bounded_census",
        "central_difference_matches_declared_affine_rule", "autograd_matches_declared_affine_rule",
        "identity_is_neutral", "decay_and_forced_contraction", "expansion_and_hidden_drift",
        "oscillation_and_nonnormal_transient"]
    checks = report["checks"]
    audit.need([row["name"] for row in checks] == check_names, "native check inventory")
    for row in checks:
        audit.check(row["passed"] is True, "recorded check: " + row["name"])
    audit.same({key: checks[0][key] for key in
                ("trajectories", "aligned_states", "native_frame_snapshots", "summary_rows")},
               {"trajectories": 14, "aligned_states": 3598, "native_frame_snapshots": 3612, "summary_rows": 154},
               "population denominators")
    audit.check(np.array_equal(state[0], np.broadcast_to(initial[0, :, None], (2, 257, 4)))
                and np.array_equal(recomputed["pair_gain"][0], np.ones(257)), "neutral identity witness")
    audit.check(np.all(np.diff(recomputed["state_l2"][1], axis=-1) < 0)
                and np.all(np.diff(recomputed["pair_gain"][[1, 6]], axis=-1) < 0)
                and np.all(np.abs(state[6, :, -1] - [0, 0, 2, 0]) <= 1e-12), "contracting witnesses")
    audit.check(np.all(np.diff(recomputed["state_l2"][2], axis=-1) > 0)
                and recomputed["pair_gain"][2, -1] > 1
                and recomputed["state_l2"][3, 0, -1] > recomputed["state_l2"][3, 0, 0]
                and np.array_equal(logits[3], np.broadcast_to(logits[3, :, :1], (2, 257, 4))),
                "expansion and readout-hidden drift witnesses")
    audit.check(np.array_equal(state[4, :, 2:], state[4, :, :-2])
                and not np.array_equal(state[4, :, 0], state[4, :, 1])
                and recomputed["spectral_radius"][5] < 1 and recomputed["operator_2_norm"][5] > 1
                and max(recomputed["pair_gain"][5]) > 1 and recomputed["pair_gain"][5, -1] < 1,
                "cycle and nonnormal transient witnesses")
    audit.need({name: sha(path) for name, path in files.items()} == input_before, "raw inputs changed")
    audit.need(sha(args.plan) == plan_sha, "plan changed")
    audit.need({name: sha(rooted(ROOT, name)) for name in bindings} == source_before, "source changed")
    audit.results.update(rows=rows, case_summaries=summaries, source_commit=commit, plan_sha256=plan_sha,
        input_sha256=input_before, implementation_sha256=bindings, auditor_sha256=sha(Path(__file__)),
        arrays_checked=len(schemas), trajectories=14, aligned_states=3598, native_frame_snapshots=3612,
        summary_rows=154, maximum_absolute_errors=audit.errors,
        tolerances={"orbit_and_metrics_atol": ATOL, "orbit_and_metrics_rtol": RTOL,
                    "finite_difference_atol": 1e-9, "finite_difference_rtol": 0,
                    "autograd_atol": 1e-12, "autograd_rtol": 0},
        available_ram_gib_at_finish=ram())
    audit.need(audit.results["available_ram_gib_at_finish"] >= 8, "RAM floor at finish")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit, started = Audit(), time.monotonic()
    try:
        run(args, audit)
    except BaseException as error:
        audit.issues.append({"exception": type(error).__name__, "message": str(error)})
    report = {"schema_version": 1, "item": 15, "status": "verified" if not audit.issues else "failed",
        "checks": audit.checks, "issues": audit.issues, "results": audit.results,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - started,
        "runtime": {"python": platform.python_version(), "numerical_threads": 1,
                    "numpy": sys.modules["numpy"].__version__ if "numpy" in sys.modules else None},
        "limits": ["Constructed known weights and two initial conditions per case; no learning or task accuracy.",
                  "Affine rules apply on the visited nonnegative orthant; Jacobians are checked at positive ones.",
                  "Finite 256-update continuations are not empirical infinite-time or learned-model guarantees.",
                  "The final analytic residual is not an additional observed native update.",
                  "Saved RNG bytes are checked for equality, without regenerating or decoding Torch RNG state.",
                  "Saved autograd values are compared with literal derivatives; this auditor does not run autograd.",
                  "Fourteen trajectories and 3598 correlated endpoints are not independent training replications."]}
    with (args.output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "checks": audit.checks, "issues": len(audit.issues)}), flush=True)
    return int(bool(audit.issues))


if __name__ == "__main__":
    raise SystemExit(main())
