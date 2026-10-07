"""Finite native dynamical controls for item15; no training or learned claim.

The hook sets an explicit initial condition after native update1. The next256
updates are untouched native recurrence. Literal affine rules hold only in the
nonnegative invariant orthant visited here, not on every ReLU activation region.
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

CASES = ("identity", "decay", "expansion", "hidden_drift", "two_cycle",
         "nonnormal_transient", "forced_contraction")
SELECTED = (0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256)
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
    "orbit_absolute_tolerance": 1e-10, "orbit_relative_tolerance": 1e-12,
    "maximum_absolute_state": 1e12, "component_limit_seconds": 120,
    "optimization_steps": 0, "trained_task_competence_assay": False,
}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_case(index):
    import torch
    from neuropixel.model import NeuroPixel
    torch.manual_seed(RECIPE["construction_seed"])
    model = NeuroPixel(4, (0, 0), c_id=4, c=4, hidden=8,
                       steps=257, fire_rate=0.5).double().eval()
    matrix = torch.eye(4, dtype=torch.float64)
    bias = torch.zeros(4, dtype=torch.float64)
    if index in (1, 6):
        matrix *= 0.5
    elif index == 2:
        matrix *= 1.05
    elif index == 3:
        bias[3] = 1
    elif index in (4, 5):
        matrix *= 0.5
        matrix[2:, 2:] = torch.tensor([[0, 1], [1, 0]] if index == 4
                                      else [[0.9, 4], [0, 0.9]], dtype=torch.float64)
    if index == 6:
        bias[2] = 1
    initial = torch.zeros((2, 4), dtype=torch.float64)
    if index != 6:
        initial[:, 2] = 1
    initial[1, 3] = RECIPE["epsilon"]
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
        model.embed.weight.copy_(torch.eye(4, dtype=torch.float64))
        model.seed.weight[:, :, 0, 0].copy_(torch.eye(4, dtype=torch.float64))
        model.f1.weight[:4, :4, 0, 0].copy_(torch.eye(4, dtype=torch.float64))
        model.f2.weight[:, :4, 0, 0].copy_(matrix - torch.eye(4, dtype=torch.float64))
        model.f2.bias.copy_(bias)
        model.read.weight[2, 2] = 1
    return model, matrix, bias, initial


def execute(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    partial, trace_buffers = {}, {}
    report = {"schema_version": 1, "item": 15, "status": "running",
              "source_commit": os.environ.get("GITHUB_SHA"),
              "plan_sha256": sha(args.plan), "recipe": RECIPE,
              "started_at_utc": ops.now(), "checks": []}
    def check(name, ok, **fields):
        report["checks"].append({"name": name, "passed": bool(ok), **fields})
        require(ok, name)
    try:
        plan = json.loads(args.plan.read_bytes())
        require(plan["item"] == 15 and plan["status"] == "frozen"
                and plan["evidence_recipe"]["native"] == RECIPE, "frozen native recipe differs")
        ops.admit(started + 120, minimum_seconds=10)
        require(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "thread environment differs")
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.version.cuda is None, "CPU-only control")
        bindings = plan["implementation_sha256"]
        require({p: sha(ROOT / p) for p in bindings} == bindings, "source bindings differ")
        arrays = {
            "canvas": np.zeros((2, 1, 1), dtype=np.int64),
            "times": np.arange(257, dtype=np.int64),
            "step_times": np.arange(1, 257, dtype=np.int64),
            "selected_times": np.asarray(SELECTED, dtype=np.int64),
            "jacobian_point": np.ones(4, dtype=np.float64),
            "jacobian_h": np.asarray(2**-20, dtype=np.float64),
        }
        shapes = {"A": (7, 4, 4), "b": (7, 4), "initial": (7, 2, 4),
                  "native_frames": (7, 2, 258, 4, 1, 1),
                  "pre_clamp": (7, 2, 4), "post_clamp": (7, 2, 4),
                  "native_final_state": (7, 2, 4), "native_final_logits": (7, 2, 4),
                  "logits": (7, 2, 257, 4), "fd_outputs": (7, 4, 2, 4),
                  "jacobian_fd": (7, 4, 4), "eigenvalues_real": (7, 4),
                  "eigenvalues_imag": (7, 4), "singular_values": (7, 4),
                  "spectral_radius": (7,), "operator_2_norm": (7,)}
        arrays.update({name: np.empty(shape, dtype=np.float64) for name, shape in shapes.items()})
        canvas = torch.zeros((2, 1, 1), dtype=torch.long)
        unchanged = []
        with torch.no_grad():
            for index, name in enumerate(CASES):
                ops.remaining(started + 120)
                model, matrix, bias, initial = build_case(index)
                arrays["A"][index], arrays["b"][index] = matrix.numpy(), bias.numpy()
                arrays["initial"][index] = initial.numpy()
                before = {k: v.detach().clone() for k, v in
                          list(model.named_parameters()) + list(model.named_buffers())}
                for key, value in before.items():
                    arrays[f"weight__{index}__{key}"] = value.numpy().copy()
                    partial[f"weight__{index}__{key}"] = value.numpy().copy()
                arrays[f"rng_before__{index}"] = torch.get_rng_state().numpy().copy()
                trace_buffers[index] = []
                def trajectory_hook(step, state):
                    ops.remaining(started + 120)
                    if step == 1:
                        arrays["pre_clamp"][index] = state[:, :, 0, 0].numpy()
                        partial[f"pre_clamp__{index}"] = state.numpy().copy()
                        state = initial[:, :, None, None].clone()
                        arrays["post_clamp"][index] = state[:, :, 0, 0].numpy()
                    trace_buffers[index].append(state.detach().clone())
                    require(torch.isfinite(state).all(), "nonfinite trajectory state")
                    require(state.abs().max() <= 1e12, "state absolute-value guard exceeded")
                    return state
                output = model(canvas, trace=True, steps=257, hook=trajectory_hook)
                arrays["native_frames"][index] = output["frames"].numpy()
                arrays["native_final_state"][index] = output["state"][:, :, 0, 0].numpy()
                arrays["native_final_logits"][index] = output["logits"].numpy()
                states = output["frames"][:, 1:, :, 0, 0]
                logits = model.read(states) @ model.dictionary().T
                logits[..., 0] = -1e4
                arrays["logits"][index] = logits.numpy()
                # Each finite-difference endpoint comes from native forward:
                # update1 is followed by the requested point injection, then
                # update2 is the original, untouched state update.
                point = torch.ones((1, 4, 1, 1), dtype=torch.float64)
                for column in range(4):
                    for sign_index, sign in enumerate((-1, 1)):
                        injected = point.clone()
                        injected[0, column, 0, 0] += sign * 2**-20
                        def point_hook(step, state, injected=injected):
                            return injected.clone() if step == 1 else state
                        result = model(canvas[:1], steps=2, hook=point_hook)
                        value = result["state"][0, :, 0, 0].numpy().copy()
                        arrays["fd_outputs"][index, column, sign_index] = value
                        partial[f"fd__{index}_{column}_{sign_index}"] = value
                fd = arrays["fd_outputs"][index]
                arrays["jacobian_fd"][index] = ((fd[:, 1] - fd[:, 0]) / (2 * 2**-20)).T
                eigen = torch.linalg.eigvals(matrix).numpy()
                order = np.lexsort((eigen.imag, eigen.real))
                arrays["eigenvalues_real"][index] = eigen.real[order]
                arrays["eigenvalues_imag"][index] = eigen.imag[order]
                singular = torch.linalg.svdvals(matrix).numpy()
                arrays["singular_values"][index] = singular
                arrays["spectral_radius"][index] = np.abs(eigen).max()
                arrays["operator_2_norm"][index] = singular[0]
                arrays[f"rng_after__{index}"] = torch.get_rng_state().numpy().copy()
                after = dict(list(model.named_parameters()) + list(model.named_buffers()))
                for key, value in after.items():
                    arrays[f"weight_after__{index}__{key}"] = value.detach().numpy().copy()
                unchanged.append(all(torch.equal(value, after[key]) for key, value in before.items()))
        arrays["states"] = arrays["native_frames"][:, :, 1:, :, 0, 0].copy()
        states = arrays["states"]
        arrays["state_l2"] = np.linalg.norm(states, axis=-1)
        arrays["step_difference_l2"] = np.linalg.norm(np.diff(states, axis=2), axis=-1)
        # Analytic residual is recorded at every t, including the final state.
        # It is not an extra observed native update or a training objective.
        residual = np.einsum("kij,kbtj->kbti", arrays["A"], states) + arrays["b"][:, None, None, :] - states
        arrays["fixedpoint_residual"] = residual
        arrays["fixedpoint_residual_l2"] = np.linalg.norm(residual, axis=-1)
        arrays["pair_l2"] = np.linalg.norm(states[:, 1] - states[:, 0], axis=-1)
        arrays["pair_gain"] = arrays["pair_l2"] / RECIPE["epsilon"]
        arrays["prediction"] = arrays["logits"].argmax(-1).astype(np.int64)
        centered = arrays["logits"] - arrays["logits"].max(axis=-1, keepdims=True)
        arrays["token2_nll"] = np.log(np.exp(centered).sum(axis=-1)) - centered[..., 2]
        rows, summaries = [], []
        for index, name in enumerate(CASES):
            gain = arrays["pair_gain"][index]
            summaries.append({
                "case": name, "spectral_radius": float(arrays["spectral_radius"][index]),
                "operator_2_norm": float(arrays["operator_2_norm"][index]),
                "peak_pair_gain": float(gain.max()), "first_peak_pair_time": int(gain.argmax()),
                "final_pair_gain": float(gain[-1]),
                "maximum_state_l2_by_trajectory": arrays["state_l2"][index].max(axis=-1).tolist(),
                "final_state_l2_by_trajectory": arrays["state_l2"][index, :, -1].tolist(),
                "final_residual_l2_by_trajectory": arrays["fixedpoint_residual_l2"][index, :, -1].tolist(),
                "maximum_jacobian_absolute_error": float(np.abs(arrays["jacobian_fd"][index] - arrays["A"][index]).max()),
            })
            for batch, trajectory in enumerate(("base", "perturbed")):
                for t in SELECTED:
                    rows.append({
                        "case": name, "trajectory": trajectory, "time": t,
                        "state_l2": float(arrays["state_l2"][index, batch, t]),
                        "step_difference_l2": None if t == 0 else float(arrays["step_difference_l2"][index, batch, t - 1]),
                        "fixedpoint_residual_l2": float(arrays["fixedpoint_residual_l2"][index, batch, t]),
                        "pair_gain": float(gain[t]),
                        "prediction": int(arrays["prediction"][index, batch, t]),
                        "token2_nll": float(arrays["token2_nll"][index, batch, t]),
                    })
        report.update(rows=rows, case_summaries=summaries, models_unchanged=unchanged,
                      optimization_steps=0, checkpoints_loaded=0)
        # Preserve every completed raw array before checking expected outcomes.
        path = args.output / "traces.npz"
        np.savez_compressed(path, **arrays)
        report.update(array_file="traces.npz", array_sha256=sha(path),
                      arrays={key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                              for key, value in arrays.items()})
        check("complete_inventory", len(rows) == 154 and states.shape == (7, 2, 257, 4),
              trajectories=14, aligned_states=3598, native_frame_snapshots=3612, summary_rows=154)
        check("native_initialization_and_alignment",
              np.array_equal(arrays["native_frames"][:, :, 0], np.zeros((7, 2, 4, 1, 1)))
              and np.array_equal(arrays["pre_clamp"], np.broadcast_to(arrays["b"][:, None], (7, 2, 4)))
              and np.array_equal(arrays["post_clamp"], arrays["initial"])
              and np.array_equal(states[:, :, 0], arrays["initial"]))
        check("native_final_state_and_logits",
              np.array_equal(states[:, :, -1], arrays["native_final_state"])
              and np.array_equal(arrays["logits"][:, :, -1], arrays["native_final_logits"]))
        check("literal_model_parameters_unchanged", all(unchanged))
        check("runtime_rng_unchanged", all(np.array_equal(arrays[f"rng_before__{i}"],
                                                        arrays[f"rng_after__{i}"]) for i in range(7)))
        check("finite_bounded_census", all(np.isfinite(value).all() for value in arrays.values())
              and np.abs(states).max() <= 1e12 and (states >= 0).all())
        check("central_difference_matches_declared_affine_rule",
              np.abs(arrays["jacobian_fd"] - arrays["A"]).max() <= 1e-9)
        check("identity_is_neutral", np.array_equal(states[0], np.broadcast_to(arrays["initial"][0, :, None], (2, 257, 4)))
              and np.array_equal(arrays["pair_gain"][0], np.ones(257)))
        check("decay_and_forced_contraction",
              np.all(np.diff(arrays["state_l2"][1], axis=-1) < 0)
              and np.all(np.diff(arrays["pair_gain"][[1, 6]], axis=-1) < 0)
              and np.allclose(states[6, :, -1], np.array([0, 0, 2, 0]), atol=1e-12, rtol=0))
        check("expansion_and_hidden_drift",
              np.all(np.diff(arrays["state_l2"][2], axis=-1) > 0)
              and arrays["pair_gain"][2, -1] > 1
              and arrays["state_l2"][3, 0, -1] > arrays["state_l2"][3, 0, 0]
              and np.array_equal(arrays["logits"][3], np.broadcast_to(arrays["logits"][3, :, :1], (2, 257, 4))))
        check("oscillation_and_nonnormal_transient",
              np.array_equal(states[4, :, 2:], states[4, :, :-2])
              and not np.array_equal(states[4, :, 1], states[4, :, 0])
              and arrays["spectral_radius"][5] < 1
              and arrays["operator_2_norm"][5] > 1
              and arrays["pair_gain"][5].max() > 1
              and arrays["pair_gain"][5, -1] < 1)
        require({p: sha(ROOT / p) for p in bindings} == bindings, "source changed")
        report.update(status="verified",
            interpretation="Finite constructed native dynamics with explicit initial-state intervention. Token2 likelihood and prediction are diagnostic readouts, not task accuracy. Local Jacobians and these trajectories imply no learned-model or global-ReLU stability guarantee.")
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        if "np" in locals() and not (args.output / "traces.npz").exists():
            try:
                for index, values in trace_buffers.items():
                    if values:
                        partial[f"observed_post_hook__{index}"] = torch.stack(values, dim=1).numpy()
                if partial:
                    path = args.output / "partial_traces.npz"
                    np.savez_compressed(path, **partial)
                    report["partial_evidence"] = {"path": path.name, "sha256": sha(path),
                        "arrays": sorted(partial), "scope": "Completed observations, including the first guarded-failure state if present; no uninitialized arrays."}
            except Exception as archive_error:
                report["partial_evidence_error"] = str(archive_error)
    report.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic() - started,
                  final_resource_observation=ops.resource_sample("item15_native_finish"))
    if report["final_resource_observation"]["available_ram_gib"] < 8:
        report.update(status="failed", final_ram_floor_violation=True)
    ops.save(args.output / "report.json", report, exclusive=True)
    print(json.dumps({"status": report["status"], "checks": len(report["checks"])}), flush=True)
    return int(report["status"] != "verified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
