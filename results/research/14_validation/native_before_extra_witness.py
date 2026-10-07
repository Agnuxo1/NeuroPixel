"""Item14 constructed native controls: repair, retention and reintroduced input.

All weights are assigned by construction. No training, checkpoint selection or
semantic benchmark is performed. Stored full traces support an independent
NumPy reconstruction of each fixed local rule and every reported transition.
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

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_fixture_model(kind):
    import torch
    from neuropixel.model import NeuroPixel
    require(kind in MODELS, "unknown constructed rule")
    torch.manual_seed(140001)
    m = NeuroPixel(8, (1, 1), c_id=8, c=8, hidden=16, steps=10, fire_rate=0.5).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.embed.weight.copy_(torch.eye(8, dtype=torch.float64))
        m.seed.weight[:, :, 0, 0].copy_(torch.eye(8, dtype=torch.float64))
        m.read.weight.copy_(torch.eye(8, dtype=torch.float64))
        if kind != "state_holder":
            for k in range(8):
                m.f1.weight[k, k, 0, 0] = 1
                m.f2.weight[k, k, 0, 0] = -1
                m.f2.weight[k, 8 + k, 0, 0] = 1
                if kind == "input_copier":
                    m.f1.weight[8 + k, 24 + k, 0, 0] = 1
                else:
                    m.perceive.weight[2 * k, 0].fill_(1 / 9)
                    m.f1.weight[8 + k, 8 + 2 * k, 0, 0] = 1
    return m

def execute(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "item": 14, "status": "running",
              "source_commit": os.environ.get("GITHUB_SHA"),
              "plan_sha256": digest(args.plan), "recipe": RECIPE,
              "started_at_utc": ops.now(), "checks": []}
    def check(name, ok, **extra):
        report["checks"].append({"name": name, "passed": bool(ok), **extra})
        require(ok, name)
    try:
        plan = json.loads(args.plan.read_bytes())
        require(plan["item"] == 14 and plan["status"] == "frozen"
                and plan["evidence_recipe"]["native"] == RECIPE, "frozen recipe differs")
        ops.admit(started + 120, minimum_seconds=10)
        require(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "thread environment differs")
        import numpy as np
        import torch
        from neuropixel.research.repair import trace_repair
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.version.cuda is None, "CPU-only fixture")
        bindings = plan["implementation_sha256"]
        require({p: digest(ROOT / p) for p in bindings} == bindings, "source bindings differ")
        cues = torch.arange(2, 8, dtype=torch.long)
        canvas = cues[:, None, None].expand(6, 3, 3).clone()
        shifted = torch.roll(cues, -1)
        future = torch.stack([canvas, torch.zeros_like(canvas),
                              shifted[:, None, None].expand(6, 3, 3).clone()])
        keep = torch.ones(4, 1, 3, 3, dtype=torch.bool)
        keep[1, 0, 1, 1] = False
        keep[2].zero_(); keep[2, 0, 1, 1] = True
        keep[3].zero_()
        arrays = {"cues": cues.numpy(), "canvas": canvas.numpy(),
                  "future": future.numpy(), "keep": keep.numpy(),
                  "prefix_states": np.empty((3, 6, 3, 8, 3, 3), dtype=np.float64),
                  "prefix_logits": np.empty((3, 6, 3, 8), dtype=np.float64),
                  "pre_damage": np.empty((3, 4, 3, 6, 8, 3, 3), dtype=np.float64),
                  "post_damage": np.empty((3, 4, 3, 6, 8, 3, 3), dtype=np.float64),
                  "recovery_states": np.empty((3, 4, 3, 6, 9, 8, 3, 3), dtype=np.float64),
                  "recovery_logits": np.empty((3, 4, 3, 6, 9, 8), dtype=np.float64)}
        native_equivalence = []
        models_unchanged = []
        with torch.no_grad():
            for mi, name in enumerate(MODELS):
                model = build_fixture_model(name)
                before = {k: v.detach().clone() for k, v in list(model.named_parameters()) + list(model.named_buffers())}
                for key, value in before.items():
                    arrays[f"weight__{mi}__{key}"] = value.numpy().copy()
                for li, lesion in enumerate(LESIONS):
                    mask = keep[li:li+1].expand(6, 1, 3, 3).clone()
                    for si, source in enumerate(SOURCES):
                        ops.remaining(started + 120)
                        out = trace_repair(model, canvas, warmup_steps=2, recovery_steps=8,
                                           keep_mask=mask, recovery_canvas=future[si])
                        arrays["pre_damage"][mi, li, si] = out["pre_damage"].numpy()
                        arrays["post_damage"][mi, li, si] = out["post_damage"].numpy()
                        arrays["recovery_states"][mi, li, si] = out["recovery_states"].numpy()
                        arrays["recovery_logits"][mi, li, si] = out["logits"].numpy()
                        if li == 0 and si == 0:
                            arrays["prefix_states"][mi] = out["prefix_states"].numpy()
                            arrays["prefix_logits"][mi] = out["prefix_logits"].numpy()
                        else:
                            require(np.array_equal(arrays["prefix_states"][mi], out["prefix_states"].numpy()),
                                    "prefix unexpectedly depends on future intervention")
                        if si == 0:
                            def hook(t, s, mask=mask):
                                return s * mask if t == 2 else s
                            actual = model(canvas, steps=10, trace=True, hook=hook)
                            err = float((actual["frames"][:, 2:] - out["recovery_states"]).abs().max())
                            le = float((actual["logits"] - out["logits"][:, -1]).abs().max())
                            native_equivalence.append({"model": name, "lesion": lesion,
                                                       "maximum_state_error": err, "maximum_logit_error": le})
                            require(err == 0 and le == 0, "held-source path differs from native forward")
                after = dict(list(model.named_parameters()) + list(model.named_buffers()))
                same = all(torch.equal(v, after[k]) for k, v in before.items())
                models_unchanged.append({"model": name, "unchanged": same})
                require(same, "fixture model mutated")
        logits = arrays["recovery_logits"]
        pred = logits.argmax(-1)
        gold = arrays["cues"][None, None, None, :, None]
        correct = pred == gold
        shifted_np = np.roll(arrays["cues"], -1)
        arrays["predictions"] = pred.astype(np.int64)
        arrays["correct"] = correct
        maximum = logits.max(axis=-1)
        arrays["nll"] = maximum + np.log(np.exp(logits - maximum[..., None]).sum(axis=-1)) - np.take_along_axis(
            logits, np.broadcast_to(gold[..., None], (*logits.shape[:-1], 1)), axis=-1)[..., 0]
        arrays["case_index"] = np.asarray([[mi, li, si, ci] for mi in range(3)
                          for li in range(4) for si in range(3) for ci in range(6)], dtype=np.int64)
        rows = []
        for mi, name in enumerate(MODELS):
            pre = arrays["prefix_logits"][mi, :, -1].argmax(-1) == arrays["cues"]
            for li, lesion in enumerate(LESIONS):
                for si, source in enumerate(SOURCES):
                    post = correct[mi, li, si, :, 0]
                    lost = pre & ~post
                    for t in range(9):
                        late = correct[mi, li, si, :, t]
                        sham = correct[mi, 0, si, :, t]
                        delta = arrays["recovery_states"][mi, li, si, :, t] - arrays["recovery_states"][mi, 0, si, :, t]
                        rows.append({"model": name, "lesion": lesion, "source": source, "recovery_step": t,
                                     "n": 6, "pre_correct": int(pre.sum()), "immediate_correct": int(post.sum()),
                                     "late_correct": int(late.sum()), "matched_sham_correct": int(sham.sum()),
                                     "initially_correct_then_lost": int(lost.sum()),
                                     "lost_then_recovered": int((lost & late).sum()),
                                     "survived_and_still_correct": int((pre & post & late).sum()),
                                     "late_matches_changed_cue": int((pred[mi, li, si, :, t] == shifted_np).sum()),
                                     "mean_nll_original_cue": float(arrays["nll"][mi, li, si, :, t].mean()),
                                     "rms_state_difference_from_time_matched_sham": float(np.sqrt(np.mean(delta * delta)))})
        report["native_equivalence"] = native_equivalence
        report["models_unchanged"] = models_unchanged
        report["rows"] = rows
        check("complete_inventory", len(rows) == 324 and arrays["case_index"].shape == (216, 4),
              trajectories=216, summary_rows=324, endpoints=1944)
        check("held_source_exact_native_equivalence", len(native_equivalence) == 12
              and all(x["maximum_state_error"] == 0 and x["maximum_logit_error"] == 0 for x in native_equivalence))
        check("source_copier_full_erasure_rebuilds", bool(correct[0, 3, 0, :, 1:].all())
              and not bool(correct[0, 3, 0, :, 0].any()))
        check("source_copier_reconstruction_is_source_dependent", not bool(correct[0, 3, 1].any())
              and bool((pred[0, 3, 2, :, 1:] == shifted_np[:, None]).all()))
        check("state_holder_survives_but_cannot_repair_center", bool(correct[1, 0, 1].all())
              and not bool(correct[1, 1, 1].any()))
        check("redundant_spatial_state_restores_categorical_readout", not bool(correct[2, 1, 1, :, 0].any())
              and bool(correct[2, 1, 1, :, 1:].all()))
        same_after_erasure = all(np.array_equal(logits[mi, 3, 1, 0], logits[mi, 3, 1, ci])
                                 for mi in range(3) for ci in range(6))
        check("total_erasure_with_common_future_is_cue_indistinguishable", same_after_erasure,
              accuracy_on_balanced_cue_set=int(correct[:, 3, 1].sum()), chance_upper_bound_for_identical_predictor=1/6)
        check("all_states_finite_and_models_unchanged", all(np.isfinite(v).all() for v in arrays.values())
              and all(x["unchanged"] for x in models_unchanged))
        require({p: digest(ROOT / p) for p in bindings} == bindings, "source changed")
        path = args.output / "traces.npz"
        np.savez_compressed(path, **arrays)
        report.update(status="verified", array_file="traces.npz", array_sha256=digest(path),
                      arrays={k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
                      optimization_steps=0, checkpoints_loaded=0,
                      interpretation="Constructed positive/negative mechanism controls, not learned task performance, long-time stability or external replication.")
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    report.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic()-started,
                  final_resource_observation=ops.resource_sample("item14_native_finish"))
    if report["final_resource_observation"]["available_ram_gib"] < 8:
        report.update(status="failed", final_ram_floor_violation=True)
    ops.save(args.output / "report.json", report, exclusive=True)
    print(json.dumps({"status": report["status"], "checks": len(report["checks"])}), flush=True)
    return int(report["status"] != "verified")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return execute(p.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
