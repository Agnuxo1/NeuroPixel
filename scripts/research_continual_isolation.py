"""Item-12 native-copy storage isolation diagnostic, not a retention benchmark.

Fresh V35 experts receive three deterministic fixture SGD updates each. Existing
experts must retain every named parameter, buffer and evaluation logit. A
separately labelled shared-reference control must be detected as non-isolated.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 120001
MODEL = {"vocab": 35, "out_pos": (2, 2), "steps": 4, "c_id": 16,
         "c": 48, "hidden": 128, "fire_rate": 0.5}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def named_tensors(model):
    # state_dict alone omits the current nonpersistent grounding buffers.
    return {**{"parameter:" + k: v for k, v in model.named_parameters()},
            **{"buffer:" + k: v for k, v in model.named_buffers()}}


def fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(named_tensors(model).items()):
        array = value.detach().cpu().contiguous().numpy()
        header = canonical({"name": name, "dtype": str(array.dtype), "shape": list(array.shape)})
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def bank_inventory(models):
    storages, rows, payload = {}, [], 0
    for slot, model in enumerate(models):
        for name, tensor in sorted(named_tensors(model).items()):
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr())
            if identity not in storages:
                storages[identity] = {"storage_id": len(storages), "bytes": storage.nbytes()}
            value = storages[identity]
            size = tensor.numel() * tensor.element_size()
            payload += size
            rows.append({"slot": slot, "name": name, "storage_id": value["storage_id"],
                         "tensor_payload_bytes": size, "storage_bytes": value["bytes"],
                         "storage_offset": tensor.storage_offset(), "shape": list(tensor.shape),
                         "dtype": str(tensor.dtype)})
    fingerprints = [fingerprint(m) for m in models]
    return {"slots": len(models), "parameters_per_expert": [sum(p.numel() for p in m.parameters()) for m in models],
            "expert_fingerprints": fingerprints,
            "whole_bank_fingerprint": hashlib.sha256(canonical(fingerprints)).hexdigest(),
            "unique_storages": len(storages), "named_tensor_references": len(rows),
            "logical_tensor_payload_bytes": payload,
            "unique_storage_bytes": sum(v["bytes"] for v in storages.values()),
            "storage_aliases": rows}


def make_model():
    from neuropixel.model import NeuroPixel
    return NeuroPixel(**MODEL).cpu().eval()


def fixture(torch):
    canvas = torch.zeros(12, 3, 3, dtype=torch.long)
    target = torch.arange(5, 17, dtype=torch.long)
    canvas[:, 0, 0] = target
    canvas[:, 2, 1] = 1
    return canvas, target


def update_fixture(model, canvas, target, torch, updates=3):
    model.eval()  # Gradient-enabled, but stochastic firing is disabled.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    losses = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        logits = model(canvas)["logits"]
        loss = torch.nn.functional.cross_entropy(logits, target)
        require(bool(torch.isfinite(loss)), "nonfinite fixture loss")
        loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                "nonfinite fixture gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def bank_logits(models, canvas, torch):
    with torch.inference_mode():
        for model in models:
            require(not model.training, "fixture evaluation must use model.eval")
        return torch.stack([model(canvas)["logits"].detach().clone() for model in models])


def snapshot(directory, name, models, canvas, torch, np):
    inventory = bank_inventory(models)
    arrays, tensor_keys = {}, {}
    for slot, model in enumerate(models):
        tensor_keys[str(slot)] = {}
        for index, (key, value) in enumerate(sorted(named_tensors(model).items())):
            array_key = f"expert{slot}_tensor{index}"
            arrays[array_key] = value.detach().cpu().numpy().copy()
            tensor_keys[str(slot)][key] = array_key
    logits = bank_logits(models, canvas, torch)
    arrays["logits"] = logits.cpu().numpy().copy()
    require(all(bool(np.isfinite(v).all()) for v in arrays.values()), "nonfinite snapshot")
    path = directory / (name + ".npz")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path),
            "inventory": inventory, "tensor_keys": tensor_keys,
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}}, logits


def run_probe(directory, torch, np):
    import copy
    torch.manual_seed(SEED)
    canvas, target = fixture(torch)
    with (directory / "fixture.npz").open("xb") as stream:
        np.savez_compressed(stream, canvas=canvas.numpy(), target=target.numpy())
        stream.flush()
        os.fsync(stream.fileno())
    models, stages, checks = [], [], []
    first = make_model()
    rng = torch.get_rng_state().clone()
    for count in range(1, 9):
        # Native growth's create branch: copy the previous complete expert.
        models.append(first if count == 1 else copy.deepcopy(models[-1]).eval())
        before, before_logits = snapshot(directory, f"stage{count}_before", models, canvas, torch, np)
        losses = update_fixture(models[-1], canvas, target, torch, updates=3)
        after, after_logits = snapshot(directory, f"stage{count}_after", models, canvas, torch, np)
        old_tensors = before["inventory"]["expert_fingerprints"][:-1] == after["inventory"]["expert_fingerprints"][:-1]
        old_outputs = torch.equal(before_logits[:-1], after_logits[:-1])
        target_changed = before["inventory"]["expert_fingerprints"][-1] != after["inventory"]["expert_fingerprints"][-1]
        logits_changed = not torch.equal(before_logits[-1], after_logits[-1])
        aliases = after["inventory"]["storage_aliases"]
        disjoint = all(len({r["slot"] for r in aliases if r["storage_id"] == sid}) == 1
                       for sid in range(after["inventory"]["unique_storages"]))
        passed = old_tensors and old_outputs and target_changed and logits_changed and disjoint
        checks.append({"name": f"stage{count}_isolation", "passed": passed,
                       "prior_expert_count": count - 1, "all_prior_full_tensors_identical": old_tensors, "all_prior_logits_identical": old_outputs,
                       "updated_expert_full_tensors_changed": target_changed,
                       "updated_expert_logits_changed": logits_changed, "cross_expert_storage_disjoint": disjoint})
        stages.append({"expert_count": count, "reported_scale": count in (1, 2, 4, 8),
                       "updates": 3, "losses": losses, "before": before, "after": after})
    alias = copy.deepcopy(models[-1]).eval()
    shared = [alias] * 2
    before, old_logits = snapshot(directory, "alias_before", shared, canvas, torch, np)
    losses = update_fixture(shared[1], canvas, target, torch, updates=3)
    after, new_logits = snapshot(directory, "alias_after", shared, canvas, torch, np)
    detected = before["inventory"]["expert_fingerprints"][0] != after["inventory"]["expert_fingerprints"][0]
    both_changed = all(not torch.equal(old_logits[i], new_logits[i]) for i in range(2))
    coherent = (after["inventory"]["expert_fingerprints"][0] == after["inventory"]["expert_fingerprints"][1]
                and torch.equal(new_logits[0], new_logits[1]))
    checks.append({"name": "shared_reference_negative_control_detected",
                   "passed": detected and both_changed and coherent,
                   "old_slot_tensor_invariance_violated": detected,
                   "both_slot_logits_changed": both_changed, "same_object_outputs_remain_identical": coherent})
    rng_final = torch.get_rng_state().clone()
    with (directory / "rng.npz").open("xb") as stream:
        np.savez_compressed(stream, initial=rng.cpu().numpy(), final=rng_final.cpu().numpy())
        stream.flush()
        os.fsync(stream.fileno())
    checks.append({"name": "fixture_updates_do_not_consume_firing_rng",
                   "passed": torch.equal(rng, rng_final)})
    report = {"schema_version": 1, "item": 12, "status": "verified" if all(c["passed"] for c in checks) else "failed",
              "created_at_utc": now(), "seed": SEED, "model": MODEL,
              "optimization": {"optimizer": "SGD", "learning_rate": 0.01, "momentum": 0.0,
                               "weight_decay": 0.0, "model_training_flag": False,
                               "positive_fixture_updates": 24, "alias_control_updates": 3},
              "fixture": {"path": "fixture.npz", "bytes": (directory / "fixture.npz").stat().st_size,
                          "sha256": sha(directory / "fixture.npz"), "n": 12,
                          "scope": "Fixed input/label fixture, not an unseen population or competency assay."},
              "rng": {"path": "rng.npz", "bytes": (directory / "rng.npz").stat().st_size,
                      "sha256": sha(directory / "rng.npz")},
              "stages": stages, "scale_inventories": {str(s["expert_count"]): s["after"]["inventory"]
                                                     for s in stages if s["reported_scale"]},
              "alias_negative_control": {"before": before, "after": after, "losses": losses},
              "checks": checks,
              "limits": [
                  "Fixture SGD establishes actual mutation and storage isolation, not useful acquisition or retention.",
                  "Only the native deepcopy/create branch is tested; updating an existing expert can change its old behavior.",
                  "Evaluation mode removes stochastic firing; these outputs are conditional on fixed fixture/configuration.",
                  "Complete old-expert tensors and outputs do not guarantee unchanged end-to-end router or mixture outputs.",
                  "Storage IDs are process-local alias classes; payload bytes exclude optimizer/autograd/allocator overhead.",
                  "The diagnostic has no task discovery, grounding mutation, new-token, repair, stability or scanner claim."
              ]}
    save_json(directory / "isolation_report.json", report)
    require(report["status"] == "verified", "a preserved isolation check failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    status = {"schema_version": 1, "item": 12, "started_at_utc": now(), "status": "running"}
    try:
        import psutil
        require(psutil.virtual_memory().available / 1024**3 >= 8, "less than 8 GiB available RAM")
        import torch
        import numpy as np
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        require(torch.version.cuda is None and torch.get_num_threads() == torch.get_num_interop_threads() == 1,
                "the diagnostic requires CPU Torch and one numerical/interop thread")
        status["runtime"] = {"torch": torch.__version__, "numpy": np.__version__,
                             "threads": 1, "interop_threads": 1, "cuda_version": torch.version.cuda}
        report = run_probe(directory, torch, np)
        status.update(status="completed", report_sha256=sha(directory / "isolation_report.json"),
                      check_count=len(report["checks"]))
    except Exception as error:
        status.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    status["completed_at_utc"] = now()
    save_json(directory / "isolation_status.json", status)
    print(json.dumps({"status": status["status"]}), flush=True)
    return int(status["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
