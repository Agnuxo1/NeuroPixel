"""Independent saved-array audit of item12 isolation; no Torch or model replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "buffer:g_mask": ([35, 1], "bool"), "buffer:g_rgb": ([35, 3], "float32"),
    "parameter:embed.weight": ([35, 16], "float32"),
    "parameter:f1.bias": ([128], "float32"), "parameter:f1.weight": ([128, 160, 1, 1], "float32"),
    "parameter:f2.bias": ([48], "float32"), "parameter:f2.weight": ([48, 128, 1, 1], "float32"),
    "parameter:perceive.weight": ([96, 1, 3, 3], "float32"),
    "parameter:read.bias": ([16], "float32"), "parameter:read.weight": ([16, 48], "float32"),
    "parameter:seed.bias": ([48], "float32"), "parameter:seed.weight": ([48, 16, 1, 1], "float32")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(values):
    digest = hashlib.sha256()
    for name, a in sorted(values.items()):
        header = canonical({"name": name, "dtype": str(a.dtype), "shape": list(a.shape)})
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(a.tobytes(order="C"))
    return digest.hexdigest()


def execute(directory, plan_path, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checks, issues, identities = [], [], {}
    result = {"schema_version": 1, "item": 12, "status": "running",
              "scope": "Independent saved-array recount; no Torch, deserialization, training or model inference.",
              "checks": checks, "issues": issues, "input_sha256": identities}

    def check(label, passed):
        checks.append({"check": label, "passed": bool(passed)})
        if not passed:
            issues.append(label)
            raise ValueError(label)

    def artifact(descriptor, expected):
        check("path:" + expected, descriptor["path"] == expected)
        file = directory / expected
        check("safe_file:" + expected, file.is_file() and not file.is_symlink()
              and file.resolve().is_relative_to(directory.resolve()))
        digest = sha(file)
        check("identity:" + expected, digest == descriptor["sha256"]
              and file.stat().st_size == descriptor["bytes"])
        identities[expected] = digest
        with np.load(file, allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        check("finite:" + expected, all(np.all(np.isfinite(v)) for v in arrays.values()))
        return arrays

    def snapshot(descriptor, expected, k, aliased=False):
        arrays = artifact(descriptor, expected)
        check("tensor_maps:" + expected, set(descriptor["tensor_keys"]) == {str(i) for i in range(k)})
        inventories = descriptor["inventory"]
        check("inventory_slots:" + expected, inventories["slots"] == k
              and inventories["parameters_per_expert"] == [29824] * k)
        values, names = [], {"logits"}
        for i in range(k):
            mapping = descriptor["tensor_keys"][str(i)]
            check(f"complete_names:{expected}:{i}", set(mapping) == set(EXPECTED))
            current = {}
            for name, (shape, dtype) in EXPECTED.items():
                key = mapping[name]
                check(f"unique_tensor_key:{expected}:{i}:{name}", key not in names)
                names.add(key)
                a = arrays[key]
                check(f"shape_dtype:{expected}:{i}:{name}", list(a.shape) == shape and str(a.dtype) == dtype)
                check(f"array_descriptor:{expected}:{i}:{name}",
                      descriptor["arrays"][key] == {"shape": shape, "dtype": dtype})
                current[name] = a
            values.append(current)
        check("exact_arrays:" + expected, set(arrays) == names and set(descriptor["arrays"]) == names)
        check("logits_shape:" + expected, arrays["logits"].shape == (k, 12, 35)
              and arrays["logits"].dtype == np.float32
              and descriptor["arrays"]["logits"] == {"shape": [k, 12, 35], "dtype": "float32"})
        hashes = [fingerprint(v) for v in values]
        check("full_fingerprints:" + expected, hashes == inventories["expert_fingerprints"])
        check("bank_fingerprint:" + expected,
              hashlib.sha256(canonical(hashes)).hexdigest() == inventories["whole_bank_fingerprint"])
        rows = inventories["storage_aliases"]
        check("storage_row_count:" + expected, len(rows) == 12 * k
              and inventories["named_tensor_references"] == len(rows))
        seen, storage = set(), {}
        for row in rows:
            key = (row["slot"], row["name"])
            check("unique_storage_record:" + expected + ":" + str(key), key not in seen)
            seen.add(key)
            a = values[row["slot"]][row["name"]]
            check("storage_payload:" + expected + ":" + str(key),
                  row["tensor_payload_bytes"] == a.nbytes and row["storage_bytes"] == a.nbytes
                  and row["storage_offset"] == 0 and row["shape"] == list(a.shape)
                  and row["dtype"] == "torch." + str(a.dtype))
            sid = row["storage_id"]
            storage.setdefault(sid, []).append(row)
        check("storage_id_inventory:" + expected, set(storage) == set(range(inventories["unique_storages"])))
        check("alias_classes:" + expected,
              all(len(rows) == (k if aliased else 1) for rows in storage.values()))
        check("alias_same_tensor_name:" + expected,
              all(len({row["name"] for row in rows}) == 1 for rows in storage.values()))
        logical = sum(a.nbytes for v in values for a in v.values())
        unique = sum(rows[0]["storage_bytes"] for rows in storage.values())
        check("payload_sums:" + expected, logical == inventories["logical_tensor_payload_bytes"]
              and unique == inventories["unique_storage_bytes"]
              and logical == 119751 * k and unique == 119751 * (1 if aliased else k))
        return values, arrays["logits"]

    try:
        check("no_Torch", "torch" not in sys.modules)
        check("RAM_floor", psutil.virtual_memory().available / 2**30 >= 8)
        plan = json.loads(plan_path.read_bytes())
        check("frozen_plan", plan["item"] == 12 and plan["status"] == "frozen")
        bindings = {name: sha(ROOT / name) for name in plan["implementation_sha256"]}
        check("source_bindings", bindings == plan["implementation_sha256"])
        report_path = directory / "isolation_report.json"
        report = json.loads(report_path.read_bytes())
        identities["isolation_report.json"] = sha(report_path)
        status_path = directory / "isolation_status.json"
        status = json.loads(status_path.read_bytes())
        identities["isolation_status.json"] = sha(status_path)
        check("completed_producer", status["status"] == "completed" and report["status"] == "verified"
              and status["report_sha256"] == identities["isolation_report.json"])
        check("declared_fixture", report["seed"] == 120001
              and report["model"] == {"vocab": 35, "out_pos": [2, 2], "steps": 4, "c_id": 16,
                                      "c": 48, "hidden": 128, "fire_rate": .5})
        check("optimization_budget", report["optimization"] == {
            "optimizer": "SGD", "learning_rate": .01, "momentum": 0., "weight_decay": 0.,
            "model_training_flag": False, "positive_fixture_updates": 24, "alias_control_updates": 3})
        fixture = artifact(report["fixture"], "fixture.npz")
        expected_canvas = np.zeros((12, 3, 3), dtype=np.int64)
        targets = np.arange(5, 17, dtype=np.int64)
        expected_canvas[:, 0, 0] = targets
        expected_canvas[:, 2, 1] = 1
        check("literal_fixture", set(fixture) == {"canvas", "target"}
              and np.array_equal(fixture["canvas"], expected_canvas)
              and np.array_equal(fixture["target"], targets))
        check("eight_stages", len(report["stages"]) == 8)
        previous = None
        for k, stage in enumerate(report["stages"], 1):
            check(f"stage_recipe:{k}", stage["expert_count"] == k and stage["updates"] == 3
                  and stage["reported_scale"] == (k in (1, 2, 4, 8))
                  and len(stage["losses"]) == 3 and all(np.isfinite(stage["losses"])))
            before, old_logits = snapshot(stage["before"], f"stage{k}_before.npz", k)
            after, new_logits = snapshot(stage["after"], f"stage{k}_after.npz", k)
            if previous:
                prev, prev_logits = previous
                for i in range(k - 1):
                    check(f"cross_stage_preservation:{k}:{i}",
                          all(np.array_equal(before[i][name], prev[i][name]) for name in EXPECTED)
                          and np.array_equal(old_logits[i], prev_logits[i]))
                check(f"copied_newest_parent:{k}",
                      all(np.array_equal(before[-1][name], prev[-1][name]) for name in EXPECTED)
                      and np.array_equal(old_logits[-1], prev_logits[-1]))
            for i in range(k - 1):
                check(f"prior_tensors_and_output_unchanged:{k}:{i}",
                      all(np.array_equal(before[i][name], after[i][name]) for name in EXPECTED)
                      and np.array_equal(old_logits[i], new_logits[i]))
            check(f"actual_updated_expert_changed:{k}",
                  any(not np.array_equal(before[-1][name], after[-1][name]) for name in EXPECTED)
                  and not np.array_equal(old_logits[-1], new_logits[-1]))
            if k in (1, 2, 4, 8):
                check(f"reported_scale_inventory:{k}",
                      report["scale_inventories"][str(k)] == stage["after"]["inventory"])
            previous = after, new_logits
        alias = report["alias_negative_control"]
        before, old_logits = snapshot(alias["before"], "alias_before.npz", 2, aliased=True)
        after, new_logits = snapshot(alias["after"], "alias_after.npz", 2, aliased=True)
        check("alias_parent_preserved_at_clone", all(np.array_equal(before[0][name], previous[0][-1][name])
                                                     for name in EXPECTED))
        check("alias_control_changed_old_slot", fingerprint(before[0]) != fingerprint(after[0])
              and all(not np.array_equal(old_logits[i], new_logits[i]) for i in range(2)))
        check("alias_control_shared_values", fingerprint(after[0]) == fingerprint(after[1])
              and np.array_equal(new_logits[0], new_logits[1]))
        rng = artifact(report["rng"], "rng.npz")
        check("saved_RNG_unchanged", set(rng) == {"initial", "final"}
              and rng["initial"].dtype == rng["final"].dtype == np.uint8
              and np.array_equal(rng["initial"], rng["final"]))
        check("producer_checks_completed", len(report["checks"]) == 10
              and all(row["passed"] for row in report["checks"]) and status["check_count"] == 10)
        check("all_inputs_unchanged", all(sha(directory / name) == digest for name, digest in identities.items()))
        check("source_unchanged", {name: sha(ROOT / name) for name in bindings} == bindings)
        result.update(status="verified", scale_inventories=report["scale_inventories"],
                      producer_checks=10, no_learned_memory_claim=True,
                      limitations=["Saved tensor/output invariance is recounted; no optimizer trajectory or model replay is independently executed.",
                                   "Storage alias IDs are reported observations from the original live process, not independently observed pointers.",
                                   "Eight copied experts use one fixed twelve-example fixture, not eight learned tasks.",
                                   "Tensor payload and unique storage sizes exclude optimizer, gradients, allocator, Python and runtime overhead."])
    except BaseException as error:
        result.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    result.update(check_count=len(checks), failed_checks=sum(not c["passed"] for c in checks),
                  wall_seconds=time.monotonic() - started,
                  final_available_ram_gib=psutil.virtual_memory().available / 2**30)
    with (output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = execute(a.input, a.plan, a.output)
    print(json.dumps({"status": result["status"], "checks": result["check_count"]}))
    return int(result["status"] != "verified")


if __name__ == "__main__":
    raise SystemExit(main())
