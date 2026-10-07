"""Independent saved-array audit for item 12; no models, producer, or metric imports.

Recounts fixed item-6 outputs only. Two seeds are descriptive; there is no new
inference, confidence interval, FWT baseline, or claim of unseen final examples.
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
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD_SOURCE = "08d0d52edd05da6835e71479f3ba4399fcbeabae"
ROLES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
ATOL = 1e-10


class Audit:
    def __init__(self):
        self.checks, self.issues, self.result = 0, [], {}
    def check(self, condition, label):
        self.checks += 1
        if not bool(condition):
            self.issues.append(label)
        return bool(condition)
    def need(self, condition, label):
        if not self.check(condition, label):
            raise ValueError(label)
    def compare(self, actual, expected, label):
        if isinstance(expected, dict):
            self.need(isinstance(actual, dict), label + ": object required")
            for key, value in expected.items():
                self.need(key in actual, label + ": missing " + key)
                self.compare(actual[key], value, label + "." + key)
        elif isinstance(expected, list):
            self.need(isinstance(actual, list) and len(actual) == len(expected), label + ": length")
            for i, value in enumerate(expected):
                self.compare(actual[i], value, label + "[" + str(i) + "]")
        elif isinstance(expected, float):
            self.check(type(actual) in (int, float) and math.isfinite(actual)
                       and abs(actual - expected) <= ATOL, label + ": numerical mismatch")
        else:
            self.check(type(actual) is type(expected) and actual == expected, label + ": mismatch")


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def file_at(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or p.as_posix() != relative:
        raise ValueError("unsafe relative file: " + relative)
    path = root / p
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("missing or indirect file: " + relative)
    return path


def hashes(root, names):
    return {name: sha(file_at(root, name)) for name in names}


def ram():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) * 1024 / 2**30


def load_arrays(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(keys):
            raise ValueError("unexpected array inventory: " + str(path))
        return {key: archive[key].copy() for key in keys}


def visible_gold(canvas, topic, query):
    """Parse four adjacent visible pairs independently of generator metadata."""
    if canvas.shape != (8, 8) or canvas.dtype.kind not in "iu":
        raise ValueError("canvas geometry/type")
    if np.any((canvas < 0) | (canvas >= 35)) or np.count_nonzero(canvas) != 9:
        raise ValueError("canvas token inventory")
    if canvas[7, 6] != query + 1 or np.count_nonzero(canvas[7]) != 1:
        raise ValueError("query or output row")
    fillers, used_rows = {}, set()
    for marker in range(1, 5):
        hits = np.argwhere(canvas[:7] == marker)
        if hits.shape != (1, 2):
            raise ValueError("missing/duplicate visible role")
        row, col = map(int, hits[0])
        if col == 7 or row in used_rows:
            raise ValueError("pair geometry")
        used_rows.add(row)
        fillers[marker] = int(canvas[row, col + 1])
    if (not all(5 + 4 * topic <= fillers[r] < 9 + 4 * topic for r in (1, 3))
            or fillers[1] == fillers[3] or not 17 <= fillers[2] < 27
            or not 27 <= fillers[4] < 35):
        raise ValueError("filler category/topic")
    return fillers[query + 1], (fillers[1] - 5, fillers[2] - 17, fillers[3] - 5)


def summary(matrix):
    stages = []
    for end in range(3):
        old = []
        for task in range(end):
            current = matrix[end][task]
            old.append({"task": task, "at_acquisition": matrix[task][task],
                        "current": current, "backward_change": current - matrix[task][task],
                        "max_past_forgetting": max(matrix[i][task] for i in range(end)) - current,
                        "post_acquisition_forgetting": max(matrix[i][task] for i in range(task, end)) - current})
        row = {"after_task": end, "tasks_seen": end + 1,
               "average_seen_accuracy": sum(matrix[end][:end + 1]) / (end + 1),
               "per_old_task": old}
        for field, child in (("BWT", "backward_change"), ("max_past_forgetting", "max_past_forgetting"),
                             ("post_acquisition_forgetting", "post_acquisition_forgetting")):
            row[field] = sum(v[child] for v in old) / len(old) if old else None
        stages.append(row)
    return {"matrix": matrix, "acquisition_diagonal": [matrix[i][i] for i in range(3)],
            "stages": stages, "final": stages[-1], "FWT": None}


def count_row(correct, roles, topic, task):
    per = {}
    for role, name in enumerate(ROLES):
        mask = (topic == task) & (roles == role)
        n, hit = int(mask.sum()), int(correct[mask].sum())
        per[name] = {"n": n, "correct": hit, "accuracy": hit / n}
    mask = topic == task
    n, hit = int(mask.sum()), int(correct[mask].sum())
    return {"n": n, "correct": hit, "accuracy": hit / n,
            "binding": (per["AGENTE"]["accuracy"] + per["PACIENTE"]["accuracy"]) / 2, "per_role": per}


def run(a, raw, plan_path, produced):
    a.result["available_ram_gib_at_admission"] = ram()
    a.need(a.result["available_ram_gib_at_admission"] >= 8, "RAM admission below 8 GiB")
    a.need("torch" not in sys.modules, "unexpected Torch import")
    plan = read_json(plan_path)
    a.need(plan["item"] == 12 and plan["status"] == "frozen", "wrong plan status/item")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    a.need(os.environ.get("GITHUB_SHA") == head, "current source commit differs")
    source_before = hashes(ROOT, plan["implementation_sha256"])
    a.compare(source_before, plan["implementation_sha256"], "implementation")
    a.need(source_before == plan["implementation_sha256"], "bound source mismatch")
    a.result["source"] = {"source_commit": head, "plan_sha256": sha(plan_path),
                          "auditor_sha256": sha(Path(__file__)), "implementation_sha256": source_before}
    spec = plan["inputs"]["growth"]
    a.need(len(spec["files_sha256"]) == 15, "expected fifteen selected raw artifacts")
    before = hashes(raw, spec["files_sha256"])
    a.result["input_sha256"] = before
    a.need(before == spec["files_sha256"], "raw input hashes differ")
    a.result["archive_identity"] = {key: spec[key] for key in ("archive_commit", "path")}
    original = read_json(file_at(raw, "archive_manifest.json"))
    a.need(original["snapshot_kind"] == "final" and original["source_commit"] == OLD_SOURCE,
           "not the frozen final item6 archive")
    for name, digest in before.items():
        if name != "archive_manifest.json":
            entry = original["files"][name]
            a.need(entry["sha256"] == digest and entry["bytes"] == file_at(raw, name).stat().st_size,
                   "original manifest descriptor: " + name)
    records = read_json(file_at(raw, "growth/growth_records.json"))
    a.need(records["status"] == "completed" and records["complete_training_stages"] == 18
           and records["source"]["git_commit"] == OLD_SOURCE, "growth status/source")
    partitions = read_json(file_at(raw, "growth/partitions.json"))
    a.need(len(partitions) == 3 and partitions == records["partitions"], "partition records disagree")
    a.need(len(records["final_datasets"]) == 3, "final dataset count")
    ys, rs = [], []
    for task in range(3):
        part = partitions[task]
        a.need(part["topic"] == task and part["split_seed"] == 0, "partition topic/seed")
        pools = {key: {tuple(t) for t in part["triples"][key]} for key in ("train", "validation", "test")}
        universe = {(x, v, y) for x in range(4 * task, 4 * task + 4)
                    for v in range(10) for y in range(4 * task, 4 * task + 4) if x != y}
        a.need(all(len(pools[k]) == len(part["triples"][k]) == part["counts"][k] for k in pools)
               and set.union(*pools.values()) == universe
               and sum(map(len, pools.values())) == 120, "partition coverage/disjointness")
        descriptor = records["final_datasets"][task]
        name = "growth/" + descriptor["path"]
        a.need(name in before and before[name] == descriptor["sha256"]
               and file_at(raw, name).stat().st_size == descriptor["bytes"], "dataset identity")
        a.need(descriptor["n"] == 1024 and descriptor["topic"] == task and descriptor["split"] == "test"
               and descriptor["split_seed"] == 0 and descriptor["sampling_seed"] == 62012 + 100 * task,
               "dataset sampling recipe")
        data = load_arrays(file_at(raw, name), ("canvas", "target", "roles"))
        canvas, target, roles = (data[k] for k in ("canvas", "target", "roles"))
        a.need(canvas.shape == (1024, 8, 8) and target.shape == roles.shape == (1024,)
               and target.dtype.kind in "iu" and roles.dtype.kind in "iu"
               and all(np.count_nonzero(roles == r) == 256 for r in range(4)), "dataset shape/balance")
        for i in range(1024):
            gold, triple = visible_gold(canvas[i], task, int(roles[i]))
            a.need(gold == int(target[i]) and triple in pools["test"], "visible gold/test membership")
        ys.append(target); rs.append(roles)
    y, roles, topics = np.concatenate(ys), np.concatenate(rs), np.repeat(np.arange(3), 1024)
    files = {p.name for p in produced.iterdir() if p.is_file()}
    a.need(files == {"report.json", "retention_rows.csv", "router_fixture.npz",
                     "predictions_s20.npy", "predictions_s21.npy"}
           and all(p.is_file() for p in produced.iterdir()), "produced file inventory")
    produced_before = hashes(produced, sorted(files))
    a.result["produced_sha256"] = produced_before
    report = read_json(file_at(produced, "report.json"))
    a.need(report["status"] == "completed" and report["item"] == 12, "producer status/item")
    a.compare(report["source"], {"source_commit": head, "plan_sha256": sha(plan_path)}, "producer source")
    a.need(report["input_sha256"] == before and report["source_implementation_sha256"] == source_before
           and report["input_unchanged"] is True and report["original_source_commit"] == OLD_SOURCE,
           "producer source/input identities")
    payload_names = {"router_fixture.npz", "predictions_s20.npy", "predictions_s21.npy"}
    a.need(set(report["payloads"]) == payload_names, "producer payload inventory")
    for name in payload_names:
        a.compare(report["payloads"][name], {"sha256": produced_before[name],
                   "bytes": file_at(produced, name).stat().st_size}, "payload " + name)
    a.need([s["seed"] for s in report["seeds"]] == [20, 21], "seed inventory/order")
    recounted = []
    for result, seed in zip(report["seeds"], (20, 21)):
        a.need(ram() >= 8, "RAM boundary below 8 GiB")
        bank_id = "snapshots_i" + str(seed)
        banks = [b for b in records["banks"] if b["bank_id"] == bank_id]
        trajectories = [t for t in records["trajectories"]
                        if t["config"]["trajectory_id"] == "fixed_sequential_i" + str(seed)]
        a.need(len(banks) == len(trajectories) == 1, "unique bank/trajectory")
        bank, trajectory = banks[0], trajectories[0]
        a.need(bank["init_seed"] == seed and len(bank["experts"]) == len(trajectory["stages"]) == 3
               and trajectory["config"]["topics"] == [0, 1, 2]
               and trajectory["config"]["stage_updates"] == 512
               and trajectory["config"]["optimizer_reset_each_stage"] is True, "training chronology")
        for i, ref in enumerate(bank["experts"]):
            name = "growth/trajectories/fixed_sequential_i" + str(seed) + "/stage" + str(i) + "/weights.pt"
            a.need("growth/" + ref["path"] == name and name in before and before[name] == ref["sha256"]
                   and file_at(raw, name).stat().st_size == ref["bytes"]
                   and ref == trajectory["stages"][i]["checkpoint"], "checkpoint lineage/hash")
        desc = bank["final_expert_outputs"]
        name = "growth/banks/" + bank_id + "/final_experts.npz"
        a.need("growth/" + desc["path"] == name and before[name] == desc["sha256"]
               and file_at(raw, name).stat().st_size == desc["bytes"], "expert output descriptor")
        arr = load_arrays(file_at(raw, name), ("log_probabilities", "scanner", "target", "role", "topic"))
        logp, scanner = arr["log_probabilities"], arr["scanner"]
        a.need(logp.shape == (3, 3072, 35) and logp.dtype.kind == "f" and np.isfinite(logp).all()
               and scanner.shape == (3, 3072) and scanner.dtype.kind == "f" and np.isfinite(scanner).all()
               and np.all((scanner >= 0) & (scanner <= 1)), "saved output schema/finiteness")
        for key, expected in (("target", y), ("role", roles), ("topic", topics)):
            a.need(arr[key].dtype.kind in "iu" and np.array_equal(arr[key], expected), "ordered " + key)
        z = logp.astype(np.float64)
        peak = z.max(axis=2)
        error = np.abs(peak + np.log(np.exp(z - peak[:, :, None]).sum(axis=2)))
        a.need(float(error.max()) <= 1e-5, "log probability normalization")
        prediction = np.argmax(z, axis=2)
        stored = np.load(file_at(produced, "predictions_s" + str(seed) + ".npy"), allow_pickle=False)
        a.need(stored.dtype.kind in "iu" and np.array_equal(stored, prediction), "produced argmax predictions")
        rows, oracle_rows = [], []
        for i in range(3):
            correct = prediction[i] == y
            for task in range(3):
                row = count_row(correct, roles, topics, task)
                positions = np.flatnonzero(topics == task)
                row.update(checkpoint_after_task=i, evaluation_task=task,
                           cross_entropy=float(-z[i, positions, y[positions]].mean()))
                rows.append(row)
        for task in range(3):
            row = count_row(np.any(prediction == y[None, :], axis=0), roles, topics, task)
            row["task"] = task
            oracle_rows.append(row)
            old = bank["hard_selection_oracle"]["per_topic"][str(task)]
            a.compare(row["correct"], old["correct"], "prior hard oracle correct")
            a.compare(row["binding"], old["macro_agent_patient_accuracy"], "prior hard oracle binding")
        measures = {}
        for name in ("accuracy", "binding", *ROLES):
            matrix = [[(rows[3 * i + j][name] if name in ("accuracy", "binding")
                        else rows[3 * i + j]["per_role"][name]["accuracy"]) for j in range(3)] for i in range(3)]
            measures[name] = summary(matrix)
        expected = {"seed": seed, "bank_id": bank_id, "checkpoints": bank["experts"],
                    "rows": rows, "measures": measures, "hard_selection_oracle_by_topic": oracle_rows}
        a.compare(result, expected, bank_id)
        recounted.append({**expected, "maximum_log_normalization_error": float(error.max())})
    literal = np.array([[[.9, .1], [.1, .9]]] + [[[.1, .9], [.9, .1]]] * 7, dtype=np.float64)
    fixture = load_arrays(file_at(produced, "router_fixture.npz"), ("target", "probabilities"))
    a.need(fixture["target"].dtype.kind in "iu" and fixture["probabilities"].dtype.kind == "f"
           and np.array_equal(fixture["target"], np.array([0, 1]))
           and np.array_equal(fixture["probabilities"], literal), "literal routing fixture")
    routing = [{"experts": k, "n": 2, "old_hard_correct": 2,
                "last_hard_correct": 2 if k == 1 else 0, "uniform_correct": u}
               for k, u in zip((1, 2, 4, 8), (2, 1, 0, 0))]
    a.compare(report["synthetic_routing"], {"sizes": [1, 2, 4, 8], "rows": routing}, "synthetic routing")
    a.result.update(seeds=recounted, synthetic_routing={"rows": routing, "scope": "Literal counterexample only."})
    a.need(hashes(raw, before) == before and hashes(ROOT, source_before) == source_before
           and hashes(produced, produced_before) == produced_before, "files changed during audit")
    a.result["available_ram_gib_at_finish"] = ram()
    a.need(a.result["available_ram_gib_at_finish"] >= 8, "RAM finish boundary below 8 GiB")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "plan", "produced", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    a = Audit()
    a.result["auditor_sha256"] = sha(Path(__file__))
    a.result["plan_sha256"] = sha(args.plan) if args.plan.is_file() else None
    try:
        run(a, args.input, args.plan, args.produced)
    except Exception as error:
        a.issues.append(type(error).__name__ + ": " + str(error))
    report = {"schema_version": 1, "item": 12, "status": "verified" if not a.issues else "failed",
              "checks": a.checks, "issues": a.issues, "results": a.result,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "threads": 1},
              "metric_absolute_tolerance": ATOL, "normalization_absolute_tolerance": 1e-5,
              "limits": ["Independent saved-array recount; no models, task sampling or checkpoint deserialization.",
                         "Selected fifteen raw artifacts only; original full archive authenticity is separately audited.",
                         "Two observed seeds and previously accessed examples; no new CI, FWT or confirmatory inference.",
                         "Checkpoint byte identity and recorded lineage do not independently verify tensor cloning.",
                         "RAM readings are boundary samples, not a continuous resource minimum.",
                         "CSV bytes are preserved in the produced inventory; numerical comparisons use report.json."]}
    with (args.output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "checks": a.checks, "issues": len(a.issues)}))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
