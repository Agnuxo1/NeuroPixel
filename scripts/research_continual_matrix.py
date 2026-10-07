"""Secondary analysis of frozen item-6 expert outputs; never retrain or replay."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.research.continual_metrics import score_matrices

ROLE_NAMES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def bound_file(root, name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == name,
            "unsafe input path")
    file = root / p
    require(file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(root.resolve()),
            "missing or indirect input")
    return file


def inputs(root, spec):
    return {name: sha(bound_file(root, name)) for name in spec["files_sha256"]}


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as source:
        require(set(source.files) == set(keys), "unexpected NPZ array inventory")
        return {key: source[key].copy() for key in keys}


def validate_dataset(root, descriptor, topic, partitions):
    path = "growth/" + descriptor["path"]
    file = bound_file(root, path)
    require(sha(file) == descriptor["sha256"] and file.stat().st_size == descriptor["bytes"],
            "dataset descriptor identity differs")
    require(descriptor["topic"] == topic and descriptor["n"] == 1024
            and descriptor["split"] == "test" and descriptor["split_seed"] == 0
            and descriptor["sampling_seed"] == 62012 + 100 * topic, "dataset recipe differs")
    data = load_npz(file, ("canvas", "target", "roles"))
    canvas, y, roles = (data[name] for name in ("canvas", "target", "roles"))
    require(canvas.shape == (1024, 8, 8) and canvas.dtype.kind in "iu"
            and np.all((canvas >= 0) & (canvas < 35)), "canvas contract differs")
    require(y.shape == roles.shape == (1024,) and y.dtype.kind in "iu" and roles.dtype.kind in "iu",
            "label arrays differ")
    query = canvas[:, 7, 6] - 1
    require(np.array_equal(roles, query) and np.all(canvas[:, 7, 7] == 0)
            and np.array_equal(np.bincount(roles, minlength=4), np.full(4, 256)),
            "query/role balance differs")
    require(np.all(np.count_nonzero(canvas, axis=(1, 2)) == 9), "occupied-cell census differs")
    fillers = np.empty((1024, 4), dtype=np.int64)
    pair_rows = np.empty((1024, 4), dtype=np.int64)
    for role in range(4):
        hits = canvas[:, :7, :] == role + 1
        require(np.all(hits.sum(axis=(1, 2)) == 1), "role marker inventory differs")
        batch, row, column = np.nonzero(hits)
        require(np.all(column < 7), "filler missing")
        fillers[batch, role] = canvas[batch, row, column + 1]
        pair_rows[batch, role] = row
    require(all(len(set(row)) == 4 for row in pair_rows), "role pairs share a row")
    require(np.all((fillers[:, (0, 2)] >= 5 + 4 * topic) & (fillers[:, (0, 2)] < 9 + 4 * topic))
            and np.all(fillers[:, 0] != fillers[:, 2])
            and np.all((fillers[:, 1] >= 17) & (fillers[:, 1] < 27))
            and np.all((fillers[:, 3] >= 27) & (fillers[:, 3] < 35)), "filler domain differs")
    require(np.array_equal(y, fillers[np.arange(1024), query]), "target not reconstructed from visible role pair")
    triples = np.stack((fillers[:, 0] - 5, fillers[:, 1] - 17, fillers[:, 2] - 5), axis=1)
    pool = {tuple(row) for row in partitions[topic]["triples"]["test"]}
    require(all(tuple(row) in pool for row in triples), "example outside frozen TEST triples")
    return data


def run(input_root, plan_path, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(plan_path.read_bytes())
    require(plan["item"] == 12 and plan["status"] == "frozen", "not frozen item12")
    require(psutil.virtual_memory().available / 2**30 >= 8, "available RAM below floor")
    require("torch" not in sys.modules, "secondary analysis must not import Torch")
    spec = plan["inputs"]["growth"]
    before = inputs(input_root, spec)
    require(before == spec["files_sha256"], "pinned input identities differ")
    source_before = {name: sha(ROOT / name) for name in plan["implementation_sha256"]}
    require(source_before == plan["implementation_sha256"], "bound source differs")
    manifest = json.loads((input_root / "archive_manifest.json").read_bytes())
    require(manifest["snapshot_kind"] == "final"
            and manifest["source_commit"] == "08d0d52edd05da6835e71479f3ba4399fcbeabae",
            "wrong original final archive")
    for name, digest in before.items():
        if name != "archive_manifest.json":
            require(manifest["files"][name]["sha256"] == digest
                    and manifest["files"][name]["bytes"] == (input_root / name).stat().st_size,
                    "original manifest mismatch: " + name)
    records = json.loads((input_root / "growth/growth_records.json").read_bytes())
    require(records["status"] == "completed" and records["complete_training_stages"] == 18
            and records["source"]["git_commit"] == manifest["source_commit"], "growth source/status differs")
    partitions_obj = json.loads((input_root / "growth/partitions.json").read_bytes())
    partitions = partitions_obj["topics"] if isinstance(partitions_obj, dict) else partitions_obj
    datasets = [validate_dataset(input_root, record, topic, partitions)
                for topic, record in enumerate(records["final_datasets"])]
    require(len(datasets) == 3, "three final datasets required")
    target = np.concatenate([d["target"] for d in datasets])
    roles = np.concatenate([d["roles"] for d in datasets])
    topics = np.repeat(np.arange(3, dtype=np.int64), 1024)
    results = []
    for seed in (20, 21):
        bank_id = f"snapshots_i{seed}"
        bank = next(row for row in records["banks"] if row["bank_id"] == bank_id)
        require(bank["init_seed"] == seed and len(bank["experts"]) == 3, "snapshot bank differs")
        trajectory = next(t for t in records["trajectories"] if t["config"]["trajectory_id"] == f"fixed_sequential_i{seed}")
        require(trajectory["config"]["topics"] == [0, 1, 2]
                and trajectory["config"]["stage_updates"] == 512
                and trajectory["config"]["optimizer_reset_each_stage"] is True, "training history differs")
        for i, reference in enumerate(bank["experts"]):
            require(reference == trajectory["stages"][i]["checkpoint"], "snapshot chronology differs")
            require(reference["path"] == f"trajectories/fixed_sequential_i{seed}/stage{i}/weights.pt"
                    and sha(input_root / "growth" / reference["path"]) == reference["sha256"],
                    "checkpoint identity differs")
            require(trajectory["stages"][i]["decision"] == {
                "action": "create", "parent": None if i == 0 else i - 1,
                "scores_used": [], "target_index": i, "threshold": None}, "copy lineage differs")
        descriptor = bank["final_expert_outputs"]
        file = input_root / "growth" / descriptor["path"]
        require(descriptor["path"] == f"banks/{bank_id}/final_experts.npz"
                and sha(file) == descriptor["sha256"] and file.stat().st_size == descriptor["bytes"],
                "saved output identity differs")
        arrays = load_npz(file, ("log_probabilities", "scanner", "target", "role", "topic"))
        require(arrays["log_probabilities"].shape == (3, 3072, 35)
                and arrays["scanner"].shape == (3, 3072)
                and np.all(np.isfinite(arrays["scanner"]))
                and np.all((arrays["scanner"] >= 0) & (arrays["scanner"] <= 1)), "output shapes differ")
        for name, expected in (("target", target), ("role", roles), ("topic", topics)):
            require(np.array_equal(arrays[name], expected), "saved output labels/order differ: " + name)
        scored = score_matrices(arrays["log_probabilities"], target, roles, topics)
        predictions = scored.pop("prediction")
        np.save(output / f"predictions_s{seed}.npy", predictions, allow_pickle=False)
        for topic, row in enumerate(scored["hard_selection_oracle_by_topic"]):
            prior = bank["hard_selection_oracle"]["per_topic"][str(topic)]
            require(row["correct"] == prior["correct"]
                    and abs(row["binding"] - prior["macro_agent_patient_accuracy"]) < 1e-12,
                    "recounted prior hard-selector bound differs")
        results.append({"seed": seed, "bank_id": bank_id, "checkpoints": bank["experts"], **scored})
    probabilities = np.repeat(np.array([[[.1, .9], [.9, .1]]], dtype=np.float64), 8, axis=0)
    probabilities[0] = [[.9, .1], [.1, .9]]
    literal_target = np.array([0, 1], dtype=np.int64)
    np.savez_compressed(output / "router_fixture.npz", target=literal_target, probabilities=probabilities)
    router_rows = []
    for k in (1, 2, 4, 8):
        p = probabilities[:k]
        router_rows.append({"experts": k, "n": 2,
                            "old_hard_correct": int((p[0].argmax(-1) == literal_target).sum()),
                            "last_hard_correct": int((p[-1].argmax(-1) == literal_target).sum()),
                            "uniform_correct": int((p.mean(0).argmax(-1) == literal_target).sum())})
    after = inputs(input_root, spec)
    require(before == after, "input changed during secondary analysis")
    require({name: sha(ROOT / name) for name in source_before} == source_before, "source changed")
    payloads = {f.name: {"bytes": f.stat().st_size, "sha256": sha(f)}
                for f in output.iterdir() if f.is_file()}
    report = {"schema_version": 1, "item": 12, "status": "completed",
              "phase": "secondary_reanalysis",
              "scope": "Retrospective new summaries of existing item6 predictions; no new checkpoint or model inference.",
              "known_prior_observations": True,
              "source": {"source_commit": os.environ.get("GITHUB_SHA"), "plan_sha256": sha(plan_path)},
              "original_source_commit": manifest["source_commit"],
              "input_sha256": before, "input_unchanged": True,
              "source_implementation_sha256": source_before, "seeds": results,
              "synthetic_routing": {"sizes": [1, 2, 4, 8], "rows": router_rows,
                  "scope": "Literal two-example output counterexample, not trained NeuroPixel predictions.",
                  "argmax_tie": "lowest class index", "all_expert_outputs_fixed": True},
              "payloads": payloads, "wall_seconds": time.monotonic() - started,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "threads": 1, "torch_imported": False,
                          "available_ram_gib": psutil.virtual_memory().available / 2**30},
              "limitations": [
                  "Only seeds20/21, split0, three noun-disjoint topics, and already accessed final examples.",
                  "Newest copied checkpoint sequence is measured; no unrecorded intermediate router trajectory is invented.",
                  "No new confirmatory interval, no seed/content replication from metric rows, and no FWT without initial baseline.",
                  "Low binding acquisition cannot support a claim of useful retention merely from small forgetting.",
                  "Checkpoint bytes are checked without deserialization; original full archive audit remains separately cited.",
                  "Synthetic routing outcomes are a constructed counterexample, not a measured learned model result."]}
    save(output / "report.json", report)
    with (output / "retention_rows.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "checkpoint_after_task", "evaluation_task",
            "n", "correct", "accuracy", "binding", "cross_entropy"))
        writer.writeheader()
        for record in results:
            for row in record["rows"]:
                writer.writerow({"seed": record["seed"], **{k: row[k] for k in writer.fieldnames if k != "seed"}})
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        report = run(args.input, args.plan, args.output)
        print(json.dumps({"status": report["status"], "seeds": len(report["seeds"]),
                          "rows": sum(len(s["rows"]) for s in report["seeds"])}))
    except BaseException as error:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            save(args.output / "failure.json", {"status": "failed", "type": type(error).__name__,
                                                "message": str(error)})
        raise


if __name__ == "__main__":
    main()
