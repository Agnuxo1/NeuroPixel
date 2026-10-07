"""Audit eligible triple pools and their overlaps, without asserting sample exposure.

The CPU Torch permutation is an explicitly recorded input, never approximated by
Python or NumPy shuffling. Universe construction, partition slicing, filtering,
disjointness, union coverage and overlap counts below use the standard library.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2, 101, 202)
HISTORICAL_SEEDS = (0, 1, 2)
PROSPECTIVE_SEEDS = (0, 101, 202)
SOURCES = {
    "neuropixel/task.py": "e088d35937040f9924264b227e70e0ba9aaf23ede1b1c6bc9f8d6c78821a156f",
    "neuropixel/research/data.py": "44cdf2b9ff492fe9eab115c8092c61c6d38bda5388babaf148380b89f2a7f94b",
    "neuropixel/research/growth_ablation.py": "25f3880569eb10168bb0979cfc1d400c65e1615b1e31f23ab37446df2e47aaa5",
}
ARCHIVED_GROWTH_SHA256 = "eff760e91613cd88a75f05231e11fe811674f4ef76d5469a412b9e5ea26f629f"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def admission():
    rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(rows["MemAvailable"].split()[0]) * 1024 / 2**30
    require(available >= 8, f"available RAM {available:.9f} GiB is below 8 GiB; audit stopped")
    return {"at_utc": now(), "available_ram_gib": available,
            "provider": "/proc/meminfo:MemAvailable", "active_threads": 1}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    return json.loads(path.read_bytes(), parse_constant=reject)


def vocabulary_from_ast(path):
    """Read public literal vocabulary definitions without importing task/Torch."""
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ("NOUNS", "VERBS", "PLACES", "ROLES"):
                values[key.lower()] = ast.literal_eval(node.value)
    require({key: len(value) for key, value in values.items()} == {"nouns": 12, "verbs": 10, "places": 8, "roles": 4},
            "public vocabulary dimensions differ")
    return values


def write_new(path, record):
    content = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    require(path.read_bytes() == content, "output readback differs")


def audit(export_path, archived_path, source_root):
    resources = [admission()]
    started = now()
    input_hashes = {"permutation_export": sha256(export_path), "archived_growth_partitions": sha256(archived_path)}
    actual_source = {name: sha256(source_root / name) for name in SOURCES}
    require(actual_source == SOURCES, "historical/research/growth source differs from the frozen definitions")
    require(input_hashes["archived_growth_partitions"] == ARCHIVED_GROWTH_SHA256, "archived item-6 growth partition hash differs")
    exported, archived = read_json(export_path), read_json(archived_path)
    require(exported["schema_version"] == 1 and exported["item"] == 7 and exported["status"] == "membership_exported",
            "membership export is not a completed item-7 input")
    require(exported["source"]["sha256"] == SOURCES, "exported source hashes differ")
    require(exported["exporter_sha256"] == sha256(source_root / "scripts/research_export_split_pools.py"),
            "membership exporter source hash differs")
    runtime = exported["runtime"]
    require(runtime["torch"] == "2.6.0+cpu" and runtime["device"] == "cpu"
            and runtime["torch_threads"] == runtime["torch_interop_threads"] == 1, "export runtime differs")
    require(exported["task_sample_called"] is False and exported["models_imported"] is False
            and exported["optimizer_updates"] == 0, "membership-only export boundary differs")
    require(exported["resource_samples"] and all(row["available_ram_gib"] >= 8 for row in exported["resource_samples"]),
            "exported RAM admission failed")
    require(exported["generator_definition"]["function"] == "torch.randperm"
            and exported["generator_definition"]["n"] == 1320
            and exported["generator_definition"]["device"] == "cpu", "permutation definition differs")
    vocabulary = vocabulary_from_ast(source_root / "neuropixel/task.py")
    require(exported["vocabulary"] == vocabulary, "exported vocabulary differs from source literals")
    universe = [(agent, verb, patient) for agent in range(12) for verb in range(10)
                for patient in range(12) if agent != patient]
    lookup = {triple: index for index, triple in enumerate(universe)}
    require(len(universe) == len(lookup) == 1320, "universe is not 1,320 unique triples")
    all_ids = set(range(1320))
    rows = exported["seeds"]
    require(len(rows) == 5 and {row["seed"] for row in rows} == set(SEEDS), "five unique required seed permutations are missing")
    definitions, direct_checks, partitions, pool_records, pool_sets = {}, [], [], [], {}

    def add_pool(revision, seed, split, ordered, topic=None):
        name = f"{revision}/seed{seed}/" + (f"topic{topic}/" if topic is not None else "") + split
        require(name not in pool_sets and len(set(ordered)) == len(ordered), "duplicate pool/triple")
        pool_sets[name] = set(ordered)
        pool_records.append({"pool_id": name, "revision": revision, "split_seed": seed, "partition": split,
                             "topic": topic, "n_triples": len(ordered), "ordered_ids_sha256": object_hash(ordered),
                             "membership_ids_sha256": object_hash(sorted(ordered)), "membership_universe_indices": sorted(ordered)})
        return name

    def check_partition(name, pools, expected_ids):
        pairs = {f"{left}|{right}": len(set(pools[left]) & set(pools[right])) for left, right in combinations(pools, 2)}
        covered = set().union(*(set(value) for value in pools.values()))
        require(not any(pairs.values()) and covered == expected_ids, f"partition overlap or incomplete coverage: {name}")
        partitions.append({"partition_id": name, "counts": {key: len(value) for key, value in pools.items()},
                           "pairwise_intersections": pairs, "union_count": len(covered), "disjoint_complete": True})

    for row in rows:
        resources.append(admission())
        seed, perm = row["seed"], row["permutation"]
        require(type(seed) is int and len(perm) == 1320 and all(type(index) is int for index in perm)
                and set(perm) == all_ids, "invalid permutation indices")
        old = {"train": perm[264:], "test": perm[:264]}
        new = {"train": perm[396:], "validation": perm[264:396], "test": perm[:264]}
        for label, reconstructed, actual in (("RoleTask", old, row["direct_role_task"]),
                                             ("ResearchRoleTask", new, row["direct_research_role_task"])):
            expected = {split: [list(universe[index]) for index in ids] for split, ids in reconstructed.items()}
            require(actual == expected, f"independent slicing differs from direct {label} constructor: seed{seed}")
            direct_checks.append({"seed": seed, "constructor": label, "all_ordered_memberships_equal": True,
                                  "counts": {key: len(values) for key, values in reconstructed.items()}})
        definitions[seed] = {"historical": old, "prospective": new}
        if seed in HISTORICAL_SEEDS:
            check_partition(f"historical80_20/seed{seed}", old, all_ids)
            for split, ids in old.items():
                add_pool("historical80_20", seed, split, ids)
        if seed in PROSPECTIVE_SEEDS:
            check_partition(f"prospective70_10_20/seed{seed}", new, all_ids)
            for split, ids in new.items():
                add_pool("prospective70_10_20", seed, split, ids)
    require(len(archived) == len(exported["growth_partitions"]) == 3, "three archived/exported growth topics required")
    growth_checks = []
    for topic in range(3):
        permitted = {index for index, (agent, _, patient) in enumerate(universe)
                     if topic * 4 <= agent < topic * 4 + 4 and topic * 4 <= patient < topic * 4 + 4}
        pools = {split: [index for index in ids if index in permitted] for split, ids in definitions[0]["prospective"].items()}
        triples = {split: [list(universe[index]) for index in ids] for split, ids in pools.items()}
        expected = {"topic": topic, "split_seed": 0, "counts": {key: len(value) for key, value in pools.items()},
                    "triples": triples, "triples_sha256": object_hash(triples)}
        require(archived[topic] == expected, f"archived growth topic{topic} differs from independent filtering")
        require(exported["growth_partitions"][topic] == expected, f"exported growth topic{topic} differs")
        check_partition(f"growth_filtered/seed0/topic{topic}", pools, permitted)
        for split, ids in pools.items():
            add_pool("growth_filtered", 0, split, ids, topic)
        growth_checks.append({"topic": topic, "n_universe": len(permitted), "counts": expected["counts"],
                              "triples_sha256": expected["triples_sha256"], "archived_ordered_memberships_equal": True})

    overlaps = []
    for left, right in combinations(sorted(pool_sets), 2):
        a, b = pool_sets[left], pool_sets[right]
        overlaps.append({"left_pool": left, "right_pool": right, "intersection_count": len(a & b),
                         "left_only_count": len(a - b), "right_only_count": len(b - a), "union_count": len(a | b)})
    training = [record["pool_id"] for record in pool_records if record["partition"] == "train"]
    groups = {
        "historical_train_seeds0_1_2": [name for name in training if name.startswith("historical80_20/")],
        "prospective_train_seeds0_101_202": [name for name in training if name.startswith("prospective70_10_20/")],
        "growth_topic_train_union": [name for name in training if name.startswith("growth_filtered/")],
        "all_declared_training_pools": training,
    }
    coverage, union_sets = {}, {}
    for label, members in groups.items():
        combined = set().union(*(pool_sets[name] for name in members))
        union_sets[label] = combined
        coverage[label] = {"pool_ids": members, "union_count": len(combined),
                           "universe_fraction": len(combined) / 1320, "membership_universe_indices": sorted(combined),
                           "outside_union_count": len(all_ids - combined), "outside_union_indices": sorted(all_ids - combined),
                           "scope": "Eligible membership union, not observed historical sample exposure."}
    final_exposure = []
    for record in pool_records:
        if record["partition"] != "test":
            continue
        name, heldout = record["pool_id"], pool_sets[record["pool_id"]]
        own_name = name[:-4] + "train"
        require(not heldout & pool_sets[own_name], "within-configuration test/train overlap")
        final_exposure.append({"final_pool_id": name, "n_triples": len(heldout), "own_training_pool_id": own_name,
            "own_train_overlap": 0,
            "overlap_with_each_training_pool": {train: len(heldout & pool_sets[train]) for train in training},
            "overlap_with_training_unions": {key: len(heldout & ids) for key, ids in union_sets.items()},
            "not_in_any_declared_training_pool": sorted(heldout - union_sets["all_declared_training_pools"]),
            "interpretation": "Cross-configuration eligibility overlap is not proof of within-run leakage or actual sampled exposure."})
    old, new = definitions[0]["historical"], definitions[0]["prospective"]
    resources.append(admission())
    require(sha256(export_path) == input_hashes["permutation_export"] and sha256(archived_path) == input_hashes["archived_growth_partitions"],
            "an input changed during the audit")
    require({name: sha256(source_root / name) for name in SOURCES} == actual_source, "task source changed during audit")
    commit = subprocess.check_output(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=source_root,
                                     stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15,
                                     env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).decode().strip()
    return {
        "schema_version": 1, "item": 7, "status": "verified", "issues": [], "started_at_utc": started, "completed_at_utc": now(),
        "audit_kind": "eligible triple-pool membership and cross-configuration overlap", "auditor_sha256": sha256(Path(__file__)),
        "source": {"git_commit": commit, "sha256": actual_source}, "export_source": exported["source"],
        "input_sha256": input_hashes, "exporter_sha256": exported["exporter_sha256"],
        "input_paths_relative_to_source_root": {"permutation_export": Path(os.path.relpath(export_path, source_root)).as_posix(),
                                                "archived_growth_partitions": Path(os.path.relpath(archived_path, source_root)).as_posix()},
        "archived_growth_provenance": {"item": 6, "archive_commit": "15e76456bc2b4cce5faec0b08fb5288fe7844547",
                                      "relative_path": "results/research/06_cloud_runs/37566497890-1/growth/partitions.json"},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "threads": 1, "torch_imported": False},
        "permutation_export_runtime": runtime, "resource_samples": resources,
        "universe": {"count": 1320, "ordered_triples_sha256": object_hash(universe), "vocabulary": vocabulary,
                     "index_definition": "Zero-based index of [(a,v,p) for a in range(12) for v in range(10) for p in range(12) if a != p].",
                     "grouping_unit": "Ordered agent/verb/patient triple; place, query role and spatial layout are outside this grouping unit."},
        "direct_constructor_checks": direct_checks, "within_partition_checks": partitions,
        "growth_archived_checks": growth_checks, "pools": pool_records, "all_pool_pair_overlaps": overlaps,
        "training_pool_union_coverage": coverage, "final_pool_training_eligibility_overlaps": final_exposure,
        "same_seed0_revision_relationship": {
            "test_memberships_identical": old["test"] == new["test"],
            "historical_train_count": len(old["train"]), "prospective_train_count": len(new["train"]),
            "prospective_validation_count": len(new["validation"]),
            "prospective_validation_previously_train_eligible": len(set(new["validation"]) & set(old["train"])),
            "historical_train_equals_prospective_train_union_validation": set(old["train"]) == set(new["train"]) | set(new["validation"]),
        },
        "sample_exposure": {"actual_minibatches_audited": False, "observed_unique_training_triples": None,
                            "historical_raw_minibatches_available_here": False,
                            "reason": "Pool membership identifies eligibility only. Historical raw minibatches are unavailable; this audit does not decode prospective sample arrays or infer unique exposures from budgets or stream hashes."},
        "limitations": [
            "Exact membership reconstruction is conditional on the recorded CPU Torch 2.6.0 permutation input. No Python/NumPy shuffle substitution is used.",
            "Ordered reconstructed pools match directly instantiated current classes for all five exported seeds and match the archived item-6 growth partitions for seed0. This does not establish byte identity to unavailable historical split lists from every old runtime.",
            "Historical study seeds are 0/1/2; prospective study seeds are 0/101/202. Other exported class/seed combinations are consistency checks, not additional declared training runs.",
            "Across seeds/revisions these are overlapping repartitions of the same 1,320 groups. Counting seed changes does not create new group identities or an unexposed external test universe.",
            "Cross-configuration training eligibility overlap does not itself demonstrate within-run leakage, full actual training exposure, selection on final scores, or parameter transfer between runs.",
            "A group outside a union of known training pools is not thereby a pristine confirmatory test: prior validation/final queries, analytical exposure, and unrecorded studies remain separate questions.",
            "No model/checkpoint import, task.sample call, new example generation, final model prediction, training, or change to historical code/gates is performed by this auditor."
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-export", type=Path, required=True)
    parser.add_argument("--archived-growth-partitions", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "refusing to overwrite a split-exposure audit")
    try:
        result = audit(args.permutation_export.resolve(), args.archived_growth_partitions.resolve(), args.source_root.resolve())
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result = {"schema_version": 1, "item": 7, "status": "rejected", "at_utc": now(),
                  "auditor_sha256": sha256(Path(__file__)), "issues": [f"{type(error).__name__}: {error}"],
                  "scope": "No membership/exposure conclusions accepted after an audit failure."}
    write_new(output, result)
    print(json.dumps({"status": result["status"], "issues": result["issues"], "output": str(output), "sha256": sha256(output)}))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
