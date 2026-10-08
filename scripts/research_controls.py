"""Run item 3's frozen elementary-control panel and retain every prediction.

Run from the repository root: python scripts/research_controls.py --threads 2
No neural model is trained or evaluated by this experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import NormalDist
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropixel.phase3 import RoleTaskFar  # noqa: E402
from neuropixel.research.controls import (  # noqa: E402
    adjacent_exact, category_probabilities, category_seeded_draw, same_row_exact,
    swap_agent_patient,
)
from neuropixel.task import NOUNS, ROLES, VERBS, RoleTask  # noqa: E402


SHAPES = {"adjacent": (8, 8), "far": (12, 12)}


def content_hash(**arrays: torch.Tensor) -> str:
    """Hash sorted array names, explicit shapes/dtypes, and little-endian values."""
    digest = hashlib.sha256()
    for name, tensor in sorted(arrays.items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<i8")
        header = json.dumps({"name": name, "shape": list(array.shape), "dtype": "<i8"},
                            sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sample_dataset(task_type, split_seed, split, sampling_seed, *, examples_per_role=None, n=None):
    """Sample the unchanged original generator; metadata remains in the evaluator."""
    if task_type not in SHAPES or split not in ("train", "test"):
        raise ValueError("unknown task or original partition")
    if (examples_per_role is None) == (n is None):
        raise ValueError("specify either examples_per_role or unbalanced n")
    task_class = RoleTask if task_type == "adjacent" else RoleTaskFar
    task = task_class(*SHAPES[task_type], seed=split_seed)
    generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
    batches = []
    roles_to_sample = range(len(ROLES)) if examples_per_role is not None else [None]
    for role in roles_to_sample:
        count = examples_per_role if role is not None else n
        if count <= 0:
            raise ValueError("sample size must be positive")
        canvas, target, meta = task.sample(count, split, generator, "cpu", query_role=role, meta=True)
        batches.append((canvas, target, meta["fillers"]))
    canvas, target, fillers = (torch.cat([batch[i] for batch in batches]) for i in range(3))
    query = canvas[:, task.query_pos[0], task.query_pos[1]]
    role_indices = (query[:, None] == task.role_ids[None, :]).to(torch.long).argmax(1)
    return task, canvas, target, role_indices, fillers


def _wilson_interval(correct, n, confidence=0.95):
    """Two-sided Wilson interval for a single marginal binary rate."""
    if n <= 0 or not 0 <= correct <= n:
        raise ValueError("Wilson counts must satisfy 0 <= correct <= n and n > 0")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, z2 = correct / n, z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = z * (p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5 / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _binary_metrics(prediction, target, roles, confidence):
    correct = prediction == target
    per_role = {}
    for index, role in enumerate(ROLES):
        select = roles == index
        n, k = int(select.sum()), int(correct[select].sum())
        per_role[role] = {"correct": k, "n": n, "accuracy": k / n,
                          "wilson_interval": _wilson_interval(k, n, confidence)}
    return {"correct": int(correct.sum()), "n": len(target),
            "accuracy": float(correct.to(torch.float64).mean()), "per_role": per_role,
            "macro_all_roles": sum(r["accuracy"] for r in per_role.values()) / 4,
            "macro_agent_patient_accuracy": (per_role["AGENTE"]["accuracy"] +
                                               per_role["PACIENTE"]["accuracy"]) / 2}


def _expected_metrics(probabilities, target, roles):
    """Score probability assigned to truth after the input-only predictor returns."""
    correct_probability = probabilities.gather(1, target[:, None]).squeeze(1)
    per_role = {}
    for index, role in enumerate(ROLES):
        p = correct_probability[roles == index]
        per_role[role] = {"n": len(p), "expected_correct": float(p.sum()),
                          "expected_accuracy": float(p.mean())}
    return {"n": len(target), "expected_correct": float(correct_probability.sum()),
            "expected_accuracy": float(correct_probability.mean()), "per_role": per_role,
            "macro_all_roles": sum(r["expected_accuracy"] for r in per_role.values()) / 4,
            "macro_agent_patient_accuracy": (per_role["AGENTE"]["expected_accuracy"] +
                                               per_role["PACIENTE"]["expected_accuracy"]) / 2}


def _triples(task, fillers):
    noun_index = {token: i for i, token in enumerate(task.v.ids(NOUNS))}
    verb_index = {token: i for i, token in enumerate(task.v.ids(VERBS))}
    return [(noun_index[a], verb_index[v], noun_index[p]) for a, v, p, _ in fillers.tolist()]


@contextmanager
def _atomic_binary(path):
    """Expose a destination only after its unique temporary file is closed and synced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as raw:
            temporary = Path(raw.name)
            yield raw
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prediction_file(path, rows):
    """Atomically write deterministic gzip JSONL, then check its complete payload."""
    digest = hashlib.sha256()
    with _atomic_binary(path) as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                payload = (json.dumps(row, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")) + "\n").encode()
                digest.update(payload)
                compressed.write(payload)
    artifact = path.read_bytes()
    if hashlib.sha256(gzip.decompress(artifact)).hexdigest() != digest.hexdigest():
        raise OSError(f"prediction artifact failed its post-write integrity check: {path}")
    return {"sha256": hashlib.sha256(artifact).hexdigest(),
            "jsonl_sha256": digest.hexdigest(), "bytes": path.stat().st_size,
            "format": "gzip-compressed UTF-8 JSONL", "gzip_mtime": 0}


def evaluate_dataset(spec, random_seed, confidence, output_dir):
    task, canvas, target, roles, fillers = sample_dataset(
        spec["task_type"], spec["split_seed"], spec["split"], spec["sampling_seed"],
        examples_per_role=spec.get("examples_per_role"), n=spec.get("n"))
    schema = {"query_pos": task.query_pos, "vocab": task.v}
    solver = adjacent_exact if spec["task_type"] == "adjacent" else same_row_exact
    exact = solver(canvas, **schema)
    probabilities = category_probabilities(canvas, **schema)
    draw = category_seeded_draw(canvas, seed=random_seed, **schema)

    # Labels below use generator fillers, not the reference predictor's answer.
    swapped = swap_agent_patient(canvas, **schema)
    swapped_fillers = fillers[:, [2, 1, 0, 3]]
    swapped_target = swapped_fillers.gather(1, roles[:, None]).squeeze(1)
    swapped_exact = solver(swapped, **schema)
    swapped_probabilities = category_probabilities(swapped, **schema)
    swapped_draw = category_seeded_draw(swapped, seed=random_seed, **schema)
    paired = (roles == 0) | (roles == 2)
    probability_unchanged = (probabilities == swapped_probabilities).all(1)
    bag_unchanged = (canvas.flatten(1).sort(1).values == swapped.flatten(1).sort(1).values).all(1)
    query_unchanged = canvas[:, *task.query_pos] == swapped[:, *task.query_pos]

    train_set, test_set = set(task.train_triples), set(task.test_triples)
    triples = _triples(task, fillers)
    reverse_triples = [(p, v, a) for a, v, p in triples]

    def membership(triple):
        if triple in train_set:
            return "train"
        if triple in test_set:
            return "test"
        raise AssertionError("diagnostic triple is outside the original generator")

    reverse_membership = [membership(triple) for triple in reverse_triples]
    if any(membership(triple) != spec["split"] for triple in triples):
        raise AssertionError("base example is in the wrong original partition")

    pair_metrics = {}
    for index, role in ((0, "AGENTE"), (2, "PACIENTE")):
        select = roles == index
        n = int(select.sum())
        base_correct = draw[select] == target[select]
        swapped_correct = swapped_draw[select] == swapped_target[select]
        pair_metrics[role] = {
            "pairs": n,
            "targets_changed": int((target[select] != swapped_target[select]).sum()),
            "exact_both_correct": int(((exact[select] == target[select]) &
                                        (swapped_exact[select] == swapped_target[select])).sum()),
            "category_both_correct": int((base_correct & swapped_correct).sum()),
            "exact_prediction_changed": int((exact[select] != swapped_exact[select]).sum()),
            "category_prediction_changed": int((draw[select] != swapped_draw[select]).sum()),
            "category_distribution_unchanged": int(probability_unchanged[select].sum()),
            "input_bag_unchanged": int(bag_unchanged[select].sum()),
            "query_unchanged": int(query_unchanged[select].sum()),
            "category_correct_members": int(base_correct.sum() + swapped_correct.sum()),
            "category_mean_within_pair_accuracy": float(
                (base_correct.to(torch.float64) + swapped_correct.to(torch.float64)).mean() / 2),
            "partition_transitions": dict(sorted(Counter(
                f"{spec['split']}->{reverse_membership[i]}" for i in range(len(target))
                if bool(select[i])).items())),
        }

    passed = bool((exact == target).all() and (swapped_exact[paired] == swapped_target[paired]).all()
                  and probability_unchanged.all() and bag_unchanged.all() and query_unchanged.all()
                  and (draw == swapped_draw).all() and (target[paired] != swapped_target[paired]).all()
                  and all(r["category_both_correct"] == 0 and r["category_correct_members"] == r["pairs"]
                          for r in pair_metrics.values()))
    if not passed:
        raise AssertionError(f"elementary-control invariant failed: {spec['id']}")

    canvas_list, swapped_list = canvas.tolist(), swapped.tolist()

    def rows():
        for i in range(len(target)):
            candidate_ids = probabilities[i].nonzero().flatten().tolist()
            row = {"example_id": f"{spec['id']}:{i:05d}", "query_role": ROLES[int(roles[i])],
                   "canvas": canvas_list[i], "target": int(target[i]), "triple": list(triples[i]),
                   "triple_membership": spec["split"], "exact_prediction": int(exact[i]),
                   "category_seeded_prediction": int(draw[i]), "category_candidates": candidate_ids,
                   "category_probabilities": probabilities[i, candidate_ids].tolist(),
                   "category_probability_of_target": float(probabilities[i, target[i]])}
            if bool(paired[i]):
                row["agent_patient_swap"] = {
                    "canvas": swapped_list[i], "target": int(swapped_target[i]),
                    "triple": list(reverse_triples[i]), "triple_membership": reverse_membership[i],
                    "exact_prediction": int(swapped_exact[i]),
                    "category_seeded_prediction": int(swapped_draw[i]),
                    "category_probability_of_target": float(swapped_probabilities[i, swapped_target[i]]),
                    "category_distribution_unchanged": bool(probability_unchanged[i]),
                    "input_bag_unchanged": bool(bag_unchanged[i]), "query_unchanged": bool(query_unchanged[i]),
                }
            yield row

    filename = f"{spec['id']}.jsonl.gz"
    artifact = _prediction_file(output_dir / "controls" / filename, rows())
    artifact["path"] = f"controls/{filename}"
    return {
        **spec, "shape": list(canvas.shape), "query_pos": list(task.query_pos),
        "original_partition_sizes": {"train": len(train_set), "test": len(test_set)},
        "dataset_sha256": content_hash(canvas=canvas, target=target, query_role_indices=roles),
        "counterfactual_sha256": content_hash(canvas=swapped[paired], target=swapped_target[paired],
                                             base_indices=torch.arange(len(target))[paired]),
        "exact_solver": solver.__name__, "exact": _binary_metrics(exact, target, roles, confidence),
        "category_expected": _expected_metrics(probabilities, target, roles),
        "category_seeded": _binary_metrics(draw, target, roles, confidence),
        "counterfactual": {"pairs": int(paired.sum()), "per_role": pair_metrics,
                           "both_members_guaranteed_in_base_partition": False,
                           "category_draw_coupling": "same seed and probability matrix; same draw per pair"},
        "prediction_artifact": artifact, "acceptance_passed": passed,
    }


def run(protocol_path, output_path, threads=2, integrity_retry_from=None):
    if not 1 <= threads <= 2:
        raise ValueError("this item's CPU reservation permits one or two threads")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(protocol_path.read_text())
    frozen_path = protocol_path.with_name("protocol_frozen.json")
    frozen = json.loads(frozen_path.read_text())
    for path, expected in frozen["files"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen protocol changed: {path}")
    control = protocol["control"]
    confidence = protocol["decision"]["confidence_level"]
    previous, retry = None, None
    if integrity_retry_from is not None:
        previous_bytes = integrity_retry_from.read_bytes()
        previous = json.loads(previous_bytes)
        differences = []
        for dataset in previous["datasets"]:
            recorded = dataset["prediction_artifact"]
            path = output_path.parent / recorded["path"]
            actual = path.read_bytes()
            actual_hash = hashlib.sha256(actual).hexdigest()
            if actual_hash != recorded["sha256"]:
                difference = {"dataset": dataset["id"], "path": recorded["path"],
                              "recorded_bytes": recorded["bytes"], "observed_bytes": len(actual),
                              "recorded_sha256": recorded["sha256"], "observed_sha256": actual_hash}
                try:
                    gzip.decompress(actual)
                except (EOFError, OSError) as error:
                    difference["decompression_error"] = f"{type(error).__name__}: {error}"
                differences.append(difference)
        retry = {"reason": "post-run artifact integrity failure; no scientific configuration change",
                 "cause": "not established; no concurrent benchmark/test writer was found",
                 "repair": "unique temporary files, close/fsync and atomic replace; complete gzip validation",
                 "previous_result_sha256": hashlib.sha256(previous_bytes).hexdigest(),
                 "previous_started_at_utc": previous["started_at_utc"],
                 "previous_completed_at_utc": previous["completed_at_utc"],
                 "previous_source_sha256": previous["source_sha256"], "observed_failures": differences}
    specs = []
    for task_type in control["task_types"]:
        for split_seed in control["split_seeds"]:
            for split in control["splits"]:
                specs.append({"id": f"{task_type}_s{split_seed}_{split}", "panel": "new_balanced_control",
                              "task_type": task_type, "split_seed": split_seed, "split": split,
                              "sampling_seed": control["sampling_seed"],
                              "examples_per_role": control["examples_per_role"]})
    specs.append({"id": "far_historical_recipe_s0_test", "panel": "retrospective_recipe_reconstruction",
                  "task_type": "far", "split_seed": 0, "split": "test",
                  "sampling_seed": protocol["historical"]["sample_seed"],
                  "n": protocol["historical"]["test_n"]})
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in specs:
        result = evaluate_dataset(spec, control["random_category_seed"], confidence, output_path.parent)
        results.append(result)
        print(json.dumps({"dataset": spec["id"], "exact_correct": result["exact"]["correct"],
                          "n": result["exact"]["n"],
                          "category_expected": result["category_expected"]["expected_accuracy"],
                          "category_seeded_correct": result["category_seeded"]["correct"],
                          "pairs": result["counterfactual"]["pairs"]}), flush=True)
    source_paths = ["neuropixel/research/controls.py", "scripts/research_controls.py", "neuropixel/task.py",
                    "neuropixel/phase3.py", "docs/research/protocol.json", "docs/research/protocol_frozen.json"]
    output = {
        "schema_version": 1, "item": 3, "protocol_id": protocol["protocol_id"],
        "provenance": "newly executed CPU controls; historical-recipe reconstruction reported separately",
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "source_commit_before_implementation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(), "device": "cpu",
                        "torch_num_threads": torch.get_num_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "frozen_control_configuration": control,
        "implementation_clarifications_before_run": {
            "recorded": "docs/research/03_controls.md, 2026-10-06, before first evaluation",
            "shapes": {k: list(v) for k, v in SHAPES.items()},
            "partition": "original 80/20 RoleTask split; no neural model is trained",
            "sampling": "reset sampling RNG once per dataset, concatenate queries in ROLES order",
            "category_draw": "reset private RNG to random_category_seed for each dataset and its swapped copy",
            "retrospective": "original Far recipe using current software; exact historical inputs unverified",
            "counterfactual": "noun queries only; target from transformed generator fillers; split crossing recorded",
        },
        "intervals": {"confidence_level": confidence, "method": "Wilson per marginal binary query role",
                      "scope": "conditional on fixed generator/schema and control; not model replication uncertainty",
                      "macro_or_pair_member_binomial_interval": False},
        "hash_format": "SHA256 of sorted names and length-prefixed JSON shape/dtype headers, then little-endian int64 bytes",
        "vocabulary": {str(i): token for i, token in enumerate(RoleTask().v.tokens)},
        "datasets": results,
        "acceptance_passed": all(r["acceptance_passed"] for r in results),
        "limitations": [
            "Domain-informed symbolic ceilings and category controls are not trained neural baselines.",
            "Balanced control samples are diagnostic; no neural hyperparameter or checkpoint is selected here.",
            "The 80/20 original generator partition is distinct from prospective neural 70/10/20 partitions.",
            "Three split panels share vocabulary, generator and sampling seed and are not independent lab replications.",
            "Original/counterfactual members are dependent and may lie in different triple partitions.",
            "Seeded category scores reflect one declared draw; expected scores average over the control's random choice.",
            "Retrospective recipe reconstruction is not a paired comparison to unavailable historical model predictions.",
            "No neural accuracy, physical energy advantage, or new mechanism is established by this item.",
        ],
    }
    if previous is not None:
        without_artifacts = lambda datasets: [{k: v for k, v in d.items() if k != "prediction_artifact"}
                                              for d in datasets]
        retry["metrics_and_dataset_hashes_identical"] = (
            without_artifacts(results) == without_artifacts(previous["datasets"]))
        retry["prediction_jsonl_hashes_identical"] = (
            [r["prediction_artifact"]["jsonl_sha256"] for r in results] ==
            [r["prediction_artifact"]["jsonl_sha256"] for r in previous["datasets"]])
        retry["intended_compressed_hashes_identical"] = (
            [r["prediction_artifact"]["sha256"] for r in results] ==
            [r["prediction_artifact"]["sha256"] for r in previous["datasets"]])
        if not retry["metrics_and_dataset_hashes_identical"] or not retry["prediction_jsonl_hashes_identical"]:
            raise AssertionError("integrity retry changed scientific outputs; retain and investigate both attempts")
        output["integrity_retry"] = retry
    with _atomic_binary(output_path) as raw:
        raw.write((json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/research/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/research/03_controls.json")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--integrity-retry-from", type=Path, default=None,
                        help="previous result manifest, retained before an artifact-integrity retry")
    args = parser.parse_args()
    result = run(args.protocol, args.output, args.threads, args.integrity_retry_from)
    print(json.dumps({"acceptance_passed": result["acceptance_passed"],
                      "datasets": len(result["datasets"]), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
