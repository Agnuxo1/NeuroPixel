#!/usr/bin/env python3
"""Independently recount item-7 saved artifacts, without importing or running RNGs.

The maximum's PMF is obtained from exact-integer CDF-power differences, unlike
the producer's survival-function moment sums. All arithmetic comparisons use
absolute tolerance 1e-12 and zero relative tolerance; IDs/counts are exact.
The output is exclusive and must be outside the manifested input directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

THREADS = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "BLIS_NUM_THREADS")
ATOL = 1e-12
SOURCE_COMMIT = "0ed43bd5bb1d8bcb166b8e63e909195cbe69f769"
MANIFEST_SHA = "e25030099e0cb631d23701b878d92f3c2d980ea68c677d157ac531e8853f755d"
PLAN_PATH = "docs/research/07_execution_plan.json"
FILES = {"analysis.json", "environment.json", "events.json", "final_counts.npz",
         "frozen_plan.json", "methodological_evidence.md", "replicates.csv",
         "resource_observations.json", "selection_counts.npz", "selection_receipt.json",
         "source_snapshot.py"}
RECIPE = {
    "name": "null_holdout_selection", "replicates": 10000, "candidate_count": 64,
    "examples_per_partition": 256, "true_accuracy": 0.5, "search_sizes": [1, 4, 16, 64],
    "bit_generator": "PCG64", "seeds": {"train": 71001, "development": 71002, "final": 71003},
    "train_rule": "descending_count_then_lower_id",
    "development_rule": "maximum_count_then_lower_id",
    "final_rule": "generate_only_after_selection_receipt", "tail_threshold": 0.6,
    "tail_count": 154, "analytic_discrepancy_standard_errors": 6.0,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("Duplicate JSON key: " + key)
            out[key] = value
        return out
    def nonfinite(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=nonfinite)


def utc(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError("Non-UTC timestamp")
    return dt


def memory_guard(report, phase):
    line = next(x for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:"))
    available = int(line.split()[1]) * 1024
    report["resource_observations"].append({"phase": phase, "available_bytes": available})
    if available < 8 * 1024**3:
        raise RuntimeError("Available RAM below 8 GiB; stopping this audit")


class Checks:
    def __init__(self, report):
        self.report = report
        self.numeric_max_error = 0.0

    def require(self, condition, label):
        self.report["checks"] += 1
        if not condition:
            raise ValueError(label)

    def exact(self, actual, expected, label):
        self.require(canonical(actual) == canonical(expected), label)

    def close(self, actual, expected, label):
        self.require(type(actual) in (int, float) and math.isfinite(actual), label + ": nonfinite/type")
        delta = abs(actual - expected)
        self.numeric_max_error = max(self.numeric_max_error, delta)
        self.require(delta <= ATOL, f"{label}: {actual!r} != {expected!r}, abs error {delta!r}")

    def nested(self, actual, expected, label):
        if isinstance(expected, dict):
            self.exact(sorted(actual), sorted(expected), label + ": keys")
            for key, value in expected.items():
                self.nested(actual[key], value, label + "." + key)
        elif isinstance(expected, list):
            self.require(len(actual) == len(expected), label + ": length")
            for i, value in enumerate(expected):
                self.nested(actual[i], value, label + f"[{i}]")
        elif type(expected) is float:
            self.close(actual, expected, label)
        else:
            self.exact(actual, expected, label)


def pmf_reference(n, m, threshold):
    """P(max=k)=(C_k**m-C_(k-1)**m)/2**(n*m), C_k=sum binomial(n,j)."""
    denominator = 1 << (n * m)
    cumulative, previous_power, probabilities = 0, 0, []
    for k in range(n + 1):
        cumulative += math.comb(n, k)
        current_power = cumulative**m
        probabilities.append((current_power - previous_power) / denominator)
        previous_power = current_power
    mean = math.fsum((k / n) * p for k, p in enumerate(probabilities))
    second = math.fsum((k / n)**2 * p for k, p in enumerate(probabilities))
    variance = second - mean**2
    final_tail = sum(math.comb(n, k) for k in range(threshold, n + 1)) / (1 << n)
    return {
        "development_mean": mean, "development_second_moment": second,
        "development_variance": variance, "final_mean": 0.5, "final_variance": 0.25 / n,
        "optimism_mean": mean - 0.5, "optimism_variance": variance + 0.25 / n,
        "development_tail_probability": math.fsum(probabilities[threshold:]),
        "final_tail_probability": final_tail,
    }, math.fsum(probabilities)


def describe_counts(counts, n):
    count, total = len(counts), sum(counts)
    mean = total / (count * n)
    # Exact integer sums avoid repeating NumPy's reduction path.
    variance = (sum(x*x for x in counts) * count - total*total) / (count * (count-1) * n*n)
    sd = math.sqrt(variance)
    se = sd / math.sqrt(count)
    return {"mean": mean, "sample_standard_deviation": sd, "monte_carlo_standard_error": se,
            "mean_monte_carlo_95_normal_interval": [mean - 1.96*se, mean + 1.96*se]}


def discrepancy(observed, expected, variance, count):
    se = math.sqrt(variance / count)
    delta = observed - expected
    return {"observed": observed, "analytic_expected": expected,
            "analytic_monte_carlo_standard_error": se, "difference": delta,
            "standard_errors": delta / se, "flagged": abs(delta) > 6 * se,
            "rule": "absolute difference strictly greater than 6 analytic MCSE; no rerun"}


def verify(args, report, check):
    root, frozen = args.input.resolve(strict=True), args.source_root.resolve(strict=True)
    memory_guard(report, "before_numpy_import")
    import numpy as np
    report["audit_environment"] = {"python": sys.version, "numpy": np.__version__,
                                   "thread_environment": {k: os.environ[k] for k in THREADS}}
    manifest_bytes = (root / "artifact_manifest.json").read_bytes()
    check.exact(sha(manifest_bytes), MANIFEST_SHA, "externally pinned manifest hash")
    manifest = read_json(root / "artifact_manifest.json")
    check.exact(sorted(p.name for p in root.iterdir()), sorted(FILES | {"artifact_manifest.json"}), "complete file inventory")
    check.exact(sorted(manifest["artifacts"]), sorted(FILES), "manifest inventory")
    for name in sorted(FILES):
        path = root / name
        check.require(path.is_file() and not path.is_symlink(), name + ": regular file")
        data = path.read_bytes()
        check.exact(manifest["artifacts"][name], {"bytes": len(data), "sha256": sha(data)}, name + ": hash and bytes")
    check.exact([manifest["schema_version"], manifest["item"], manifest["status"]], [1, 7, "completed"], "manifest status")
    head = subprocess.run(["git", "-C", str(frozen), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()
    check.exact(head, SOURCE_COMMIT, "frozen checkout HEAD")
    plan_bytes = (root / "frozen_plan.json").read_bytes()
    plan = read_json(root / "frozen_plan.json")
    snapshots = {PLAN_PATH: "frozen_plan.json", "scripts/research_simulate_holdout_selection.py": "source_snapshot.py",
                 "docs/research/07_methodological_evidence.md": "methodological_evidence.md"}
    for relative, name in snapshots.items():
        committed = subprocess.run(["git", "-C", str(frozen), "show", f"{SOURCE_COMMIT}:{relative}"],
                                   check=True, capture_output=True, timeout=10).stdout
        check.require(committed == (frozen / relative).read_bytes() == (root / name).read_bytes(), relative + ": exact frozen bytes")
    check.exact(plan["simulation"], RECIPE, "fixed scientific recipe")
    check.exact([plan["schema_version"], plan["item"], plan["status"]], [1, 7, "frozen"], "plan status")
    check.exact(plan["resource_limits"], {"minimum_available_memory_gib": 8.0, "numerical_threads": 1}, "resource policy")
    source_hashes = {path: sha((frozen / path).read_bytes()) for path in snapshots if path != PLAN_PATH}
    check.exact(plan["implementation_sha256"], source_hashes, "source hash map")
    plan_hash = sha(plan_bytes)
    check.exact(manifest["plan_sha256"], plan_hash, "manifest plan identity")
    environment, receipt, analysis, events, resources = [read_json(root / name) for name in (
        "environment.json", "selection_receipt.json", "analysis.json", "events.json", "resource_observations.json")]
    for label, document in (("environment", environment), ("receipt", receipt), ("analysis", analysis)):
        check.exact(document["plan_sha256"], plan_hash, label + ": plan hash")
        check.exact(document["source_sha256"], source_hashes, label + ": source hashes")
    check.exact(environment["execution_git_head"], SOURCE_COMMIT, "recorded execution HEAD")
    check.exact([environment["numpy"], environment["bit_generator"], environment["device"], environment["numerical_threads"]],
                ["2.3.5", "PCG64", "cpu", 1], "execution environment")
    check.exact(environment["thread_environment"], {k: "1" for k in THREADS}, "producer thread policy")
    for document in (receipt, analysis):
        check.exact(document["environment_sha256"], manifest["artifacts"]["environment.json"]["sha256"], "environment link")
    check.exact(receipt["selection_artifact"], {"path": "selection_counts.npz", "sha256": manifest["artifacts"]["selection_counts.npz"]["sha256"]}, "selection link")
    check.exact([receipt["schema_version"], receipt["item"], receipt["event"], receipt["replicates"], receipt["search_sizes"],
                 receipt["final_rng_constructed"], receipt["final_draws_completed"]],
                [1, 7, "selection_frozen_before_final_generation", 10000, [1, 4, 16, 64], False, False], "receipt state")
    check.exact([e["event"] for e in events], ["started", "selection_receipt_persisted", "final_generation_started", "final_generation_completed"], "gate events")
    monotonic = [events[0]["monotonic_ns"], receipt["monotonic_ns"]] + [e["monotonic_ns"] for e in events[1:]]
    check.require(all(type(x) is int for x in monotonic) and all(a < b for a, b in zip(monotonic, monotonic[1:])), "strict monotonic gate ordering")
    times = [utc(plan["freeze_utc"]), utc(events[0]["at_utc"]), utc(receipt["at_utc"])] + [utc(e["at_utc"]) for e in events[1:]] + [utc(analysis["completed_at_utc"])]
    check.require(all(a < b for a, b in zip(times, times[1:])), "strict UTC freeze/selection/final ordering")
    receipt_hash = manifest["artifacts"]["selection_receipt.json"]["sha256"]
    check.exact(events[1]["sha256"], receipt_hash, "persisted receipt link")
    check.exact(events[2]["selection_receipt_sha256"], receipt_hash, "final gate receipt link")
    check.exact(analysis["selection_receipt_sha256"], receipt_hash, "analysis receipt link")
    check.exact(analysis["selection_counts_sha256"], receipt["selection_artifact"]["sha256"], "analysis selection link")
    check.exact(analysis["started_at_utc"], events[0]["at_utc"], "start timestamp")
    check.require(len(resources) == 5, "all producer resource observations")
    for row in resources:
        check.require(type(row["available_bytes"]) is int and row["available_bytes"] >= 8 * 1024**3, "producer RAM floor")
        check.close(row["available_gib"], row["available_bytes"] / 1024**3, "producer RAM units")
    report["provenance"] = {"source_commit": SOURCE_COMMIT, "artifact_manifest_sha256": MANIFEST_SHA,
                             "plan_sha256": plan_hash, "source_sha256": source_hashes,
                             "all_manifested_files_verified": len(FILES), "gate_order_verified": True}
    memory_guard(report, "before_array_recount")
    with np.load(root / "selection_counts.npz", allow_pickle=False) as handle:
        selection = {key: handle[key] for key in handle.files}
    with np.load(root / "final_counts.npz", allow_pickle=False) as handle:
        final = {key: handle[key] for key in handle.files}
    check.exact(sorted(selection), sorted(["train_counts", "development_counts", "train_order", "winner_ids", "selected_development_counts", "search_sizes"]), "selection array keys")
    check.exact(sorted(final), sorted(["final_counts", "selected_final_counts", "winner_ids", "search_sizes"]), "final array keys")
    for section in (selection, final):
        for key, value in section.items():
            shape = (4,) if key == "search_sizes" else ((10000, 4) if key in ("winner_ids", "selected_development_counts", "selected_final_counts") else (10000, 64))
            dtype = "|u1" if key in ("winner_ids", "train_order") else "<u2"
            check.exact(list(value.shape), list(shape), key + ": shape")
            check.exact(value.dtype.str, dtype, key + ": dtype")
            upper = 63 if key in ("winner_ids", "train_order") else 256
            check.require(bool((value <= upper).all()), key + ": value range")
        check.require(np.array_equal(section["search_sizes"], [1, 4, 16, 64]), "search size array")
    train, dev = selection["train_counts"], selection["development_counts"]
    winners = selection["winner_ids"]
    check.require(np.array_equal(winners, final["winner_ids"]), "winners unchanged after FINAL")
    digest = sha((canonical({"dtype": winners.dtype.str, "shape": list(winners.shape)}) + "\n").encode() + winners.tobytes(order="C"))
    check.exact(receipt["winner_ids_sha256"], digest, "receipt winner digest")
    check.exact(analysis["winner_ids_sha256"], digest, "analysis winner digest")
    for i in range(10000):
        order = sorted(range(64), key=lambda candidate: (-int(train[i, candidate]), candidate))
        check.require(selection["train_order"][i].tolist() == order, f"TRAIN order replicate {i}")
        for col, m in enumerate((1, 4, 16, 64)):
            winner = min(order[:m], key=lambda candidate: (-int(dev[i, candidate]), candidate))
            check.require(int(winners[i, col]) == winner, f"DEV shortlist/tie winner {i}/{m}")
            check.require(int(selection["selected_development_counts"][i, col]) == int(dev[i, winner]), f"selected DEV gather {i}/{m}")
            check.require(int(final["selected_final_counts"][i, col]) == int(final["final_counts"][i, winner]), f"selected FINAL gather {i}/{m}")
    check.require(bool((np.diff(selection["selected_development_counts"].astype(np.int32), axis=1) >= 0).all()), "nested DEV monotonicity")
    report["selection"] = {"replicates": 10000, "train_order_entries": 640000,
                           "shortlists_and_winners": 40000, "final_gathers": 40000,
                           "winner_ids_sha256": digest, "winner_ids_unchanged": True}
    memory_guard(report, "before_csv_and_summary_recount")
    with (root / "replicates.csv").open(newline="") as stream:
        rows = csv.DictReader(stream)
        check.exact(rows.fieldnames, ["replicate", "search_size", "winner_id", "train_winner_count", "development_count", "final_count", "examples", "development_accuracy", "final_accuracy", "paired_optimism"], "CSV columns")
        row_count = 0
        for row_count, row in enumerate(rows, start=1):
            i, col = divmod(row_count - 1, 4)
            check.require(i < 10000, "CSV has extra rows")
            winner = int(winners[i, col])
            d, f = int(dev[i, winner]), int(final["final_counts"][i, winner])
            expected_ints = [i, (1, 4, 16, 64)[col], winner, int(train[i, winner]), d, f, 256]
            check.exact([int(row[key]) for key in rows.fieldnames[:7]], expected_ints, f"CSV integer fields row {row_count}")
            for key, value in zip(rows.fieldnames[7:], (d/256, f/256, (d-f)/256)):
                check.close(float(row[key]), value, f"CSV {key} row {row_count}")
        check.exact(row_count, 40000, "all CSV rows")
    check.exact(analysis["recipe"], RECIPE, "analysis recipe")
    check.exact([analysis["schema_version"], analysis["item"], analysis["status"], analysis["h1_eligible"], analysis["replicate_count"], analysis["csv_rows"]], [1, 7, "completed", False, 10000, 40000], "analysis scope/counts")
    check.exact([s["search_size"] for s in analysis["summaries"]], [1, 4, 16, 64], "all four summaries")
    expected_summaries, flags = [], []
    for col, m in enumerate((1, 4, 16, 64)):
        analytic, mass = pmf_reference(256, m, 154)
        check.close(mass, 1.0, f"PMF normalization M={m}")
        dev_counts = [int(x) for x in selection["selected_development_counts"][:, col]]
        final_counts = [int(x) for x in final["selected_final_counts"][:, col]]
        values = {"development": dev_counts, "final": final_counts,
                  "paired_optimism": [d-f for d, f in zip(dev_counts, final_counts)]}
        descriptive = {name: describe_counts(counts, 256) for name, counts in values.items()}
        checks, tails = {}, {}
        for name, prefix in (("development", "development"), ("final", "final"), ("paired_optimism", "optimism")):
            checks[name + "_mean"] = discrepancy(descriptive[name]["mean"], analytic[prefix + "_mean"], analytic[prefix + "_variance"], 10000)
        for name in ("development", "final"):
            count = sum(value >= 154 for value in values[name])
            fraction = count/10000
            probability = analytic[name + "_tail_probability"]
            sd = math.sqrt(count * (10000-count) / (10000 * 9999))
            tails[name] = {"count": count, "replicates": 10000, "fraction": fraction,
                           "analytic_probability": probability, "observed_sample_standard_deviation": sd,
                           "observed_monte_carlo_standard_error": sd/100}
            checks[name + "_tail"] = discrepancy(fraction, probability, probability * (1-probability), 10000)
        for name, result in checks.items():
            if result["flagged"]:
                flags.append({"search_size": m, "check": name, **result})
        expected = {"search_size": m, "analytic": analytic, "descriptive": descriptive,
                    "tails": tails, "analytic_discrepancy_checks": checks}
        check.nested(analysis["summaries"][col], expected, f"summary M={m}")
        expected_summaries.append(expected)
    check.nested(analysis["analytic_discrepancy_flags"], flags, "all six-SE flags")
    check.exact(analysis["analytic_status"], "flagged_no_rerun" if flags else "within_prespecified_diagnostic_bands", "analytic status")
    report["recount"] = {"csv_rows": 40000, "csv_cells": 400000, "summaries": expected_summaries,
                         "analytic_checks": 20, "tails": 8, "six_se_flags": flags}
    memory_guard(report, "after_recount")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in THREADS:
        os.environ[key] = "1"
    output = args.output.absolute()
    if output.resolve().is_relative_to(args.input.resolve()):
        parser.error("Audit output must be outside the manifested input directory")
    if output.exists():
        parser.error("Refusing to overwrite an existing audit")
    report = {"schema_version": 1, "item": 7, "status": "running", "checks": 0, "issues": [],
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "verifier_sha256": sha(Path(__file__).read_bytes()),
              "input": str(args.input.resolve()), "source_root": str(args.source_root.resolve()),
              "arithmetic_tolerance": {"atol": ATOL, "rtol": 0.0}, "resource_observations": [],
              "method": "Exact-integer CDF-power-difference PMF; integer-sum empirical moments; independent Python selection; no RNG or producer imports.",
              "limits": ["This recount verifies saved artifacts and recorded gate order, not independent custody or absence of outside access.",
                         "No random streams are regenerated; distributional independence is the frozen design assumption.",
                         "The simulation does not establish NeuroPixel model performance."]}
    check = Checks(report)
    try:
        verify(args, report, check)
        report["status"] = "verified"
    except BaseException as exc:
        report["status"] = "failed"
        report["issues"].append({"type": type(exc).__name__, "message": str(exc)})
    report["maximum_absolute_arithmetic_difference"] = check.numeric_max_error
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "checks": report["checks"], "issues": report["issues"],
                      "output": str(output), "sha256": sha(output.read_bytes())}))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
