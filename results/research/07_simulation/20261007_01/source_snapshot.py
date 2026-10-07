#!/usr/bin/env python3
"""Item 7 null selection illustration; never run before the source/plan freeze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


SCRIPT_PATH = "scripts/research_simulate_holdout_selection.py"
NOTE_PATH = "docs/research/07_methodological_evidence.md"
REQUIRED_NUMPY_VERSION = "2.3.5"
EXPECTED_SIMULATION = {
    "name": "null_holdout_selection",
    "replicates": 10000,
    "candidate_count": 64,
    "examples_per_partition": 256,
    "true_accuracy": 0.5,
    "search_sizes": [1, 4, 16, 64],
    "bit_generator": "PCG64",
    "seeds": {"train": 71001, "development": 71002, "final": 71003},
    "train_rule": "descending_count_then_lower_id",
    "development_rule": "maximum_count_then_lower_id",
    "final_rule": "generate_only_after_selection_receipt",
    "tail_threshold": 0.6,
    "tail_count": 154,
    "analytic_discrepancy_standard_errors": 6.0,
}
EXPECTED_RESOURCES = {
    "minimum_available_memory_gib": 8.0,
    "numerical_threads": 1,
}
PLAN_FIELDS = {
    "schema_version", "item", "status", "freeze_utc", "simulation",
    "implementation_sha256", "resource_limits",
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strict_json(data):
    def reject_constant(value):
        raise ValueError("Non-finite JSON constant: " + value)

    def unique_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("Duplicate JSON key: " + key)
            out[key] = value
        return out

    return json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_pairs)


def parse_utc(value):
    if not isinstance(value, str):
        raise ValueError("freeze_utc must be an explicit UTC timestamp")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError("freeze_utc must be UTC")
    return dt


def validate_plan(plan, source_root):
    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS:
        raise ValueError("Unknown or incomplete item-7 plan schema")
    if canonical([plan["schema_version"], plan["item"], plan["status"]]) != canonical([1, 7, "frozen"]):
        raise ValueError("Expected schema_version=1, item=7, status=frozen")
    if canonical(plan["simulation"]) != canonical(EXPECTED_SIMULATION):
        raise ValueError("Scientific recipe differs from the source-fixed design")
    if canonical(plan["resource_limits"]) != canonical(EXPECTED_RESOURCES):
        raise ValueError("Resource contract differs from the source-fixed design")
    if parse_utc(plan["freeze_utc"]) > datetime.now(timezone.utc):
        raise ValueError("Freeze timestamp is in the future")
    hashes = plan["implementation_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != {SCRIPT_PATH, NOTE_PATH}:
        raise ValueError("Exactly the runner and methodological note must be anchored")
    snapshots = {}
    for relative, expected in hashes.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid source SHA-256: " + relative)
        path = source_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing/nonregular source: " + relative)
        data = path.read_bytes()
        if sha256_bytes(data) != expected:
            raise ValueError("Source differs from the frozen plan: " + relative)
        snapshots[relative] = data
    return snapshots


def available_memory():
    """MemAvailable includes reclaimable Linux pages; no model/runtime import."""
    path = Path("/proc/meminfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) != 3 or fields[2] != "kB":
                    raise RuntimeError("Unexpected MemAvailable units")
                return int(fields[1]) * 1024, "proc_meminfo_MemAvailable"
    try:
        import psutil
        return int(psutil.virtual_memory().available), "psutil_available"
    except ImportError as exc:
        raise RuntimeError("Cannot establish available RAM; refusing numerical work") from exc


def memory_guard(phase, observations):
    count, method = available_memory()
    if count < 0:
        raise RuntimeError("Invalid available-memory measurement")
    record = {
        "phase": phase, "at_utc": utc_now(), "available_bytes": count,
        "available_gib": count / (1024 ** 3), "method": method,
    }
    observations.append(record)
    if count < 8 * (1024 ** 3):
        raise RuntimeError("Available RAM is below the frozen 8-GiB margin")
    return record


def atomic_exclusive(path, writer):
    """Commit a complete file without ever replacing an existing destination."""
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)  # Atomic exclusive name creation, same filesystem.
        temp.unlink()
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temp.exists():
            temp.unlink()


def save_bytes(path, data):
    atomic_exclusive(path, lambda handle: handle.write(data))


def save_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    save_bytes(path, data)
    return sha256_bytes(data)


def array_digest(array):
    header = canonical({"dtype": array.dtype.str, "shape": list(array.shape)}).encode()
    return sha256_bytes(header + b"\n" + array.tobytes(order="C"))


def selection_from_counts(train, development, search_sizes, np):
    """Pure selection: this routine cannot access a FINAL array or generator."""
    if train.shape != development.shape or train.ndim != 2:
        raise ValueError("TRAIN and DEV shapes must match")
    candidate_count = train.shape[1]
    if list(search_sizes) != sorted(set(search_sizes)) or not all(
        type(m) is int and 1 <= m <= candidate_count for m in search_sizes
    ):
        raise ValueError("Invalid search sizes")
    # Initial columns are candidate IDs. Stable sorting therefore breaks ties by ID.
    order = np.argsort(-train.astype(np.int32), axis=1, kind="stable")
    winners, chosen = [], []
    for m in search_sizes:
        ids = order[:, :m]
        scores = np.take_along_axis(development, ids, axis=1)
        maximum = scores.max(axis=1)
        winner = np.where(scores == maximum[:, None], ids, candidate_count).min(axis=1)
        winners.append(winner)
        chosen.append(maximum)
    return order, np.stack(winners, axis=1), np.stack(chosen, axis=1)


def analytic_reference(n, m, threshold_count):
    """Exact-integer binomial CDF, with floating-point moment summation; p=1/2."""
    if any(type(x) is not int for x in (n, m, threshold_count)):
        raise ValueError("Analytical parameters must be integers")
    if n < 1 or m < 1 or not 1 <= threshold_count <= n:
        raise ValueError("Analytical parameters outside range")
    denominator = 2 ** n
    cumulative = 0
    first, second = [], []
    threshold_cdf = None
    for k in range(n):
        cumulative += math.comb(n, k)
        cdf = cumulative / denominator
        tail = -math.expm1(m * math.log(cdf)) if cdf < 1.0 else 0.0
        first.append(tail)
        second.append((2 * k + 1) * tail)
        if k == threshold_count - 1:
            threshold_cdf = cdf
    mean = math.fsum(first) / n
    second_moment = math.fsum(second) / (n * n)
    variance = second_moment - mean * mean
    if variance < -1e-12 or threshold_cdf is None:
        raise ArithmeticError("Invalid analytic moment or threshold")
    variance = max(0.0, variance)  # Only suppress floating-point cancellation.
    dev_tail = -math.expm1(m * math.log(threshold_cdf)) if threshold_cdf < 1 else 0.0
    final_tail = 1.0 - threshold_cdf
    return {
        "development_mean": mean,
        "development_variance": variance,
        "development_second_moment": second_moment,
        "final_mean": 0.5,
        "final_variance": 0.25 / n,
        "optimism_mean": mean - 0.5,
        "optimism_variance": variance + 0.25 / n,
        "development_tail_probability": dev_tail,
        "final_tail_probability": final_tail,
    }


def describe(values, np):
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    mcse = sd / math.sqrt(len(values))
    return {
        "mean": mean, "sample_standard_deviation": sd, "monte_carlo_standard_error": mcse,
        "mean_monte_carlo_95_normal_interval": [mean - 1.96 * mcse, mean + 1.96 * mcse],
    }


def discrepancy(observed, expected, variance, replicates):
    se = math.sqrt(variance / replicates)
    delta = observed - expected
    # Record a finite JSON value even in a degenerate analytical case.
    z = delta / se if se > 0 else None
    return {
        "observed": observed, "analytic_expected": expected,
        "analytic_monte_carlo_standard_error": se, "difference": delta,
        "standard_errors": z, "flagged": abs(delta) > 6.0 * se,
        "rule": "absolute difference strictly greater than 6 analytic MCSE; no rerun",
    }


def git_head(source_root):
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", value) else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def run(plan_path, output):
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    source_root = Path(__file__).resolve().parents[1]
    plan_path = plan_path.resolve(strict=True)
    plan_bytes = plan_path.read_bytes()
    plan = strict_json(plan_bytes)
    source_snapshots = validate_plan(plan, source_root)
    plan_hash = sha256_bytes(plan_bytes)
    observations = []
    memory_guard("before_numpy_import", observations)
    import numpy as np
    if np.__version__ != REQUIRED_NUMPY_VERSION:
        raise RuntimeError("The frozen simulation requires NumPy " + REQUIRED_NUMPY_VERSION)

    memory_guard("before_output_creation", observations)
    # Existing directories, files, or symlinks fail; there is no resume/overwrite mode.
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    clock_start = time.monotonic()
    events = [{"event": "started", "at_utc": started, "monotonic_ns": time.monotonic_ns()}]
    try:
        def source_still_frozen():
            if plan_path.read_bytes() != plan_bytes:
                raise RuntimeError("Frozen plan changed during execution")
            validate_plan(plan, source_root)

        environment = {
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "device": "cpu", "bit_generator": "PCG64", "numerical_threads": 1,
            "thread_environment": {key: os.environ[key] for key in THREAD_VARIABLES},
            "execution_git_head": git_head(source_root),
            "plan_sha256": plan_hash, "source_sha256": plan["implementation_sha256"],
            "interpretation": "Public reproducible random streams; no claim of blinded custody.",
        }
        save_bytes(output / "frozen_plan.json", plan_bytes)
        save_bytes(output / "source_snapshot.py", source_snapshots[SCRIPT_PATH])
        save_bytes(output / "methodological_evidence.md", source_snapshots[NOTE_PATH])
        environment_hash = save_json(output / "environment.json", environment)
        settings = EXPECTED_SIMULATION
        r, k, n = settings["replicates"], settings["candidate_count"], settings["examples_per_partition"]
        sizes = settings["search_sizes"]
        memory_guard("before_training_and_development_draws", observations)
        train_rng = np.random.Generator(np.random.PCG64(settings["seeds"]["train"]))
        development_rng = np.random.Generator(np.random.PCG64(settings["seeds"]["development"]))
        train = train_rng.binomial(n, 0.5, size=(r, k)).astype(np.uint16)
        development = development_rng.binomial(n, 0.5, size=(r, k)).astype(np.uint16)
        order, winner, selected_development = selection_from_counts(train, development, sizes, np)
        order, winner = order.astype(np.uint8), winner.astype(np.uint8)
        if not np.all(np.diff(selected_development.astype(np.int32), axis=1) >= 0):
            raise ArithmeticError("Nested development maxima are not monotone")
        selection_path = output / "selection_counts.npz"
        atomic_exclusive(selection_path, lambda handle: np.savez_compressed(
            handle, train_counts=train, development_counts=development,
            train_order=order, winner_ids=winner,
            selected_development_counts=selected_development,
            search_sizes=np.asarray(sizes, dtype=np.uint16),
        ))
        selection_hash = sha256_bytes(selection_path.read_bytes())
        receipt = {
            "schema_version": 1, "item": 7, "event": "selection_frozen_before_final_generation",
            "at_utc": utc_now(), "monotonic_ns": time.monotonic_ns(),
            "replicates": r, "search_sizes": sizes, "winner_ids_sha256": array_digest(winner),
            "selection_artifact": {"path": selection_path.name, "sha256": selection_hash},
            "plan_sha256": plan_hash, "source_sha256": plan["implementation_sha256"],
            "environment_sha256": environment_hash,
            "final_rng_constructed": False, "final_draws_completed": False,
        }
        receipt_hash = save_json(output / "selection_receipt.json", receipt)
        if sha256_bytes((output / "selection_receipt.json").read_bytes()) != receipt_hash:
            raise RuntimeError("Selection receipt read-back failed")
        events.append({"event": "selection_receipt_persisted", "at_utc": utc_now(),
                       "monotonic_ns": time.monotonic_ns(), "sha256": receipt_hash})
        source_still_frozen()
        memory_guard("before_final_rng_construction", observations)
        final_start = {"event": "final_generation_started", "at_utc": utc_now(),
                       "monotonic_ns": time.monotonic_ns(), "selection_receipt_sha256": receipt_hash}
        if final_start["monotonic_ns"] <= events[-1]["monotonic_ns"]:
            raise RuntimeError("Final generation did not follow receipt persistence")
        events.append(final_start)
        # The FINAL generator is first constructed here, after the durable receipt.
        final_rng = np.random.Generator(np.random.PCG64(settings["seeds"]["final"]))
        final_counts = final_rng.binomial(n, 0.5, size=(r, k)).astype(np.uint16)
        selected_final = np.take_along_axis(final_counts, winner, axis=1)
        if array_digest(winner) != receipt["winner_ids_sha256"]:
            raise RuntimeError("Winner identities changed after final access")
        atomic_exclusive(output / "final_counts.npz", lambda handle: np.savez_compressed(
            handle, final_counts=final_counts, selected_final_counts=selected_final,
            winner_ids=winner, search_sizes=np.asarray(sizes, dtype=np.uint16),
        ))
        events.append({"event": "final_generation_completed", "at_utc": utc_now(),
                       "monotonic_ns": time.monotonic_ns()})
        memory_guard("before_analytic_summary", observations)
        summaries, flagged = [], []
        for column, m in enumerate(sizes):
            dev_score = selected_development[:, column].astype(np.float64) / n
            final_score = selected_final[:, column].astype(np.float64) / n
            optimism = dev_score - final_score
            analytic = analytic_reference(n, m, settings["tail_count"])
            descriptive = {
                "development": describe(dev_score, np), "final": describe(final_score, np),
                "paired_optimism": describe(optimism, np),
            }
            checks = {}
            for name, values, prefix in (
                ("development_mean", dev_score, "development"),
                ("final_mean", final_score, "final"),
                ("paired_optimism_mean", optimism, "optimism"),
            ):
                checks[name] = discrepancy(
                    float(values.mean()), analytic[prefix + "_mean"],
                    analytic[prefix + "_variance"], r,
                )
            tails = {}
            for name, values in (("development", selected_development[:, column]),
                                 ("final", selected_final[:, column])):
                probability = analytic[name + "_tail_probability"]
                count = int(np.count_nonzero(values >= settings["tail_count"]))
                observed_probability = count / r
                observed_sd = math.sqrt(r * observed_probability * (1 - observed_probability) / (r - 1))
                tails[name] = {"count": count, "replicates": r, "fraction": count / r,
                               "analytic_probability": probability,
                               "observed_sample_standard_deviation": observed_sd,
                               "observed_monte_carlo_standard_error": observed_sd / math.sqrt(r)}
                checks[name + "_tail"] = discrepancy(count / r, probability,
                                                     probability * (1 - probability), r)
            for name, check in checks.items():
                if check["flagged"]:
                    flagged.append({"search_size": m, "check": name, **check})
            summaries.append({"search_size": m, "analytic": analytic,
                              "descriptive": descriptive, "tails": tails,
                              "analytic_discrepancy_checks": checks})
        means = [row["analytic"]["development_mean"] for row in summaries]
        if abs(means[0] - 0.5) > 1e-12 or any(a > b + 1e-12 for a, b in zip(means, means[1:])):
            raise ArithmeticError("Analytic identity failed")

        def write_csv(handle):
            wrapper = io.TextIOWrapper(handle, encoding="utf-8", newline="", write_through=True)
            writer = csv.writer(wrapper)
            writer.writerow(["replicate", "search_size", "winner_id", "train_winner_count",
                             "development_count", "final_count", "examples",
                             "development_accuracy", "final_accuracy", "paired_optimism"])
            for replicate in range(r):
                for column, m in enumerate(sizes):
                    candidate = int(winner[replicate, column])
                    dev_value = int(selected_development[replicate, column])
                    final_value = int(selected_final[replicate, column])
                    writer.writerow([replicate, m, candidate, int(train[replicate, candidate]),
                                     dev_value, final_value, n, dev_value / n, final_value / n,
                                     (dev_value - final_value) / n])
            wrapper.flush()
            wrapper.detach()

        atomic_exclusive(output / "replicates.csv", write_csv)
        source_still_frozen()
        result = {
            "schema_version": 1, "item": 7, "status": "completed",
            "scientific_scope": "illustrative null simulation; no NeuroPixel performance claim",
            "h1_eligible": False, "recipe": settings, "plan_sha256": plan_hash,
            "source_sha256": plan["implementation_sha256"],
            "environment_sha256": environment_hash, "selection_receipt_sha256": receipt_hash,
            "selection_counts_sha256": selection_hash, "winner_ids_sha256": array_digest(winner),
            "summaries": summaries, "analytic_discrepancy_flags": flagged,
            "analytic_status": "flagged_no_rerun" if flagged else "within_prespecified_diagnostic_bands",
            "replicate_count": r, "csv_rows": r * len(sizes),
            "mean_interval_scope": "Monte Carlo precision; normal approximation, not model uncertainty",
            "dependence": "Independent replicates; overlapping shortlists and shared candidate final counts across M.",
            "limitations": [
                "Equal null true accuracies and independent candidate errors are deliberate assumptions.",
                "Candidate selection is adaptive; candidate construction is fixed.",
                "No model fitting, bootstrap, nested-CV run, or reusable-holdout algorithm is performed.",
                "Public seeds/hashes do not establish blindness.",
                "No condition or seed is selected using FINAL; discrepancies never trigger reruns.",
            ],
            "started_at_utc": started, "completed_at_utc": utc_now(),
            "wall_seconds": time.monotonic() - clock_start,
        }
        save_json(output / "analysis.json", result)
        save_json(output / "events.json", events)
        save_json(output / "resource_observations.json", observations)
        artifacts = {}
        for path in sorted(output.iterdir()):
            if path.is_file():
                data = path.read_bytes()
                artifacts[path.name] = {"bytes": len(data), "sha256": sha256_bytes(data)}
        manifest_hash = save_json(output / "artifact_manifest.json", {
            "schema_version": 1, "item": 7, "status": "completed",
            "plan_sha256": plan_hash, "artifacts": artifacts,
            "self_hash_scope": "This manifest does not hash itself.",
        })
        print(json.dumps({"status": "completed", "output_dir": str(output),
                          "analytic_flags": len(flagged), "artifact_manifest_sha256": manifest_hash}))
        return 0
    except Exception as exc:
        failure = {
            "status": "failed", "item": 7, "at_utc": utc_now(), "plan_sha256": plan_hash,
            "error_type": type(exc).__name__, "error": str(exc), "events": events,
            "resource_observations": observations,
            "retry_policy": "Preserve this attempt; no automatic retry or overwrite.",
        }
        try:
            save_json(output / "failure.json", failure)
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args.plan, args.output_dir)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
