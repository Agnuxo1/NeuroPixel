"""Independently audit saved item-6 growth artifacts and tabulate fixed contrasts.

Only JSON, NPZ and file hashes are read. This program does not import the study
modules or Torch, deserialize checkpoints, regenerate datasets, fit a gate, or
execute models. A verified report means the saved evidence is internally
consistent within the explicitly reported verification scope.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
POLICIES = ("fixed_sequential", "adaptive_novelty", "adaptive_resonance")
KINDS = ("snapshots", "duplicate_slots", "single_final", "adaptive_novelty", "adaptive_resonance")
ROUTERS = ("uniform", "random", "scanner", "learned")
SEEDS = (20, 21)
PARAMETERS = 29824
T_975_DF1 = 12.706204736432095
HEX = re.compile(r"[0-9a-f]{64}")
EXPECTED_SOURCE = {
    "neuropixel/model.py": "564045fff799185e15b921873225d344014a4d4bcae6df6deacaf42f6e4bd701",
    "neuropixel/task.py": "e088d35937040f9924264b227e70e0ba9aaf23ede1b1c6bc9f8d6c78821a156f",
    "neuropixel/research/models.py": "646810a33a8ee97a8e66cafb61b418b3fee544394ef71123229e6ab0342c54ee",
    "neuropixel/research/data.py": "44cdf2b9ff492fe9eab115c8092c61c6d38bda5388babaf148380b89f2a7f94b",
    "neuropixel/research/experiment.py": "0a95e445e0654eb3666b3efc12a7bc594a237142dc8a7a2dc9527fac274f41ad",
    "scripts/research_train.py": "e3a8955a8561cf913a99310c4af3916d738b7f2a8290dc756c47df3d6872c92a",
    "docs/research/protocol.json": "41eea054d2be595469a8ffaa32ae4290ac26ebb1c27a20c9a3586dd4c0630290",
}
MISSING_SOURCE = {"scripts/research_queue_worker.py":
                  "73759c106c56775462c19c1cba9dfd82b0acac3c38467edf2e4c0af868c7be73"}


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise VerificationError(f"nonfinite JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def contained(root, relative):
    """Resolve only an explicit relative artifact path inside its declared root."""
    require(isinstance(relative, str) and bool(relative), "empty artifact path")
    path = PurePosixPath(relative.replace("\\", "/"))
    require(not path.is_absolute() and ".." not in path.parts
            and not any(":" in part for part in path.parts), f"unsafe relative path: {relative}")
    result = (Path(root) / Path(*path.parts)).resolve()
    require(result.is_relative_to(Path(root).resolve()), f"artifact escapes root: {relative}")
    return result


def utc(value):
    require(isinstance(value, str), "missing UTC timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "timestamp is not UTC")
    return result


def same(actual, expected, label, tolerance=1e-9):
    """Check exact structure and counts; tolerate only floating arithmetic noise."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{label}: keys differ")
        for key in expected:
            same(actual[key], expected[key], f"{label}.{key}", tolerance)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{label}: list length differs")
        for i, value in enumerate(expected):
            same(actual[i], value, f"{label}[{i}]", tolerance)
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
                f"{label}: numeric value differs")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{label}: value differs")


def array_same(actual, expected, label, tolerance=0):
    require(actual.shape == expected.shape, f"{label}: shape differs")
    if tolerance:
        require(np.all(np.isfinite(actual)) and np.allclose(actual, expected, atol=tolerance, rtol=tolerance),
                f"{label}: values differ")
    else:
        require(np.array_equal(actual, expected), f"{label}: values differ")


def tensor_digest(arrays):
    digest = hashlib.sha256()
    for name in ("canvas", "target", "roles"):
        array = np.ascontiguousarray(arrays[name], dtype="<i8")
        header = json.dumps({"name": name, "shape": list(array.shape), "dtype": "<i8"},
                            sort_keys=True).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def features(canvas):
    n = len(canvas)
    result = np.zeros((n, 40), dtype=np.float64)
    result[np.arange(n)[:, None], canvas.reshape(n, -1)] = 1
    result[:, 0] = 0
    result[:, 35:39] = canvas[:, 7, 6, None] == np.arange(1, 5)[None, :]
    result[:, 39] = 1
    return result


def logsumexp(values, axis):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)), axis=axis)


def counts(correct, roles, topics, nll=None):
    def selected(mask):
        per_role = {}
        for index, role in enumerate(ROLES):
            values = correct[mask & (roles == index)]
            require(len(values) > 0, "a scored topic has no examples for one role")
            per_role[role] = {"n": len(values), "correct": int(values.sum()), "accuracy": float(values.mean())}
        values = correct[mask]
        return {"n": len(values), "correct": int(values.sum()), "accuracy": float(values.mean()),
                "per_role": per_role,
                "macro_all_roles": sum(row["accuracy"] for row in per_role.values()) / 4,
                "macro_agent_patient_accuracy": (per_role["AGENTE"]["accuracy"] + per_role["PACIENTE"]["accuracy"]) / 2}
    result = selected(np.ones(len(correct), dtype=bool))
    result["per_topic"] = {str(i): selected(topics == i) for i in range(3)}
    if nll is not None:
        result["cross_entropy"] = float(np.mean(nll))
    return result


def independent_decision(policy, stage, novelty, resonance, tau):
    require(len(novelty) == len(resonance) and len(novelty) <= stage, "invalid diagnostic expert count")
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
                for v in novelty + resonance), "invalid diagnostic probability")
    k = len(novelty)
    if stage == 0:
        require(k == 0 and tau is None, "stage A must start empty")
        return {"action": "create", "parent": None, "target_index": 0, "scores_used": [], "threshold": None}
    require(k >= 1, "later stage has no retained expert")
    if policy == "fixed_sequential":
        require(k == stage, "fixed snapshot lineage lost a stage")
        return {"action": "create", "parent": k - 1, "target_index": k, "scores_used": [], "threshold": None}
    if policy == "adaptive_novelty":
        values = [round(v, 4) for v in novelty]
        parent = values.index(min(values))
        threshold, create = 0.1, values[parent] > 0.1
    else:
        require(policy == "adaptive_resonance" and isinstance(tau, (int, float))
                and math.isfinite(tau) and 0 <= tau <= 0.75, "invalid fixed resonance threshold")
        values = [round(v, 4) for v in resonance]
        parent = values.index(max(values))
        threshold, create = tau, values[parent] < tau
    return {"action": "create" if create else "update", "parent": parent,
            "target_index": k if create else parent, "scores_used": values, "threshold": threshold}


def route(logp, router, scanner, x, gate=None):
    """Recompute a declared policy from saved expert probabilities, never targets."""
    e, n, _ = logp.shape
    if router == "uniform":
        weights = np.full((n, e), 1 / e)
        picks = np.zeros(n, dtype=np.int64)
    elif router == "random":
        picks = np.random.default_rng(62013).integers(e, size=n, dtype=np.int64)
        weights = np.eye(e)[picks]
    elif router == "scanner":
        picks = scanner.argmax(axis=0)
        weights = np.eye(e)[picks]
    else:
        require(router == "learned" and gate is not None, "invalid routing policy")
        logits = x @ gate
        weights = np.exp(logits - logsumexp(logits, 1)[:, None])
        picks = weights.argmax(1)
    if router in ("random", "scanner"):
        mixed = logp[picks, np.arange(n), :]
    else:
        with np.errstate(divide="ignore"):
            mixed = logsumexp(logp.transpose(1, 0, 2) + np.log(weights)[:, :, None], 1)
    return mixed, weights, picks


def diagnostic_statistics(own, scanner, canvas):
    """Recount the already saved probabilities using the frozen float32 policy."""
    require(own.dtype == np.float32 and scanner.dtype == np.float32
            and own.shape == canvas.shape and scanner.shape == (len(canvas),),
            "diagnostic probability shape/dtype differs")
    require(np.isfinite(own).all() and np.isfinite(scanner).all()
            and np.all((own >= 0) & (own <= 1)) and np.all((scanner >= 0) & (scanner <= 1)),
            "invalid diagnostic probability")
    occupied = canvas != 0
    counts = occupied.sum((1, 2))
    require(np.all(counts > 0), "empty diagnostic input")
    recalculated = (own * occupied.astype(np.float32)).sum((1, 2), dtype=np.float32) / counts.astype(np.float32)
    require(np.allclose(scanner, recalculated, atol=1e-6, rtol=0),
            "scanner differs from occupied-cell probability mean")
    hits = int(np.sum((own < np.float32(.5)) & occupied))
    novelty = float(np.float32(hits) / np.float32(occupied.sum()))
    return {"novelty": novelty, "mean_resonance_recount": float(scanner.mean(dtype=np.float32)),
            "occupied_cells": int(occupied.sum()), "novel_cells": hits,
            "maximum_scanner_recount_difference": float(np.abs(scanner - recalculated).max())}


def paired_summary(values):
    require(len(values) == 2 and all(math.isfinite(v) for v in values), "exactly two finite seed effects required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = T_975_DF1 * sd / math.sqrt(2)
    return {"n_initializations": 2, "raw_seed_effects": dict(zip(map(str, SEEDS), values)),
            "mean": mean, "sample_standard_deviation": sd,
            "student_t_95_df1_unclipped": [mean - half, mean + half],
            "interval_scope": "Exploratory two-initialization interval conditional on one split and these fixed examples; normal seed-effect assumption is untestable with n=2.",
            "decision": "descriptive only; no H1 decision or multiple-comparison selection"}


class Audit:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.hashes = {}
        self.source = self.controller = None
        self.final_phase_allowed = False

    def path(self, relative):
        return contained(self.root, relative)

    def json(self, relative):
        if relative == "growth_records.json" or relative.startswith("evaluations/"):
            require(self.final_phase_allowed, "final outcome records remain closed until development verification passes")
        path = self.path(relative)
        self.hashes[relative] = sha256(path)
        return read_json(path)

    def artifact(self, record, expected=None):
        require(isinstance(record, dict) and HEX.fullmatch(record.get("sha256", "")), "artifact lacks a SHA-256")
        relative = record.get("path")
        if expected is not None:
            require(relative == expected, f"artifact location differs: {relative}")
        path = self.path(relative)
        actual = sha256(path)
        require(actual == record["sha256"], f"artifact hash differs: {relative}")
        require(path.stat().st_size == record.get("bytes"), f"artifact byte count differs: {relative}")
        self.hashes[relative] = actual
        return path

    def npz(self, record, keys, expected=None):
        relative = record.get("path", "")
        if (relative.startswith("evaluations/") or relative.endswith("/final_experts.npz")
                or "/split0_test_" in relative):
            require(self.final_phase_allowed, "final arrays remain closed until development verification passes")
        path = self.artifact(record, expected)
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == set(keys), f"array keys differ: {record['path']}")
            values = {key: saved[key] for key in keys}
        require(all(value.dtype.kind in "biuf" for value in values.values()), "unsupported NPZ dtype")
        require(all(np.isfinite(value).all() for value in values.values()), "nonfinite NPZ value")
        return values

    def provenance(self, record):
        require(record.get("source") == self.source and record.get("controller_source") == self.controller,
                "artifact source differs from execution manifest")

    def checkpoint(self, record):
        require(record.get("parameter_count") == PARAMETERS and HEX.fullmatch(record.get("tensor_sha256", "")),
                "checkpoint parameter count or tensor fingerprint differs")
        self.artifact(record)

    def dataset(self, record, topic, split, partitions, diagnostic=False):
        n, seed = (512, 62010 + 100 * topic) if diagnostic else (
            (512, 62011 + 100 * topic) if split == "train" else (1024, 62012 + 100 * topic))
        require(record.get("topic") == topic and record.get("split") == split and record.get("n") == n
                and record.get("sampling_seed") == seed, "dataset recipe differs")
        expected = f"datasets/topic{topic}/" + ("diagnostic_train.npz" if diagnostic else f"split0_{split}_n{n}_seed{seed}.npz")
        arrays = self.npz(record, ("canvas",) if diagnostic else ("canvas", "target", "roles"), expected)
        canvas = arrays["canvas"]
        require(canvas.shape == (n, 8, 8) and canvas.dtype.kind in "iu"
                and np.all((canvas >= 0) & (canvas < 35)), "invalid saved canvas")
        query = canvas[:, 7, 6] - 1
        require(np.all((query >= 0) & (query < 4)) and np.all(canvas[:, 7, 7] == 0), "invalid query/output cells")
        require(np.all(np.count_nonzero(canvas, axis=(1, 2)) == 9), "unexpected occupied cell count")
        fillers = np.empty((n, 4), dtype=np.int64)
        pair_rows = np.empty((n, 4), dtype=np.int64)
        for role in range(4):
            hits = canvas[:, :7, :] == role + 1
            require(np.all(hits.sum((1, 2)) == 1), "missing or repeated visible role")
            batch, row, column = np.nonzero(hits)
            require(np.all(column < 7), "role has no right-hand filler")
            fillers[batch, role] = canvas[batch, row, column + 1]
            pair_rows[batch, role] = row
        require(all(len(set(row)) == 4 for row in pair_rows), "two role pairs share a row")
        require(np.all((fillers[:, [0, 2]] >= 5 + 4 * topic) & (fillers[:, [0, 2]] < 9 + 4 * topic))
                and np.all(fillers[:, 0] != fillers[:, 2])
                and np.all((fillers[:, 1] >= 17) & (fillers[:, 1] < 27))
                and np.all((fillers[:, 3] >= 27) & (fillers[:, 3] < 35)), "filler vocabulary/topic differs")
        triples = np.stack((fillers[:, 0] - 5, fillers[:, 1] - 17, fillers[:, 2] - 5), 1)
        membership = {tuple(row) for row in partitions[topic]["triples"][split]}
        require(all(tuple(row) in membership for row in triples), "dataset contains a triple outside its split")
        if diagnostic:
            require(record.get("targets_retained") is False, "diagnostic unexpectedly retained labels")
            expected_counts = {name: int(np.sum(query == i)) for i, name in enumerate(ROLES)}
            require(record.get("query_role_counts") == expected_counts, "diagnostic query counts differ")
            require(record.get("role_balance") == "natural sampling; no forced role balance",
                    "diagnostic balance description differs")
        else:
            require(record.get("split_seed") == 0, "dataset split seed differs")
            array_same(arrays["roles"], query, "dataset roles")
            array_same(arrays["target"], fillers[np.arange(n), query], "visible-pair target")
            require(np.array_equal(np.bincount(query, minlength=4), np.full(4, n // 4)), "frozen dataset role counts differ")
            require(tensor_digest(arrays) == record.get("content_sha256"), "canonical dataset content hash differs")
        return arrays


    def diagnostic(self, descriptor, canvas, input_record, checkpoint, expected_path, post_stage_a=False):
        require(descriptor["input_dataset"] == input_record, "diagnostic input reference differs")
        checkpoint_key = "checkpoint_after_stage" if post_stage_a else "checkpoint_before_intervention"
        require(descriptor[checkpoint_key] == checkpoint, "diagnostic checkpoint differs from retained lineage")
        require(descriptor["recount_atol"] == 1e-6 and descriptor["recount_rtol"] == 0,
                "diagnostic recount tolerance changed")
        arrays = self.npz(descriptor, ("own_token_probability", "scanner"), expected_path)
        recounted = diagnostic_statistics(arrays["own_token_probability"], arrays["scanner"], canvas)
        require(descriptor["novelty"] == recounted["novelty"], "novelty differs from float32 occupied-cell fraction")
        require(abs(descriptor["mean_resonance"] - recounted["mean_resonance_recount"]) <= 1e-6,
                "resonance differs from saved scanner reduction")
        return recounted

    def expert_arrays(self, record, bank, target, roles, topics, phase):
        expected = f"banks/{bank['bank_id']}/" + ("gate_fit_experts.npz" if phase == "fit" else "final_experts.npz")
        arrays = self.npz(record, ("log_probabilities", "scanner", "target", "role", "topic"), expected)
        for key, values in (("target", target), ("role", roles), ("topic", topics)):
            array_same(arrays[key], values, f"expert {phase} {key}")
        logp = np.asarray(arrays["log_probabilities"], dtype=np.float64)
        scores = np.asarray(arrays["scanner"], dtype=np.float64)
        require(logp.shape == (len(bank["experts"]), len(target), 35), "expert output shape differs")
        require(scores.shape == logp.shape[:2] and np.all((scores >= 0) & (scores <= 1)), "scanner shape/range differs")
        require(np.allclose(logsumexp(logp, 2), 0, atol=1e-5, rtol=0), "expert log probabilities are not normalized")
        return logp, scores


def verify_partitions(partitions):
    require(len(partitions) == 3 and [p["topic"] for p in partitions] == [0, 1, 2], "topic partition inventory differs")
    for topic, record in enumerate(partitions):
        require(record["split_seed"] == 0 and set(record["triples"]) == {"train", "validation", "test"},
                "partition recipe differs")
        pools = record["triples"]
        digest = hashlib.sha256(json.dumps(pools, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(digest == record["triples_sha256"], "partition membership hash differs")
        sets = {}
        for split, rows in pools.items():
            require(all(len(row) == 3 and all(type(v) is int for v in row) for row in rows), "invalid partition triple")
            sets[split] = set(map(tuple, rows))
            require(len(sets[split]) == len(rows) == record["counts"][split], "partition count/duplicates differ")
        require(not any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))),
                "partition overlap")
        expected = {(a, b, c) for a in range(4 * topic, 4 * topic + 4) for b in range(10)
                    for c in range(4 * topic, 4 * topic + 4) if a != c}
        require(set.union(*sets.values()) == expected, "partition does not cover the declared topic")


def audit_growth(directory, source_root=ROOT):
    source_root = Path(source_root).resolve()
    audit = Audit(directory)
    manifest, gate, bundle = (audit.json(name) for name in
                              ("run_manifest.json", "evaluation_gate.json", "training_bundle.json"))
    events = audit.json("events.json")
    audit.source, audit.controller = manifest["source"], manifest["controller_source"]
    audit.provenance(gate)
    audit.provenance(bundle)
    require(audit.source.get("sha256") == EXPECTED_SOURCE
            and audit.source.get("unavailable_expected_sha256") == MISSING_SOURCE
            and audit.source.get("historical_files_verified") == 7
            and audit.source.get("historical_files_expected") == 8, "historical source recovery scope differs")
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", audit.source.get("git_commit", "")), "execution HEAD missing")
    require(not (source_root / "source_snapshot.json").exists(), "analysis checkout contains stale source snapshot")
    require(all(HEX.fullmatch(value) for value in audit.controller.values()), "invalid controller digest")
    require(all(relative not in EXPECTED_SOURCE or digest == EXPECTED_SOURCE[relative]
                for relative, digest in audit.controller.items()), "conflicting historical/controller hashes")
    for relative, digest in {**EXPECTED_SOURCE, **audit.controller}.items():
        require(sha256(contained(source_root, relative)) == digest, f"analysis checkout source differs: {relative}")
    require(not any(contained(source_root, p).exists() for p in MISSING_SOURCE), "a declared unavailable source was silently replaced")
    recovery = audit.source["recovery_manifest"]
    require(recovery["path"] == "docs/research/06_recovery_manifest.json"
            and sha256(contained(source_root, recovery["path"])) == recovery["sha256"], "recovery manifest hash differs")
    recovered = read_json(contained(source_root, recovery["path"]))
    require(recovered["available_historical_sha256"] == EXPECTED_SOURCE
            and recovered["unavailable_historical_sha256"] == MISSING_SOURCE, "recovery manifest scope differs")
    plan = read_json(source_root / "docs/research/06_execution_plan.json")
    require(plan.get("status") == "frozen" and plan.get("item") == 6, "execution plan is not frozen")
    expected_controller = dict(plan["implementation_sha256"])
    expected_controller["docs/research/06_execution_plan.json"] = sha256(source_root / "docs/research/06_execution_plan.json")
    require(audit.controller == expected_controller, "runtime source map differs from frozen plan")
    spec = plan["growth"]
    require(manifest["item"] == 6 and manifest["panel"] == "growth" and manifest["specification"] == spec,
            "growth manifest scope/specification differs")
    require(all(spec[key] == value for key, value in {
        "planned_trajectories": 6, "planned_training_stages": 18, "planned_optimizer_updates": 9216,
        "planned_gate_fits": 8, "planned_routing_evaluations": 34, "planned_preallocation_checks": 2}.items()),
        "frozen growth inventory differs")
    require(spec["h1_eligible"] is False, "growth cannot reopen H1")
    require(spec["diagnostic_recount_atol"] == 1e-6
            and spec["diagnostic_role_balance"] == "natural sampling; no forced role balance",
            "diagnostic measurement contract differs")
    require(gate["completed_trajectories"] == 6 and gate["completed_training_stages"] == 18
            and gate["completed_gate_fits"] == 8, "incomplete final authorization gate")
    development = gate["development_artifacts_sha256"]
    require(isinstance(development, dict) and len(development) > 0, "empty development artifact inventory")
    for relative, digest in development.items():
        require(not (relative.startswith("evaluations/") or relative.endswith("/final_experts.npz")
                     or "/split0_test_" in relative or relative == "growth_records.json"),
                "final-phase artifact already existed in the development authorization inventory")
        require(HEX.fullmatch(digest) and sha256(audit.path(relative)) == digest,
                f"development artifact changed: {relative}")
        audit.hashes[relative] = digest
    for required in ("run_manifest.json", "training_bundle.json", "partitions.json"):
        require(required in development, f"development gate omitted {required}")
    timestamps = [utc(row["at_utc"]) for row in events]
    require(timestamps == sorted(timestamps), "event timestamps are not monotonic")
    names = [row["event"] for row in events]
    require(names[0] == "growth_started" and names[-1] == "growth_completed"
            and "growth_failed" not in names, "growth event transcript is incomplete")
    require(names.count("growth_final_evaluation_authorized") == 1
            and names.count("growth_final_datasets_opened") == 1
            and names.count("growth_stage_completed") == 18
            and names.count("growth_gate_fitted") == 8
            and names.count("growth_final_saved") == 34, "event inventory differs")
    authorization = names.index("growth_final_evaluation_authorized")
    opening = names.index("growth_final_datasets_opened")
    require(authorization < opening and utc(gate["at_utc"]) == timestamps[authorization],
            "final dataset was opened before authorization")
    require({k: v for k, v in events[authorization].items() if k != "event"} == gate, "authorization event differs from gate")
    require(all(i < authorization for i, name in enumerate(names) if name in ("growth_stage_completed", "growth_gate_fitted"))
            and all(i > opening for i, name in enumerate(names) if name == "growth_final_saved"),
            "development/final event ordering differs")
    partitions = audit.json("partitions.json")
    verify_partitions(partitions)
    development_event = [row for row in events if row["event"] == "development_datasets_saved"]
    require(len(development_event) == 1, "development dataset event must be unique")
    diagnostic_records, fit_records = development_event[0]["diagnostic"], development_event[0]["gate_fit"]
    require(len(diagnostic_records) == len(fit_records) == 3, "development dataset inventory differs")
    require(utc(development_event[0]["at_utc"]) <= timestamps[authorization], "development dataset event occurs too late")
    diagnostics, fit_arrays = [], []
    for topic in range(3):
        diagnostics.append(audit.dataset(diagnostic_records[topic], topic, "train", partitions, True))
        fit_arrays.append(audit.dataset(fit_records[topic], topic, "train", partitions))
    expected_configs = {row["trajectory_id"]: row for row in spec["trajectories"]}
    require(len(expected_configs) == 6 and set(expected_configs) ==
            {f"{policy}_i{seed}" for seed in SEEDS for policy in POLICIES}, "trajectory IDs differ")
    trajectories = bundle["trajectories"]
    require(len(trajectories) == 6, "trajectory bundle differs")
    require({p.parent.name for p in (audit.root / "trajectories").glob("*/training.json")} == set(expected_configs),
            "extra or missing trajectory directories")
    trajectory_map, stream_hashes, stage_a, decisions, diagnostic_recounts = {}, {}, {}, [], []
    for trajectory in trajectories:
        config = trajectory["config"]
        ident, seed, policy = config["trajectory_id"], config["init_seed"], config["policy"]
        require(ident not in trajectory_map and config == expected_configs[ident], "duplicate/changed trajectory")
        require(trajectory["status"] == "completed" and trajectory["final_test_accessed"] is False, "incomplete training trajectory")
        audit.provenance(trajectory)
        require(audit.json(f"trajectories/{ident}/training.json") == trajectory, "trajectory file differs")
        require(len(trajectory["stages"]) == 3, "incomplete training stages")
        references, previous_tau = [], None
        for stage, record in enumerate(trajectory["stages"]):
            require(record["stage"] == record["topic"] == stage and record["final_test_accessed"] is False,
                    "stage identity/final-data flag differs")
            require(record["diagnostic_dataset"] == diagnostic_records[stage], "diagnostic reference differs")
            require(audit.json(f"trajectories/{ident}/stage{stage}/training.json") == record, "stage file differs")
            require(len(record["raw_novelty"]) == len(references), "stage diagnostic expert count differs")
            descriptors = record["diagnostic_expert_outputs"]
            require(len(descriptors) == len(references), "diagnostic probability artifact count differs")
            for expert, descriptor in enumerate(descriptors):
                require(descriptor["expert_index"] == expert, "diagnostic expert index differs")
                recount = audit.diagnostic(
                    descriptor, diagnostics[stage]["canvas"], diagnostic_records[stage], references[expert],
                    f"trajectories/{ident}/stage{stage}/diagnostic_expert{expert}.npz")
                require(record["raw_novelty"][expert] == descriptor["novelty"]
                        and record["raw_resonance"][expert] == descriptor["mean_resonance"],
                        "heuristic raw scalar differs from probability descriptor")
                diagnostic_recounts.append({"trajectory": ident, "stage": stage, "expert": expert,
                                            "phase": "before_intervention", **recount})
            decision = independent_decision(policy, stage, record["raw_novelty"], record["raw_resonance"], previous_tau)
            same(record["decision"], decision, "heuristic decision")
            if policy == "adaptive_resonance":
                tau = record["tau_after_stage"]
                require(isinstance(tau, (float, int)) and math.isfinite(tau) and 0 <= tau <= .75, "invalid post-stage threshold")
                if stage:
                    require(tau == previous_tau, "resonance threshold changed after stage A")
                else:
                    descriptor = record["post_stage_a_diagnostic"]
                    recount = audit.diagnostic(
                        descriptor, diagnostics[0]["canvas"], diagnostic_records[0], record["checkpoint"],
                        f"trajectories/{ident}/stage0/post_stage_a_diagnostic.npz", post_stage_a=True)
                    require(record["post_stage_a_mean_resonance"] == descriptor["mean_resonance"],
                            "post-stage-A reference mean differs")
                    same(tau, .75 * record["post_stage_a_mean_resonance"], "stage-A threshold multiplier", 1e-12)
                    diagnostic_recounts.append({"trajectory": ident, "stage": 0, "expert": 0,
                                                "phase": "after_stage_A", **recount})
                previous_tau = tau
            else:
                require(record["tau_after_stage"] is None, "unexpected resonance threshold")
            if stage != 0 or policy != "adaptive_resonance":
                require(record["post_stage_a_diagnostic"] is None and record["post_stage_a_mean_resonance"] is None,
                        "unexpected post-stage-A measurement")
            checkpoint = record["checkpoint"]
            require(checkpoint["path"] == f"trajectories/{ident}/stage{stage}/weights.pt", "stage checkpoint location differs")
            audit.checkpoint(checkpoint)
            if decision["action"] == "create":
                require(decision["target_index"] == len(references), "created expert slot differs")
                references.append(checkpoint)
            else:
                references[decision["target_index"]] = checkpoint
            require(record["experts_after_stage"] == len(references), "retained expert count differs")
            require(isinstance(record["training_seconds"], (int, float)) and math.isfinite(record["training_seconds"])
                    and record["training_seconds"] >= 0, "invalid stage training duration")
            require(record["optimizer_updates"] == 512 and record["train_sample_seed"] == 9002 + 100 * stage
                    and record["firing_seed"] == 9001 + 100 * stage, "stage training recipe differs")
            require(HEX.fullmatch(record["train_stream_sha256"]), "missing training stream digest")
            stream_hashes.setdefault(stage, record["train_stream_sha256"])
            require(stream_hashes[stage] == record["train_stream_sha256"], "training examples differ across trajectories")
            if stage == 0:
                stage_a.setdefault(seed, checkpoint["tensor_sha256"])
                require(stage_a[seed] == checkpoint["tensor_sha256"], "shared stage-A tensor fingerprint differs")
            curve = record["curve"]
            require([row["update"] for row in curve] == [128, 256, 384, 512], "stage objective trace inventory differs")
            for row in curve:
                require(all(math.isfinite(row[k]) and row[k] >= 0 for k in
                            ("answer_cross_entropy", "school_cross_entropy", "objective", "gradient_norm_before_clip")),
                        "invalid stage objective trace")
                same(row["objective"], row["answer_cross_entropy"] + .3 * row["school_cross_entropy"],
                     "stage objective", tolerance=2e-6)
            matching = [row for row in events if row["event"] == "growth_stage_completed"
                        and row["trajectory"] == ident and row["stage"] == stage]
            require(len(matching) == 1 and matching[0]["checkpoint"] == checkpoint
                    and matching[0]["decision"] == decision and matching[0]["experts"] == len(references), "stage event differs")
            margin = None
            if stage and policy != "fixed_sequential":
                parent = decision["parent"]
                raw = (record["raw_novelty"] if policy == "adaptive_novelty" else record["raw_resonance"])[parent]
                rounded, threshold = decision["scores_used"][parent], decision["threshold"]
                sign = 1 if policy == "adaptive_novelty" else -1
                raw_margin, rounded_margin = sign * (raw - threshold), sign * (rounded - threshold)
                margin = {"selected_expert_raw_score": raw, "selected_expert_rounded_score": rounded,
                          "threshold": threshold, "raw_creation_margin": raw_margin,
                          "rounded_creation_margin": rounded_margin,
                          "strictly_positive_means_create": True,
                          "rounding_changes_comparison_for_selected_expert": (raw_margin > 0) != (rounded_margin > 0)}
                require((rounded_margin > 0) == (decision["action"] == "create"), "heuristic margin contradicts decision")
            decisions.append({"threshold_margin": margin, "trajectory": ident, "seed": seed, "policy": policy, "stage": stage,
                              "raw_novelty": record["raw_novelty"], "raw_resonance": record["raw_resonance"],
                              "decision": decision, "tau": previous_tau, "experts_after": len(references)})
        require(trajectory["final_experts"] == references and trajectory["expert_count"] == len(references)
                and trajectory["optimizer_updates"] == 1536, "final retained lineage differs")
        same(trajectory["training_seconds"], sum(row["training_seconds"] for row in trajectory["stages"]), "trajectory time", 1e-8)
        trajectory_map[ident] = trajectory
    banks = bundle["banks"]
    require(len(banks) == 10, "bank inventory differs")
    bank_map, bank_accounting, bundle_map = {}, {}, {row["bank_id"]: row for row in banks}
    require(set(bundle_map) == {f"{kind}_i{seed}" for kind in KINDS for seed in SEEDS}, "bank IDs differ")
    for bank in banks:
        ident, kind, seed = bank["bank_id"], bank["kind"], bank["init_seed"]
        require(ident not in bank_map and ident == f"{kind}_i{seed}", "duplicate/invalid bank ID")
        base = bundle_map[ident]
        require(all(bank[key] == value for key, value in base.items()), "bank changed after authorization")
        fixed = trajectory_map[f"fixed_sequential_i{seed}"]
        checkpoints = [row["checkpoint"] for row in fixed["stages"]]
        expected = ([checkpoints[i] for i in {"snapshots": (0, 1, 2), "duplicate_slots": (0, 1, 1), "single_final": (2,)}[kind]]
                    if kind in ("snapshots", "duplicate_slots", "single_final")
                    else trajectory_map[f"{kind}_i{seed}"]["final_experts"])
        require(bank["experts"] == expected and bank["fit_gate"] == (kind != "single_final"), "bank composition differs")
        unique = {row["sha256"]: row for row in expected}
        bank_accounting[ident] = {"logical_expert_slots": len(expected), "nominal_expert_parameters": len(expected) * PARAMETERS,
                                  "unique_checkpoint_files": len(unique),
                                  "unique_checkpoint_file_bytes": sum(r["bytes"] for r in unique.values()),
                                  "unique_checkpoint_tensor_parameters": len(unique) * PARAMETERS}
        bank_map[ident] = bank
    preallocation = bundle["preallocation_checks"]
    require(len(preallocation) == 2, "preallocation inventory differs")
    require({row["init_seed"] for row in preallocation} == set(SEEDS), "preallocation seeds differ")
    for row in preallocation:
        refs = bank_map[f"snapshots_i{row['init_seed']}"]["experts"]
        require(row["status"] == "verified" and row["kind"] == "construction_equivalence_not_replication"
                and row["ordered_tensor_sha256"] == [ref["tensor_sha256"] for ref in refs]
                and row["same_physical_storage_bytes_claimed"] is False, "preallocation evidence differs")
    def concatenate(arrays):
        return {name: np.concatenate([row[name] for row in arrays]) for name in ("canvas", "target", "roles")}
    fit = concatenate(fit_arrays)
    fit_topics = np.repeat(np.arange(3), 512)
    fit_features = features(fit["canvas"])
    gates, gate_rows, fit_seen_outputs = {}, [], {}
    for ident, bank in bank_map.items():
        if not bank["fit_gate"]:
            continue
        descriptor = bank["gate"]
        require(audit.json(f"banks/{ident}/gate.json") == descriptor and descriptor["final_test_accessed"] is False,
                "gate fitting descriptor differs")
        require(descriptor["data"] == fit_records, "gate used undeclared development data")
        require(descriptor["updates"] == 128 and descriptor["learning_rate"] == .1 and descriptor["l2"] == .0001
                and descriptor["parameter_count"] == 40 * len(bank["experts"]), "gate fitting recipe differs")
        logp, _scanner = audit.expert_arrays(descriptor["expert_outputs"], bank, fit["target"], fit["roles"], fit_topics, "fit")
        for index, reference in enumerate(bank["experts"]):
            fingerprint = reference["tensor_sha256"]
            if fingerprint in fit_seen_outputs:
                old_logp, old_scanner = fit_seen_outputs[fingerprint]
                array_same(logp[index], old_logp, "shared gate-fitting expert output")
                array_same(_scanner[index], old_scanner, "shared gate-fitting scanner output")
            else:
                fit_seen_outputs[fingerprint] = (logp[index], _scanner[index])
        weights = audit.npz(descriptor["weights"], ("weights",), f"banks/{ident}/gate_weights.npz")["weights"]
        require(weights.shape == (40, len(bank["experts"])) and weights.dtype == np.float64, "gate weight shape/dtype differs")
        trace = descriptor["trace"]
        require([row["update"] for row in trace] == [0, 32, 64, 96, 128], "gate objective trace inventory differs")
        require(all(math.isfinite(row[key]) and row[key] >= 0 for row in trace for key in ("objective", "cross_entropy")), "invalid gate trace")
        uniform, _, _ = route(logp, "uniform", _scanner, fit_features)
        learned, _, _ = route(logp, "learned", _scanner, fit_features, weights)
        target = fit["target"]
        initial_nll = float(-uniform[np.arange(len(target)), target].mean())
        final_nll = float(-learned[np.arange(len(target)), target].mean())
        same(trace[0]["objective"], initial_nll, "initial gate objective")
        same(trace[0]["cross_entropy"], initial_nll, "initial gate NLL")
        same(trace[-1]["cross_entropy"], final_nll, "final gate NLL")
        same(trace[-1]["objective"], final_nll + .00005 * float(np.square(weights).sum()), "final gate objective")
        matching = [row for row in events if row["event"] == "growth_gate_fitted" and row["bank"] == ident]
        require(len(matching) == 1 and matching[0]["weights"] == descriptor["weights"], "gate fitting event differs")
        gates[ident] = weights
        gate_rows.append({"bank_id": ident, "parameters": int(weights.size), "initial_nll": initial_nll,
                          "final_nll": final_nll, "trace": trace})
    require(len(gates) == 8, "gate fitting inventory differs")
    # Before opening final arrays, require every development reference we read in the
    # frozen hash map. The map itself is validated before any NPZ is opened above.
    required_development = set(audit.hashes) - {"growth_records.json", "events.json", "evaluation_gate.json"}
    require(required_development <= set(development), "authorization gate omitted a development artifact")
    # Only now open the aggregate result containing final outcome metrics.
    audit.final_phase_allowed = True
    result = audit.json("growth_records.json")
    require(result.get("status") == "completed" and result.get("item") == 6 and result.get("panel") == "growth",
            "growth result is not completed item 6")
    audit.provenance(result)
    require(events == result["events"] and result["specification"] == spec, "result transcript/specification differs")
    require(manifest["environment"] == result["environment"], "execution environment differs")
    require(result["h1_eligible"] is False and result["inference_latency_seconds"] is None
            and result["physical_energy_joules"] is None, "unexpected H1/physical-cost claim")
    require(result["partitions"] == partitions and result["trajectories"] == trajectories
            and result["preallocation_checks"] == preallocation, "result development evidence differs")
    require(result["diagnostic_datasets"] == diagnostic_records and result["gate_fit_datasets"] == fit_records,
            "result development dataset records differ")
    require(len(result["banks"]) == 10 and {b["bank_id"] for b in result["banks"]} == set(bank_map),
            "final bank inventory differs")
    banks = result["banks"]
    for bank in banks:
        ident = bank["bank_id"]
        require(all(bank[key] == value for key, value in bank_map[ident].items()), "bank changed after authorization")
        require(all(bank[key] == value for key, value in bank_accounting[ident].items()), "bank accounting differs")
        bank_map[ident] = bank
    require(len(result["final_datasets"]) == 3 and result["final_datasets"] == events[opening]["datasets"],
            "final dataset transcript differs")
    final_arrays = [audit.dataset(result["final_datasets"][i], i, "test", partitions) for i in range(3)]
    final = concatenate(final_arrays)
    target, roles, topics = final["target"], final["roles"], np.repeat(np.arange(3), 1024)
    x = features(final["canvas"])
    outputs = {}
    for ident, bank in bank_map.items():
        logp, scanner = audit.expert_arrays(bank["final_expert_outputs"], bank, target, roles, topics, "final")
        oracle = counts(np.any(logp.argmax(2) == target[None, :], axis=0), roles, topics)
        same(bank["hard_selection_oracle"], oracle, "hard-selection oracle")
        require(bank["oracle_scope"] == spec["hard_selection_oracle_scope"], "oracle interpretation changed")
        outputs[ident] = (logp, scanner)
    # Cached expert outputs must be identical wherever the same checkpoint is used.
    seen_outputs = {}
    for ident, bank in bank_map.items():
        logp, scanner = outputs[ident]
        for index, reference in enumerate(bank["experts"]):
            key = reference["tensor_sha256"]
            if key in seen_outputs:
                old_logp, old_scanner = seen_outputs[key]
                array_same(logp[index], old_logp, "shared expert output")
                array_same(scanner[index], old_scanner, "shared scanner output")
            else:
                seen_outputs[key] = (logp[index], scanner[index])
    evaluations = result["evaluations"]
    require(len(evaluations) == 34, "incomplete routing inventory")
    expected_evaluation_dirs = {f"{bank['bank_id']}_{router}" for bank in banks for router in
                                (("uniform",) if bank["kind"] == "single_final" else ROUTERS)}
    require({p.parent.name for p in (audit.root / "evaluations").glob("*/evaluation.json")} == expected_evaluation_dirs,
            "extra or missing routing evaluation directories")
    evaluation_map, tables, routing_use = {}, [], []
    for evaluation in evaluations:
        ident, router = evaluation["bank_id"], evaluation["router"]
        key = (ident, router)
        require(key not in evaluation_map and ident in bank_map
                and router in (("uniform",) if bank_map[ident]["kind"] == "single_final" else ROUTERS),
                "duplicate/unexpected routing evaluation")
        bank = bank_map[ident]
        require(evaluation["status"] == "completed" and evaluation["kind"] == bank["kind"]
                and evaluation["init_seed"] == bank["init_seed"], "routing status/identity differs")
        audit.provenance(evaluation)
        require(audit.json(f"evaluations/{ident}_{router}/evaluation.json") == evaluation, "evaluation JSON differs")
        require(utc(evaluation["evaluated_at_utc"]) >= timestamps[opening], "evaluation timestamp predates final opening")
        arrays = audit.npz(evaluation["predictions"], ("prediction", "target", "role", "topic", "confidence", "nll",
                           "expert_pick", "routing_weights"), f"evaluations/{ident}_{router}/predictions.npz")
        for name, values in (("target", target), ("role", roles), ("topic", topics)):
            array_same(arrays[name], values, f"routing {name}")
        logp, scanner = outputs[ident]
        mixed, weights, picks = route(logp, router, scanner, x, gates.get(ident))
        predicted, nll = mixed.argmax(1), -mixed[np.arange(len(target)), target]
        array_same(arrays["prediction"], predicted, "routed prediction")
        array_same(arrays["expert_pick"], picks, "routing argmax/pick")
        array_same(arrays["routing_weights"], weights, "routing weights", 1e-9)
        array_same(arrays["confidence"], np.exp(mixed).max(1), "routed confidence", 1e-9)
        array_same(arrays["nll"], nll, "routed NLL", 1e-9)
        metrics = counts(predicted == target, roles, topics, nll)
        same(evaluation["metrics"], metrics, "routing metrics")
        slots = len(bank["experts"])
        require(evaluation["routing_frequency_source"] == ("expert_pick" if router in ("random", "scanner") else "routing_weights")
                and evaluation["expert_pick_is_hard_selection"] == (router in ("random", "scanner")),
                "routing frequency interpretation differs")
        require(evaluation["expert_slots"] == slots and evaluation["nominal_expert_parameters"] == slots * PARAMETERS
                and evaluation["gate_fitted_parameter_count"] == (40 * slots if router == "learned" else 0)
                and evaluation["unique_expert_evaluations_required_per_input"] == bank["unique_checkpoint_files"]
                and evaluation["counterfactual_policy_outputs_from_same_expert_tensor"] is True, "evaluation cost accounting differs")
        matching = [row for row in events if row["event"] == "growth_final_saved"
                    and row["bank"] == ident and row["router"] == router]
        require(len(matching) == 1 and utc(matching[0]["at_utc"]) >= utc(evaluation["evaluated_at_utc"]),
                "routing completion event differs")
        evaluation_map[key] = {"metrics": metrics, "nll": nll, "correct": predicted == target}
        for topic in ("all", "0", "1", "2"):
            mask = np.ones(len(target), dtype=bool) if topic == "all" else topics == int(topic)
            selected = metrics if topic == "all" else metrics["per_topic"][topic]
            row = {"bank_id": ident, "seed": bank["init_seed"], "kind": bank["kind"], "router": router,
                   "topic": topic, "n": selected["n"], "correct": selected["correct"], "accuracy": selected["accuracy"],
                   "binding": selected["macro_agent_patient_accuracy"], "macro_all_roles": selected["macro_all_roles"],
                   "cross_entropy": float(nll[mask].mean()), "expert_slots": slots,
                   "nominal_expert_parameters": slots * PARAMETERS,
                   "unique_checkpoint_parameters": bank["unique_checkpoint_tensor_parameters"],
                   "gate_parameters": evaluation["gate_fitted_parameter_count"]}
            for role in ROLES:
                row.update({f"{role}_{key}": value for key, value in selected["per_role"][role].items()})
            tables.append(row)
            routing_use.append({"bank_id": ident, "router": router, "topic": topic,
                                "mean_routing_weights": weights[mask].mean(0).tolist(),
                                "argmax_or_hard_pick_counts": np.bincount(picks[mask], minlength=slots).tolist(),
                                "interpretation": "Mixture usage is mean_routing_weights; argmax is not fractional utilization."})
    require(set(evaluation_map) == {(bank["bank_id"], router) for bank in banks for router in
            (("uniform",) if bank["kind"] == "single_final" else ROUTERS)}, "routing cases missing")
    require(result["complete_training_stages"] == 18 and result["complete_gate_fits"] == 8
            and result["complete_routing_evaluations"] == 34 and result["total_optimizer_updates"] == 9216,
            "result completion totals differ")
    same(result["total_training_seconds"], sum(t["training_seconds"] for t in trajectories), "total training seconds", 1e-8)
    require(math.isfinite(result["total_wall_seconds"]) and result["total_wall_seconds"] >= 0, "invalid wall time")
    contrasts = []
    comparisons = []
    for kind in ("snapshots", "duplicate_slots", "adaptive_novelty", "adaptive_resonance"):
        for router in ("uniform", "random", "learned"):
            comparisons.append((f"{kind}:scanner_minus_{router}", kind, "scanner", kind, router,
                                "Routing counterfactual on the identical bank and saved expert outputs."))
    for router in ROUTERS:
        comparisons.append((f"snapshots_{router}_minus_single_final_uniform", "snapshots", router, "single_final", "uniform",
                            "Retained snapshot bank versus final sequential expert; capacity and retention differ."))
        comparisons.append((f"snapshots_minus_duplicate_slots:{router}", "snapshots", router, "duplicate_slots", router,
                            "ABC versus ABB at the same nominal slot count; unique functional capacity differs."))
        for policy in ("adaptive_novelty", "adaptive_resonance"):
            comparisons.append((f"{policy}_minus_fixed_snapshots:{router}", policy, router, "snapshots", router,
                                "Allocation-policy contrast; condition on observed K and lineage. Capacity is an outcome, not matched."))
    for label, left_kind, left_router, right_kind, right_router, scope in comparisons:
        for topic in ("all", "0", "1", "2"):
            for measure in ("accuracy", "macro_all_roles", "macro_agent_patient_accuracy", "cross_entropy"):
                effects, capacities, raw_values = [], [], []
                for seed in SEEDS:
                    left_id, right_id = f"{left_kind}_i{seed}", f"{right_kind}_i{seed}"
                    left, right = evaluation_map[(left_id, left_router)], evaluation_map[(right_id, right_router)]
                    if measure == "cross_entropy":
                        mask = np.ones(len(target), dtype=bool) if topic == "all" else topics == int(topic)
                        lv, rv = float(left["nll"][mask].mean()), float(right["nll"][mask].mean())
                    else:
                        lm = left["metrics"] if topic == "all" else left["metrics"]["per_topic"][topic]
                        rm = right["metrics"] if topic == "all" else right["metrics"]["per_topic"][topic]
                        lv, rv = lm[measure], rm[measure]
                    effects.append(lv - rv)
                    raw_values.append({"seed": seed, "left": lv, "right": rv, "difference": lv - rv})
                    capacities.append({"seed": seed, "left_slots": len(bank_map[left_id]["experts"]),
                                       "right_slots": len(bank_map[right_id]["experts"]),
                                       "left_unique_checkpoints": bank_map[left_id]["unique_checkpoint_files"],
                                       "right_unique_checkpoints": bank_map[right_id]["unique_checkpoint_files"]})
                contrasts.append({"contrast": label, "topic": topic, "measure": measure,
                                  "direction": "left minus right; negative cross-entropy favors left",
                                  "scope": scope, "capacities_by_seed": capacities, "values_by_seed": raw_values,
                                  **paired_summary(effects)})
    try:
        analysis_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True,
                                                stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        analysis_head = None
    return {"schema_version": 1, "item": 6, "panel": "growth", "status": "verified", "issues": [],
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_head": audit.source["git_commit"], "analysis_checkout_head": analysis_head,
            "source": audit.source, "controller_source": audit.controller,
            "analyzer_sha256": sha256(__file__), "numpy_version": np.__version__,
            "verified_counts": {"trajectories": 6, "training_stages": 18, "gate_fits": 8,
                                "routing_evaluations": 34, "preallocation_checks": 2},
            "input_sha256": dict(sorted(audit.hashes.items())), "rows": tables,
            "routing_usage": routing_use, "decisions": decisions, "diagnostic_recounts": diagnostic_recounts,
            "gate_objectives": gate_rows,
            "contrasts": contrasts, "training_seconds": result["total_training_seconds"],
            "optimizer_updates": 9216, "h1_eligible": False,
            "limitations": [
                "No models, checkpoints or task generators are executed. Serialized checkpoint file hashes are verified; tensor hashes and parent-copy lineage are checked as recorded links, not recomputed tensors.",
                "Diagnostic own-token probabilities are recounted over occupied inputs with prospective atol1e-6, rtol0 for float32 scanner/mean reductions; novelty is the exact float32 fraction. The model that produced those probabilities is not rerun.",
                "Heuristic choices use the original saved scalars, Python round(value,4), strict comparisons and first-index ties; recount tolerance never changes a decision. The 0.75 post-stage-A threshold and raw/rounded margins are checked separately.",
                "The saved final gate objective and zero-weight initial objective are recounted; the 128-step weight optimization trajectory is not refitted.",
                "Partition disjointness, full topic coverage, and visible-pair targets are checked without reproducing the original Torch split permutation.",
                "Mean routing weights describe mixture usage. The stored expert_pick for mixtures is an argmax convention, not actual hard selection.",
                "All declared contrasts are reported. Two initialization seeds at one split do not support broad population claims; Student-t intervals have one degree of freedom, assume normal seed effects, and are intentionally unclipped.",
                "Adaptive versus fixed differences include observed capacity and lineage. ABC versus ABB matches nominal slots, not unique functional capacity. No H1 test, corrected significance claim, timing benchmark or energy measurement is produced.",
            ]}


def write_new(path, content):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing growth_records.json.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Checkout containing the exact recorded source/controller bytes.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    paths = [output / name for name in ("06_growth_analysis.json", "06_growth_runs.csv", "06_growth_contrasts.csv")]
    require(not any(path.exists() for path in paths), "refusing to overwrite an existing analysis artifact")
    try:
        report = audit_growth(args.input, args.source_root)
    except (VerificationError, OSError, ValueError, KeyError, TypeError) as error:
        report = {"schema_version": 1, "item": 6, "panel": "growth", "status": "rejected",
                  "issues": [f"{type(error).__name__}: {error}"], "h1_eligible": False,
                  "analyzer_sha256": sha256(__file__),
                  "audited_at_utc": datetime.now(timezone.utc).isoformat()}
        output.mkdir(parents=True, exist_ok=True)
        write_new(paths[0], json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"status": "rejected", "issues": report["issues"]}), file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    write_new(paths[0], json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with paths[1].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    contrast_rows = []
    for row in report["contrasts"]:
        contrast_rows.append({key: row[key] for key in ("contrast", "topic", "measure", "mean", "sample_standard_deviation")}
                             | {"seed20": row["raw_seed_effects"]["20"], "seed21": row["raw_seed_effects"]["21"],
                                "ci95_low": row["student_t_95_df1_unclipped"][0],
                                "ci95_high": row["student_t_95_df1_unclipped"][1], "scope": row["scope"]})
    with paths[2].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)
    print(json.dumps({"status": "verified", "issues": 0, "routing_evaluations": 34,
                      "metric_rows": len(report["rows"]), "contrasts": len(report["contrasts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
