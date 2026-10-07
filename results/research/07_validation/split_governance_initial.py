"""Prospective dataset identity and one-use final-evaluation contracts.

This module prevents accidental overlap, drift and reopening through this API.
It does not hide public data, authenticate provenance, or prevent arbitrary code
from reading files or creating another directory. History remains an explicit,
auditable assertion. Historical experiments and their samplers are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


PARTITIONS = ("train", "validation", "final")
SELECTION_RULE = ("binding_accuracy_desc", "cross_entropy_asc",
                  "parameter_count_asc", "candidate_id_asc")
EXPOSURE_USES = {"training_pool", "validation_feedback", "final_feedback", "unknown"}


class ContractError(ValueError):
    """A declared data, selection or evaluation invariant was not met."""


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    return value


def _sha(value, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ContractError(f"{name} must be a lowercase SHA-256")
    return value


def _positive(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _keys(value, expected, name: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ContractError(f"{name} has missing or unexpected fields")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    def bad_constant(value):
        raise ContractError(f"nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_pairs, parse_constant=bad_constant)


def write_exclusive(path: Path, value) -> None:
    """Never replace a receipt; a partial failed write also blocks reuse."""
    payload = canonical_bytes(value) + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def make_manifest(*, dataset_id: str, specification: dict,
                  partitions: dict, history: dict) -> dict:
    """Record ordered examples and declared group units, without loading labels.

    Each partition contains artifact_sha256, artifact_bytes and ordered records.
    A record has record_id, group_id and content_sha256 (including its label and
    all evaluation-relevant fields). Repeated content within a partition is
    retained; duplicate content or groups across partitions are rejected.
    """
    _text(dataset_id, "dataset_id")
    required = {"generator_id", "generator_source_sha256", "vocabulary_sha256",
                "shape", "generation_parameters", "sampling_seed", "rng_runtime",
                "grouping_rule", "grouping_source_sha256", "transformations",
                "preprocessing_fit_partitions"}
    _keys(specification, required, "specification")
    for key in ("generator_id", "grouping_rule", "rng_runtime"):
        _text(specification[key], key)
    for key in ("generator_source_sha256", "vocabulary_sha256", "grouping_source_sha256"):
        _sha(specification[key], key)
    if (not isinstance(specification["shape"], list) or not specification["shape"]
            or any(type(x) is not int or x <= 0 for x in specification["shape"])):
        raise ContractError("shape must contain positive integer dimensions")
    if type(specification["sampling_seed"]) is not int:
        raise ContractError("sampling_seed must be an integer")
    if not isinstance(specification["generation_parameters"], dict):
        raise ContractError("generation_parameters must be an object")
    if not isinstance(specification["transformations"], list):
        raise ContractError("transformations must be a list")
    if specification["preprocessing_fit_partitions"] not in ([], ["train"]):
        raise ContractError("fitted preprocessing may use only the training partition")
    _keys(partitions, PARTITIONS, "partitions")
    all_ids, owners_by_group, owners_by_content = set(), {}, {}
    normalized, partition_digests = {}, {}
    for split in PARTITIONS:
        part = partitions[split]
        _keys(part, {"artifact_sha256", "artifact_bytes", "records"}, split)
        _sha(part["artifact_sha256"], f"{split}.artifact_sha256")
        _positive(part["artifact_bytes"], f"{split}.artifact_bytes")
        if not isinstance(part["records"], list) or not part["records"]:
            raise ContractError(f"{split} must contain at least one record")
        records = []
        for record in part["records"]:
            _keys(record, {"record_id", "group_id", "content_sha256"}, "record")
            rid = _text(record["record_id"], "record_id")
            gid = _text(record["group_id"], "group_id")
            content = _sha(record["content_sha256"], "content_sha256")
            if rid in all_ids:
                raise ContractError(f"duplicate record_id: {rid}")
            all_ids.add(rid)
            for key, owners, label in ((gid, owners_by_group, "group"),
                                       (content, owners_by_content, "content")):
                if key in owners and owners[key] != split:
                    raise ContractError(f"{label} overlaps partitions: {key}")
                owners[key] = split
            records.append(dict(record))
        normalized[split] = {**part, "records": records}
        partition_digests[split] = digest(normalized[split])
    _keys(history, {"scope", "coverage", "evidence_sha256", "exposures"}, "history")
    _text(history["scope"], "history.scope")
    _sha(history["evidence_sha256"], "history.evidence_sha256")
    if history["coverage"] not in {"complete_within_declared_scope", "partial", "unknown"}:
        raise ContractError("history.coverage is not recognized")
    if not isinstance(history["exposures"], list):
        raise ContractError("history.exposures must be a list")
    exposed = set()
    for row in history["exposures"]:
        _keys(row, {"group_id", "uses"}, "exposure")
        gid = _text(row["group_id"], "exposure.group_id")
        if gid in exposed:
            raise ContractError("duplicate group in exposure history")
        if (not isinstance(row["uses"], list) or not row["uses"]
                or len(set(row["uses"])) != len(row["uses"])
                or not set(row["uses"]) <= EXPOSURE_USES):
            raise ContractError("exposure uses are invalid")
        exposed.add(gid)
    final_groups = {x["group_id"] for x in normalized["final"]["records"]}
    conflicts = sorted(exposed & final_groups)
    manifest = {"schema_version": 1, "dataset_id": dataset_id,
                "specification": specification, "partitions": normalized,
                "partition_sha256": partition_digests, "history": history,
                "previously_exposed_final_groups": conflicts,
                "no_conflict_in_declared_complete_history": (
                    history["coverage"] == "complete_within_declared_scope" and not conflicts)}
    # JSON round trip detaches every nested input and rejects NaN/Infinity.
    manifest = json.loads(canonical_bytes(manifest))
    manifest["identity_sha256"] = digest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> dict:
    try:
        rebuilt = make_manifest(dataset_id=manifest["dataset_id"],
                                specification=manifest["specification"],
                                partitions=manifest["partitions"], history=manifest["history"])
    except (KeyError, TypeError) as exc:
        raise ContractError("malformed dataset manifest") from exc
    if rebuilt != manifest:
        raise ContractError("dataset manifest identity or derived fields changed")
    return rebuilt


def _relative_file(workspace: Path, relative: str) -> Path:
    _text(relative, "relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError("artifact paths must stay inside the study workspace")
    resolved = (workspace / path).resolve()
    if not resolved.is_relative_to(workspace.resolve()) or not resolved.is_file():
        raise ContractError("artifact is missing or leaves the workspace")
    return resolved


def _denominators(value):
    _keys(value, {"agent", "patient", "all"}, "denominators")
    for key in value:
        _positive(value[key], f"denominators.{key}")
    if value["all"] < value["agent"] + value["patient"]:
        raise ContractError("role denominators exceed the total")


class FinalStudy:
    """A local one-use gate. The callback performs evaluation on verified bytes."""

    def __init__(self, workspace: Path, directory: Path):
        self.workspace = Path(workspace).resolve()
        self.directory = Path(directory).resolve()
        if not self.directory.is_relative_to(self.workspace):
            raise ContractError("study directory must stay inside its workspace")

    def freeze(self, *, manifest: dict, protocol: dict, candidates: list,
               source_paths: list[str]) -> dict:
        """Select solely from a fixed candidate inventory and validation metrics."""
        manifest = validate_manifest(manifest)
        _keys(protocol, {"protocol_id", "mode", "candidate_ids", "selection_rule",
                         "validation_artifact_sha256", "validation_denominators"}, "protocol")
        _text(protocol["protocol_id"], "protocol_id")
        if protocol["mode"] not in {"exploratory", "confirmatory"}:
            raise ContractError("protocol mode must be explicit")
        if (protocol["mode"] == "confirmatory"
                and not manifest["no_conflict_in_declared_complete_history"]):
            raise ContractError("final groups have prior exposure or incomplete declared history")
        ids = protocol["candidate_ids"]
        if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids)):
            raise ContractError("candidate inventory must contain unique nonempty IDs")
        if protocol["selection_rule"] != list(SELECTION_RULE):
            raise ContractError("selection rule differs from the declared deterministic rule")
        val_sha = manifest["partitions"]["validation"]["artifact_sha256"]
        if protocol["validation_artifact_sha256"] != val_sha:
            raise ContractError("protocol validation population differs from manifest")
        _denominators(protocol["validation_denominators"])
        fields = {"candidate_id", "status", "configuration_path", "configuration_sha256",
                  "checkpoint_path", "checkpoint_sha256", "validation_artifact_sha256",
                  "binding_accuracy", "cross_entropy", "parameter_count", "denominators"}
        if not isinstance(candidates, list) or len(candidates) != len(ids):
            raise ContractError("candidate inventory is incomplete")
        observed, artifact_hashes = set(), {}
        for candidate in candidates:
            _keys(candidate, fields, "candidate")
            cid = _text(candidate["candidate_id"], "candidate_id")
            if cid not in ids or cid in observed or candidate["status"] != "completed":
                raise ContractError("candidate is unexpected, duplicated or incomplete")
            observed.add(cid)
            if (candidate["validation_artifact_sha256"] != val_sha
                    or candidate["denominators"] != protocol["validation_denominators"]):
                raise ContractError("candidates must share the declared validation population")
            for key in ("binding_accuracy", "cross_entropy"):
                value = candidate[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ContractError(f"{key} must be finite and nonnegative")
            if candidate["binding_accuracy"] > 1:
                raise ContractError("binding accuracy must lie in [0, 1]")
            _positive(candidate["parameter_count"], "parameter_count")
            for prefix in ("configuration", "checkpoint"):
                relative = candidate[prefix + "_path"]
                expected = _sha(candidate[prefix + "_sha256"], prefix + "_sha256")
                if file_digest(_relative_file(self.workspace, relative)) != expected:
                    raise ContractError(f"candidate {prefix} hash differs: {cid}")
                artifact_hashes[relative] = expected
        if not isinstance(source_paths, list) or not source_paths or len(set(source_paths)) != len(source_paths):
            raise ContractError("source_paths must contain a nonempty unique source inventory")
        for relative in source_paths:
            artifact_hashes[relative] = file_digest(_relative_file(self.workspace, relative))
        selected = min(candidates, key=lambda row: (-row["binding_accuracy"], row["cross_entropy"],
                                                  row["parameter_count"], row["candidate_id"]))
        receipt = {"schema_version": 1, "frozen_at_utc": _now(), "manifest": manifest,
                   "protocol": protocol, "protocol_sha256": digest(protocol),
                   "candidates": candidates, "candidates_sha256": digest(candidates),
                   "selected_candidate_id": selected["candidate_id"],
                   "artifact_hashes": artifact_hashes,
                   "gate_implementation_sha256": file_digest(Path(__file__)),
                   "guarantee": "local accidental-drift/reopening protection; no blinding or independent custody"}
        receipt = json.loads(canonical_bytes(receipt))
        self.directory.mkdir(parents=True, exist_ok=True)
        if any((self.directory / name).exists() for name in
               ("final_access.json", "final_result.json", "final_failure.json")):
            raise ContractError("final-phase evidence already exists in this study")
        write_exclusive(self.directory / "freeze.json", receipt)
        return receipt

    def evaluate(self, *, expected_freeze_sha256: str, final_artifact: str,
                 callback: Callable[[bytes, dict], dict], max_artifact_bytes: int = 256 * 1024**2) -> dict:
        """Consume access before reading final bytes, even if evaluation fails.

        The caller must preserve expected_freeze_sha256 outside this mutable
        directory (e.g. in the previously committed execution plan). Final bytes
        are read once and passed directly to the callback to avoid a second read
        racing with changes to the data file. Callback output must be finite JSON.
        """
        _sha(expected_freeze_sha256, "expected_freeze_sha256")
        _positive(max_artifact_bytes, "max_artifact_bytes")
        freeze_path = self.directory / "freeze.json"
        if file_digest(freeze_path) != expected_freeze_sha256:
            raise ContractError("freeze receipt changed")
        frozen = read_json(freeze_path)
        validate_manifest(frozen["manifest"])
        if (digest(frozen["protocol"]) != frozen["protocol_sha256"]
                or digest(frozen["candidates"]) != frozen["candidates_sha256"]
                or file_digest(Path(__file__)) != frozen["gate_implementation_sha256"]):
            raise ContractError("frozen protocol, candidates or gate implementation changed")
        for relative, expected in frozen["artifact_hashes"].items():
            if file_digest(_relative_file(self.workspace, relative)) != expected:
                raise ContractError(f"frozen source/configuration/checkpoint changed: {relative}")
        access = {"schema_version": 1, "opened_at_utc": _now(),
                  "freeze_sha256": expected_freeze_sha256,
                  "dataset_identity_sha256": frozen["manifest"]["identity_sha256"],
                  "selected_candidate_id": frozen["selected_candidate_id"],
                  "final_artifact": final_artifact,
                  "access_is_consumed_on_failure": True}
        # Exclusive publication precedes even final file stat/path resolution.
        write_exclusive(self.directory / "final_access.json", access)
        try:
            final = frozen["manifest"]["partitions"]["final"]
            path = _relative_file(self.workspace, final_artifact)
            if path.stat().st_size != final["artifact_bytes"] or final["artifact_bytes"] > max_artifact_bytes:
                raise ContractError("final artifact size changed or exceeds the declared read limit")
            payload = path.read_bytes()
            if len(payload) != final["artifact_bytes"] or hashlib.sha256(payload).hexdigest() != final["artifact_sha256"]:
                raise ContractError("final artifact content differs from the frozen identity")
            # Give the callback a detached receipt so it cannot mutate this record.
            result = callback(payload, json.loads(canonical_bytes(frozen)))
            if not isinstance(result, dict):
                raise ContractError("final evaluator must return a JSON object")
            outcome = {"status": "completed", "completed_at_utc": _now(),
                       "freeze_sha256": expected_freeze_sha256,
                       "access_sha256": file_digest(self.directory / "final_access.json"),
                       "result": result}
            write_exclusive(self.directory / "final_result.json", outcome)
            return outcome
        except BaseException as exc:
            # No deletion or silent retry, including interrupt/loader failures.
            failure = {"status": "failed_after_access", "at_utc": _now(),
                       "freeze_sha256": expected_freeze_sha256,
                       "error_type": type(exc).__name__, "message": str(exc)[:500]}
            write_exclusive(self.directory / "final_failure.json", failure)
            raise
