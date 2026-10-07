"""Audit existing item-4 pilot artifacts without generating data or running models.

The verifier deliberately imports no training, dataset or model implementation.
All artifact locations are derived from --pilot; recorded Windows paths are
metadata only. Use --scope selection to avoid reading final evaluation files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import sys
import tempfile
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
SOURCE_PATHS = (
    "neuropixel/model.py", "neuropixel/task.py", "neuropixel/research/models.py",
    "neuropixel/research/data.py", "neuropixel/research/experiment.py",
    "scripts/research_train.py", "scripts/research_queue_worker.py",
    "docs/research/protocol.json",
)


class ArtifactError(ValueError):
    """An artifact is missing, malformed or inconsistent with its evidence."""

    def __init__(self, code, artifact, message):
        super().__init__(message)
        self.code, self.artifact = code, artifact


def require(condition, code, artifact, message):
    if not condition:
        raise ArtifactError(code, artifact, message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _time(value, artifact):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.utcoffset() is not None, "timestamp", artifact,
                "Timestamps must include a UTC offset.")
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError) as error:
        raise ArtifactError("timestamp", artifact, "Invalid timestamp.") from error


def _close(left, right):
    return (isinstance(left, (int, float)) and not isinstance(left, bool)
            and math.isfinite(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12))


def _metrics(correct, roles, nll=None):
    """Count decisions, then average the two nominal role rates for binding."""
    counts = [(int(np.count_nonzero(correct[roles == role])), int(np.count_nonzero(roles == role)))
              for role in range(4)]
    require(all(n for _, n in counts), "roles", "predictions", "Every role must occur.")
    rates = [k / n for k, n in counts]
    result = {
        "correct": sum(k for k, _ in counts), "n": sum(n for _, n in counts),
        "accuracy": sum(k for k, _ in counts) / len(correct),
        "macro_all_roles": sum(rates) / 4,
        "macro_agent_patient_accuracy": (rates[0] + rates[2]) / 2,
        "per_role": {name: {"correct": k, "n": n, "accuracy": k / n}
                     for name, (k, n) in zip(ROLES, counts)},
    }
    if nll is not None:
        result["cross_entropy"] = float(np.asarray(nll, dtype=np.float64).mean())
    return result


def _compare_metrics(recorded, recomputed, artifact):
    for key, value in recomputed.items():
        if key == "per_role":
            require(set(recorded[key]) == set(ROLES), "metric_roles", artifact,
                    "Reported role names differ from the protocol.")
            for role, row in value.items():
                _compare_metrics(recorded[key][role], row, f"{artifact}:{role}")
        elif key in ("n", "correct"):
            require(type(recorded[key]) is int and recorded[key] == value,
                    "metric_count", artifact, f"Reported {key} differs from counted decisions.")
        else:
            require(_close(recorded[key], value), "metric_value", artifact,
                    f"Reported {key} differs from the independently recomputed value.")


def _content_digest(arrays):
    """Recreate the documented integer-array digest without Torch or generators."""
    digest = hashlib.sha256()
    for name in ("canvas", "target", "roles"):
        array = np.ascontiguousarray(arrays[name], dtype="<i8")
        header = json.dumps({"name": name, "shape": list(array.shape), "dtype": "<i8"},
                            sort_keys=True).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def _config(protocol, family, rate):
    data, training = protocol["data"], protocol["training"]
    return {
        "phase": "item4_pilot", "protocol_id": protocol["protocol_id"],
        "family": family, "vocab": 35, "height": data["height"], "width": data["width"],
        "split_seed": data["split_seeds"][0], "init_seed": training["pilot_init_seed"],
        "learning_rate": rate, "updates": training["updates"],
        "batch_size": training["batch_size"], "steps": training["steps"],
        "weight_decay": training["weight_decay"], "gradient_clip": training["gradient_clip"],
        "school_weight": training["primary_school_weight"],
        "train_sample_seed": data["train_sample_seed"],
        "update_random_seed": training["update_random_seed"], "variant": {},
    }


def _run_name(config):
    return (f"{config['family']}_s{config['split_seed']}_i{config['init_seed']}"
            f"_lr{config['learning_rate']:g}")


def rank_validation(runs, families, rates):
    """Select only from validation counts/loss, using the frozen tie breakers."""
    require(len(runs) == len(families) * len(rates), "trial_count", "selection.json",
            "All declared trials are required for selection.")
    chosen = {}
    for family in families:
        trials = [run for run in runs if run["config"]["family"] == family]
        require(sorted(run["config"]["learning_rate"] for run in trials) == sorted(rates),
                "trial_rates", "selection.json", "A family has a missing or duplicated learning rate.")
        chosen[family] = sorted(trials, key=lambda run: (
            -run["validation"]["macro_agent_patient_accuracy"],
            run["validation"]["cross_entropy"], run["config"]["learning_rate"]))[0]
    comparator = sorted((row for family, row in chosen.items() if family != "neuropixel"),
                        key=lambda run: (-run["validation"]["macro_agent_patient_accuracy"],
                                         run["validation"]["cross_entropy"], run["parameter_count"],
                                         run["config"]["family"]))[0]
    return chosen, comparator["config"]["family"]


class PilotAudit:
    """Read and cross-check one copied pilot directory with portable identities."""

    def __init__(self, pilot, source_root, expected_source_commit, scope, checkpoints):
        self.pilot, self.source_root = Path(pilot).resolve(), Path(source_root).resolve()
        self.expected_source_commit, self.scope = expected_source_commit, scope
        self.checkpoints = checkpoints
        self.files, self.issues, self.runs, self.scored = {}, [], {}, {}
        self.environment = None
        self.checkpoint_signatures = {}
        self.source = None
        self.final_files_read = False

    def local_path(self, relative, *, source=False):
        root = self.source_root if source else self.pilot
        parts = PurePosixPath(relative)
        require(not parts.is_absolute() and ".." not in parts.parts and "\\" not in relative,
                "unsafe_path", relative, "Only relative artifact identities are permitted.")
        path = root.joinpath(*parts.parts).resolve()
        require(path.is_relative_to(root), "unsafe_path", relative,
                "An artifact symlink escapes its declared root.")
        return path

    def read(self, relative, *, source=False):
        path = self.local_path(relative, source=source)
        identity = ("source/" if source else "pilot/") + relative
        require(path.is_file(), "missing_artifact", identity, "Required artifact is absent.")
        payload = path.read_bytes()
        record = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        require(identity not in self.files or self.files[identity] == record,
                "changed_during_audit", identity, "Artifact changed during verification.")
        self.files[identity] = record
        return payload

    def json(self, relative, *, source=False):
        return json.loads(self.read(relative, source=source).decode("utf-8"),
                          object_pairs_hook=_object, parse_constant=_invalid_constant)

    def protected(self, artifact, operation):
        try:
            return operation()
        except ArtifactError as error:
            self.issues.append({"code": error.code, "artifact": error.artifact, "message": str(error)})
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError,
                EOFError, ImportError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
            # Do not copy absolute paths or arbitrary serialized data into the portable report.
            self.issues.append({"code": "invalid_artifact", "artifact": artifact,
                                "message": f"Artifact could not be validated ({type(error).__name__})."})
        return None

    def artifact(self, relative, record):
        payload = self.read(relative)
        actual = self.files["pilot/" + relative]
        require(actual["sha256"] == record["sha256"], "sha256", relative,
                "File bytes do not match the recorded SHA-256.")
        if "bytes" in record:
            require(type(record["bytes"]) is int and actual["bytes"] == record["bytes"],
                    "file_size", relative, "File size differs from the recorded size.")
        if "path" in record:
            basename = str(record["path"]).replace("\\", "/").rsplit("/", 1)[-1]
            require(basename == PurePosixPath(relative).name, "artifact_name", relative,
                    "Recorded file name differs from the expected local artifact name.")
        return payload

    def arrays(self, relative, record):
        with np.load(io.BytesIO(self.artifact(relative, record)), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def specification(self):
        frozen = self.json("docs/research/protocol_frozen.json", source=True)
        for relative, expected in frozen["files"].items():
            require(hashlib.sha256(self.read(relative, source=True)).hexdigest() == expected,
                    "frozen_protocol", relative, "A frozen protocol file changed.")
        self.protocol = self.json("docs/research/protocol.json", source=True)
        self.plan = self.json("docs/research/04_execution_plan.json", source=True)
        require(self.plan["protocol_id"] == self.protocol["protocol_id"] == frozen["protocol_id"],
                "protocol_id", "source", "Protocol identities differ.")
        require(self.plan["protocol_sha256"] == frozen["files"]["docs/research/protocol.json"],
                "protocol_hash", "source", "Execution plan refers to a different protocol.")
        training = self.protocol["training"]
        require(training["lr_selection"] == ["validation_binding_accuracy_desc",
                "validation_cross_entropy_asc", "learning_rate_asc"]
                and self.protocol["decision"]["baseline_selection"]["exclude"] == "neuropixel"
                and self.protocol["decision"]["baseline_selection"]["order"] == [
                    "binding_accuracy_desc", "cross_entropy_asc", "parameters_asc", "family_name_asc"],
                "unsupported_protocol", "source", "This verifier does not implement the declared ranking rule.")
        require(len(training["families"]) == 4 and len(training["learning_rates"]) == 2,
                "unsupported_protocol", "source", "Item 4 requires four families and two learning rates.")
        self.expected_hashes = {relative: hashlib.sha256(self.read(relative, source=True)).hexdigest()
                                for relative in SOURCE_PATHS}
        return True

    def verify_source(self, source, artifact):
        commit = source["git_commit"]
        require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
                "source_commit", artifact, "A full source commit is required.")
        require(self.expected_source_commit is None or commit == self.expected_source_commit,
                "source_commit", artifact, "Recorded source commit differs from the requested commit.")
        require(source["sha256"] == self.expected_hashes, "source_hashes", artifact,
                "Source hashes differ from the supplied frozen source directory.")
        require(self.source is None or source == self.source, "source_identity", artifact,
                "Runs, selection and summary must share exactly one source record.")
        self.source = source

    def verify_environment(self, environment, artifact):
        require(self.environment is None or environment == self.environment,
                "environment_identity", artifact, "Runs and summary have different measured environments.")
        require(type(environment["threads"]) is int and 1 <= environment["threads"] <= 4,
                "environment_threads", artifact, "The declared CPU thread reservation was exceeded.")
        require(environment["deterministic_algorithms"] is True
                and environment["cudnn_benchmark"] is False
                and environment["cudnn_deterministic"] is True
                and environment["matmul_tf32"] is False and environment["cudnn_tf32"] is False
                and environment["cublas_workspace_config"] == ":4096:8",
                "environment_determinism", artifact, "Runtime determinism flags differ from the frozen runner.")
        require(all(environment["versions"].get(name) for name in ("numpy", "torch", "psutil", "pytest", "scipy"))
                and environment["python"] and environment["platform"] and environment["device"],
                "environment_metadata", artifact, "Required runtime provenance is missing.")
        if str(environment["device"]).startswith("cuda"):
            require(environment["cuda_runtime"] and environment["gpu_name"],
                    "environment_metadata", artifact, "CUDA provenance is missing.")
        self.environment = environment

    def dataset(self, record, split, n, seed):
        relative = f"datasets/split0_{split}_n{n}_seed{seed}.npz"
        require(all(record[key] == expected for key, expected in {
            "split": split, "split_seed": 0, "sampling_seed": seed, "n": n}.items()),
            "dataset_config", relative, "Dataset metadata differs from the frozen sample configuration.")
        arrays = self.arrays(relative, record)
        require(set(arrays) == {"canvas", "target", "roles"}, "array_keys", relative,
                "Dataset arrays are missing or unexpected.")
        shapes = {"canvas": (n, self.protocol["data"]["height"], self.protocol["data"]["width"]),
                  "target": (n,), "roles": (n,)}
        for key, values in arrays.items():
            require(values.shape == shapes[key] and values.dtype.kind == "i" and values.dtype.itemsize == 8,
                    "array_schema", relative, f"Invalid shape or integer dtype for {key}.")
        require(np.array_equal(arrays["roles"], np.repeat(np.arange(4), n // 4)),
                "dataset_roles", relative, "The frozen dataset must contain four balanced role blocks.")
        require(np.all((arrays["canvas"] >= 0) & (arrays["canvas"] < 35))
                and np.all((arrays["target"] >= 0) & (arrays["target"] < 35)),
                "token_range", relative, "Dataset token IDs exceed the frozen vocabulary.")
        require(_content_digest(arrays) == record["content_sha256"], "dataset_content_hash", relative,
                "The dataset content digest does not match its arrays.")
        return arrays

    def predictions(self, relative, record, dataset, recorded_metrics):
        arrays = self.arrays(relative, record)
        require(set(arrays) == {"prediction", "target", "role", "confidence", "nll"},
                "array_keys", relative, "Prediction arrays are missing or unexpected.")
        n = len(dataset["target"])
        require(all(values.shape == (n,) for values in arrays.values()), "array_shape", relative,
                "Every prediction array must match the frozen dataset length.")
        for key in ("prediction", "target", "role"):
            require(arrays[key].dtype.kind == "i" and arrays[key].dtype.itemsize == 8,
                    "array_dtype", relative, f"{key} must contain signed 64-bit integers.")
        require(np.array_equal(arrays["target"], dataset["target"])
                and np.array_equal(arrays["role"], dataset["roles"]),
                "prediction_alignment", relative, "Targets or ordered roles differ from the common dataset.")
        require(np.all((arrays["prediction"] >= 0) & (arrays["prediction"] < 35)),
                "token_range", relative, "Predicted token IDs exceed the frozen vocabulary.")
        for key in ("confidence", "nll"):
            require(arrays[key].dtype.kind == "f" and np.all(np.isfinite(arrays[key])),
                    "prediction_finite", relative, f"{key} must contain finite floating-point values.")
        require(np.all((arrays["confidence"] >= 0) & (arrays["confidence"] <= 1))
                and np.all(arrays["nll"] >= 0), "prediction_range", relative,
                "Confidence or per-example negative log likelihood is outside its valid range.")
        metrics = _metrics(arrays["prediction"] == arrays["target"], arrays["role"], arrays["nll"])
        _compare_metrics(recorded_metrics, metrics, relative)
        self.scored[relative] = metrics
        return metrics

    def checkpoint(self, relative, expected_hash, parameter_count, family):
        payload = self.artifact(relative, {"sha256": expected_hash})
        if not self.checkpoints:
            return {"content_inspected": False}
        import torch

        torch.set_num_threads(2)
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        require(isinstance(state, dict) and state and all(isinstance(key, str) for key in state),
                "checkpoint_schema", relative, "Checkpoint must be a nonempty tensor state dictionary.")
        require(all(isinstance(value, torch.Tensor) and value.is_floating_point()
                    and bool(torch.isfinite(value).all()) for value in state.values()),
                "checkpoint_tensors", relative, "Checkpoint contains a non-finite or unexpected state value.")
        elements = sum(value.numel() for value in state.values())
        # All frozen item-4 buffers are non-persistent, so the state contains parameters only.
        require(elements == parameter_count, "checkpoint_size", relative,
                "Checkpoint element count differs from the frozen parameter count.")
        signature = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                     for key, value in sorted(state.items())}
        require(family not in self.checkpoint_signatures or self.checkpoint_signatures[family] == signature,
                "checkpoint_signature", relative, "A family's two trials have different tensor signatures.")
        self.checkpoint_signatures[family] = signature
        return {"content_inspected": True, "tensors": len(state), "elements": elements,
                "signature_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()}

    def training(self, config):
        name = _run_name(config)
        artifact = f"{name}/training.json"
        row = self.json(artifact)
        require(row["schema_version"] == 1 and row["status"] == "completed", "training_incomplete", artifact,
                "Every predeclared trial must have a completed training record.")
        require(row["config"] == config, "training_config", artifact,
                "Training configuration differs from its frozen trial.")
        self.verify_source(row["source"], artifact)
        self.verify_environment(row["environment"], artifact)
        initial = self.json(f"{name}/configuration.json")
        require(initial == {"schema_version": 1, "status": "started", "config": config,
                            "source": row["source"], "started_at_utc": row["started_at_utc"]},
                "configuration_identity", artifact, "Initial and completed configuration records differ.")
        family = config["family"]
        require(row["resolved_model"] == self.plan["families"][family]
                and row["parameter_count"] == self.plan["families"][family]["parameters"],
                "model_config", artifact, "Resolved model differs from the frozen execution plan.")
        require(row["final_test_accessed"] is False, "training_test_access", artifact,
                "Training did not explicitly record that final test was unopened.")
        require(_time(row["started_at_utc"], artifact) <= _time(row["completed_at_utc"], artifact),
                "training_time", artifact, "Training completion precedes its start.")
        require(all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) and row[key] > 0
                    for key in ("training_seconds", "total_wall_seconds"))
                and row["total_wall_seconds"] >= row["training_seconds"],
                "training_duration", artifact, "Recorded training durations are invalid.")
        curve = row["curve"]
        expected_updates = sorted(set(range(128, config["updates"] + 1, 128)) | {config["updates"]})
        require([entry["update"] for entry in curve] == expected_updates, "training_updates", artifact,
                "Learning curve does not cover every scheduled log and the final declared update.")
        for entry in curve:
            require(all(isinstance(entry[key], (int, float)) and math.isfinite(entry[key]) and entry[key] >= 0
                        for key in ("mean_training_loss_since_last_log", "last_answer_loss",
                                    "last_unclipped_gradient_norm", "elapsed_training_seconds")),
                    "training_curve", artifact, "Learning curve contains invalid numbers.")
        data = self.protocol["data"]
        validation = self.dataset(row["validation_dataset"], "validation", data["validation_n"],
                                  data["validation_sample_seed"])
        diagnostic = self.protocol["training"]["optimization_diagnostic"]
        self.dataset(row["training_probe_dataset"], diagnostic["probe_split"], diagnostic["n"],
                     diagnostic["sample_seed"])
        metrics = self.predictions(f"{name}/validation_predictions.npz", row["validation_predictions"],
                                   validation, row["validation"])
        probe = row["train_probe"]
        counts = [probe["per_role"][role]["correct"] for role in ROLES]
        per_role_n = diagnostic["n"] // 4
        require(all(type(k) is int and 0 <= k <= per_role_n for k in counts),
                "probe_counts", artifact, "Training probe counts are invalid.")
        # Only aggregate probe counts were saved; this is an internal consistency check.
        correct = np.concatenate([np.arange(per_role_n) < k for k in counts])
        _compare_metrics(probe, _metrics(correct, np.repeat(np.arange(4), per_role_n)), artifact + ":probe")
        require(type(row["optimization_budget_limited"]) is bool and row["optimization_budget_limited"]
                == (probe["macro_agent_patient_accuracy"] < diagnostic["budget_limited_below_binding_accuracy"]),
                "optimization_flag", artifact, "Optimization diagnostic flag differs from its reported counts.")
        checkpoint = self.checkpoint(f"{name}/weights.pt", row["weights_sha256"], row["parameter_count"], family)
        self.runs[name] = row
        return {"run": name, "family": family, "learning_rate": config["learning_rate"],
                "validation": metrics, "checkpoint": checkpoint,
                "training_probe_counts_consistent": True}

    def selection(self):
        protocol = self.protocol
        families, rates = protocol["training"]["families"], protocol["training"]["learning_rates"]
        ranking_rows = [{**row, "validation": self.scored[f"{name}/validation_predictions.npz"]}
                        for name, row in self.runs.items()]
        ranked, reference = rank_validation(ranking_rows, families, rates)
        chosen = {family: self.runs[_run_name(row["config"])] for family, row in ranked.items()}
        selection = self.json("selection.json")
        self.verify_source(selection["source"], "selection.json")
        require(selection["selection_inputs"] == "validation only; no final test metrics",
                "selection_inputs", "selection.json", "Selection input declaration differs from the frozen runner.")
        require(selection["selected_configs"] == {family: row["config"] for family, row in chosen.items()},
                "selection_learning_rate", "selection.json", "Selected learning rates violate validation ranking.")
        require(selection["selected_validation"] == {family: row["validation"] for family, row in chosen.items()},
                "selection_metrics", "selection.json", "Selection metrics differ from their training records.")
        require(selection["primary_reference_family"] == reference, "selection_family", "selection.json",
                "Reference family violates the predeclared validation ranking.")
        self.selected, self.selection_record = chosen, selection
        self.selection_hash = self.files["pilot/selection.json"]["sha256"]
        self.events = self.json("events.json")
        expected = ["pilot_started"] + ["training_completed"] * 8 + ["selection_frozen"]
        actual = [event["event"] for event in self.events]
        if self.scope == "complete":
            expected += ["final_evaluation_opened", "pilot_completed"]
            require(actual == expected, "event_order", "events.json",
                    "A complete pilot requires training, frozen selection, test opening, then completion.")
        else:
            require(actual[:10] == expected and actual[10:] in ([], ["final_evaluation_opened"],
                    ["final_evaluation_opened", "pilot_completed"]), "event_order", "events.json",
                    "Events do not contain a valid selection-first prefix.")
        times = [_time(event["at_utc"], "events.json") for event in self.events]
        require(times == sorted(times), "event_time", "events.json", "Event timestamps are not monotonic.")
        recorded_runs = [event["run"] for event in self.events[1:9]]
        require(len(set(recorded_runs)) == 8 and set(recorded_runs) == set(self.runs),
                "event_training_runs", "events.json", "Training completion events do not identify all eight trials.")
        frozen_time = _time(selection["frozen_at_utc"], "selection.json")
        require(times[8] <= frozen_time <= times[9], "selection_time", "selection.json",
                "Selection was not frozen between the last training event and its freeze event.")
        for event, event_time in zip(self.events[1:9], times[1:9]):
            require(_time(self.runs[event["run"]]["completed_at_utc"], event["run"]) <= event_time,
                    "training_event_time", "events.json", "A completion event precedes its training record.")
        require(self.events[9]["sha256"] == self.selection_hash, "selection_hash", "events.json",
                "The freeze event is not linked to the current selection bytes.")
        return {"selected_runs": {family: _run_name(row["config"]) for family, row in chosen.items()},
                "primary_reference_family": reference, "selection_sha256": self.selection_hash,
                "recorded_selection_precedes_test": True,
                "test_open_event_present": len(self.events) > 10}

    def final(self):
        summary = self.json("pilot_summary.json")
        require(summary["schema_version"] == 1 and summary["item"] == 4 and summary["status"] == "completed"
                and summary["protocol_id"] == self.protocol["protocol_id"],
                "summary_status", "pilot_summary.json", "Pilot summary is incomplete or identifies another experiment.")
        self.verify_source(summary["source"], "pilot_summary.json")
        self.verify_environment(summary["environment"], "pilot_summary.json")
        require(summary["events"] == self.events and summary["selection"] == self.selection_record
                and summary["selection_sha256"] == self.selection_hash,
                "summary_identity", "pilot_summary.json", "Summary differs from the persisted selection or events.")
        require(set(summary["results"]) == set(self.selected), "final_families", "pilot_summary.json",
                "Final results must contain exactly the four selected families.")
        require(_close(summary["total_training_seconds"], sum(row["training_seconds"] for row in self.runs.values())),
                "summary_duration", "pilot_summary.json", "Total training time does not sum all eight trials.")
        require(summary["inference_latency_seconds"] is None and summary["physical_energy_joules"] is None,
                "unmeasured_resources", "pilot_summary.json", "The frozen runner does not measure latency or energy.")
        # This gate is reached only after all training and selection evidence passed.
        data = self.protocol["data"]
        self.final_files_read = True
        dataset = self.dataset(summary["final_dataset"], "test", data["test_n"], data["test_sample_seed"])
        require(self.events[10]["dataset_sha256"] == summary["final_dataset"]["content_sha256"],
                "test_open_hash", "events.json", "Test opening event refers to a different dataset.")
        final_results = {}
        for family, training in self.selected.items():
            name = _run_name(training["config"])
            relative = f"{name}/final_evaluation.json"
            row = self.json(relative)
            require(row == summary["results"][family], "final_summary_identity", relative,
                    "Final record differs from the summary copy.")
            require(row["family"] == family and row["config"] == training["config"]
                    and row["weights_sha256"] == training["weights_sha256"]
                    and row["selection_sha256"] == self.selection_hash
                    and row["parameter_count"] == training["parameter_count"]
                    and row["training_seconds"] == training["training_seconds"]
                    and row["optimization_budget_limited"] == training["optimization_budget_limited"],
                    "final_training_identity", relative, "Final record is not tied to its selected training artifact.")
            evaluated = _time(row["evaluated_at_utc"], relative)
            require(_time(self.events[10]["at_utc"], "events.json") <= evaluated
                    <= _time(self.events[11]["at_utc"], "events.json"),
                    "final_evaluation_time", relative, "Evaluation timestamp is outside the test-open/completion interval.")
            final_results[family] = self.predictions(f"{name}/final_predictions.npz", row["predictions"],
                                                      dataset, row["metrics"])
        return final_results

    def report(self):
        started = datetime.now(timezone.utc).isoformat()
        result = {"schema_version": 1, "item": 4, "scope": self.scope,
                  "started_at_utc": started, "expected_source_commit": self.expected_source_commit,
                  "verifier": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "python": sys.version.split()[0], "numpy": np.__version__,
                               "checkpoint_content_inspection": self.checkpoints},
                  "training_runs": [], "selection": None, "final_metrics": None}
        if self.protected("source", self.specification):
            result["protocol_id"] = self.protocol["protocol_id"]
            families, rates = self.protocol["training"]["families"], self.protocol["training"]["learning_rates"]
            expected = {_run_name(_config(self.protocol, family, rate)) for family in families for rate in rates}
            found = {path.parent.relative_to(self.pilot).as_posix() for path in self.pilot.glob("*/training.json")}
            if found != expected:
                self.issues.append({"code": "trial_inventory", "artifact": "pilot",
                                    "message": "Training record inventory is not exactly the eight declared trials.",
                                    "missing": sorted(expected - found), "unexpected": sorted(found - expected)})
            for family in families:
                for rate in rates:
                    config = _config(self.protocol, family, rate)
                    row = self.protected(_run_name(config), lambda config=config: self.training(config))
                    if row is not None:
                        result["training_runs"].append(row)
            if not self.issues:
                result["selection"] = self.protected("selection.json", self.selection)
            if not self.issues and self.scope == "complete":
                result["final_metrics"] = self.protected("final evaluation", self.final)
        result.update(status="verified" if not self.issues else "failed", issues=self.issues,
                      verified_training_runs=len(result["training_runs"]), expected_training_runs=8,
                      source=self.source, environment=self.environment,
                      final_prediction_artifacts_read=self.final_files_read,
                      inspected_files=dict(sorted(self.files.items())),
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      limitations=[
                          "Verification reads existing evidence only; it does not train, instantiate models or generate examples.",
                          "Matching hashes and timestamps establish artifact consistency, not independent proof of the recorded execution history.",
                          "Accuracy, per-role counts and binding are independently recounted from saved predictions; cross-entropy is the mean saved NLL.",
                          "Saved confidence/NLL cannot be checked against model logits without inference, which is outside this verifier.",
                          "Training-probe aggregate counts are checked internally; per-example probe predictions were not saved.",
                          "Bootstrap and Wilson intervals are not recomputed; this report verifies point metrics and selection only.",
                          "Dataset labels are checked for identity across artifacts, not regenerated from the task implementation.",
                          "Checkpoint signatures are compared within family; architecture semantics are not inferred from tensor names.",
                          "No scientific conclusion or item-5 replication decision is made by artifact verification.",
                      ])
        if not self.checkpoints:
            result["limitations"].append("Checkpoint byte hashes were checked, but tensor content inspection was explicitly disabled.")
        return result


def verify_pilot(pilot, *, source_root=ROOT, expected_source_commit=None,
                 scope="complete", inspect_checkpoints=True):
    """Return a JSON-compatible audit; leave every pilot input unchanged."""
    if scope not in ("selection", "complete"):
        raise ValueError("scope must be selection or complete")
    if expected_source_commit is not None and re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is None:
        raise ValueError("expected source commit must be a full 40-character lowercase SHA")
    return PilotAudit(pilot, source_root, expected_source_commit, scope, inspect_checkpoints).report()


def write_report(path, report, *, pilot, source_root=ROOT):
    """Atomically save a portable report outside the pilot input directory."""
    path = Path(path).resolve()
    if path.is_relative_to(Path(pilot).resolve()):
        raise ValueError("report output must be outside the pilot input directory")
    source_root = Path(source_root).resolve()
    reserved = [source_root / relative for relative in SOURCE_PATHS]
    reserved += [source_root / "docs/research/protocol_frozen.json",
                 source_root / "docs/research/04_execution_plan.json", Path(__file__).resolve()]
    if path in reserved or (path.exists() and path.suffix.lower() != ".json"):
        raise ValueError("report output would overwrite a source or non-report artifact")
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if path.read_bytes() != payload:
        raise OSError("verification report did not persist correctly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True, help="Local copied pilot artifact directory")
    parser.add_argument("--output", type=Path, required=True, help="Report JSON outside the pilot directory")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Frozen source used by the pilot")
    parser.add_argument("--expected-source-commit", help="Optional exact training source commit")
    parser.add_argument("--scope", choices=("selection", "complete"), default="complete")
    parser.add_argument("--skip-checkpoint-content", action="store_true")
    args = parser.parse_args()
    report = verify_pilot(args.pilot, source_root=args.source_root,
                          expected_source_commit=args.expected_source_commit, scope=args.scope,
                          inspect_checkpoints=not args.skip_checkpoint_content)
    write_report(args.output, report, pilot=args.pilot, source_root=args.source_root)
    print(json.dumps({"status": report["status"], "scope": report["scope"],
                      "verified_training_runs": report["verified_training_runs"],
                      "issues": len(report["issues"])}))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
