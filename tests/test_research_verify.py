"""Corrupt synthetic saved artifacts; never instantiate a model or a task."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_verify as verifier


FAMILIES = ("neuropixel", "standard_nca", "convgru", "relative_transformer")
ROLES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
COMMIT = "a" * 40


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_record(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    # Deliberately nonexistent Windows roots must not be followed by the verifier.
    return {"path": "D:\\synthetic\\copied_pilot\\" + path.name,
            "sha256": digest(path), "bytes": path.stat().st_size}


def known_metrics(counts, n_per_role, nll=0.25):
    """Build an oracle from prescribed counts, separately from the NPZ decisions."""
    return {"n": n_per_role * 4, "correct": sum(counts),
            "accuracy": sum(counts) / (4 * n_per_role),
            "macro_all_roles": sum(counts) / (4 * n_per_role),
            "macro_agent_patient_accuracy": (counts[0] + counts[2]) / (2 * n_per_role),
            "cross_entropy": nll,
            "per_role": {role: {"n": n_per_role, "correct": k, "accuracy": k / n_per_role}
                         for role, k in zip(ROLES, counts)}}


def fixture_dataset(pilot, split, n, seed):
    roles = np.repeat(np.arange(4, dtype=np.int64), n // 4)
    target = np.repeat(np.array([5, 17, 6, 27], dtype=np.int64), n // 4)
    arrays = {"canvas": np.zeros((n, 8, 8), dtype=np.int64), "target": target, "roles": roles}
    record = array_record(pilot / "datasets" / f"split0_{split}_n{n}_seed{seed}.npz", **arrays)
    # The digest's published binary format is encoded independently in this fixture.
    content = bytearray()
    for key, values in arrays.items():
        header = json.dumps({"dtype": "<i8", "name": key, "shape": list(values.shape)}, sort_keys=True).encode()
        content.extend(len(header).to_bytes(8, "little"))
        content.extend(header)
        content.extend(values.astype("<i8").tobytes())
    record.update(split=split, split_seed=0, sampling_seed=seed, n=n,
                  content_sha256=hashlib.sha256(content).hexdigest())
    return arrays, record


def fixture_predictions(path, dataset, counts, nll=0.25):
    target, roles = dataset["target"], dataset["roles"]
    prediction = np.zeros_like(target)
    for role, k in enumerate(counts):
        selected = np.flatnonzero(roles == role)[:k]
        prediction[selected] = target[selected]
    return array_record(path, prediction=prediction, target=target, role=roles,
                        confidence=np.full(len(target), 0.5, dtype=np.float32),
                        nll=np.full(len(target), nll, dtype=np.float32))


@pytest.fixture
def pilot(tmp_path):
    """Write arithmetic-only fixtures using the real frozen artifact dimensions."""
    torch.set_num_threads(2)
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    source = {"git_commit": COMMIT,
              "sha256": {relative: digest(ROOT / relative) for relative in verifier.SOURCE_PATHS}}
    plan = read_json(ROOT / "docs/research/04_execution_plan.json")
    environment = {
        "python": "3.13.0", "platform": "Windows synthetic fixture", "device": "cuda",
        "versions": {name: "fixture" for name in ("numpy", "torch", "psutil", "pytest", "scipy")},
        "threads": 4, "cuda_runtime": "12.4", "gpu_name": "Synthetic device",
        "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "matmul_tf32": False, "cudnn_tf32": False,
    }
    validation, validation_record = fixture_dataset(pilot, "validation", 2048, 61001)
    _, probe_record = fixture_dataset(pilot, "train", 512, 61006)
    final, final_record = fixture_dataset(pilot, "test", 4096, 61002)
    rows = {}
    events = [{"event": "pilot_started", "at_utc": "2026-10-07T00:00:00+00:00"}]
    for family_index, family in enumerate(FAMILIES):
        for rate_index, rate in enumerate((0.001, 0.003)):
            index = family_index * 2 + rate_index
            config = {
                "phase": "item4_pilot", "protocol_id": "NP-SCI-20261006-v1", "family": family,
                "vocab": 35, "height": 8, "width": 8, "split_seed": 0, "init_seed": 7,
                "learning_rate": rate, "updates": 1024, "batch_size": 64, "steps": 16,
                "weight_decay": 0.0001, "gradient_clip": 1.0, "school_weight": 0.0,
                "train_sample_seed": 9002, "update_random_seed": 9001, "variant": {},
            }
            name = f"{family}_s0_i7_lr{rate:g}"
            directory = pilot / name
            directory.mkdir()
            counts = (384, 512, 320, 512) if family == "neuropixel" and rate_index == 1 else (256, 512, 256, 512)
            nll = (0.125 if family == "neuropixel" and rate_index == 0 else
                   0.5 if family == "standard_nca" and rate_index == 0 else 0.25)
            prediction_record = fixture_predictions(directory / "validation_predictions.npz", validation, counts, nll)
            parameters = plan["families"][family]["parameters"]
            torch.save({"fixture_parameter": torch.full((parameters,), float(index))}, directory / "weights.pt")
            initial = {"schema_version": 1, "status": "started", "config": config,
                       "source": source, "started_at_utc": f"2026-10-07T00:{index + 1:02}:00+00:00"}
            write_json(directory / "configuration.json", initial)
            row = {**initial, "status": "completed", "completed_at_utc": f"2026-10-07T00:{index + 1:02}:20+00:00",
                   "environment": environment, "resolved_model": plan["families"][family],
                   "parameter_count": parameters, "training_seconds": 10.0, "total_wall_seconds": 20.0,
                   "final_test_accessed": False, "validation_dataset": validation_record,
                   "training_probe_dataset": probe_record, "validation_predictions": prediction_record,
                   "validation": known_metrics(counts, 512, nll),
                   "train_probe": known_metrics((128, 128, 128, 128), 128),
                   "optimization_budget_limited": False, "weights_sha256": digest(directory / "weights.pt"),
                   "curve": [{"update": update, "mean_training_loss_since_last_log": 0.25,
                              "last_answer_loss": 0.25, "last_unclipped_gradient_norm": 0.1,
                              "elapsed_training_seconds": update / 128.0} for update in range(128, 1025, 128)]}
            write_json(directory / "training.json", row)
            rows[name] = row
            events.append({"event": "training_completed", "run": name,
                           "at_utc": f"2026-10-07T00:{index + 1:02}:21+00:00"})
    chosen_rates = {"neuropixel": 0.003, "standard_nca": 0.003, "convgru": 0.001, "relative_transformer": 0.001}
    chosen = {family: rows[f"{family}_s0_i7_lr{rate:g}"] for family, rate in chosen_rates.items()}
    selection = {"frozen_at_utc": "2026-10-07T00:09:00+00:00", "source": source,
                 "selection_inputs": "validation only; no final test metrics",
                 "selected_configs": {family: row["config"] for family, row in chosen.items()},
                 "selected_validation": {family: row["validation"] for family, row in chosen.items()},
                 "primary_reference_family": "convgru"}
    write_json(pilot / "selection.json", selection)
    selection_hash = digest(pilot / "selection.json")
    events.extend([{"event": "selection_frozen", "at_utc": "2026-10-07T00:09:01+00:00", "sha256": selection_hash},
                   {"event": "final_evaluation_opened", "at_utc": "2026-10-07T00:10:00+00:00",
                    "dataset_sha256": final_record["content_sha256"]},
                   {"event": "pilot_completed", "at_utc": "2026-10-07T00:12:00+00:00"}])
    finals = {}
    final_counts = {"neuropixel": (768, 1024, 512, 1024), "standard_nca": (1024, 1024, 1024, 1024),
                    "convgru": (1024, 0, 0, 1024), "relative_transformer": (0, 1024, 0, 1024)}
    for family, row in chosen.items():
        directory = pilot / f"{family}_s0_i7_lr{chosen_rates[family]:g}"
        prediction_record = fixture_predictions(directory / "final_predictions.npz", final, final_counts[family])
        result = {"family": family, "config": row["config"], "metrics": known_metrics(final_counts[family], 1024),
                  "weights_sha256": row["weights_sha256"], "predictions": prediction_record,
                  "selection_sha256": selection_hash, "evaluated_at_utc": "2026-10-07T00:11:00+00:00",
                  "parameter_count": row["parameter_count"], "training_seconds": 10.0,
                  "optimization_budget_limited": False}
        write_json(directory / "final_evaluation.json", result)
        finals[family] = result
    write_json(pilot / "events.json", events)
    write_json(pilot / "pilot_summary.json", {
        "schema_version": 1, "item": 4, "status": "completed", "protocol_id": "NP-SCI-20261006-v1",
        "source": source, "environment": environment, "selection": selection, "selection_sha256": selection_hash,
        "events": events, "final_dataset": final_record, "results": finals,
        "total_training_seconds": 80.0, "inference_latency_seconds": None, "physical_energy_joules": None,
    })
    return pilot


def check(pilot, **kwargs):
    return verifier.verify_pilot(pilot, expected_source_commit=COMMIT, **kwargs)


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_complete_copied_pilot_recounts_known_answers_without_running_models(pilot, tmp_path):
    before = {path.relative_to(pilot).as_posix(): digest(path) for path in pilot.rglob("*") if path.is_file()}
    report = check(pilot)
    assert report["status"] == "verified", report["issues"]
    assert report["verified_training_runs"] == 8
    assert report["selection"]["primary_reference_family"] == "convgru"
    assert report["selection"]["selected_runs"]["neuropixel"].endswith("lr0.003")
    assert report["selection"]["selected_runs"]["standard_nca"].endswith("lr0.003")
    result = report["final_metrics"]["neuropixel"]
    assert result["correct"] == 3328 and result["n"] == 4096
    assert result["accuracy"] == 0.8125 and result["macro_agent_patient_accuracy"] == 0.625
    assert result["per_role"]["PACIENTE"] == {"correct": 512, "n": 1024, "accuracy": 0.5}
    assert all(row["checkpoint"]["content_inspected"] for row in report["training_runs"])
    assert before == {path.relative_to(pilot).as_posix(): digest(path) for path in pilot.rglob("*") if path.is_file()}
    output = tmp_path / "portable_report.json"
    verifier.write_report(output, report, pilot=pilot)
    assert read_json(output) == report
    assert str(tmp_path) not in output.read_text() and "D:\\synthetic" not in output.read_text()


def test_selection_scope_never_opens_final_files(pilot, monkeypatch):
    real_read = verifier.PilotAudit.read

    def guard(self, relative, **kwargs):
        assert "final" not in relative and "_test_" not in relative and relative != "pilot_summary.json"
        return real_read(self, relative, **kwargs)

    monkeypatch.setattr(verifier.PilotAudit, "read", guard)
    report = check(pilot, scope="selection")
    assert report["status"] == "verified", report["issues"]
    assert report["final_metrics"] is None and report["final_prediction_artifacts_read"] is False


@pytest.mark.parametrize("filename", ["weights.pt", "validation_predictions.npz"])
def test_corrupted_binary_is_rejected_before_final_access(pilot, filename):
    path = pilot / "neuropixel_s0_i7_lr0.001" / filename
    path.write_bytes(path.read_bytes()[:31])
    report = check(pilot)
    assert report["status"] == "failed" and "sha256" in codes(report)
    assert report["final_prediction_artifacts_read"] is False


def test_wrong_metric_is_detected_even_with_intact_hashes(pilot):
    path = pilot / "convgru_s0_i7_lr0.001/training.json"
    row = read_json(path)
    row["validation"]["macro_agent_patient_accuracy"] = 0.9
    write_json(path, row)
    assert "metric_value" in codes(check(pilot))


def test_changed_targets_with_refreshed_prediction_hash_are_rejected(pilot):
    directory = pilot / "relative_transformer_s0_i7_lr0.001"
    path = directory / "validation_predictions.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["target"][0] = 8
    record = array_record(path, **arrays)
    row = read_json(directory / "training.json")
    row["validation_predictions"] = record
    write_json(directory / "training.json", row)
    assert "prediction_alignment" in codes(check(pilot))


@pytest.mark.parametrize("change,expected", [("rate", "selection_learning_rate"), ("family", "selection_family")])
def test_selection_cannot_follow_bad_rate_or_best_final_result(pilot, change, expected):
    path = pilot / "selection.json"
    selection = read_json(path)
    if change == "rate":
        inferior = read_json(pilot / "neuropixel_s0_i7_lr0.001/training.json")
        selection["selected_configs"]["neuropixel"] = inferior["config"]
        selection["selected_validation"]["neuropixel"] = inferior["validation"]
    else:
        # This family has perfect final accuracy, but loses the validation tie.
        selection["primary_reference_family"] = "standard_nca"
    write_json(path, selection)
    assert expected in codes(check(pilot))


@pytest.mark.parametrize("change,expected", [("order", "event_order"), ("time", "event_time"), ("hash", "selection_hash")])
def test_test_opening_requires_intact_prior_selection_evidence(pilot, change, expected):
    path = pilot / "events.json"
    events = read_json(path)
    if change == "order":
        events[9], events[10] = events[10], events[9]
    elif change == "time":
        events[10]["at_utc"] = "2026-10-07T00:08:30+00:00"
    else:
        events[9]["sha256"] = "0" * 64
    write_json(path, events)
    report = check(pilot)
    assert expected in codes(report)
    assert report["final_prediction_artifacts_read"] is False


@pytest.mark.parametrize("change,expected", [("source", "source_hashes"), ("environment", "environment_identity"),
                                           ("missing", "trial_inventory")])
def test_mixed_or_missing_trials_are_rejected(pilot, change, expected):
    path = pilot / "convgru_s0_i7_lr0.003/training.json"
    row = read_json(path)
    if change == "source":
        row["source"]["sha256"]["neuropixel/research/data.py"] = "0" * 64
    elif change == "environment":
        row["environment"]["versions"]["torch"] = "different"
    else:
        path.unlink()
    if change != "missing":
        write_json(path, row)
    assert expected in codes(check(pilot))


def test_safe_checkpoint_content_detects_nan_after_hash_refresh(pilot):
    directory = pilot / "convgru_s0_i7_lr0.001"
    path = directory / "weights.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["fixture_parameter"][0] = float("nan")
    torch.save(state, path)
    row = read_json(directory / "training.json")
    row["weights_sha256"] = digest(path)
    write_json(directory / "training.json", row)
    assert "checkpoint_tensors" in codes(check(pilot))


def test_malformed_checkpoint_with_matching_hash_returns_a_failure_report(pilot):
    directory = pilot / "convgru_s0_i7_lr0.001"
    path = directory / "weights.pt"
    path.write_bytes(b"not a tensor checkpoint")
    row = read_json(directory / "training.json")
    row["weights_sha256"] = digest(path)
    write_json(directory / "training.json", row)
    report = check(pilot)
    assert report["status"] == "failed" and "invalid_artifact" in codes(report)
    assert report["final_prediction_artifacts_read"] is False


def test_checkpoint_symlink_cannot_escape_pilot_root(pilot, tmp_path):
    path = pilot / "neuropixel_s0_i7_lr0.001/weights.pt"
    outside = tmp_path / "outside.pt"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("This operating system does not permit creating symlinks.")
    assert "unsafe_path" in codes(check(pilot))


def test_final_metrics_are_recounted_and_report_cannot_overwrite_inputs(pilot):
    directory = pilot / "neuropixel_s0_i7_lr0.003"
    final = read_json(directory / "final_evaluation.json")
    final["metrics"]["correct"] += 1
    summary = read_json(pilot / "pilot_summary.json")
    summary["results"]["neuropixel"] = final
    write_json(directory / "final_evaluation.json", final)
    write_json(pilot / "pilot_summary.json", summary)
    report = check(pilot)
    assert "metric_count" in codes(report)
    before = digest(pilot / "selection.json")
    with pytest.raises(ValueError, match="outside the pilot"):
        verifier.write_report(pilot / "selection.json", report, pilot=pilot)
    assert digest(pilot / "selection.json") == before


def test_family_name_breaks_equal_parameter_tie_and_lr_breaks_equal_loss():
    # No final metrics are read by ranking, even when supplied as distractions.
    rows = [{"config": {"family": family, "learning_rate": rate}, "parameter_count": 100,
             "validation": {"macro_agent_patient_accuracy": 0.5, "cross_entropy": 0.25},
             "final_accuracy": 1.0 if family == "standard_nca" else 0.0}
            for family in reversed(FAMILIES) for rate in (0.003, 0.001)]
    chosen, family = verifier.rank_validation(rows, FAMILIES, (0.001, 0.003))
    assert family == "convgru"
    assert all(row["config"]["learning_rate"] == 0.001 for row in chosen.values())
    with pytest.raises(verifier.ArtifactError):
        verifier.rank_validation(rows[:-1], FAMILIES, (0.001, 0.003))


def test_source_commit_can_be_required_or_inferred_from_consistent_records(pilot):
    assert verifier.verify_pilot(pilot, scope="selection")["status"] == "verified"
    report = verifier.verify_pilot(pilot, expected_source_commit="b" * 40, scope="selection")
    assert "source_commit" in codes(report)
