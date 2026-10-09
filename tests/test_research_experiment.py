"""Protect prospective selection, interrupted evidence and role-aware scoring."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropixel.research import experiment as exp
from scripts.research_train import load_protocol, pilot_config, run_pilot


def test_macro_binding_does_not_hide_nominal_failure():
    roles = np.repeat(np.arange(4), [4, 16, 4, 16])
    target = np.ones(len(roles), dtype=int)
    prediction = np.where(np.isin(roles, [1, 3]), 1, 0)
    result = exp.classification_metrics(prediction, target, roles, intervals=True, repetitions=100)
    assert result["accuracy"] == .8
    assert result["macro_all_roles"] == .5
    assert result["macro_agent_patient_accuracy"] == 0
    assert result["binding_interval_95"] == [0, 0]


def test_selection_uses_validation_and_prespecified_ties():
    families = ["neuropixel", "standard_nca", "convgru", "relative_transformer"]
    runs = []
    for family in families:
        for lr in [.001, .003]:
            binding = .9 if family == "neuropixel" else .8
            loss = .2
            if family == "standard_nca":
                binding, loss = (.6, 1.0) if lr == .001 else (.5, .01)
            runs.append({"status": "completed", "config": {"family": family, "learning_rate": lr},
                         "validation": {"macro_agent_patient_accuracy": binding, "cross_entropy": loss},
                         "parameter_count": 27000 if family == "convgru" else 30000,
                         "final_metric_must_be_ignored": 1.0 if family == "standard_nca" else 0.0})
    selected = exp.select_pilot(runs, families, [.001, .003])
    assert selected["primary_reference_family"] == "convgru"
    assert all(config["learning_rate"] == .001 for config in selected["selected_configs"].values())
    with pytest.raises(ValueError, match="all predeclared"):
        exp.select_pilot(runs[:-1], families, [.001, .003])
    failed = copy.deepcopy(runs)
    failed[0]["status"] = "failed"
    with pytest.raises(ValueError, match="all predeclared"):
        exp.select_pilot(failed, families, [.001, .003])


def test_atomic_write_keeps_prior_evidence_on_interruption(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"prior verified evidence")
    with pytest.raises(KeyboardInterrupt):
        with exp.atomic_binary(path) as stream:
            stream.write(b"incomplete replacement")
            raise KeyboardInterrupt
    assert path.read_bytes() == b"prior verified evidence"
    assert list(tmp_path.iterdir()) == [path]


def test_pilot_cannot_silently_reopen_final_evaluation(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text('{"immutable": true}')
    with pytest.raises(RuntimeError, match="explicit recovery"):
        run_pilot(tmp_path, torch.device("cpu"), {}, 8)
    assert json.loads(path.read_text()) == {"immutable": True}


def test_partial_run_is_not_overwritten(tmp_path, monkeypatch):
    prior = tmp_path / "configuration.json"
    prior.write_text('{"prior": true}')
    monkeypatch.setattr(exp, "source_record", lambda: {})
    monkeypatch.setattr(exp, "environment_record", lambda device: {})
    with pytest.raises(RuntimeError, match="interrupted run directory"):
        exp.train_one({}, tmp_path, torch.device("cpu"), {})
    assert json.loads(prior.read_text()) == {"prior": True}


def test_training_smoke_preserves_weights_and_never_opens_test(tmp_path, monkeypatch):
    """Two-update software check on a tiny validation sample; not a scientific run."""
    torch.set_num_threads(2)
    protocol = copy.deepcopy(load_protocol())
    protocol["data"].update(validation_n=8, validation_sample_seed=1234567)
    config = pilot_config(protocol, "neuropixel", .001)
    config.update(phase="unit_test_not_scientific", updates=2, batch_size=4, steps=2)
    monkeypatch.setattr(exp, "resource_sample", lambda *a, **k: {"unit_test_only": True})
    real_dataset = exp.dataset_artifact
    seen = []

    def guard(task, split, *args, **kwargs):
        seen.append(split)
        assert split in ("train", "validation")
        return real_dataset(task, split, *args, **kwargs)

    monkeypatch.setattr(exp, "dataset_artifact", guard)
    result = exp.train_one(config, tmp_path / "software_smoke", torch.device("cpu"), protocol)
    assert result["status"] == "completed" and result["final_test_accessed"] is False
    assert seen == ["validation", "train"]
    assert len(result["curve"]) == 1 and result["curve"][0]["update"] == 2
    assert exp.file_sha256(tmp_path / "software_smoke/weights.pt") == result["weights_sha256"]
