"""Verify panel membership and prevent unnoticed recipe or evaluation changes."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replication_runner", ROOT / "scripts/research_replicate.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def inputs():
    protocol = json.loads((ROOT / "docs/research/protocol.json").read_text())
    selection = json.loads((ROOT / "results/research/04_experiment/pilot/selection.json").read_text())
    return protocol, selection


def test_exact_panel_union_preserves_selected_recipes():
    protocol, selection = inputs()
    original = json.dumps(selection, sort_keys=True)
    configs = runner.replication_configs(protocol, selection, {"controller": "fixed"})
    actual = [(c["family"], c["init_seed"], c["split_seed"]) for c in configs]
    expected = {(f, seed, 0) for f in protocol["training"]["families"] for seed in (10, 11, 12, 13, 14)}
    expected |= {(f, 10, split) for f in protocol["training"]["families"] for split in (101, 202)}
    assert len(actual) == 28 and set(actual) == expected
    assert json.dumps(selection, sort_keys=True) == original
    for config in configs:
        base = selection["selected_configs"][config["family"]]
        assert config["train_sample_seed"] == 9002
        assert config["update_random_seed"] == 9001
        for key in base.keys() - {"phase", "init_seed", "split_seed"}:
            assert config[key] == base[key]
    for family in protocol["training"]["families"]:
        assert actual.count((family, 10, 0)) == 1


def test_tampered_selection_rejected_before_source_check(tmp_path, monkeypatch):
    path = tmp_path / "selection.json"
    path.write_text('{"primary_reference_family":"neuropixel"}')
    monkeypatch.setattr(runner, "source_record", lambda: pytest.fail("source should not be read"))
    with pytest.raises(ValueError, match="exact selection"):
        runner.load_selection(path)


def test_source_drift_rejected_without_final_data_access(monkeypatch):
    path = ROOT / "results/research/04_experiment/pilot/selection.json"
    monkeypatch.setattr(runner, "source_record", lambda: {"sha256": {"changed": "value"}})
    with pytest.raises(ValueError, match="source file changed"):
        runner.load_selection(path)


def test_frozen_evaluation_gate_blocks_reentry(tmp_path, monkeypatch):
    (tmp_path / "evaluation_gate.json").write_text("{}")
    monkeypatch.setattr(runner, "load_selection", lambda _: pytest.fail("selection should not reopen"))
    with pytest.raises(RuntimeError, match="Final-phase evidence"):
        runner.run_replications(tmp_path, Path("unused"), torch.device("cpu"), {}, 8)
