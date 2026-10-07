"""Focused item-6 contract tests; all tensor inputs are synthetic.

These tests are implementation checks, not study runs or hyperparameter trials.
Torch is imported only by the tensor fixture; the inventory and admission checks
also run in a fresh process to verify that their path stays standard-library-only.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OPS = _load(ROOT / "neuropixel/research/ablations.py", "test_item6_operations")
CTRL = _load(ROOT / "scripts/research_ablations.py", "test_item6_controller")
PROTOCOL = json.loads((ROOT / "docs/research/protocol.json").read_text(encoding="utf-8"))


@pytest.fixture
def cpu_torch():
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 2))
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            yield torch
    finally:
        torch.set_num_threads(previous_threads)


def _config(panel="factorial", *, school=0.0, tied=True, reinject=True, seed=20):
    return deepcopy(next(
        row for row in OPS.core_training_configs(PROTOCOL)
        if row["panel"] == panel and row["school_weight"] == school
        and row["variant"]["tied"] == tied and row["variant"]["reinject"] == reinject
        and row["init_seed"] == seed
    ))


def _canvas(torch, n=2):
    canvas = (torch.arange(n * 64, dtype=torch.long).reshape(n, 8, 8) % 34) + 1
    canvas[:, 0, 0] = 0
    return canvas


def _legacy_model(config):
    from neuropixel.research.models import build_model
    return build_model(
        config["family"], vocab=config["vocab"], h=config["height"], w=config["width"],
        steps=config["steps"], **config["variant"])


def _same_state(torch, left, right):
    assert left.state_dict().keys() == right.state_dict().keys()
    for key in left.state_dict():
        assert torch.equal(left.state_dict()[key], right.state_dict()[key]), key


def test_inventory_and_low_ram_admission_do_not_import_torch(tmp_path):
    code = r"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
root, forbidden_output = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("isolated_item6", root / "scripts/research_ablations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
inventory = module.inventory(module.load_protocol())
assert inventory["planned_training_runs"] == 26
assert inventory["planned_evaluation_cases"] == 34
assert "torch" not in sys.modules
sys.modules["psutil"] = SimpleNamespace(
    virtual_memory=lambda: SimpleNamespace(available=7 * 2**30))
sys.argv = ["research_ablations.py", "--output", str(forbidden_output)]
try:
    module.main()
except RuntimeError as error:
    assert "below the declared floor" in str(error)
else:
    raise AssertionError("low-RAM execution was admitted")
assert "torch" not in sys.modules
assert not forbidden_output.exists()
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT), str(tmp_path / "must_not_exist")],
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_exact_inventory_and_checkpoint_reuse():
    rows = OPS.core_training_configs(PROTOCOL)
    by_id = {row["run_id"]: row for row in rows}
    assert len(rows) == len(by_id) == 26
    counts = {panel: sum(row["panel"] == panel for row in rows)
              for panel in ("factorial", "recurrence", "damage", "optimization")}
    assert counts == {"factorial": 16, "recurrence": 4, "damage": 2, "optimization": 4}
    assert sum(row["updates"] for row in rows) == 55296
    assert all("controller_source" not in row for row in rows)
    assert all(row["split_seed"] == 0 and row["learning_rate"] == 0.003
               and row["batch_size"] == 64 for row in rows)
    assert {(row["variant"]["tied"], row["school_weight"], row["variant"]["reinject"],
             row["init_seed"]) for row in rows if row["panel"] == "factorial"} == {
        (tied, school, reinject, seed)
        for tied in (False, True) for school in (0.0, 0.3)
        for reinject in (False, True) for seed in (20, 21)
    }
    assert all(row["school_weight"] == 0 for row in rows if row["steps"] in (1, 4))
    assert all(row["updates"] == 8192 for row in rows if row["panel"] == "optimization")
    cases = OPS.core_evaluation_cases(rows)
    assert len(cases) == len({row["evaluation_id"] for row in cases}) == 34
    truncated = [row for row in cases if row["kind"] == "deployment_truncation"]
    assert len(truncated) == 4
    assert all(by_id[row["run_id"]]["panel"] == "factorial"
               and by_id[row["run_id"]]["steps"] == 16
               and by_id[row["run_id"]]["school_weight"] == 0 for row in truncated)
    lesions = [row for row in cases if row["damage"] is not None]
    assert len(lesions) == 4
    assert all(row["damage"] == OPS.EVAL_DAMAGE for row in lesions)


@pytest.mark.parametrize("tied,reinject", [(True, True), (False, False)])
def test_zero_damage_is_exact_legacy_factory(cpu_torch, tied, reinject):
    torch = cpu_torch
    config = _config(tied=tied, reinject=reinject)
    zero = deepcopy(config)
    zero["training_damage"] = {**OPS.TRAIN_DAMAGE, "apply_probability": 0.0}
    torch.manual_seed(123)
    reference = _legacy_model(config).eval()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    actual = OPS.training_model(zero).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    _same_state(torch, actual, reference)
    canvas = _canvas(torch)
    with torch.no_grad():
        expected, observed = reference(canvas), actual(canvas)
    assert torch.equal(expected["state"], observed["state"])
    assert torch.equal(expected["logits"], observed["logits"])


def test_active_damage_preserves_initialization_and_global_firing_rng(cpu_torch):
    torch = cpu_torch
    config = _config(panel="damage")
    # Boundary settings expose the intervention exactly on synthetic inputs.
    config["training_damage"] = {**OPS.TRAIN_DAMAGE, "apply_probability": 1.0,
                                "erase_probability": 1.0}
    torch.manual_seed(123)
    reference = _legacy_model(config)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    damaged = OPS.training_model(config)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    _same_state(torch, damaged, reference)
    with torch.no_grad():
        for model in (reference, damaged):
            model.f2.weight.zero_()
            model.f2.bias.zero_()
    canvas = _canvas(torch)
    reference.train()
    damaged.train()
    torch.manual_seed(901)
    clean = reference(canvas)
    firing_after_clean = torch.get_rng_state().clone()
    torch.manual_seed(901)
    erased = damaged(canvas)
    assert torch.equal(torch.get_rng_state(), firing_after_clean)
    assert torch.count_nonzero(clean["state"]) > 0
    assert torch.count_nonzero(erased["state"]) == 0
    # Development evaluation neither applies lesions nor advances their private RNG.
    damage_rng = damaged.damage_generator.get_state().clone()
    reference.eval()
    damaged.eval()
    with torch.no_grad():
        assert torch.equal(reference(canvas)["state"], damaged(canvas)["state"])
    assert torch.equal(damaged.damage_generator.get_state(), damage_rng)


def test_damage_path_has_finite_nonzero_backward(cpu_torch):
    torch = cpu_torch
    config = _config(panel="damage")
    config["training_damage"] = {**OPS.TRAIN_DAMAGE, "apply_probability": 1.0}
    torch.manual_seed(123)
    model = OPS.training_model(config).train()
    output = model(_canvas(torch))
    loss = torch.nn.functional.cross_entropy(output["logits"], torch.tensor([1, 2]))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert torch.isfinite(loss)
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_evaluation_lesion_is_indexed_by_example_not_batch(cpu_torch):
    torch = cpu_torch
    config = _config()
    torch.manual_seed(123)
    model = OPS.training_model(config).eval()
    with torch.no_grad():
        model.f2.weight.zero_()
        model.f2.bias.zero_()
    canvas = _canvas(torch, n=5)
    case = {"steps_override": None, "damage": deepcopy(OPS.EVAL_DAMAGE)}
    keep = OPS.evaluation_keep_mask(5, 8, 8, case["damage"])
    global_rng = torch.get_rng_state().clone()
    assert torch.equal(keep, OPS.evaluation_keep_mask(5, 8, 8, case["damage"]))
    assert torch.equal(global_rng, torch.get_rng_state())
    full = OPS.evaluation_view(model, config, case, keep).eval()
    chunked = OPS.evaluation_view(model, config, case, keep).eval()
    with torch.no_grad():
        expected_state = model(canvas)["state"] * keep
        complete = full(canvas)
        parts = [chunked(canvas[:2]), chunked(canvas[2:])]
    assert full.offset == chunked.offset == 5
    assert torch.equal(complete["state"], expected_state)
    assert torch.equal(torch.cat([part["state"] for part in parts]), expected_state)
    torch.testing.assert_close(
        torch.cat([part["logits"] for part in parts]), complete["logits"],
        rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="mask inventory"):
        chunked(canvas[:1])


def test_deployment_truncation_uses_same_weights(cpu_torch):
    torch = cpu_torch
    config = _config()
    torch.manual_seed(123)
    model = OPS.training_model(config).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    canvas = _canvas(torch)
    for steps in (1, 4):
        view = OPS.evaluation_view(
            model, config, {"steps_override": steps, "damage": None}).eval()
        with torch.no_grad():
            expected, actual = model(canvas, steps=steps), view(canvas)
        assert torch.equal(expected["state"], actual["state"])
        assert torch.equal(expected["logits"], actual["logits"])
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_adapter_reports_school_and_restores_after_failure(cpu_torch):
    from neuropixel.research import experiment

    config = _config(school=0.3)
    old_factory, old_metadata = experiment.model_from_config, experiment.model_config
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with OPS.training_factory_adapter(config):
            metadata = experiment.model_config(
                config["family"], vocab=35, h=8, w=8, steps=16, **config["variant"])
            assert metadata["school_weight"] == 0.3
            assert metadata["school_state_times"] == [3, 7, 11, 15]
            assert "plus_occupied_token_lens" in metadata["primary_supervision"]
            assert metadata["parameters"] == 29824
            with pytest.raises(RuntimeError, match="already active"):
                with OPS.training_factory_adapter(config):
                    pass
            raise RuntimeError("synthetic failure")
    assert experiment.model_from_config is old_factory
    assert experiment.model_config is old_metadata
    with OPS.training_factory_adapter(config):
        pass
    assert experiment.model_from_config is old_factory


def test_plan_refuses_unfrozen_status_without_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(CTRL, "ROOT", tmp_path)
    monkeypatch.setattr(CTRL, "load_protocol", lambda: PROTOCOL)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"item": 6, "status": "draft"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be frozen"):
        CTRL.load_execution_plan(path)


def test_plan_refuses_implementation_hash_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(CTRL, "ROOT", tmp_path)
    monkeypatch.setattr(CTRL, "load_protocol", lambda: PROTOCOL)
    hashes = {}
    for relative in sorted(CTRL.REQUIRED_IMPLEMENTATION):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source\n", encoding="utf-8")
        hashes[relative] = CTRL.sha256(path)
    hashes["neuropixel/research/ablations.py"] = "0" * 64
    plan = {"item": 6, "status": "frozen", **CTRL.inventory(PROTOCOL),
            "implementation_sha256": hashes}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen implementation changed"):
        CTRL.load_execution_plan(path)


def test_twenty_five_complete_runs_never_open_final_data(cpu_torch, tmp_path, monkeypatch):
    from neuropixel.research import experiment

    plan = CTRL.inventory(PROTOCOL)
    source = {"git_commit": "a" * 40, "sha256": {}}
    monkeypatch.setattr(CTRL, "load_execution_plan", lambda path: (PROTOCOL, plan, {}, {}))
    monkeypatch.setattr(CTRL, "source_record", lambda: source)
    monkeypatch.setattr(CTRL, "training_metadata", lambda config: {"parameters": 1})
    seen, final_access = [], []

    def fake_training(config, output, device, protocol, **kwargs):
        assert experiment.source_record() == source
        seen.append(config["run_id"])
        return {
            "status": "completed" if len(seen) < 26 else "failed",
            "config": config, "source": source, "final_test_accessed": False,
            "resolved_model": {"parameters": 1}, "parameter_count": 1,
            "validation_dataset": {"content_sha256": CTRL.VALIDATION_CONTENT_SHA256},
            "training_probe_dataset": {"content_sha256": CTRL.PROBE_CONTENT_SHA256},
            "environment": {"device": "synthetic"},
        }

    def forbidden_dataset(*args, **kwargs):
        final_access.append((args, kwargs))
        raise AssertionError("final data were accessed before the full inventory completed")

    monkeypatch.setattr(experiment, "train_one", fake_training)
    monkeypatch.setattr(experiment, "dataset_artifact", forbidden_dataset)
    old_factory, old_metadata = experiment.model_from_config, experiment.model_config
    previous_source = experiment.source_record
    output = tmp_path / "core"
    with pytest.raises(RuntimeError, match="did not complete"):
        CTRL.run_ablations(output, tmp_path / "unused_plan.json", SimpleNamespace(type="cpu"), 8.0)
    assert len(seen) == 26 and len(set(seen)) == 26
    assert not final_access
    assert not (output / "evaluation_gate.json").exists()
    assert not (output / "evaluations").exists()
    assert experiment.model_from_config is old_factory
    assert experiment.model_config is old_metadata
    assert experiment.source_record is previous_source


def test_missing_old_freeze_requires_explicit_recovery_record(tmp_path, monkeypatch):
    monkeypatch.setattr(CTRL, "ROOT", tmp_path)
    protocol_path = tmp_path / "docs/research/protocol.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_bytes((ROOT / "docs/research/protocol.json").read_bytes())
    recovery_path = tmp_path / CTRL.SOURCE_RECOVERY_RECORD
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.write_text(json.dumps({
        "available_historical_sha256": {},
        "unavailable_historical_sha256": CTRL.UNAVAILABLE_HISTORICAL_SHA256,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source recovery provenance"):
        CTRL.load_protocol()
    recovery_path.write_text(json.dumps({
        "available_historical_sha256": CTRL.AVAILABLE_HISTORICAL_SHA256,
        "unavailable_historical_sha256": CTRL.UNAVAILABLE_HISTORICAL_SHA256,
        "note": "Synthetic explicit recovery; no old freeze is recreated.",
    }), encoding="utf-8")
    assert CTRL.load_protocol() == PROTOCOL
    assert not (tmp_path / "docs/research/protocol_frozen.json").exists()


def test_source_record_declares_seven_files_and_rejects_a_placeholder(tmp_path, monkeypatch):
    for relative in CTRL.AVAILABLE_HISTORICAL_SHA256:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    recovery_path = tmp_path / CTRL.SOURCE_RECOVERY_RECORD
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.write_text(json.dumps({
        "available_historical_sha256": CTRL.AVAILABLE_HISTORICAL_SHA256,
        "unavailable_historical_sha256": CTRL.UNAVAILABLE_HISTORICAL_SHA256,
        "known_previous_source_commit": "fbf576f0dcfd0149d2969eccb61e442c41c7d47f",
    }), encoding="utf-8")
    monkeypatch.setattr(CTRL, "ROOT", tmp_path)
    monkeypatch.setattr(CTRL.subprocess, "check_output", lambda *args, **kwargs: "b" * 40 + "\n")
    record = CTRL.source_record()
    assert record["git_commit"] == "b" * 40
    assert record["sha256"] == CTRL.AVAILABLE_HISTORICAL_SHA256
    assert len(record["sha256"]) == record["historical_files_verified"] == 7
    assert record["historical_files_expected"] == 8
    assert record["unavailable_expected_sha256"] == CTRL.UNAVAILABLE_HISTORICAL_SHA256
    assert record["recovery_manifest"]["sha256"] == CTRL.sha256(recovery_path)
    model_path = tmp_path / "neuropixel/research/models.py"
    original = model_path.read_bytes()
    model_path.write_bytes(original + b"\n")
    with pytest.raises(RuntimeError, match="available historical source changed"):
        CTRL.source_record()
    model_path.write_bytes(original)
    missing_path = tmp_path / next(iter(CTRL.UNAVAILABLE_HISTORICAL_SHA256))
    assert not missing_path.exists()
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.write_text("# An invented historical worker must never be accepted.\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="declared unavailable exists"):
        CTRL.source_record()


def test_source_adapter_restores_original_function_on_failure(cpu_torch, monkeypatch):
    from neuropixel.research import experiment

    previous = experiment.source_record
    replacement = lambda: {
        "git_commit": "c" * 40, "sha256": {},
        "unavailable_expected_sha256": dict(CTRL.UNAVAILABLE_HISTORICAL_SHA256),
    }
    monkeypatch.setattr(CTRL, "source_record", replacement)
    marker = "_item6_source_adapter_active"
    assert not hasattr(experiment, marker)
    with pytest.raises(RuntimeError, match="synthetic provenance failure"):
        with CTRL.training_source_adapter():
            assert experiment.source_record is replacement
            assert experiment.source_record()["git_commit"] == "c" * 40
            with pytest.raises(RuntimeError, match="already active"):
                with CTRL.training_source_adapter():
                    pass
            raise RuntimeError("synthetic provenance failure")
    assert experiment.source_record is previous
    assert not hasattr(experiment, marker)


def test_original_evaluator_sets_deployment_model_to_eval(cpu_torch):
    torch = cpu_torch
    from neuropixel.research.experiment import evaluate

    config = _config()
    model = OPS.training_model(config)
    observed_training_modes = []
    handle = model.register_forward_pre_hook(
        lambda module, inputs: observed_training_modes.append(module.training))
    view = OPS.evaluation_view(
        model, config, {"steps_override": 1, "damage": None})
    canvas = _canvas(torch, n=4)
    with pytest.raises(RuntimeError, match="must only be evaluated"):
        view(canvas)
    dataset = (canvas, torch.tensor([1, 2, 3, 4]), torch.arange(4))
    try:
        metrics, predictions = evaluate(view, dataset, torch.device("cpu"), batch_size=2)
    finally:
        handle.remove()
    assert observed_training_modes == [False, False]
    assert metrics["n"] == 4 and len(predictions["prediction"]) == 4
