"""Focused tests for item-6 growth decisions, routing and frozen topic subsets."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "neuropixel/research/growth_ablation.py"
_SPEC = importlib.util.spec_from_file_location("growth_operations_under_test", _PATH)
growth = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(growth)


def test_exact_growth_inventory_has_six_paired_trajectories():
    plan = growth.growth_inventory()
    assert plan["planned_trajectories"] == 6
    assert plan["planned_training_stages"] == 18
    assert plan["planned_optimizer_updates"] == 9216
    assert plan["planned_gate_fits"] == 8
    assert plan["planned_routing_evaluations"] == 34
    assert plan["planned_preallocation_checks"] == 2
    assert {(row["policy"], row["init_seed"]) for row in plan["trajectories"]} == {
        (policy, seed) for policy in ("fixed_sequential", "adaptive_novelty", "adaptive_resonance")
        for seed in (20, 21)}
    assert [row["diagnostic_seed"] for row in plan["stages"]] == [62010, 62110, 62210]
    assert all(row["stage_updates"] == 512 and row["optimizer_reset_each_stage"]
               for row in plan["trajectories"])
    assert plan["diagnostic_role_balance"] == "natural sampling; no forced role balance"
    assert plan["diagnostic_recount_atol"] == 1e-6
    assert plan["h1_eligible"] is False


@pytest.mark.parametrize("policy", growth.POLICIES)
def test_empty_bank_creates_only_the_first_expert(policy):
    assert growth.growth_decision(policy, 0, [], []) == {
        "action": "create", "parent": None, "target_index": 0,
        "scores_used": [], "threshold": None}


def test_novelty_rounding_strict_threshold_and_parent_tie():
    stay = growth.growth_decision("adaptive_novelty", 1, [0.100049], [0.9])
    grow = growth.growth_decision("adaptive_novelty", 1, [0.10006], [0.9])
    tied = growth.growth_decision("adaptive_novelty", 2, [0.20003, 0.20001], [0.1, 0.9])
    assert (stay["action"], stay["target_index"]) == ("update", 0)
    assert (grow["action"], grow["target_index"]) == ("create", 1)
    assert tied["parent"] == 0
    assert tied["scores_used"] == [0.2, 0.2]
    assert tied["target_index"] == 2


def test_resonance_preserves_unrounded_tau_and_separate_criterion():
    decision = growth.growth_decision("adaptive_resonance", 1, [0.0], [0.71274], tau=0.712725)
    assert decision["scores_used"] == [0.7127]
    assert decision["threshold"] == 0.712725
    assert decision["action"] == "create"
    equality = growth.growth_decision("adaptive_resonance", 1, [0.9], [0.7], tau=0.7)
    assert equality["action"] == "update"
    tied = growth.growth_decision("adaptive_resonance", 2, [0.9, 0.0], [0.80001, 0.80002], tau=0.7)
    assert tied["parent"] == 0


def test_fixed_lineage_copies_latest_snapshot():
    decision = growth.growth_decision("fixed_sequential", 2, [0.0, 1.0], [1.0, 0.0])
    assert decision["parent"] == 1 and decision["target_index"] == 2
    with pytest.raises(ValueError):
        growth.growth_decision("fixed_sequential", 2, [0.1], [0.9])


def test_visible_features_are_binary_and_have_no_label_or_topic_argument():
    canvas = np.zeros((2, 2, 3), dtype=np.int64)
    canvas[:, 0, :2] = 7
    canvas[:, 1, 2] = [3, 1]
    features = growth.input_features(canvas, (1, 2), [1, 2, 3, 4])
    assert features.shape == (2, 40)
    assert np.array_equal(features[:, 7], [1, 1])
    assert np.array_equal(features[:, 0], [0, 0])
    assert np.array_equal(features[:, 35:39], [[0, 0, 1, 0], [1, 0, 0, 0]])
    assert np.array_equal(features[:, -1], [1, 1])
    assert {"target", "topic", "metadata"}.isdisjoint(inspect.signature(growth.input_features).parameters)
    assert {"target", "topic", "metadata"}.isdisjoint(inspect.signature(growth.route_experts).parameters)


def test_scanner_is_unrounded_hard_selection_with_first_tie():
    lp = np.log(np.array([[[0.8, 0.2], [0.8, 0.2]],
                          [[0.1, 0.9], [0.1, 0.9]]]))
    result = growth.route_experts(lp, "scanner", scanner=[[0.5, 0.50001], [0.5, 0.50002]])
    assert np.array_equal(result["expert_pick"], [0, 1])
    assert np.array_equal(result["prediction"], [0, 1])


def test_seeded_random_selector_is_reproducible_without_global_rng_mutation():
    lp = np.log(np.full((3, 20, 2), 0.5))
    before = np.random.get_state()
    first = growth.route_experts(lp, "random", random_seed=62013)
    second = growth.route_experts(lp, "random", random_seed=62013)
    after = np.random.get_state()
    assert np.array_equal(first["expert_pick"], second["expert_pick"])
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_probability_mixture_can_exceed_hard_selection_oracle():
    lp = np.log(np.array([[[0.40, 0.35, 0.25]], [[0.25, 0.35, 0.40]]]))
    assert not np.any(lp.argmax(axis=-1) == 1)
    assert growth.route_experts(lp, "uniform")["prediction"][0] == 1


def test_gate_gradient_agrees_with_independent_finite_difference():
    x = np.array([[1.0, 1.0], [0.0, 1.0], [0.5, 1.0]])
    lp = np.log(np.array([[[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]],
                          [[0.2, 0.8], [0.9, 0.1], [0.4, 0.6]]]))
    target = np.array([0, 1, 1])
    weights = np.array([[0.2, -0.1], [0.3, 0.15]])
    _, gradient, _ = growth.gate_objective(x, lp, target, weights)
    epsilon = 1e-6
    for row in range(2):
        for column in range(2):
            plus, minus = weights.copy(), weights.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numeric = (growth.gate_objective(x, lp, target, plus)[0]
                       - growth.gate_objective(x, lp, target, minus)[0]) / (2 * epsilon)
            assert gradient[row, column] == pytest.approx(numeric, rel=1e-6, abs=1e-8)


def test_gate_fit_is_fixed_deterministic_and_does_not_modify_experts():
    x = np.zeros((8, 40))
    x[:4, 0], x[:, -1] = 1, 1
    probabilities = np.array([[[0.9, 0.1]] * 4 + [[0.1, 0.9]] * 4,
                              [[0.1, 0.9]] * 4 + [[0.9, 0.1]] * 4])
    lp, target = np.log(probabilities), np.zeros(8, dtype=np.int64)
    original = lp.copy()
    first = growth.fit_gate(x, lp, target)
    second = growth.fit_gate(x, lp, target)
    assert first["updates"] == 128 and first["parameter_count"] == 80
    assert first["trace"][-1]["cross_entropy"] < first["trace"][0]["cross_entropy"]
    assert np.array_equal(first["weights"], second["weights"])
    assert np.array_equal(lp, original)


def test_partition_manifest_detects_overlap_without_models():
    triples = [(a, v, p) for a in range(4) for v in range(10) for p in range(4) if a != p]
    task = SimpleNamespace(topic_index=0, split_seed=0, train_triples=triples[:84],
                           validation_triples=triples[84:96], test_triples=triples[96:])
    record = growth.partition_manifest(task)
    assert record["counts"] == {"train": 84, "validation": 12, "test": 24}
    task.test_triples = [triples[0]] + triples[97:]
    with pytest.raises(ValueError, match="overlap"):
        growth.partition_manifest(task)


def test_topic_tasks_preserve_original_partition_membership():
    # This dataset-only test runs after the worker's RAM admission; no model is built.
    from neuropixel.research.data import ResearchRoleTask

    original = ResearchRoleTask(8, 8, seed=0)
    for topic_index, nouns in enumerate(growth.TOPICS):
        task = growth.make_topic_task(topic_index)
        for attribute in ("train_triples", "validation_triples", "test_triples"):
            expected = tuple(triple for triple in getattr(original, attribute)
                             if triple[0] in nouns and triple[2] in nouns)
            assert getattr(task, attribute) == expected
        assert sum(growth.partition_manifest(task)["counts"].values()) == 120


@pytest.fixture
def growth_controller():
    path = _PATH.parents[2] / "scripts/research_growth_ablation.py"
    spec = importlib.util.spec_from_file_location("growth_controller_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_development_fixture():
    """Supply completed metadata only; callbacks never construct study data."""
    trajectories = []
    for config in growth.growth_inventory()["trajectories"]:
        trajectories.append({
            "config": config, "status": "completed", "final_test_accessed": False,
            "optimizer_updates": 1536,
            "stages": [{"stage": stage, "optimizer_updates": 512, "final_test_accessed": False}
                       for stage in range(3)],
        })
    banks = []
    for seed in (20, 21):
        for kind in ("snapshots", "duplicate_slots", "single_final",
                     "adaptive_novelty", "adaptive_resonance"):
            bank = {"bank_id": f"{kind}_i{seed}", "kind": kind, "init_seed": seed,
                    "fit_gate": kind != "single_final"}
            if bank["fit_gate"]:
                bank["gate"] = {"updates": 128, "final_test_accessed": False,
                                "weights": {"path": "metadata_fixture.npz"}}
            banks.append(bank)
    equivalence = [{"init_seed": seed, "status": "verified"} for seed in (20, 21)]
    return trajectories, banks, equivalence


def test_complete_growth_gate_authorizes_before_the_final_callback(growth_controller):
    calls = []

    def authorize():
        calls.append("authorize")
        return "authorization"

    def final_data():
        calls.append("test")
        return "saved_dataset_placeholder"

    result = growth_controller.enter_final_phase(
        *complete_development_fixture(), authorize, final_data)
    assert result == ("authorization", "saved_dataset_placeholder")
    assert calls == ["authorize", "test"]


@pytest.mark.parametrize("missing", ("training_stage", "fitted_gate"))
def test_incomplete_growth_never_authorizes_or_opens_test(growth_controller, missing):
    trajectories, banks, equivalence = complete_development_fixture()
    if missing == "training_stage":
        trajectories[-1]["stages"].pop(1)
    else:
        del banks[0]["gate"]
    calls = []
    with pytest.raises(RuntimeError, match="stages|fitted gates"):
        growth_controller.enter_final_phase(
            trajectories, banks, equivalence,
            lambda: calls.append("authorize"), lambda: calls.append("test"))
    assert calls == []


@pytest.mark.parametrize("drift", ("source", "implementation_hash"))
def test_source_or_hash_drift_blocks_final_data(growth_controller, monkeypatch, tmp_path, drift):
    source = {"git_commit": "frozen-commit", "sha256": {"scientific.py": "frozen-source"}}
    implementation = tmp_path / "controller.py"
    implementation.write_text("original implementation\n", encoding="utf-8")
    controller = {"controller.py": growth_controller.core.sha256(implementation)}
    actual = dict(source)
    monkeypatch.setattr(growth_controller.core, "source_record", lambda: actual)
    monkeypatch.setattr(growth_controller.core, "relative_source", lambda path: tmp_path / path)
    growth_controller.verify_source_integrity(source, controller)
    if drift == "source":
        actual["git_commit"] = "changed-commit"
    else:
        implementation.write_text("changed implementation\n", encoding="utf-8")
    opened = []
    with pytest.raises(RuntimeError, match="changed"):
        growth_controller.enter_final_phase(
            *complete_development_fixture(),
            lambda: growth_controller.verify_source_integrity(source, controller),
            lambda: opened.append("test"))
    assert opened == []


@pytest.mark.parametrize("mode", ("describe", "low_ram"))
def test_preadmission_paths_never_import_torch_in_a_fresh_process(mode, tmp_path):
    runner = _PATH.parents[2] / "scripts/research_growth_ablation.py"
    code = r"""
import importlib.abc
import importlib.util
import json
import sys
from types import SimpleNamespace

class ForbidTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Torch was imported before admission")
        return None

assert "torch" not in sys.modules
sys.meta_path.insert(0, ForbidTorch())
spec = importlib.util.spec_from_file_location("fresh_growth_controller", sys.argv[1])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
mode, output = sys.argv[2], sys.argv[3]
if mode == "describe":
    runner.core.load_protocol = lambda: {}
    sys.argv = ["growth", "--describe"]
    runner.main()
else:
    runner.load_execution_plan = lambda path: None
    sys.modules["psutil"] = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=7 * 2**30))
    sys.argv = ["growth", "--output", output, "--device", "cpu",
                "--threads", "2", "--minimum-ram-gib", "8"]
    try:
        runner.main()
    except RuntimeError as error:
        assert "available RAM is below the declared floor" in str(error)
    else:
        raise AssertionError("the low-RAM guard did not reject execution")
    print(json.dumps({"low_ram_rejected": True}))
assert "torch" not in sys.modules
"""
    output = tmp_path / "must_not_be_created"
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run([sys.executable, "-c", code, str(runner), mode, str(output)],
                            capture_output=True, text=True, timeout=30, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    if mode == "describe":
        assert data["planned_trajectories"] == 6
        assert data["planned_training_stages"] == 18
        assert data["planned_gate_fits"] == 8
    else:
        assert data == {"low_ram_rejected": True}
    assert not output.exists()
