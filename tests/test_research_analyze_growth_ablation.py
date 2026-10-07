"""Synthetic numerical and path checks for the independent growth auditor.

No study outputs, model code, Torch, training or task generators are used.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "independent_growth_audit", ROOT / "scripts/research_analyze_growth_ablation.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_decisions_use_saved_scalar_rounding_and_first_tie():
    result = audit.independent_decision(
        "adaptive_novelty", 2, [.100049, .100001], [.7, .8], None)
    assert result == {"action": "update", "parent": 0, "target_index": 0,
                      "scores_used": [.1, .1], "threshold": .1}
    result = audit.independent_decision(
        "adaptive_novelty", 1, [.10006], [.7], None)
    assert result["action"] == "create"
    result = audit.independent_decision(
        "adaptive_resonance", 2, [.2, .4], [.69996, .70001], .700001)
    assert result["scores_used"] == [.7, .7]
    assert result["parent"] == 0 and result["action"] == "create"
    result = audit.independent_decision(
        "adaptive_resonance", 1, [.2], [.7], .7)
    assert result["action"] == "update"


def test_probability_mixtures_and_hard_routes_are_independent_of_labels():
    probabilities = np.array([[[.8, .2], [.6, .4]], [[.1, .9], [.3, .7]]])
    logp = np.log(probabilities)
    scanner = np.array([[.2, .9], [.7, .1]])
    x = np.zeros((2, 40))
    mixed, weights, picks = audit.route(logp, "uniform", scanner, x)
    assert np.allclose(np.exp(mixed), [[.45, .55], [.45, .55]])
    assert np.array_equal(picks, [0, 0])
    assert np.allclose(weights, .5)
    mixed, weights, picks = audit.route(logp, "scanner", scanner, x)
    assert np.array_equal(picks, [1, 0])
    assert np.allclose(np.exp(mixed), probabilities[picks, np.arange(2)])
    learned, weights, _ = audit.route(logp, "learned", scanner, x, np.zeros((40, 2)))
    assert np.allclose(np.exp(learned), [[.45, .55], [.45, .55]])
    random_one = audit.route(logp, "random", scanner, x)
    random_two = audit.route(logp, "random", scanner, x)
    assert np.array_equal(random_one[2], random_two[2])


def test_visible_features_encode_bag_query_and_bias_only():
    canvas = np.zeros((2, 8, 8), dtype=np.int64)
    canvas[0, 0, :2] = [1, 5]
    canvas[1, 3, 4:6] = [1, 5]
    canvas[:, 7, 6] = 3
    x = audit.features(canvas)
    assert x.shape == (2, 40)
    assert np.array_equal(x[0], x[1])
    assert np.array_equal(np.flatnonzero(x[0]), [1, 3, 5, 37, 39])
    assert np.all(x[:, 0] == 0)


def test_two_seed_interval_keeps_raw_units_and_is_not_clipped():
    result = audit.paired_summary([0., 1.])
    assert result["n_initializations"] == 2
    assert result["raw_seed_effects"] == {"20": 0., "21": 1.}
    assert result["mean"] == .5
    half = audit.T_975_DF1 / 2
    assert np.allclose(result["student_t_95_df1_unclipped"], [.5 - half, .5 + half])
    assert result["student_t_95_df1_unclipped"][0] < -1
    with pytest.raises(audit.VerificationError):
        audit.paired_summary([.5] * 4)


def test_counts_use_agent_patient_macro_and_separate_topics():
    roles = np.tile(np.arange(4), 3)
    topics = np.repeat(np.arange(3), 4)
    correct = np.tile([True, True, False, True], 3)
    result = audit.counts(correct, roles, topics, np.full(12, math.log(2)))
    assert result["accuracy"] == .75
    assert result["macro_agent_patient_accuracy"] == .5
    assert result["per_role"]["PACIENTE"]["correct"] == 0
    assert all(row["n"] == 4 for row in result["per_topic"].values())
    assert result["cross_entropy"] == pytest.approx(math.log(2))


def test_path_and_array_integrity_refuse_escapes_and_mismatch(tmp_path):
    assert audit.contained(tmp_path, "banks/example.npz") == tmp_path / "banks/example.npz"
    for relative in ("../outside", "/absolute", "C:/outside", "banks/../../outside"):
        with pytest.raises(audit.VerificationError):
            audit.contained(tmp_path, relative)
    with pytest.raises(audit.VerificationError):
        audit.array_same(np.array([1, 2]), np.array([2, 1]), "target alignment")
    with pytest.raises(audit.VerificationError):
        audit.array_same(np.array([np.nan]), np.array([0.]), "confidence", 1e-9)


def test_diagnostic_recount_uses_only_occupied_cells_and_strict_threshold():
    canvas = np.zeros((2, 8, 8), dtype=np.int64)
    canvas[:, 0, :2] = [1, 5]
    own = np.zeros((2, 8, 8), dtype=np.float32)
    own[:, 0, :2] = [.2, .8]
    scanner = np.full(2, .5, dtype=np.float32)
    result = audit.diagnostic_statistics(own, scanner, canvas)
    assert result["novelty"] == .5
    assert result["mean_resonance_recount"] == .5
    assert result["occupied_cells"] == 4 and result["novel_cells"] == 2
    own[:, 0, :2] = .5
    assert audit.diagnostic_statistics(own, scanner, canvas)["novelty"] == 0
    with pytest.raises(audit.VerificationError, match="scanner differs"):
        audit.diagnostic_statistics(own, scanner + np.float32(.0001), canvas)


def test_final_metadata_and_arrays_are_closed_without_gate(tmp_path):
    reader = audit.Audit(tmp_path)
    with pytest.raises(audit.VerificationError, match="final outcome records remain closed"):
        reader.json("growth_records.json")
    for path in ("evaluations/snapshots_i20_uniform/predictions.npz",
                 "banks/snapshots_i20/final_experts.npz",
                 "datasets/topic0/split0_test_n1024_seed62012.npz"):
        with pytest.raises(audit.VerificationError, match="final arrays remain closed"):
            reader.npz({"path": path}, ())
