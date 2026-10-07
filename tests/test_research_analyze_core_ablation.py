"""Synthetic tests only: independent item-6 scoring, signs, scales, gates and output.

No models, Torch, real datasets, training artifacts or GPU are accessed.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research_analyze_core_ablation.py"
spec = importlib.util.spec_from_file_location("independent_core_audit", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def synthetic_score(seed, a, b, c):
    return (0.2 + 0.01 * (seed - 20) + .03 * a + .04 * b + .05 * c
            + .02 * a * b - .01 * a * c + .01 * b * c + .008 * a * b * c)


def synthetic_rows():
    configs, cases = audit.expected_inventory()
    by_id = {c["run_id"]: c for c in configs}
    rows = []
    for case in cases:
        config = by_id[case["run_id"]]
        seed = config["init_seed"]
        a, b, c = int(config["variant"]["tied"]), int(config["school_weight"] > 0), int(config["variant"]["reinject"])
        baseline = synthetic_score(seed, 1, 0, 1)
        if config["panel"] == "factorial":
            value = synthetic_score(seed, a, b, c)
        elif config["panel"] == "recurrence":
            value = baseline + {1: -.07, 4: -.03}[config["steps"]]
        elif config["panel"] == "damage":
            value = baseline - .02
        else:
            value = synthetic_score(seed, 1, b, 1) + {0: .08, 1: .04}[b]
        if case["kind"] == "deployment_truncation":
            value = baseline + {1: -.11, 4: -.05}[case["steps_override"]]
        elif case["kind"] == "fixed_lesion":
            value = baseline - (.06 if config["panel"] == "damage" else .15)
        rows.append({"config": config, "evaluation_case": case,
                     "recounted_metrics": {"macro_agent_patient_accuracy": value}})
    return rows


def effects():
    records, _ = audit.build_contrasts(synthetic_rows())
    return {row["contrast"]: row for row in records}


def prediction_fixture():
    roles = np.repeat(np.arange(4), 4)
    targets = np.full(16, 5, dtype=np.int64)
    prediction = targets.copy()
    prediction[[0, 4, 5, 8, 9, 10, 12, 13, 14, 15]] = 6
    correct = prediction == targets
    true_p = np.where(correct, .7, .2)
    arrays = {"prediction": prediction, "target": targets, "role": roles,
              "confidence": np.full(16, .7, dtype=np.float32),
              "nll": -np.log(true_p).astype(np.float32)}
    return arrays, {"target": targets.copy(), "roles": roles.copy()}


def test_inventory_has_only_approved_26_runs_and_34_evaluations():
    configs, cases = audit.expected_inventory()
    assert len(configs) == len({c["run_id"] for c in configs}) == 26
    assert len(cases) == len({c["evaluation_id"] for c in cases}) == 34
    assert sum(c["updates"] for c in configs) == 55296
    assert sum(c["updates"] * c["batch_size"] for c in configs) == 3538944
    assert {p: sum(c["panel"] == p for c in configs)
            for p in ("factorial", "recurrence", "damage", "optimization")} == {
                "factorial": 16, "recurrence": 4, "damage": 2, "optimization": 4}
    assert sum(c["kind"] == "deployment_truncation" for c in cases) == 4
    assert sum(c["kind"] == "fixed_lesion" for c in cases) == 4


@pytest.mark.parametrize("name,expected", [
    ("tying", .037), ("school", .057), ("reinjection", .052),
    ("tying_by_school", .024), ("tying_by_reinjection", -.006),
    ("school_by_reinjection", .014), ("tying_by_school_by_reinjection", .008),
    ("tying__school0_reinjection0", .03), ("tying__school1_reinjection1", .048),
    ("tying_by_school__reinjection0", .02), ("tying_by_school__reinjection1", .028),
])
def test_factorial_raw_differences_have_declared_sign_and_scale(name, expected):
    row = effects()[name]
    assert row["values"] == pytest.approx([expected, expected], abs=1e-14)
    assert row["n"] == 2 and row["df"] == 1


def test_all_factorial_conditionals_are_present():
    rows, _ = audit.build_contrasts(synthetic_rows())
    assert sum(r["panel"] == "factorial" for r in rows) == 7
    assert sum(r["panel"] == "factorial_conditional" for r in rows) == 18
    assert len(rows) == 38


@pytest.mark.parametrize("name,expected", [
    ("trained_T1_minus_trained_T16", -.07),
    ("trained_T4_minus_trained_T16", -.03),
    ("same_T16_weights_eval_T1_minus_eval_T16", -.11),
    ("same_T16_weights_eval_T4_minus_eval_T16", -.05),
    ("damage_training_minus_clean_training__eval_lesion0", -.02),
    ("damage_training_minus_clean_training__eval_lesion1", .09),
    ("eval_lesion_minus_clean__training_damage0", -.15),
    ("eval_lesion_minus_clean__training_damage1", -.04),
    ("training_damage_by_eval_lesion", .11),
    ("updates8192_minus1024__school0", .08),
    ("updates8192_minus1024__school1", .04),
    ("budget_by_school", -.04),
    ("school03_minus0__updates8192", .038),
])
def test_secondary_panels_do_not_conflate_interventions(name, expected):
    assert effects()[name]["values"] == pytest.approx([expected, expected], abs=1e-14)


def test_all_four_damage_cells_are_preserved():
    _, cells = audit.build_contrasts(synthetic_rows())
    assert len(cells) == 4
    table = {(c["training_damage"], c["evaluation_lesion"]): c for c in cells}
    baseline = synthetic_score(20, 1, 0, 1)
    assert table[0, 0]["values"][0] == pytest.approx(baseline)
    assert table[0, 1]["values"][0] == pytest.approx(baseline - .15)
    assert table[1, 0]["values"][0] == pytest.approx(baseline - .02)
    assert table[1, 1]["values"][0] == pytest.approx(baseline - .06)


def test_t_interval_uses_two_seed_units_ddof_one_and_is_not_clipped():
    row = audit.summarize_pair({20: -1.0, 21: 1.0})
    assert row["mean"] == 0
    assert row["sample_sd"] == pytest.approx(math.sqrt(2))
    assert row["df"] == 1
    assert row["t_critical_95"] == pytest.approx(12.7062047361747)
    assert row["t_interval_95"] == pytest.approx([-audit.T_CRITICAL_DF1, audit.T_CRITICAL_DF1])
    assert row["t_interval_95"][0] < -1 and row["t_interval_95"][1] > 1


def test_zero_variance_interval_and_exact_pairing():
    row = audit.summarize_pair({21: .5, 20: .5})
    assert row["seeds"] == [20, 21]
    assert row["sample_sd"] == 0
    assert row["t_interval_95"] == [.5, .5]
    for bad in ({20: .5}, {20: .5, 22: .6}, {20: float("nan"), 21: .5}):
        with pytest.raises(audit.AuditError):
            audit.summarize_pair(bad)


def test_missing_or_duplicate_cells_do_not_become_complete_pairs():
    rows = synthetic_rows()
    with pytest.raises(KeyError):
        audit.build_contrasts(rows[:-1])
    with pytest.raises(audit.AuditError, match="duplicate statistical cell"):
        audit.build_contrasts(rows + [rows[0]])


def test_scoring_recounts_roles_binding_nll_and_saved_metrics():
    arrays, dataset = prediction_fixture()
    score = audit.score_arrays(arrays, dataset, "synthetic")
    assert score["correct"] == 6
    assert score["accuracy"] == .375
    assert score["macro_agent_patient_accuracy"] == .5
    assert [score["per_role"][role]["accuracy"] for role in audit.ROLES] == [.75, .5, .25, 0]
    assert score["cross_entropy"] == pytest.approx(arrays["nll"].astype(np.float64).mean())
    audit.compare_metrics(score, score, "synthetic")
    wrong = deepcopy(score)
    wrong["macro_agent_patient_accuracy"] += .01
    with pytest.raises(audit.AuditError, match="macro_agent_patient_accuracy"):
        audit.compare_metrics(wrong, score, "synthetic")


@pytest.mark.parametrize("change,match", [
    ("target", "target alignment"), ("role", "role alignment"),
    ("prediction", "prediction out of vocabulary"), ("confidence_nan", "nonfinite confidence"),
    ("nll_inf", "nonfinite nll"), ("confidence_low", "maximum probability below uniform"),
    ("nll_inconsistent", "true-label probability exceeds"),
])
def test_prediction_corruption_is_rejected(change, match):
    arrays, dataset = prediction_fixture()
    if change == "target":
        arrays["target"] = arrays["target"].copy()
        arrays["target"][0] = 6
    elif change == "role":
        arrays["role"] = arrays["role"].copy()
        arrays["role"][0] = 1
    elif change == "prediction":
        arrays["prediction"][0] = 35
    elif change == "confidence_nan":
        arrays["confidence"][0] = np.nan
    elif change == "nll_inf":
        arrays["nll"][0] = np.inf
    elif change == "confidence_low":
        arrays["confidence"][:] = 0
        arrays["nll"][:] = 100
    elif change == "nll_inconsistent":
        arrays["nll"][0] = .01
    with pytest.raises(audit.AuditError, match=match):
        audit.score_arrays(arrays, dataset, "corrupt")


def test_role_or_scalar_metric_corruption_is_rejected():
    arrays, dataset = prediction_fixture()
    metrics = audit.score_arrays(arrays, dataset, "synthetic")
    for key in ("correct", "n", "cross_entropy", "accuracy"):
        changed = deepcopy(metrics)
        changed[key] += 1
        with pytest.raises(audit.AuditError):
            audit.compare_metrics(changed, metrics, "corrupt")


def test_probe_flags_have_counts_only_scope():
    counts = (128, 127, 126, 125)
    probe = {
        "correct": sum(counts), "n": 512, "accuracy": sum(counts) / 512,
        "macro_all_roles": sum(counts) / 512,
        "macro_agent_patient_accuracy": (counts[0] + counts[2]) / 256,
        "cross_entropy": .1,
        "per_role": {name: {"correct": n, "n": 128, "accuracy": n / 128,
                            "wilson_95": audit.wilson95(n, 128)}
                     for name, n in zip(audit.ROLES, counts)},
    }
    assert audit.audit_reported_probe(probe)["correct"] == sum(counts)
    probe["per_role"]["AGENTE"]["correct"] = 129
    with pytest.raises(audit.AuditError):
        audit.audit_reported_probe(probe)


def test_final_npz_gate_precedes_any_file_read(tmp_path):
    reader = audit.Reader(tmp_path)
    with pytest.raises(audit.AuditError, match="gate has not passed"):
        reader.arrays("not_even_present.npz", "0" * 64, final=True)
    assert reader.inputs == {}


def test_reader_prevents_escape_and_detects_digest_change(tmp_path):
    reader = audit.Reader(tmp_path)
    with pytest.raises(audit.AuditError, match="unsafe"):
        reader.path("../outside.json")
    p = tmp_path / "file.json"
    p.write_text("{}")
    expected = hashlib.sha256(b"{}").hexdigest()
    reader.artifact("file.json", expected)
    p.write_text('{"changed":true}')
    with pytest.raises(audit.AuditError, match="changed during audit"):
        reader.artifact("file.json", expected)


def test_no_torch_or_project_imports_are_in_analyzer():
    tree = ast.parse(SOURCE.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith(("torch", "neuropixel")) for name in imports)


def test_atomic_failure_report_and_output_hashes(tmp_path):
    report = {"status": "failed", "issues": ["synthetic missing arm"], "contrasts": []}
    audit.write_outputs(report, tmp_path)
    assert json.loads((tmp_path / "core_analysis.json").read_text()) == report
    assert not (tmp_path / "core_scores.csv").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_training_gate_failure_precedes_final_result_json(tmp_path, monkeypatch):
    configs, cases = audit.expected_inventory()
    source, controller = {}, {}
    manifest = {
        "item": 6, "panel": "core", "planned_training_runs": 26, "planned_evaluation_cases": 34,
        "source": source, "controller_source": controller,
        "planned_runs": [dict(c, controller_source=controller) for c in configs],
        "planned_evaluations": cases, "selection_policy": "none; all prespecified configurations retained",
        "training_curve_observables": {
            "mean_training_loss_since_last_log": "total window mean",
            "last_answer_loss": "last point", "school_loss": "absent", "prohibited_derivation": "no subtraction",
        },
    }
    gate = {"complete_training_runs": 26, "source": source, "controller_source": controller,
            "run_manifest_sha256": "0" * 64}
    for name, value in (("run_manifest.json", manifest), ("evaluation_gate.json", gate), ("events.json", [])):
        (tmp_path / name).write_text(json.dumps(value))
    # Isolate the temporal boundary from the separate historical-source checks.
    monkeypatch.setattr(audit, "check_source", lambda *args: None)
    reader = audit.Reader(tmp_path)
    with pytest.raises(audit.AuditError, match="gate manifest hash differs"):
        audit.audit_graph(reader, tmp_path)
    assert "ablation_records.json" not in reader.inputs
    assert reader.final_arrays_opened is False
