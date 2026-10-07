"""Check replication inference using prescribed paired outcomes, without models."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from neuropixel.research.statistics import (
    describe_values, paired_binding_interval, summarize_replications,
)


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "macro_agent_patient_accuracy"
REFERENCE = "relative_transformer"


@pytest.fixture
def protocol():
    return json.loads((ROOT / "docs/research/protocol.json").read_text())


def metrics(binding):
    return {"accuracy": binding, "macro_all_roles": binding, PRIMARY: binding,
            "cross_entropy": 1.0 - binding,
            "per_role": {role: {"accuracy": binding} for role in ("AGENTE", "ACCION", "PACIENTE", "LUGAR")}}


@pytest.fixture
def records(protocol):
    configurations = [(0, seed) for seed in (10, 11, 12, 13, 14)] + [(101, 10), (202, 10)]
    rows = []
    for family in protocol["training"]["families"]:
        for split, initialization in configurations:
            if family == "neuropixel":
                binding = (60 + initialization - 10) / 64 if split == 0 else (0.125 if split == 101 else 0.25)
            elif family == REFERENCE:
                binding = 0.8125 if split == 0 else 0.9375
            else:
                binding = 1.0 if family == "standard_nca" else 0.75
            rows.append({"status": "completed", "config": {"family": family, "split_seed": split,
                         "init_seed": initialization, "controller_source": {"irrelevant_extra_metadata": True}},
                         "metrics": metrics(binding)})
    return rows


def summarize(records, protocol):
    return summarize_replications(records, REFERENCE, protocol)


def set_initialization_values(records, family, values):
    for row in records:
        config = row["config"]
        if config["family"] == family and config["split_seed"] == 0:
            row["metrics"] = metrics(values[config["init_seed"] - 10])


def test_known_paired_t_interval_uses_five_seed_differences(records, protocol):
    before = copy.deepcopy(records)
    result = summarize(records, protocol)
    panel = result["panels"]["initialization"]
    paired = panel["paired_differences"][REFERENCE]
    assert paired["values"] == [0.125, 0.140625, 0.15625, 0.171875, 0.1875]
    assert paired["mean"] == 0.15625 and paired["range"] == [0.125, 0.1875]
    # For these five equally spaced differences, sample variance is exactly 5/8192.
    expected_sd = math.sqrt(5 / 8192)
    half_width = 2.7764451051977987 * expected_sd / math.sqrt(5)
    assert paired["sample_sd"] == pytest.approx(expected_sd)
    assert paired["interval_95"] == pytest.approx([0.15625 - half_width, 0.15625 + half_width])
    assert paired["degrees_of_freedom"] == 4 and paired["n"] == 5
    assert paired["paired_np_mean"] == 0.96875
    assert result["decision"]["positive_support"] is True
    assert result["decision"]["status"] == "supported"
    assert records == before
    json.dumps(result, allow_nan=False)


def test_panels_overlap_but_are_not_pooled_and_other_baselines_do_not_select_h1(records, protocol):
    result = summarize(list(reversed(records)), protocol)
    assert result["record_inventory"]["expected_unique_runs"] == 28
    assert result["record_inventory"]["received"] == 28
    assert result["record_inventory"]["shared_panel_configurations_per_family"] == [{"split_seed": 0, "init_seed": 10}]
    split = result["panels"]["split"]
    assert split["seeds"] == [0, 101, 202]
    assert split["families"]["neuropixel"]["metrics"][PRIMARY]["values"] == [0.9375, 0.125, 0.25]
    assert split["paired_differences"][REFERENCE]["n"] == 3
    assert split["paired_differences"][REFERENCE]["degrees_of_freedom"] == 2
    assert split["paired_differences"][REFERENCE]["mean"] < 0
    assert result["panels"]["initialization"]["paired_differences"]["standard_nca"]["mean"] < 0
    assert result["decision"]["positive_support"] is True
    assert result["decision"]["uses_split_panel"] is False


@pytest.mark.parametrize("missing_family", ["neuropixel", REFERENCE])
def test_missing_primary_pair_cannot_be_replaced_by_split_runs(records, protocol, missing_family):
    kept = [row for row in records if not (row["config"]["family"] == missing_family
                                          and row["config"]["split_seed"] == 0 and row["config"]["init_seed"] == 14)]
    result = summarize(kept, protocol)
    assert result["decision"]["n_complete_pairs"] == 4
    assert result["decision"]["eligible"] is False and result["decision"]["positive_support"] is False
    assert result["decision"]["status"] == "incomplete"
    paired = result["panels"]["initialization"]["paired_differences"][REFERENCE]
    assert paired["values"][-1] is None and paired["missing_pairs"][0]["seed"] == 14


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), 1.5, -0.1])
def test_nonfinite_or_invalid_binding_blocks_positive_support_and_serializes_null(records, protocol, invalid):
    row = next(row for row in records if row["config"] == {
        "family": "neuropixel", "split_seed": 0, "init_seed": 12,
        "controller_source": {"irrelevant_extra_metadata": True}})
    row["metrics"][PRIMARY] = invalid
    result = summarize(records, protocol)
    assert result["decision"]["status"] == "incomplete" and result["decision"]["positive_support"] is False
    assert result["panels"]["initialization"]["paired_differences"][REFERENCE]["values"][2] is None
    assert any(row["reason"] == "invalid_primary_metric" for row in result["panels"]["initialization"]["unavailable_runs"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kind", ["failed", "duplicate"])
def test_failed_or_duplicate_trial_is_never_cherry_picked(records, protocol, kind):
    selected = next(row for row in records if row["config"]["family"] == "neuropixel"
                    and row["config"]["split_seed"] == 0 and row["config"]["init_seed"] == 13)
    if kind == "failed":
        selected["status"] = "failed"
    else:
        records.append(copy.deepcopy(selected))
    result = summarize(records, protocol)
    assert result["decision"]["positive_support"] is False
    assert result["decision"]["n_complete_pairs"] == 4
    if kind == "duplicate":
        assert result["record_inventory"]["duplicate_keys"] == [{"family": "neuropixel", "split_seed": 0, "init_seed": 13}]


def test_one_pair_and_zero_pairs_have_no_t_interval(records, protocol):
    one_pair = [row for row in records if row["config"]["family"] in ("neuropixel", REFERENCE)
                and row["config"]["split_seed"] == 0 and row["config"]["init_seed"] == 10]
    result = summarize(one_pair, protocol)
    paired = result["panels"]["initialization"]["paired_differences"][REFERENCE]
    assert paired["n"] == 1 and paired["mean"] == 0.125
    assert paired["sample_sd"] is None and paired["interval_95"] is None
    assert paired["degrees_of_freedom"] is None and not result["decision"]["positive_support"]
    empty = summarize([], protocol)
    paired = empty["panels"]["initialization"]["paired_differences"][REFERENCE]
    assert paired["n"] == 0 and paired["mean"] is None and paired["range"] is None
    assert empty["decision"]["status"] == "incomplete"
    json.dumps(empty, allow_nan=False)


def test_constant_pairs_have_explicit_zero_sd_and_unclipped_intervals():
    constant = describe_values([0.125] * 5, interval=True)
    assert constant["sample_sd"] == 0 and constant["interval_95"] == [0.125, 0.125]
    wide = describe_values([-1.0, 1.0], interval=True)
    assert wide["degrees_of_freedom"] == 1
    assert wide["interval_95"] == pytest.approx([-12.706204736432095, 12.706204736432095])
    nulls = describe_values([None, float("nan"), float("inf")], interval=True)
    assert nulls["values"] == [None, None, None] and nulls["interval_95"] is None


@pytest.mark.parametrize("np_values,reference_values,failed_check", [
    ([0.875] * 5, [0.75] * 5, "mean_np_at_least_threshold"),
    ([0.9375] * 5, [0.90625] * 5, "mean_advantage_at_least_threshold"),
    ([0.9375] * 5, [0.75, 0.75, 0.75, 1.0, 1.0], "ci_lower_bound_strictly_above_threshold"),
])
def test_each_h1_condition_is_required(records, protocol, np_values, reference_values, failed_check):
    set_initialization_values(records, "neuropixel", np_values)
    set_initialization_values(records, REFERENCE, reference_values)
    decision = summarize(records, protocol)["decision"]
    assert decision["eligible"] is True and decision["positive_support"] is False
    assert decision["status"] == "not_supported" and decision["checks"][failed_check] is False
    assert sum(not value for value in decision["checks"].values()) == 1


def test_zero_lower_endpoint_does_not_satisfy_strict_positive_bound(records, protocol):
    set_initialization_values(records, "neuropixel", [0.9375] * 5)
    set_initialization_values(records, REFERENCE, [0.9375] * 5)
    decision = summarize(records, protocol)["decision"]
    assert decision["interval_95"] == [0.0, 0.0]
    assert decision["checks"]["ci_lower_bound_strictly_above_threshold"] is False


def test_reference_is_fixed_and_unrelated_records_do_not_enter_panels(records, protocol):
    with pytest.raises(ValueError, match="frozen"):
        summarize_replications(records, "convgru", protocol)
    extra = copy.deepcopy(records[0])
    extra["config"]["init_seed"] = 7
    result = summarize(records + [extra], protocol)
    assert result["record_inventory"]["excluded_records"][0]["reason"] == "outside_declared_panels"
    assert result["decision"]["n_complete_pairs"] == 5 and result["decision"]["positive_support"] is True


def test_paired_bootstrap_retains_correlation_and_nominal_macro_weight():
    roles = np.array([0, 0] + [2] * 8 + [1, 3])
    correct = np.array([True, False] * 6)
    same = paired_binding_interval(correct, correct, roles)
    assert same["estimate"] == 0 and same["interval_95"] == [0, 0]
    np_correct = roles == 0
    reference_correct = roles == 2
    opposite = paired_binding_interval(np_correct, reference_correct, roles)
    # Role 0 has +1 and role 2 has -1. The macro mean is zero, not the pooled -0.6.
    assert opposite["estimate"] == 0 and opposite["interval_95"] == [0, 0]
    assert opposite["per_role"]["AGENTE"] == {"n": 2, "mean_difference": 1.0}
    assert opposite["per_role"]["PACIENTE"] == {"n": 8, "mean_difference": -1.0}


def test_bootstrap_artifacts_have_known_signed_distribution_and_private_rng():
    roles = np.array([0, 0, 2, 2])
    target = np.array([5, 6, 7, 8])
    np_artifact = {"prediction": np.array([5, 0, 7, 0]), "target": target, "role": roles}
    reference = {"prediction": np.array([0, 6, 7, 8]), "target": target.copy(), "role": roles.copy()}
    np.random.seed(111)
    before = np.random.get_state()
    result = paired_binding_interval(np_artifact, reference, roles)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert result["estimate"] == -0.25
    # Enumerating the 16 ordered within-role resamples gives endpoint masses 1/16.
    assert result["interval_95"] == [-1.0, 0.5]
    assert result["seed"] == 61003 and result["repetitions"] == 2000
    assert "not training-run uncertainty" in result["scope"]
    np.random.seed(999)
    np.random.random(100)
    assert paired_binding_interval(np_artifact, reference, roles) == result


@pytest.mark.parametrize("change", ["targets", "roles", "class_ids", "nonfinite", "missing_role"])
def test_bootstrap_rejects_unpaired_or_unscored_inputs(change):
    roles = np.array([0, 0, 2, 2])
    target = np.array([5, 6, 7, 8])
    np_artifact = {"prediction": target.copy(), "target": target, "role": roles}
    reference = {"prediction": target.copy(), "target": target.copy(), "role": roles.copy()}
    if change == "targets":
        reference["target"][0] = 6
    elif change == "roles":
        reference["role"][0] = 2
    elif change == "class_ids":
        np_artifact, reference = target, target.copy()
    elif change == "nonfinite":
        np_artifact["prediction"] = np.array([5, 6, np.nan, 8])
    else:
        roles[:] = 0
    with pytest.raises(ValueError):
        paired_binding_interval(np_artifact, reference, roles)
