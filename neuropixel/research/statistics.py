"""Separate seed-panel summaries and conditional paired bootstrap intervals.

This module consumes evaluation records or saved decisions only. It neither
loads files nor imports model, training or dataset implementations.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np
from scipy.stats import t as student_t


ROLES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
PRIMARY_METRIC = "macro_agent_patient_accuracy"
FROZEN_REFERENCE_FAMILY = "relative_transformer"
METRIC_PATHS = {
    "accuracy": ("accuracy",),
    "macro_all_roles": ("macro_all_roles",),
    PRIMARY_METRIC: (PRIMARY_METRIC,),
    "cross_entropy": ("cross_entropy",),
    **{f"role_{role}_accuracy": ("per_role", role, "accuracy") for role in ROLES},
}


def _finite_value(record, path):
    value = record.get("metrics", {})
    try:
        for key in path:
            value = value[key]
    except (KeyError, TypeError):
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None
    value = float(value)
    if value < 0 or (path != ("cross_entropy",) and value > 1):
        return None
    return value


def describe_values(values, *, confidence_level=0.95, interval=False):
    """Describe finite observations while retaining missing slots as JSON null.

    SD is the sample SD with denominator n-1. A t interval is undefined for
    fewer than two observations. Constant observations have an explicit zero
    SD and degenerate interval. Intervals are never clipped to an accuracy range.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if interval and confidence_level != 0.95:
        raise ValueError("The frozen paired interval uses confidence_level=0.95")
    cleaned = [float(value) if isinstance(value, Real) and not isinstance(value, bool)
               and math.isfinite(value) else None for value in values]
    available = [value for value in cleaned if value is not None]
    n = len(available)
    mean = math.fsum(available) / n if n else None
    sd = None
    if n >= 2:
        sd = (0.0 if min(available) == max(available)
              else math.sqrt(math.fsum((value - mean) ** 2 for value in available) / (n - 1)))
    result = {
        "values": cleaned, "n": n, "expected_n": len(cleaned),
        "complete": n == len(cleaned) and n > 0,
        "mean": mean, "sample_sd": sd,
        "range": [min(available), max(available)] if n else None,
    }
    if interval:
        bounds = None
        if n >= 2:
            half_width = float(student_t.ppf((1 + confidence_level) / 2, df=n - 1)) * sd / math.sqrt(n)
            bounds = [mean - half_width, mean + half_width]
        result.update(confidence_level=confidence_level, interval_method="paired differences Student t",
                      degrees_of_freedom=n - 1 if n >= 2 else None, interval_95=bounds)
    return result


def _run_key(record):
    config = record.get("config")
    if not isinstance(config, Mapping):
        return None
    family, split, initialization = (config.get("family"), config.get("split_seed"), config.get("init_seed"))
    if (not isinstance(family, str) or not isinstance(split, Integral) or isinstance(split, bool)
            or not isinstance(initialization, Integral) or isinstance(initialization, bool)):
        return None
    return family, int(split), int(initialization)


def _key_record(key):
    return {"family": key[0], "split_seed": key[1], "init_seed": key[2]}


def _panel(axis, seeds, fixed_seed, families, indexed, duplicates, confidence):
    pairs = [(fixed_seed, seed) if axis == "init_seed" else (seed, fixed_seed) for seed in seeds]
    family_results, unavailable = {}, []
    for family in families:
        run_rows, metric_values = [], {name: [] for name in METRIC_PATHS}
        for seed, (split, initialization) in zip(seeds, pairs):
            key = (family, split, initialization)
            row = indexed.get(key)
            if key in duplicates:
                reason = "duplicate_run"
            elif row is None:
                reason = "missing_run"
            elif row.get("status") != "completed":
                reason = "run_not_completed"
            elif _finite_value(row, METRIC_PATHS[PRIMARY_METRIC]) is None:
                reason = "invalid_primary_metric"
            else:
                reason = None
            run_row = {"seed": seed, "split_seed": split, "init_seed": initialization,
                       "primary_usable": reason is None, "reason": reason}
            run_rows.append(run_row)
            if reason is not None:
                unavailable.append({**_key_record(key), "reason": reason})
            available_record = row is not None and key not in duplicates and row.get("status") == "completed"
            for name, path in METRIC_PATHS.items():
                metric_values[name].append(_finite_value(row, path) if available_record else None)
        family_results[family] = {
            "runs": run_rows,
            "metrics": {name: describe_values(values) for name, values in metric_values.items()},
        }
    differences = {}
    np_values = family_results["neuropixel"]["metrics"][PRIMARY_METRIC]["values"]
    for reference in families:
        if reference == "neuropixel":
            continue
        reference_values = family_results[reference]["metrics"][PRIMARY_METRIC]["values"]
        deltas, paired_np, paired_reference, missing_pairs = [], [], [], []
        for seed, np_value, reference_value in zip(seeds, np_values, reference_values):
            if np_value is None or reference_value is None:
                deltas.append(None)
                missing_pairs.append({"seed": seed, "neuropixel_unavailable": np_value is None,
                                      "reference_unavailable": reference_value is None})
            else:
                deltas.append(np_value - reference_value)
                paired_np.append(np_value)
                paired_reference.append(reference_value)
        difference = describe_values(deltas, confidence_level=confidence, interval=True)
        difference.update(metric=PRIMARY_METRIC, direction="neuropixel minus reference",
                          reference_family=reference, seeds=list(seeds), missing_pairs=missing_pairs,
                          paired_np_mean=math.fsum(paired_np) / len(paired_np) if paired_np else None,
                          paired_reference_mean=(math.fsum(paired_reference) / len(paired_reference)
                                                 if paired_reference else None))
        differences[reference] = difference
    return {
        "axis": axis, "seeds": list(seeds),
        "fixed_split_seed" if axis == "init_seed" else "fixed_init_seed": fixed_seed,
        "families": family_results, "paired_differences": differences,
        "unavailable_runs": unavailable, "complete": not unavailable,
    }


def summarize_replications(records, reference_family, protocol):
    """Summarize the frozen five-initialization and three-split panels separately.

    The caller must first verify/recount saved evaluation artifacts. Records are
    identified by (family, split_seed, init_seed), never by list position. Extra
    configuration provenance fields are retained by the caller and are ignored
    here. Duplicate keys are excluded rather than choosing a favorable record.
    Only completed, finite binding pairs can enter the H1 calculation.
    """
    if reference_family != FROZEN_REFERENCE_FAMILY:
        raise ValueError("Item 5 uses the relative_transformer reference frozen by the item-4 pilot")
    families = list(protocol["training"]["families"])
    if len(set(families)) != len(families) or "neuropixel" not in families or reference_family not in families:
        raise ValueError("Protocol families must be unique and contain NeuroPixel and the fixed reference")
    decision = protocol["decision"]
    if decision["primary_metric"] != PRIMARY_METRIC or decision["primary_panel"] != "initialization":
        raise ValueError("Unsupported primary metric or decision panel")
    confidence = decision["confidence_level"]
    if confidence != 0.95:
        raise ValueError("This frozen item-5 analysis requires a two-sided 95% interval")
    initialization = protocol["replication"]["initialization_panel"]
    split = protocol["replication"]["split_panel"]
    init_seeds, split_seeds = list(initialization["init_seeds"]), list(split["split_seeds"])
    if (len(set(init_seeds)) != len(init_seeds) or len(set(split_seeds)) != len(split_seeds)
            or not init_seeds or not split_seeds):
        raise ValueError("Panel seeds must be nonempty and unique")
    expected_pairs = ({(initialization["split_seed"], seed) for seed in init_seeds}
                      | {(seed, split["init_seed"]) for seed in split_seeds})
    expected_keys = {(family, split_seed, init_seed) for family in families
                     for split_seed, init_seed in expected_pairs}
    records = list(records)
    indexed, duplicates, excluded = {}, set(), []
    for index, row in enumerate(records):
        key = _run_key(row) if isinstance(row, Mapping) else None
        if key not in expected_keys:
            excluded.append({"index": index, "reason": "invalid_run_identity" if key is None else "outside_declared_panels",
                             **(_key_record(key) if key is not None else {})})
            continue
        if key in indexed:
            duplicates.add(key)
        else:
            indexed[key] = row
    panels = {
        "initialization": _panel("init_seed", init_seeds, initialization["split_seed"], families,
                                  indexed, duplicates, confidence),
        "split": _panel("split_seed", split_seeds, split["init_seed"], families,
                         indexed, duplicates, confidence),
    }
    primary = panels["initialization"]["paired_differences"][reference_family]
    complete = (len(init_seeds) == decision["required_complete_pairs"] == 5
                and primary["complete"] and primary["n"] == 5)
    thresholds = {
        "minimum_np_binding": decision["minimum_np_accuracy"],
        "minimum_mean_advantage": decision["point_estimate_advantage_screen"],
        "strict_ci_lower_bound": decision["directional_ci_lower_bound"],
    }
    checks = {
        "five_complete_finite_pairs": complete,
        "mean_np_at_least_threshold": (primary["paired_np_mean"] is not None
                                       and primary["paired_np_mean"] >= thresholds["minimum_np_binding"]),
        "mean_advantage_at_least_threshold": (primary["mean"] is not None
                                              and primary["mean"] >= thresholds["minimum_mean_advantage"]),
        "ci_lower_bound_strictly_above_threshold": (primary["interval_95"] is not None
                                                    and primary["interval_95"][0] > thresholds["strict_ci_lower_bound"]),
    }
    supported = all(checks.values())
    overlap = sorted({(initialization["split_seed"], seed) for seed in init_seeds}
                     & {(seed, split["init_seed"]) for seed in split_seeds})
    return {
        "schema_version": 1, "item": 5, "protocol_id": protocol["protocol_id"],
        "primary_metric": PRIMARY_METRIC, "reference_family": reference_family,
        "panels": panels,
        "decision": {
            "hypothesis": "H1", "panel": "initialization", "reference_family": reference_family,
            "eligible": complete, "positive_support": supported,
            "status": "supported" if supported else "not_supported" if complete else "incomplete",
            "checks": checks, "thresholds": thresholds,
            "n_complete_pairs": primary["n"], "mean_np_binding": primary["paired_np_mean"],
            "mean_advantage": primary["mean"], "interval_95": primary["interval_95"],
            "degrees_of_freedom": primary["degrees_of_freedom"], "uses_split_panel": False,
        },
        "record_inventory": {
            "received": len(records), "expected_unique_runs": len(expected_keys),
            "present_unique_runs": len(indexed), "duplicate_keys": [_key_record(key) for key in sorted(duplicates)],
            "excluded_records": excluded,
            "shared_panel_configurations_per_family": [{"split_seed": s, "init_seed": i} for s, i in overlap],
        },
        "limitations": [
            "Panels overlap and vary different factors; their seven unique configurations per family are not pooled as IID observations.",
            "The initialization t interval conditions on the frozen split; the split panel conditions on initialization 10.",
            "Student t intervals assume approximately independent, normally distributed run differences; five or three seeds provide limited precision.",
            "Available incomplete-panel summaries remain descriptive; missing, duplicate or non-finite primary pairs block positive H1 support.",
            "The fixed Transformer comparison alone determines H1; comparisons with other baselines are descriptive and are not multiplicity-adjusted.",
            "This module does not validate source, checkpoints or dataset hashes; the caller must verify and recount the input artifacts first.",
            "Conditional example-bootstrap intervals do not enter the across-initialization H1 decision.",
        ],
    }


def _paired_correctness(pred_np, pred_ref, roles):
    mappings = isinstance(pred_np, Mapping), isinstance(pred_ref, Mapping)
    if any(mappings) and not all(mappings):
        raise ValueError("Both inputs must use the same form: artifact mappings or correctness vectors")
    if all(mappings):
        targets = [np.asarray(prediction["target"]) for prediction in (pred_np, pred_ref)]
        predictions = [np.asarray(prediction["prediction"]) for prediction in (pred_np, pred_ref)]
        if any(values.shape != roles.shape or values.dtype.kind not in "biu" for values in targets + predictions):
            raise ValueError("Prediction and target arrays must be aligned integer vectors")
        if not np.array_equal(targets[0], targets[1]):
            raise ValueError("Paired prediction artifacts must contain identical ordered targets")
        for artifact in (pred_np, pred_ref):
            for key in ("role", "roles"):
                if key in artifact and not np.array_equal(np.asarray(artifact[key]), roles):
                    raise ValueError("Artifact roles differ from the supplied ordered roles")
        return predictions[0] == targets[0], predictions[1] == targets[1]
    correct = [np.asarray(prediction) for prediction in (pred_np, pred_ref)]
    if any(values.shape != roles.shape or values.dtype.kind != "b" for values in correct):
        raise ValueError("Raw inputs must be boolean correctness vectors; class IDs require target mappings")
    return correct[0], correct[1]


def paired_binding_interval(pred_np, pred_ref, roles, seed=61003, repetitions=2000):
    """Bootstrap NP-reference binding differences using common within-role draws.

    Inputs are either mappings with prediction/target arrays (and optional role
    arrays), or boolean correctness vectors. Nominal roles 0 and 2 each receive
    weight one half regardless of their sample counts. Randomness is private.
    This percentile interval conditions on these checkpoints and examples; it
    is not an interval over training runs and is never used for the H1 decision.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, Integral) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    roles = np.asarray(roles)
    if roles.ndim != 1 or not len(roles) or roles.dtype.kind not in "iu" or not np.isin(roles, [0, 1, 2, 3]).all():
        raise ValueError("roles must be a nonempty integer vector containing role IDs 0 through 3")
    if not np.any(roles == 0) or not np.any(roles == 2):
        raise ValueError("Both agent role 0 and patient role 2 must occur")
    correct_np, correct_ref = _paired_correctness(pred_np, pred_ref, roles)
    differences = correct_np.astype(np.int8) - correct_ref.astype(np.int8)
    rng = np.random.default_rng(int(seed))
    samples = np.zeros(int(repetitions), dtype=np.float64)
    role_results = {}
    for role in (0, 2):
        values = differences[roles == role]
        indices = rng.integers(len(values), size=(int(repetitions), len(values)))
        samples += values[indices].mean(axis=1) / 2
        role_results[ROLES[role]] = {"n": len(values), "mean_difference": float(values.mean())}
    point = sum(row["mean_difference"] for row in role_results.values()) / 2
    return {
        "metric": PRIMARY_METRIC, "direction": "neuropixel minus reference",
        "estimate": point, "interval_95": np.quantile(samples, [0.025, 0.975], method="linear").tolist(),
        "method": "paired percentile bootstrap stratified by nominal role",
        "confidence_level": 0.95, "seed": int(seed), "repetitions": int(repetitions),
        "per_role": role_results, "common_resampling_indices": True,
        "scope": "conditional on these checkpoints and evaluated examples; not training-run uncertainty",
    }
