"""Independent saved-artifact recount for item 9; no generator/model imports.

Only the complete Stage-B study is analyzed here. The separate item-9 preflight
auditor handles Stage A. No checkpoint is deserialized and no data is generated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"
import numpy as np


CONDITIONS = ("base", "swap_queried_agent_patient", "swap_other_agent_patient",
              "relabel_events", "query_switch", "layout_permutation")
FAMILIES = ("neuropixel", "relative_transformer")
ROLES = ("agent", "action", "patient", "place")
SEEDS = (40, 41, 42, 43, 44)
FLOAT32_TOLERANCE = 16 * 2**-23
NLL_ATOL, NLL_RTOL = 1e-10, 1e-12
OFFLINE_RUNTIME = {"python": "3.12.14", "numpy": "2.3.5", "scipy": "1.17.0"}
ARRAY_FIELDS = {"canvas", "target", "base_target", "role", "query_event", "base_query_event",
                "group_index", "condition_index", "changed_gold", "group_id", "record_id", "pair_id"}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json(payload):
    def invalid(value):
        raise AuditError(f"nonfinite JSON value: {value}")
    return json.loads(payload, object_pairs_hook=_object, parse_constant=invalid)


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def near(actual, expected, label, *, atol=FLOAT32_TOLERANCE, rtol=FLOAT32_TOLERANCE):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), f"{label}: shape/nonfinite")
    require(np.allclose(a, b, rtol=rtol, atol=atol), f"{label}: numerical discrepancy")


class Reader:
    """Read local files and retain a byte/hash inventory, with safe relative paths."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}

    def path(self, name):
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "artifact path must be relative")
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root) and path.is_file(), f"missing/outside artifact: {name}")
        self.inputs[relative.as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        return path

    def verify(self, record):
        require(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}, "artifact schema differs")
        require(type(record["bytes"]) is int and record["bytes"] >= 0 and valid_hash(record["sha256"]), "artifact identity invalid")
        path = self.path(record["path"])
        require(self.inputs[Path(record["path"]).as_posix()] == {k: record[k] for k in ("bytes", "sha256")},
                f"artifact byte/hash mismatch: {record['path']}")
        return path

    def json(self, name):
        path = self.path(name)
        payload = path.read_bytes()
        return decode_json(gzip.decompress(payload) if path.suffix == ".gz" else payload)

    def arrays(self, name):
        path = self.path(name)
        with np.load(path, allow_pickle=False) as source:
            require(len(source.files) == len(set(source.files)), "duplicate NPZ keys")
            return {key: source[key] for key in source.files}


def category(role):
    return range(5, 17) if role in (0, 2) else range(17, 27) if role == 1 else range(27, 35)


def parse_visible(canvas):
    """Independent row scanner: no generator, metadata target, or scenario input."""
    canvas = np.asarray(canvas)
    require(canvas.shape == (10, 8) and canvas.dtype.kind in "iu", "visible canvas must be integer10x8")
    require(((canvas >= 0) & (canvas < 37)).all(), "visible token outside vocabulary")
    query = canvas[9]
    require(np.array_equal(query[[0, 1, 2, 3, 4, 7]], np.zeros(6, dtype=int)), "query padding/output differs")
    require(int(query[5]) in (35, 36) and int(query[6]) in (1, 2, 3, 4), "invalid visible query")
    facts, positions = {}, {}
    for row_index in range(9):
        columns = np.flatnonzero(canvas[row_index])
        if len(columns) == 0:
            continue
        require(len(columns) == 3 and int(columns[-1])-int(columns[0]) == 2, "malformed fact triple")
        start = int(columns[0])
        event_token, role_token, filler = map(int, canvas[row_index, columns])
        require(event_token in (35, 36) and role_token in (1, 2, 3, 4), "event/role fact order differs")
        key = event_token-35, role_token-1
        require(key not in facts and filler in category(key[1]), "duplicate fact or filler category mismatch")
        facts[key], positions[key] = filler, (row_index, start)
    require(set(facts) == {(event, role) for event in range(2) for role in range(4)}, "incomplete eight-fact inventory")
    require(len(set(facts.values())) == 8, "fillers must be distinct")
    bag = {"nouns": sorted(v for (_, role), v in facts.items() if role in (0, 2)),
           "verbs": sorted(v for (_, role), v in facts.items() if role == 1),
           "places": sorted(v for (_, role), v in facts.items() if role == 3)}
    event, role = int(query[5])-35, int(query[6])-1
    return {"facts": facts, "positions": positions, "event": event, "role": role,
            "gold": facts[event, role], "bag": bag, "group_id": compact_hash(bag)}


def _integer_vector(arrays, name, n, lower, upper):
    value = np.asarray(arrays[name])
    require(value.shape == (n,) and value.dtype.kind in "iu" and ((value >= lower) & (value < upper)).all(),
            f"invalid integer vector: {name}")


def validate_dataset(arrays, conditions, groups, *, split=None):
    """Verify ordered grouping, visible truth, transformations and paired labels."""
    require(tuple(conditions) in (("base",), CONDITIONS), "unknown/reordered condition inventory")
    require(type(groups) is int and groups > 0 and set(arrays) == ARRAY_FIELDS, "dataset fields/groups differ")
    n = len(conditions)*groups*8
    require(arrays["canvas"].shape == (n, 10, 8) and arrays["canvas"].dtype.kind in "iu", "dataset canvas shape differs")
    for name, upper in (("target", 37), ("base_target", 37), ("role", 4), ("query_event", 2),
                        ("base_query_event", 2), ("group_index", groups), ("condition_index", len(conditions))):
        _integer_vector(arrays, name, n, 0, upper)
    require(arrays["changed_gold"].shape == (n,) and arrays["changed_gold"].dtype == np.bool_, "changed_gold must be boolean")
    for name in ("group_id", "record_id", "pair_id"):
        require(arrays[name].shape == (n,) and arrays[name].dtype.kind == "U"
                and all(valid_hash(str(value)) for value in arrays[name]), f"invalid {name}")
    require(len(set(arrays["record_id"])) == n, "duplicate record IDs")
    require(np.array_equal(arrays["condition_index"], np.repeat(np.arange(len(conditions)), groups*8)), "condition order differs")
    require(np.array_equal(arrays["group_index"], np.tile(np.repeat(np.arange(groups), 8), len(conditions))), "group order differs")
    require(np.array_equal(arrays["role"], np.tile(np.arange(4), len(conditions)*groups*2)), "role order differs")
    require(np.array_equal(arrays["base_query_event"], np.tile(np.repeat(np.arange(2), 4), len(conditions)*groups)), "anchor query order differs")
    base = [parse_visible(arrays["canvas"][group*8]) for group in range(groups)]
    require(len({row["group_id"] for row in base}) == groups, "bags are not unique")
    parsed = []
    for index in range(n):
        group, ci = int(arrays["group_index"][index]), int(arrays["condition_index"][index])
        event, role = int(arrays["base_query_event"][index]), int(arrays["role"][index])
        condition, anchor = conditions[ci], base[group]
        visible = parse_visible(arrays["canvas"][index])
        require((visible["event"], visible["role"]) == (int(arrays["query_event"][index]), role), "visible query/metadata differ")
        require(visible["gold"] == int(arrays["target"][index]), "visible gold differs from target")
        require(visible["group_id"] == str(arrays["group_id"][index]) == anchor["group_id"], "visible bag/group identity differs")
        if split is not None:
            bucket = int(visible["group_id"], 16) % 100
            observed_split = "train" if bucket < 70 else "validation" if bucket < 85 else "final"
            require(split == observed_split, "visible group belongs to another split")
        gold = anchor["facts"][event, role]
        require(int(arrays["base_target"][index]) == gold, "base target differs from visible anchor")
        require(bool(arrays["changed_gold"][index]) == (visible["gold"] != gold), "changed-gold flag differs")
        base_index = group*8 + event*4 + role
        require(arrays["pair_id"][index] == arrays["pair_id"][base_index], "paired identity differs")
        expected_facts = dict(anchor["facts"])
        expected_query = event
        expected_canvas = arrays["canvas"][base_index].copy()
        if condition in CONDITIONS[1:3]:
            affected = event if condition == CONDITIONS[1] else 1-event
            a, p = (affected, 0), (affected, 2)
            expected_facts[a], expected_facts[p] = expected_facts[p], expected_facts[a]
            for key in (a, p):
                r, c = anchor["positions"][key]
                expected_canvas[r, c+2] = expected_facts[key]
        elif condition == "relabel_events":
            expected_facts = {(1-e, r): filler for (e, r), filler in expected_facts.items()}
            for (e, _), (r, c) in anchor["positions"].items():
                expected_canvas[r, c] = 36-e
            expected_query = 1-event
            expected_canvas[9, 5] = 35+expected_query
        elif condition == "query_switch":
            expected_query = 1-event
            expected_canvas[9, 5] = 35+expected_query
        require(visible["facts"] == expected_facts and visible["event"] == expected_query, "transformation semantics differ")
        if condition == "layout_permutation":
            require(not np.array_equal(arrays["canvas"][index, :9], expected_canvas[:9]), "layout intervention is identity")
            first = ci*groups*8 + group*8
            require(np.array_equal(arrays["canvas"][index, :9], arrays["canvas"][first, :9]), "layout varies across queries")
        else:
            require(np.array_equal(arrays["canvas"][index], expected_canvas), "transformation changes undeclared cells")
        if condition == "base":
            require(np.array_equal(arrays["canvas"][index, :9], arrays["canvas"][group*8, :9]), "base layout varies across queries")
        parsed.append(visible)
    return parsed


def recount_predictions(values, arrays):
    n = len(arrays["target"])
    required = {"logits", "pred", "nll", "target", "role", "group_index", "record_id"}
    if "condition_index" in values:
        required.add("condition_index")
    require(set(values) == required, "prediction NPZ fields differ")
    for name in required - {"logits", "pred", "nll"}:
        require(np.array_equal(values[name], arrays[name]), f"ordered prediction alignment differs: {name}")
    logits = values["logits"]
    require(logits.shape == (n, 37) and logits.dtype == np.float32 and np.isfinite(logits).all(), "logits must be finite float32[N,37]")
    require(np.array_equal(logits[:, 0], np.full(n, -10000, dtype=np.float32)), "PAD logit policy differs")
    _integer_vector(values, "pred", n, 0, 37)
    require(np.array_equal(values["pred"], logits.argmax(axis=1)), "saved prediction differs from first-index argmax")
    require(values["nll"].shape == (n,) and values["nll"].dtype.kind == "f"
            and np.isfinite(values["nll"]).all() and (values["nll"] >= 0).all(), "NLL shape/nonfinite/negative")
    doubles = logits.astype(np.float64)
    shifted = doubles-doubles.max(axis=1, keepdims=True)
    nll = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(n), arrays["target"]]
    near(values["nll"], nll, "saved vs independently recomputed NLL", atol=NLL_ATOL, rtol=NLL_RTOL)
    return values["pred"], nll, float(np.max(np.abs(values["nll"]-nll)))


def point_metrics(pred, target, roles, nll, groups):
    correct = np.asarray(pred) == np.asarray(target)
    role_accuracy = [float(correct[roles == role].mean()) for role in range(4)]
    block_ids = np.unique(groups)
    require(len(correct) == len(block_ids)*8, "metric population is not eight queries per bag")
    all_eight = [bool(correct[groups == group].all()) for group in block_ids]
    return {"n": len(correct), "groups": len(block_ids), "correct": int(correct.sum()),
            "global": float(correct.mean()), "per_role": role_accuracy,
            "binding": statistics.mean([role_accuracy[0], role_accuracy[2]]),
            "macro": statistics.mean(role_accuracy), "cross_entropy": float(np.mean(nll, dtype=np.float64)),
            "all_eight": statistics.mean(all_eight)}


def compare_metrics(saved, recounted):
    require(set(saved) == set(recounted), "saved metric schema differs")
    for name in ("n", "groups", "correct"):
        require(type(saved[name]) is int and saved[name] == recounted[name], f"saved {name} count differs")
    for name in set(recounted)-{"n", "groups", "correct"}:
        if name == "cross_entropy":
            near(saved[name], recounted[name], "saved cross entropy", atol=NLL_ATOL, rtol=NLL_RTOL)
        else:
            require(saved[name] == recounted[name], f"saved {name} differs")


def verify_controls(saved, arrays, parsed, majority):
    names = {"symbolic", "role_only", "event_category", "bag_category", "train_role_majority"}
    require(set(saved) == names, "control names differ")
    expected = {name: [] for name in names}
    for index, visible in enumerate(parsed):
        facts, event, role = visible["facts"], visible["event"], visible["role"]
        gold = int(arrays["target"][index])
        candidates = {
            "symbolic": [facts[event, role]],
            "role_only": [value for (_, r), value in facts.items() if r == role],
            "event_category": [value for (e, _), value in facts.items() if e == event and value in category(role)],
            "bag_category": [value for value in facts.values() if value in category(role)],
        }
        for name, tokens in candidates.items():
            expected[name].append(float(gold in tokens)/len(tokens))
        expected["train_role_majority"].append(float(majority[str(role)] == gold))
    result = {}
    for name in sorted(names):
        values = np.asarray(expected[name], dtype=np.float64)
        require(saved[name].shape == values.shape and np.array_equal(saved[name], values), f"input-only control differs: {name}")
        result[name] = {"n": len(values), "expected_correct": float(values.sum()),
                        "global": float(values.mean()), "binding": float(values[np.isin(arrays["role"], [0, 2])].mean()),
                        "per_role": [float(values[arrays["role"] == role].mean()) for role in range(4)],
                        "scope": "analytic answer probability, not sampled predictions"}
    return result


def paired_seed_interval(left, right):
    """Five training-realization differences; conditional on recipe and dataset."""
    require(set(left) == set(SEEDS) == set(right), "exactly five matched seeds40..44 are required")
    values = [float(left[seed])-float(right[seed]) for seed in SEEDS]
    require(all(math.isfinite(value) for value in values), "nonfinite initialized-run difference")
    from scipy.stats import t
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = float(t.ppf(.975, 4))*sd/math.sqrt(5)
    return {"seeds": list(SEEDS), "left": [float(left[s]) for s in SEEDS], "right": [float(right[s]) for s in SEEDS],
            "differences": values, "n": 5, "df": 4, "mean": mean, "sample_sd": sd,
            "range": [min(values), max(values)], "t95": [mean-half, mean+half],
            "scope": "training realizations vary initialization and input/firing streams; conditional on fixed data and selected recipe; H1 unchanged"}


def bootstrap_indices(groups, repetitions=2000, seed=94001):
    require(type(groups) is int and groups > 0 and type(repetitions) is int and repetitions > 0, "invalid bootstrap dimensions")
    return np.random.Generator(np.random.PCG64(seed)).integers(0, groups, size=(repetitions, groups), dtype=np.int64)


def ratio_interval(numerators, denominators, indices):
    numerator = np.asarray(numerators, dtype=np.float64)
    denominator = np.asarray(denominators, dtype=np.float64)
    require(numerator.shape == denominator.shape and numerator.ndim == 1 and np.isfinite(numerator).all()
            and np.isfinite(denominator).all() and (denominator >= 0).all(), "invalid bag-level ratio inputs")
    require(indices.ndim == 2 and indices.shape[1] == len(numerator) and indices.dtype.kind in "iu"
            and ((indices >= 0) & (indices < len(numerator))).all(), "invalid shared bag bootstrap indices")
    total = float(denominator.sum())
    if total == 0:
        return {"numerator": float(numerator.sum()), "denominator": 0, "estimate": None, "bag_bootstrap95": None}
    sampled_denominator = denominator[indices].sum(axis=1)
    require((sampled_denominator > 0).all(), "bootstrap ratio includes a zero-denominator resample")
    samples = numerator[indices].sum(axis=1)/sampled_denominator
    return {"numerator": float(numerator.sum()), "denominator": total,
            "estimate": float(numerator.sum()/total),
            "bag_bootstrap95": np.quantile(samples, [.025, .975], method="linear").tolist()}


def condition_diagnostics(pred, arrays, nll, indices, memory_check=None):
    groups, conditions = indices.shape[1], len(np.unique(arrays["condition_index"]))
    correctness = (pred == arrays["target"]).reshape(conditions, groups, 8)
    guesses, changed = pred.reshape(conditions, groups, 8), arrays["changed_gold"].reshape(conditions, groups, 8)
    losses = nll.reshape(conditions, groups, 8)
    result, per_bag = {}, {}
    for ci, condition in enumerate(CONDITIONS[:conditions]):
        if memory_check is not None:
            memory_check("bootstrap_condition_"+condition)
        current = correctness[ci]
        names = {"global": np.ones(8, dtype=bool), "binding": np.array([1, 0, 1, 0]*2, dtype=bool)}
        names.update({"role_"+role: np.tile(np.arange(4) == ri, 2) for ri, role in enumerate(ROLES)})
        rows, blocks = {}, {}
        for name, mask in names.items():
            blocks[name] = current[:, mask].mean(axis=1)
            rows[name] = ratio_interval(blocks[name], np.ones(groups), indices)
            rows[name]["units"] = "numerator=sum of per-bag query-accuracy means; denominator=number of bags"
        blocks["all_eight"] = current.all(axis=1).astype(float)
        blocks["cross_entropy"] = losses[ci].mean(axis=1)
        for name in ("all_eight", "cross_entropy"):
            rows[name] = ratio_interval(blocks[name], np.ones(groups), indices)
            rows[name]["units"] = ("numerator=number of entirely correct bags; denominator=number of bags" if name == "all_eight"
                                   else "numerator=sum of per-bag mean query NLL; denominator=number of bags")
        joint = correctness[0] & current
        equal = guesses[0] == guesses[ci]
        paired = {}
        for name, mask in {"all": np.ones(8, dtype=bool), **{r: np.tile(np.arange(4) == i, 2) for i, r in enumerate(ROLES)}}.items():
            eligible = np.broadcast_to(mask, (groups, 8))
            invariant, change = eligible & ~changed[ci], eligible & changed[ci]
            paired[name] = {
                "joint_correct": ratio_interval((joint & eligible).sum(1), eligible.sum(1), indices),
                "invariant_prediction_equal": ratio_interval((equal & invariant).sum(1), invariant.sum(1), indices),
                "changed_prediction_different": ratio_interval((~equal & change).sum(1), change.sum(1), indices)}
            for value in paired[name].values():
                value["units"] = "eligible query counts; bootstrap resamples whole bags"
        result[condition] = {"scores": rows, "counterfactual_vs_base": paired}
        per_bag[condition] = blocks
    return result, per_bag


def duplicate_agreement(arrays, values):
    first, repeated, identical_predictions, max_abs = {}, 0, 0, 0.0
    for index, canvas in enumerate(arrays["canvas"]):
        key = canvas.astype(np.uint8).tobytes()
        if key not in first:
            first[key] = index
            continue
        other = first[key]
        require(arrays["target"][index] == arrays["target"][other], "same visible canvas has conflicting gold")
        repeated += 1
        identical_predictions += int(values["pred"][index] == values["pred"][other])
        error = float(np.max(np.abs(values["logits"][index].astype(float)-values["logits"][other].astype(float))))
        max_abs = max(max_abs, error)
        near(values["logits"][index], values["logits"][other], "duplicate-canvas logits")
    require(repeated == identical_predictions, "duplicate visible canvas has different argmax predictions")
    return {"nominal_rows": len(arrays["target"]), "unique_canvases": len(first),
            "duplicate_rows_compared_to_first": repeated, "equal_argmax": identical_predictions,
            "maximum_absolute_logit_difference": max_abs,
            "scope": "duplicate agreement is not independent accuracy evidence"}


def content_digest(canvas, target):
    digest = hashlib.sha256()
    for name, value in (("canvas", canvas), ("target", target)):
        a = np.ascontiguousarray(value, dtype="<i8")
        header = (json.dumps({"name": name, "shape": list(a.shape), "dtype": "<i8"},
                             sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(a.tobytes())
    return digest.hexdigest()


def scenario_inventory(scenarios, *, expected_split):
    """Validate saved semantic records, not regenerate random examples."""
    groups, majority_counts = [], [Counter() for _ in range(4)]
    for scenario in scenarios:
        require(set(scenario) == {"schema_version", "group_id", "bag", "facts", "layout"}
                and scenario["schema_version"] == 1, "saved scenario schema differs")
        require(len(scenario["facts"]) == len(scenario["layout"]) == 8, "saved scenario fact count differs")
        canvas = np.zeros((10, 8), dtype=np.int64)
        occupied_rows = set()
        for fact, slot in zip(scenario["facts"], scenario["layout"]):
            require(set(fact) == {"event", "role", "filler"} and set(slot) == {"row", "start"}, "scenario fact/layout fields differ")
            require(all(type(x) is int for x in (*fact.values(), *slot.values())), "scenario indices must be integers")
            e, r, filler, row, start = fact["event"], fact["role"], fact["filler"], slot["row"], slot["start"]
            require(e in (0, 1) and r in range(4) and 0 <= row < 9 and 0 <= start < 6 and row not in occupied_rows,
                    "invalid scenario geometry")
            occupied_rows.add(row)
            canvas[row, start:start+3] = [35+e, 1+r, filler]
            majority_counts[r][filler] += 1
        canvas[9, 5:7] = [35, 1]
        parsed = parse_visible(canvas)
        require(parsed["bag"] == scenario["bag"] and parsed["group_id"] == scenario["group_id"], "saved scenario bag identity differs")
        bucket = int(parsed["group_id"], 16) % 100
        split = "train" if bucket < 70 else "validation" if bucket < 85 else "final"
        require(split == expected_split, "saved scenario split differs")
        groups.append(scenario["group_id"])
    require(len(groups) == len(set(groups)), "saved scenario bags repeat")
    majority = {str(role): min(counts, key=lambda token: (-counts[token], token)) for role, counts in enumerate(majority_counts)}
    return groups, majority


def align_scenarios(scenarios, arrays, records=None):
    """Check saved assignments, content IDs and optional full final JSON records."""
    for group, scenario in enumerate(scenarios):
        visible = parse_visible(arrays["canvas"][group*8])
        expected = {(fact["event"], fact["role"]): fact["filler"] for fact in scenario["facts"]}
        positions = {(fact["event"], fact["role"]): (slot["row"], slot["start"])
                     for fact, slot in zip(scenario["facts"], scenario["layout"])}
        require(visible["facts"] == expected and visible["positions"] == positions, "scenario/base array differs")
    if records is not None:
        require(isinstance(records, list) and len(records) == len(arrays["target"]), "record list differs")
    for index in range(len(arrays["target"])):
        group, ci = int(arrays["group_index"][index]), int(arrays["condition_index"][index])
        scenario_id = compact_hash(scenarios[group])
        pair_id = compact_hash({"scenario_id": scenario_id,
            "base_query_event": int(arrays["base_query_event"][index]), "query_role": int(arrays["role"][index])})
        record_id = compact_hash({"pair_id": pair_id, "condition": CONDITIONS[ci], "canvas": arrays["canvas"][index].tolist()})
        require(arrays["pair_id"][index] == pair_id and arrays["record_id"][index] == record_id, "computed pair/record identity differs")
        if records is not None:
            row = records[index]
            for key in arrays:
                name = "query_role" if key == "role" else key
                actual = arrays[key][index]
                actual = actual.tolist() if isinstance(actual, np.ndarray) else actual.item()
                require(row.get(name) == actual, f"record JSON/NPZ mismatch: {name}")
            require(row["scenario_id"] == scenario_id and row["condition"] == CONDITIONS[ci], "record scenario/condition differs")


def training_stream(path, expected_updates, summary, batch_size, bag_count):
    sequence, digest = [], hashlib.sha256()
    with Path(path).open() as stream:
        for index, line in enumerate(stream, 1):
            row = decode_json(line)
            require(row["step"] == index and valid_hash(row["input_sha256"]), "training log step/digest invalid")
            require(all(type(row[name]) in (int, float) and math.isfinite(row[name]) and row[name] >= 0
                        for name in ("loss", "gradient_norm_before_clip", "elapsed_seconds")), "nonfinite/negative training log")
            ids = row["training_group_indices"]
            require(isinstance(ids, list) and len(ids) == batch_size and all(type(x) is int and 0 <= x < bag_count for x in ids),
                    "training bag index inventory differs")
            sequence.append((row["input_sha256"], tuple(ids)))
            digest.update(bytes.fromhex(row["input_sha256"]))
    require(len(sequence) == expected_updates and digest.hexdigest() == summary["training_input_sequence_sha256"], "training stream length/aggregate differs")
    return sequence


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "naive timestamp")
    return parsed


class StudyAudit:
    def __init__(self, root, recipe_path, source_root=None):
        self.reader = Reader(root)
        self.cloud = Reader(Path(root).parent)
        self.recipe_path = Path(recipe_path).resolve()
        self.recipe = decode_json(self.recipe_path.read_bytes())
        self.source_root = Path(source_root).resolve() if source_root else None
        self.report = {"schema_version": 1, "item": 9, "status": "in_progress", "issues": [],
            "analysis_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_source_sha256": sha(__file__), "recipe_sha256": sha(self.recipe_path),
            "numerical_tolerance": {
                "duplicate_logits": {"absolute": FLOAT32_TOLERANCE, "relative": FLOAT32_TOLERANCE,
                    "reason": "16 float32 eps for saved-logit/library roundoff"},
                "nll_and_cross_entropy": {"absolute": NLL_ATOL, "relative": NLL_RTOL,
                    "reason": "Both implementations evaluate double-precision log-sum-exp from identical saved float32 logits, treated as exact inputs"},
                "exact": "argmax, labels, counts, identities and control probabilities"},
            "limits": ["No model execution, checkpoint deserialization or data resampling by the scientific generator.",
                "Checkpoint verification covers saved bytes/hashes, not tensor contents.",
                "Logged matching input streams are artifact evidence, not instrumentation of every arithmetic operation.",
                "Remote custody and archive/worker/resource guarantees require the separate operational audit.",
                "Bag bootstrap is conditional on fixed checkpoints/examples; seed t intervals concern five training realizations, not isolated initialization or complete pipeline uncertainty.",
                "Item-5 H1 remains unchanged; secondary diagnostics do not establish broad architecture superiority."]}

    def memory(self, phase):
        fields = dict((line.split(':')[0], int(line.split()[1])) for line in Path('/proc/meminfo').read_text().splitlines())
        row = {"at_utc": datetime.now(timezone.utc).isoformat(), "phase": phase, "available_ram_gib": fields["MemAvailable"]/1024**2}
        self.report.setdefault("resource_samples", []).append(row)
        require(row["available_ram_gib"] >= 8, "analysis RAM below8GiB at "+phase)

    def development(self, context):
        r, recipe = self.reader, self.recipe
        manifest_path = r.verify(context["development_manifest"])
        require(manifest_path == r.root/"development_data/manifest.json", "development manifest path differs")
        manifest = r.json("development_data/manifest.json")
        require(manifest["identities"] == context["development_identities"], "development identities differ from context")
        for artifact in manifest["files"].values():
            r.verify(artifact)
        ledger = r.json("development_data/fixture_exposure.json")
        require(manifest["excluded_fixture_groups"] == ledger["excluded_groups"], "fixture exclusion list differs")
        excluded = set(ledger["excluded_groups"])
        require(all(valid_hash(x) for x in excluded), "invalid excluded group hash")
        arrays, groups, scenarios = {}, {}, {}
        counts = {"train": recipe["data"]["train_bags"], "validation": recipe["data"]["validation_bags"],
                  "probe": recipe["data"]["probe_bags"], "memorization": recipe["data"]["memorization_bags"]}
        for name, count in counts.items():
            scenarios[name] = r.json(f"development_data/{name}_scenarios.json.gz")
            require(len(scenarios[name]) == count == manifest["counts"][name], "development bag count differs")
            groups[name], fitted = scenario_inventory(scenarios[name], expected_split="validation" if name == "validation" else "train")
            if name == "train":
                require(fitted == manifest["train_only_role_majority"], "train-only majority differs from saved training facts")
            if name != "train":
                arrays[name] = r.arrays(f"development_data/{name}.npz")
                validate_dataset(arrays[name], ("base",), count, split="validation" if name == "validation" else "train")
                align_scenarios(scenarios[name], arrays[name])
                require(content_digest(arrays[name]["canvas"], arrays[name]["target"]) == manifest["identities"][name+".npz"], "development tensor identity differs")
            key = name+"_scenarios.json.gz"
            require(sha(r.path("development_data/"+key)) == manifest["identities"][key], "saved scenario identity differs")
        require(not (set(groups["train"]) & set(groups["validation"])), "development groups overlap")
        require(not ((set(groups["train"]) | set(groups["validation"])) & excluded), "fixture bag appears in development")
        for name in ("probe", "memorization"):
            require(groups[name] == groups["train"][:counts[name]], "training-derived panel bag inventory differs")
        require(sha(r.path("development_data/fixture_exposure.json")) == manifest["identities"]["fixture_exposure.json"], "ledger identity differs")
        self.report["development"] = {"counts": counts, "excluded_fixture_groups": len(excluded), "within_split_disjoint": True,
                                      "probe_and_memorization_are_training_subsets": True}
        return manifest, arrays, groups

    def run(self):
        r, recipe = self.reader, self.recipe
        require(recipe["item"] == 9 and tuple(recipe["data"]["final_conditions"]) == CONDITIONS
                and tuple(recipe["training"]["primary_initializations"]) == SEEDS, "scientific recipe inventory differs")
        require(recipe["analysis"]["bootstrap_repetitions"] == 2000 and recipe["analysis"]["bootstrap_seed"] == 94001
                and recipe["analysis"]["paired_seed_t_df"] == 4, "frozen statistical recipe differs")
        context = r.json("context.json")
        require(context["recipe_sha256"] == sha(self.recipe_path), "recipe differs from study context")
        plan = self.cloud.json("execution_plan.json")
        require(sha(self.cloud.path("execution_plan.json")) == context["plan_sha256"] and plan["item"] == 9
                and plan["phase"] == "study" and plan["status"] == "frozen" and plan["recipe_sha256"] == context["recipe_sha256"], "plan/context identity differs")
        actual_runtime = {"python": platform.python_version(), "numpy": np.__version__, "scipy": importlib.metadata.version("scipy")}
        require(actual_runtime == OFFLINE_RUNTIME and plan["offline_analysis_runtime"] == OFFLINE_RUNTIME,
                "offline analysis runtime differs from its prospective Stage-B binding")
        if self.source_root:
            for path, expected in plan["implementation_sha256"].items():
                candidate = (self.source_root/path).resolve()
                require(candidate.is_relative_to(self.source_root) and sha(candidate) == expected, "provided source tree differs: "+path)
        rates = plan["pilot_evidence"]["selected_learning_rates"]
        require(set(rates) == set(FAMILIES) and all(rates[f] in recipe["training"]["pilot_learning_rates"] for f in FAMILIES), "selected rate inventory differs")
        require(context["pilot_evidence_sha256"] == plan["pilot_evidence"]["receipt_sha256"], "pilot evidence binding differs")
        manifest, development, dev_groups = self.development(context)
        inventory = r.json("final_inventory.json")
        expected_ids = [f"{family}_seed{seed}" for seed in SEEDS for family in FAMILIES]
        require(inventory["context"] == context and inventory["selected_learning_rates"] == rates
                and inventory["final_recipe"] == recipe["data"] and inventory["final_data_generated"] is False,
                "final inventory context differs")
        require([row["run_id"] for row in inventory["entries"]] == expected_ids, "missing/duplicate/reordered final inventory")
        index = r.json("primary_runs.json")
        r.verify(inventory["primary_index"])
        require(index["status"] == "completed" and index["runs"] == expected_ids
                and index["statuses"] == dict.fromkeys(expected_ids, "completed"), "primary index incomplete")
        training, streams, recounted = {}, {}, {}
        for entry in inventory["entries"]:
            run_id, family, seed = entry["run_id"], entry["family"], entry["initialization"]
            self.memory("training_recount_"+run_id)
            require(run_id == f"{family}_seed{seed}" and family in FAMILIES and seed in SEEDS
                    and entry["learning_rate"] == rates[family] and entry["updates"] == recipe["training"]["primary_updates"], "inventory recipe mismatch")
            r.verify(entry["summary"])
            r.verify(entry["checkpoint"])
            summary = r.json(f"runs/{run_id}/run.json")
            require(summary["context"] == context and summary["status"] == "completed" and summary["phase"] == "primary"
                    and summary["updates_completed"] == entry["updates"] and summary["run_id"] == run_id
                    and summary["family"] == family and summary["initialization"] == seed
                    and summary["learning_rate"] == rates[family], "training summary differs")
            require(entry["checkpoint"] == summary["artifacts"]["checkpoint"], "inventory checkpoint differs from trained checkpoint")
            config = summary["configuration"]
            expected_config = {"family": family, "model": recipe["models"][family], "vocab": 37,
                "height": 10, "width": 8, "out_pos": [9, 7], "initialization": seed,
                "learning_rate": rates[family], "optimizer": "AdamW", "weight_decay": recipe["training"]["weight_decay"],
                "gradient_clip": recipe["training"]["gradient_clip"], "batch_size": recipe["training"]["batch_size"],
                "loss": recipe["training"]["loss"], "scheduled_updates": entry["updates"], "memorization": False}
            require(set(config) == set(expected_config)|{"trainable_parameters"}
                    and all(config[key] == value for key, value in expected_config.items())
                    and type(config["trainable_parameters"]) is int and config["trainable_parameters"] > 0, "scientific configuration differs")
            require(r.json(f"runs/{run_id}/configuration.json") == config, "configuration file differs")
            for artifact in summary["artifacts"].values():
                r.verify(artifact)
            streams[run_id] = training_stream(r.verify(summary["artifacts"]["training_log"]), entry["updates"], summary,
                                             recipe["training"]["batch_size"], recipe["data"]["train_bags"])
            recounted[run_id] = {}
            for name in ("probe", "validation"):
                values = r.arrays(summary["artifacts"][name+"_predictions"]["path"])
                pred, nll, error = recount_predictions(values, development[name])
                points = point_metrics(pred, development[name]["target"], development[name]["role"], nll, development[name]["group_index"])
                compare_metrics(summary[name], points)
                recounted[run_id][name] = {"metrics": points, "maximum_absolute_nll_difference": error}
            require(summary["probe_below_95_percent_binding"] == (recounted[run_id]["probe"]["metrics"]["binding"] < recipe["training"]["training_probe_binding_flag_below"]), "probe diagnostic flag differs")
            training[run_id] = summary
        paired_streams = []
        for seed in SEEDS:
            left, right = (f"{family}_seed{seed}" for family in FAMILIES)
            require(streams[left] == streams[right], f"matched-family per-update inputs differ at seed{seed}")
            paired_streams.append({"seed": seed, "updates": len(streams[left]),
                                   "input_sequence_sha256": training[left]["training_input_sequence_sha256"], "equal": True})
        self.report.update(training_recount=recounted, paired_training_streams=paired_streams)
        archive = r.json("gate_archive_receipt.json")
        access = r.json("final_access.json")
        require(archive["final_inventory_sha256"] == sha(r.path("final_inventory.json")) == access["final_inventory_sha256"], "gate/access inventory identity differs")
        require(archive["source_commit"] == context["source_commit"] == access["source_commit"]
                and archive["results_branch"] == "research/scientific-validation-2026-10-07-cloud-results"
                and isinstance(archive["archive_commit"], str) and len(archive["archive_commit"]) == 40,
                "archive provenance differs")
        r.verify(access["archive_receipt"])
        require(access["consumed_before_final_generation"] is True and access["retry_allowed"] is False, "final access is not consume-once")
        require(timestamp(inventory["created_at_utc"]) <= timestamp(archive["archived_at_utc"]) <= timestamp(access["claimed_at_utc"]), "gate event ordering differs")
        for name in ("train", "final"):
            began, completed = r.json(name+"_started.json"), r.json(name+"_completed.json")
            require(completed["status"] == "completed" and all(began["context"][key] == context[key] for key in began["context"]), "phase completion/context differs")
            require(began["runtime"]["python"] == recipe["runtime"]["python"]
                    and began["runtime"]["versions"] == recipe["runtime"]["versions"]
                    and began["runtime"]["intra_threads"] == 2 and began["runtime"]["inter_threads"] == 1
                    and began["runtime"]["torch_cuda_version"] is None, "recorded numerical environment differs")
        require(timestamp(r.json("train_completed.json")["at_utc"]) <= timestamp(archive["archived_at_utc"])
                <= timestamp(r.json("final_started.json")["at_utc"]) <= timestamp(access["claimed_at_utc"])
                <= timestamp(r.json("final_completed.json")["at_utc"]), "phase/final access chronology differs")
        self.report["gate"] = {"verified_local_hash_links": True, "chronology_verified": True,
                               "archive_commit_claim": archive["archive_commit"], "source_commit": context["source_commit"],
                               "remote_confirmation_scope": "checked by separate operational archive audit"}
        return self.final(context, inventory, manifest, dev_groups, recounted)

    def final(self, context, inventory, development_manifest, dev_groups, recounted):
        r, recipe = self.reader, self.recipe
        cfg = recipe["data"]
        manifest = r.json("final_data/manifest.json")
        require(manifest["created_after_access_sha256"] == sha(r.path("final_access.json")), "final manifest access binding differs")
        require(manifest["groups"] == cfg["final_bags"] and manifest["conditions"] == list(CONDITIONS)
                and manifest["nominal_rows"] == cfg["final_bags"]*48, "final population size differs")
        for artifact in manifest["files"]:
            r.verify(artifact)
        scenarios = r.json("final_data/scenarios.json.gz")
        ids, _ = scenario_inventory(scenarios, expected_split="final")
        require(len(ids) == cfg["final_bags"] and not(set(ids) & (set(dev_groups["train"]) | set(dev_groups["validation"])
                | set(development_manifest["excluded_fixture_groups"]))), "final bags overlap development/exposure")
        arrays = r.arrays("final_data/examples.npz")
        parsed = validate_dataset(arrays, CONDITIONS, cfg["final_bags"], split="final")
        align_scenarios(scenarios, arrays, r.json("final_data/records.json.gz"))
        require(content_digest(arrays["canvas"], arrays["target"]) == manifest["input_target_digest"], "final input/target digest differs")
        controls = verify_controls(r.arrays("final_data/controls.npz"), arrays, parsed, development_manifest["train_only_role_majority"])
        multiplicity = Counter(canvas.astype(np.uint8).tobytes() for canvas in arrays["canvas"])
        distribution = {str(k): v for k, v in sorted(Counter(multiplicity.values()).items())}
        self.report["observed_duplicate_structure"] = {"unique_canvases": len(multiplicity),
            "multiplicity_counts": distribution, "expected_unique": 40*cfg["final_bags"],
            "expected_multiplicity_counts": {"1": 32*cfg["final_bags"], "2": 8*cfg["final_bags"]}}
        require(len(multiplicity) == 40*cfg["final_bags"] == manifest["unique_canvases"]
                and distribution == {"1": 32*cfg["final_bags"], "2": 8*cfg["final_bags"]}
                and distribution == manifest["duplicate_multiplicity_counts"], "unique-canvas duplicate structure differs")
        self.report["final_dataset"] = {"groups": len(ids), "nominal_rows": len(arrays["target"]),
            "unique_canvases": len(multiplicity), "duplicate_multiplicity_counts": distribution,
            "independent_visible_truth_and_transformations_verified": True, "controls": controls}
        summary = r.json("final_summary.json")
        require(summary["status"] == "completed" and summary["context"] == context
                and summary["final_inventory_sha256"] == sha(r.path("final_inventory.json"))
                and summary["final_access_sha256"] == sha(r.path("final_access.json")), "final summary identity differs")
        require([row["run_id"] for row in summary["runs"]] == [row["run_id"] for row in inventory["entries"]], "final run inventory differs")
        indices = bootstrap_indices(cfg["final_bags"], 2000, 94001)
        self.report["bootstrap"] = {"algorithm": "NumPy Generator(PCG64(94001)).integers int64", "seed": 94001,
            "repetitions": 2000, "groups": cfg["final_bags"], "common_indices_for_all_models_conditions_queries": True,
            "indices": indices.tolist(), "indices_little_endian_int64_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
            "scope": "percentile95 conditional on fixed checkpoints and sampled bags; no new training-realization draws"}
        final, bag_values = {}, {}
        for saved, entry in zip(summary["runs"], inventory["entries"]):
            run_id = entry["run_id"]
            self.memory("final_recount_"+run_id)
            require(saved == r.json(f"final_predictions/{run_id}.json") and saved["checkpoint"] == entry["checkpoint"]
                    and saved["family"] == entry["family"] and saved["initialization"] == entry["initialization"], "final run/checkpoint metadata differs")
            r.verify(saved["predictions"])
            values = r.arrays(saved["predictions"]["path"])
            require("condition_index" in values, "final predictions omit condition alignment")
            pred, nll, nll_error = recount_predictions(values, arrays)
            points = {}
            for ci, condition in enumerate(CONDITIONS):
                pick = arrays["condition_index"] == ci
                points[condition] = point_metrics(pred[pick], arrays["target"][pick], arrays["role"][pick], nll[pick], arrays["group_index"][pick])
                compare_metrics(saved["condition_metrics"][condition], points[condition])
            diagnostics, blocks = condition_diagnostics(pred, arrays, nll, indices, self.memory)
            final[run_id] = {"family": entry["family"], "seed": entry["initialization"], "metrics": points,
                "conditional_bag_diagnostics": diagnostics, "duplicate_agreement": duplicate_agreement(arrays, values),
                "maximum_absolute_nll_difference": nll_error}
            bag_values[run_id] = blocks
        left = {seed: final[f"neuropixel_seed{seed}"]["metrics"]["base"]["binding"] for seed in SEEDS}
        right = {seed: final[f"relative_transformer_seed{seed}"]["metrics"]["base"]["binding"] for seed in SEEDS}
        self.report.update(final_recount=final, primary_paired_training_realizations=paired_seed_interval(left, right))
        comparisons = []
        for seed in SEEDS:
            self.memory("paired_bag_intervals_seed"+str(seed))
            for condition in CONDITIONS:
                row = {"seed": seed, "condition": condition, "left": FAMILIES[0], "right": FAMILIES[1], "metrics": {}}
                for metric in ("global", "binding", "all_eight", "cross_entropy", *["role_"+role for role in ROLES]):
                    difference = bag_values[f"neuropixel_seed{seed}"][condition][metric]-bag_values[f"relative_transformer_seed{seed}"][condition][metric]
                    row["metrics"][metric] = ratio_interval(difference, np.ones(len(difference)), indices)
                    row["metrics"][metric]["units"] = "numerator=sum of paired per-bag metric differences; denominator=number of bags"
                comparisons.append(row)
        self.report["paired_checkpoint_bag_differences"] = comparisons
        self.report["family_training_realizations"] = {family: {
            condition: {metric: self.describe([final[f"{family}_seed{seed}"]["metrics"][condition][metric] for seed in SEEDS])
                        for metric in ("global", "binding", "all_eight", "cross_entropy")}
            for condition in CONDITIONS} for family in FAMILIES}
        screen = recipe["analysis"]["high_competence_screen"]
        checks = {"all_five_base_binding": all(left[s] >= screen["all_five_base_binding_at_least"] for s in SEEDS),
            "all_five_base_all_eight": all(final[f"neuropixel_seed{s}"]["metrics"]["base"]["all_eight"] >= screen["all_five_base_all_eight_at_least"] for s in SEEDS),
            "every_condition_every_seed_binding": all(final[f"neuropixel_seed{s}"]["metrics"][c]["binding"] >= screen["every_condition_every_seed_binding_at_least"] for s in SEEDS for c in CONDITIONS)}
        self.report["operational_competence_screen"] = {"criteria": screen, "checks": checks, "met": all(checks.values()),
            "scope": "exploratory fixed-template operational screen; not H1 or a significance test"}
        self.memory("analysis_completed")
        self.report["status"] = "verified"
        return self.report

    @staticmethod
    def describe(values):
        return {"values": values, "n": len(values), "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values), "range": [min(values), max(values)]}

    def finish(self):
        self.report.update(inputs=self.reader.inputs, parent_cloud_inputs=self.cloud.inputs,
            analysis_completed_at_utc=datetime.now(timezone.utc).isoformat(),
            analysis_environment={"python": platform.python_version(), "numpy": np.__version__,
                "scipy": importlib.metadata.version("scipy"),
                "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}})
        return self.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed Stage-B study directory")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite analysis evidence")
    memory = dict((line.split(':')[0], int(line.split()[1])) for line in Path('/proc/meminfo').read_text().splitlines())
    require(memory["MemAvailable"] >= 8*1024**2, "analysis admission requires8GiB available RAM")
    audit = StudyAudit(args.input, args.recipe, args.source_root)
    audit.report["available_ram_gib_at_admission"] = memory["MemAvailable"]/1024**2
    try:
        audit.run()
    except Exception as error:
        audit.report["status"] = "failed"
        audit.report["issues"].append({"type": type(error).__name__, "message": str(error)})
    result = audit.finish()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "issues": len(result["issues"]), "output": str(args.output), "sha256": sha(args.output)}))
    return int(result["status"] != "verified")


if __name__ == "__main__":
    raise SystemExit(main())
