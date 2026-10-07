"""Independent item-13 saved-prediction audit; no producer or project imports.

Recounts the irrelevant-event swap against its paired original input. Reuses
already accessed item-9 outcomes and adds no confidence interval or hypothesis.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
OLD = "83fe135e301f76bc0c74e30c66bb18e067ca5959"
CONDITIONS = ("base", "swap_queried_agent_patient", "swap_other_agent_patient",
              "relabel_events", "query_switch", "layout_permutation")
FAMILIES = ("neuropixel", "relative_transformer")
SEEDS = (40, 41, 42, 43, 44)
SUBSETS = ("all", "binding", "agent", "action", "patient", "place")


class Audit:
    def __init__(self):
        self.checks, self.issues, self.results = 0, [], {}
    def check(self, yes, label):
        self.checks += 1
        if not bool(yes):
            self.issues.append(label)
        return bool(yes)
    def need(self, yes, label):
        if not self.check(yes, label):
            raise ValueError(label)
    def same(self, observed, expected, label):
        if isinstance(expected, dict):
            self.need(isinstance(observed, dict), label + ": object")
            for key, value in expected.items():
                self.need(key in observed, label + ": missing " + key)
                self.same(observed[key], value, label + "." + key)
        elif isinstance(expected, list):
            self.need(isinstance(observed, list) and len(observed) == len(expected), label + ": length")
            for i, value in enumerate(expected):
                self.same(observed[i], value, label + "[" + str(i) + "]")
        elif isinstance(expected, float):
            self.check(type(observed) in (int, float) and math.isfinite(observed)
                       and abs(observed-expected) <= 1e-12, label + ": number")
        else:
            self.check(type(observed) is type(expected) and observed == expected, label + ": value")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def path_at(root, name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or p.as_posix() != name:
        raise ValueError("unsafe path")
    p = root / p
    if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError("missing or indirect file: " + name)
    return p


def inventory(root, names):
    return {name: sha(path_at(root, name)) for name in names}


def read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def ram():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) * 1024 / 2**30


def npz(path, names):
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(names):
            raise ValueError("unexpected NPZ keys")
        return {name: stored[name].copy() for name in names}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decode(canvas):
    """Read event-token starts, then verify complete coverage of occupied cells."""
    if canvas.shape != (10, 8) or canvas.dtype != np.int64 or np.any((canvas < 0) | (canvas > 36)):
        raise ValueError("bad canvas schema")
    event, role = int(canvas[9, 5])-35, int(canvas[9, 6])-1
    if event not in (0, 1) or role not in range(4) or np.count_nonzero(canvas[9]) != 2:
        raise ValueError("bad current query/readout row")
    values, locations, used = {}, {}, set()
    starts = np.argwhere((canvas[:9] == 35) | (canvas[:9] == 36))
    if len(starts) != 8:
        raise ValueError("expected eight fact starts")
    for r0, c0 in starts:
        r, c = int(r0), int(c0)
        if c > 5 or any(old[0] == r for old in locations.values()):
            raise ValueError("fact row/column geometry")
        key = (int(canvas[r, c])-35, int(canvas[r, c+1])-1)
        filler = int(canvas[r, c+2])
        if key[1] not in range(4) or key in values:
            raise ValueError("duplicate/invalid fact role")
        valid = 5 <= filler < 17 if key[1] in (0, 2) else 17 <= filler < 27 if key[1] == 1 else 27 <= filler < 35
        if not valid:
            raise ValueError("filler category")
        values[key], locations[key] = filler, (r, c)
        used.update((r, col) for col in range(c, c+3))
    occupied = {tuple(map(int, pos)) for pos in np.argwhere(canvas[:9] != 0)}
    if used != occupied or len(set(values.values())) != 8:
        raise ValueError("unparsed cells or duplicate fillers")
    bag = {"nouns": sorted(v for (e, r), v in values.items() if r in (0, 2)),
           "verbs": sorted(v for (e, r), v in values.items() if r == 1),
           "places": sorted(v for (e, r), v in values.items() if r == 3)}
    return values, locations, (event, role), canonical(bag)


def table(first, second, target):
    buckets = Counter()
    equal_wrong = 0
    for b, d, y in zip(first, second, target):
        buckets[("C" if b == y else "W") + ("C" if d == y else "W")] += 1
        equal_wrong += int(b == d and b != y)
    n = len(target)
    cc, cw, wc, ww = (buckets[k] for k in ("CC", "CW", "WC", "WW"))
    clean, shifted, equal = cc+cw, cc+wc, cc+equal_wrong
    ratio = lambda x, denominator: x/denominator if denominator else None
    return {"n": n, "base_correct": clean, "transformed_correct": shifted,
            "CC": cc, "CW": cw, "WC": wc, "WW": ww, "prediction_equal": equal,
            "equal_wrong": equal_wrong, "base_accuracy": clean/n, "transformed_accuracy": shifted/n,
            "joint_accuracy": cc/n, "agreement_rate": equal/n, "equal_wrong_rate": equal_wrong/n,
            "retained_given_base_correct": ratio(cc, clean), "corrupted_given_base_correct": ratio(cw, clean),
            "repaired_given_base_wrong": ratio(wc, n-clean), "stable_wrong_given_base_wrong": ratio(equal_wrong, n-clean)}


def execute(a, raw, plan_path, produced):
    global np
    a.results["available_ram_gib_at_admission"] = ram()
    a.need(a.results["available_ram_gib_at_admission"] >= 8, "RAM admission below 8 GiB")
    import numpy as np
    a.need("torch" not in sys.modules, "unexpected Torch import")
    plan = read(plan_path)
    a.need(plan["item"] == 13 and plan["status"] == "frozen", "plan status/item")
    a.need(plan["evidence_recipe"]["condition_mapping"] == list(CONDITIONS), "frozen condition mapping")
    cfg = plan["evidence_recipe"]["secondary"]
    a.same(cfg, {"original_source_commit": OLD, "seeds": list(SEEDS), "families": list(FAMILIES),
                "contrast": ["base", "swap_other_agent_patient"], "paired_examples_per_run": 2048,
                "shared_bags": 256, "selected_input_files": 12, "new_model_inference": False,
                "new_confidence_intervals": False}, "recipe")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    a.need(head == os.environ.get("GITHUB_SHA"), "current source mismatch")
    impl = inventory(ROOT, plan["implementation_sha256"])
    a.need(impl == plan["implementation_sha256"], "implementation hashes")
    a.results["source"] = {"source_commit": head, "plan_sha256": sha(plan_path), "implementation_sha256": impl}
    spec = plan["inputs"]["distractor"]
    ids = [f"{family}_seed{seed}" for family in FAMILIES for seed in SEEDS]
    names = ["archive_manifest.json", "study/final_data/examples.npz"] + [f"study/final_predictions/{rid}.npz" for rid in ids]
    a.need(set(names) == set(spec["files_sha256"]), "twelve selected artifacts required")
    inputs = inventory(raw, names); a.results["input_sha256"] = inputs
    a.need(inputs == spec["files_sha256"], "input hashes")
    original = read(raw / "archive_manifest.json")
    a.need(original["final"] is True and original["snapshot_kind"] == "final" and original["source_commit"] == OLD,
           "original final source")
    for name in names[1:]:
        entry = original["files"][name]
        a.need(entry["sha256"] == inputs[name] and entry["bytes"] == (raw/name).stat().st_size, "old manifest: " + name)
    output_names = {"artifact_manifest.json", "report.json", "transitions.csv", "paired_outcomes.npz"}
    a.need({p.name for p in produced.iterdir()} == output_names, "produced inventory")
    before = inventory(produced, sorted(output_names)); a.results["produced_sha256"] = before
    manifest = read(produced / "artifact_manifest.json")
    a.need(manifest["item"] == 13 and set(manifest["files"]) == output_names-{"artifact_manifest.json"}, "payload manifest")
    for name, entry in manifest["files"].items():
        a.same(entry, {"sha256": before[name], "bytes": (produced/name).stat().st_size}, "payload " + name)
    report = read(produced / "report.json")
    a.same(report, {"item": 13, "status": "completed", "source": a.results["source"], "original_source_commit": OLD,
                   "input_sha256": inputs, "condition_mapping": list(CONDITIONS),
                   "contrast": ["base", "swap_other_agent_patient"], "shared_bags": 256, "pairs_per_run": 2048,
                   "archive_identity": {"archive_commit": spec["archive_commit"], "path": spec["path"]}}, "report identity")
    keys = ("canvas", "target", "base_target", "role", "query_event", "base_query_event", "group_index",
            "condition_index", "changed_gold", "group_id", "record_id", "pair_id")
    data = npz(raw / names[1], keys)
    a.need(data["canvas"].shape == (12288, 10, 8) and all(data[k].shape == (12288,) for k in keys[1:]), "data shapes")
    a.need(all(data[k].dtype == np.int64 for k in keys[1:8])
           and data["changed_gold"].dtype == np.bool_
           and all(data[k].dtype.kind == "U" for k in keys[9:]), "metadata dtypes")
    a.need(np.array_equal(data["condition_index"], np.repeat(np.arange(6), 2048)), "condition order")
    bidx = np.where(data["condition_index"] == CONDITIONS.index("base"))[0]
    didx = np.where(data["condition_index"] == CONDITIONS.index("swap_other_agent_patient"))[0]
    for key in ("target", "base_target", "role", "query_event", "base_query_event", "group_index", "group_id", "pair_id"):
        a.need(np.array_equal(data[key][bidx], data[key][didx]), "paired " + key)
    a.need(np.array_equal(data["role"][bidx], np.tile(np.arange(4), 512))
           and np.array_equal(data["query_event"][bidx], np.tile(np.repeat(np.arange(2), 4), 256))
           and np.array_equal(data["group_index"][bidx], np.repeat(np.arange(256), 8)), "pair order")
    a.need(len(set(data["pair_id"][bidx])) == 2048 and len(set(data["group_id"][bidx])) == 256
           and all(len(set(data["group_id"][bidx][8*g:8*g+8])) == 1 for g in range(256)), "bag/pair identity")
    a.need(not data["changed_gold"][np.r_[bidx, didx]].any(), "unchanged-target condition")
    for base, altered in zip(bidx, didx):
        facts, places, query, gid = decode(data["canvas"][base])
        other, other_places, other_query, other_gid = decode(data["canvas"][altered])
        a.need(query == other_query == (int(data["query_event"][base]), int(data["role"][base]))
               and places == other_places and gid == other_gid == str(data["group_id"][base])
               and int(gid, 16) % 100 >= 85, "visible query/group/layout")
        for key in facts:
            expected = facts[(key[0], 2-key[1])] if key[0] != query[0] and key[1] in (0, 2) else facts[key]
            a.need(other[key] == expected, "declared irrelevant-event swap")
        a.need(facts[query] == other[query] == int(data["target"][base]) == int(data["base_target"][base]),
               "independent visible gold")
        for index in (base, altered):
            identity = canonical({"pair_id": str(data["pair_id"][index]),
                                  "condition": CONDITIONS[int(data["condition_index"][index])],
                                  "canvas": data["canvas"][index].tolist()})
            a.need(identity == str(data["record_id"][index]), "record identity")
    rows, bp, dp, maximum_nll_error = [], [], [], 0.0
    for rid in ids:
        a.need(ram() >= 8, "RAM model-array boundary")
        values = npz(raw / f"study/final_predictions/{rid}.npz",
                     ("logits", "pred", "nll", "target", "role", "group_index", "condition_index", "record_id"))
        for key in ("target", "role", "group_index", "condition_index", "record_id"):
            a.need(values[key].dtype == data[key].dtype and np.array_equal(values[key], data[key]), "prediction alignment " + key)
        logits, predictions, nll = values["logits"], values["pred"], values["nll"]
        a.need(logits.dtype == np.float32 and logits.shape == (12288, 37) and np.isfinite(logits).all()
               and np.all(logits[:, 0] == -10000), "logit schema/PAD")
        a.need(predictions.shape == nll.shape == (12288,) and predictions.dtype == np.int64
               and nll.dtype == np.float64 and np.isfinite(nll).all() and (nll >= 0).all()
               and np.array_equal(predictions, np.argmax(logits, axis=1)), "argmax/NLL schema")
        select = np.r_[bidx, didx]; z = logits[select].astype(np.float64)
        losses = np.logaddexp.reduce(z, axis=1) - z[np.arange(4096), data["target"][select]]
        maximum_nll_error = max(maximum_nll_error, float(np.max(np.abs(losses-nll[select]))))
        a.need(np.allclose(losses, nll[select], atol=1e-10, rtol=1e-12), "selected NLL recount")
        first, second, truth, roles = predictions[bidx], predictions[didx], data["target"][bidx], data["role"][bidx]
        bp.append(first.copy()); dp.append(second.copy())
        family, seed = rid.rsplit("_seed", 1)
        for subset in SUBSETS:
            selected_roles = range(4) if subset == "all" else (0, 2) if subset == "binding" else (SUBSETS.index(subset)-2,)
            selected_rows = [i for i, r in enumerate(roles) if r in selected_roles]
            rows.append({"run_id": rid, "family": family, "seed": int(seed), "subset": subset,
                         **table(first[selected_rows], second[selected_rows], truth[selected_rows])})
    a.same(report["rows"], rows, "transition rows")
    summaries = []
    for family in FAMILIES:
        for subset in SUBSETS:
            by_seed = {r["seed"]: r for r in rows if r["family"] == family and r["subset"] == subset}
            out = {}
            for metric in table(np.array([0]), np.array([0]), np.array([0])):
                values = [by_seed[seed][metric] for seed in SEEDS]
                out[metric] = {"values_by_seed": {str(s): v for s, v in zip(SEEDS, values)},
                               "n_defined": sum(v is not None for v in values),
                               "mean": sum(values)/5 if None not in values else None}
            summaries.append({"family": family, "subset": subset, "n_training_realizations": 5, "metrics": out})
    a.same(report["family_summaries"], summaries, "descriptive family means")
    paired = npz(produced / "paired_outcomes.npz",
                 ("base_prediction", "transformed_prediction", "target", "role", "group_index", "pair_id", "run_id"))
    expected_arrays = {"base_prediction": np.stack(bp), "transformed_prediction": np.stack(dp),
                       "target": data["target"][bidx], "role": data["role"][bidx],
                       "group_index": data["group_index"][bidx], "pair_id": data["pair_id"][bidx], "run_id": np.asarray(ids)}
    for key, value in expected_arrays.items():
        a.need(np.array_equal(paired[key], value) and paired[key].dtype == value.dtype, "paired payload " + key)
    with (produced / "transitions.csv").open(newline="", encoding="utf-8") as stream:
        saved_csv = list(csv.DictReader(stream))
    a.need(len(saved_csv) == len(rows), "CSV row count")
    for saved, expected in zip(saved_csv, rows):
        a.need(set(saved) == set(expected), "CSV columns")
        for key, value in expected.items():
            a.need(saved[key] == "" if value is None else
                   saved[key] == value if isinstance(value, str) else float(saved[key]) == value, "CSV value " + key)
    a.results.update(rows=rows, family_summaries=summaries, maximum_selected_nll_error=maximum_nll_error)
    a.need(inventory(raw, names) == inputs and inventory(ROOT, impl) == impl
           and inventory(produced, output_names) == before, "inputs/source/outputs unchanged")
    a.results["available_ram_gib_at_finish"] = ram()
    a.need(a.results["available_ram_gib_at_finish"] >= 8, "RAM finish boundary")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "plan", "produced", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    a = Audit()
    a.results["auditor_sha256"] = sha(Path(__file__))
    a.results["plan_sha256"] = sha(args.plan) if args.plan.is_file() else None
    try:
        execute(a, args.input, args.plan, args.produced)
    except Exception as error:
        a.issues.append(type(error).__name__ + ": " + str(error))
    report = {"schema_version": 1, "item": 13, "status": "verified" if not a.issues else "failed",
              "checks": a.checks, "issues": a.issues, "results": a.results,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__ if "np" in globals() else None,
                          "numerical_threads": 1},
              "limits": ["Independent saved-array recount, not an external replication or new experiment.",
                         "No producer/project/Torch imports, model execution, checkpoint loading or generator sampling.",
                         "Baseline already contains irrelevant facts; no claim of robustness to adding distractors.",
                         "Five saved realizations and shared bags; descriptive means only, no additional interval or test.",
                         "Conditional retention retains its base-correct denominator; null means denominator zero.",
                         "Twelve selected raw artifacts are verified; original full custody audit is separate.",
                         "RAM observations are sampled boundaries, not an independently monitored peak/minimum."]}
    with (args.output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "checks": a.checks, "issues": len(a.issues)}))
    return int(bool(a.issues))


if __name__ == "__main__":
    raise SystemExit(main())
