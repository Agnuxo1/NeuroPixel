"""Secondary item-9 irrelevant-event permutation recount; no model execution.

The baseline already contains the other event. These previously observed finite
pairs test perturbing existing irrelevant facts, not adding unseen distractors.
All summaries are descriptive across the five saved training realizations.
"""
from __future__ import annotations
import argparse
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
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
OLD_SOURCE = "83fe135e301f76bc0c74e30c66bb18e067ca5959"
CONDITIONS = ["base", "swap_queried_agent_patient", "swap_other_agent_patient",
              "relabel_events", "query_switch", "layout_permutation"]
FAMILIES = ("neuropixel", "relative_transformer")
SEEDS = (40, 41, 42, 43, 44)
SUBSETS = ("all", "binding", "agent", "action", "patient", "place")
ROLES = ("agent", "action", "patient", "place")


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def locate(root, name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == name, "unsafe path")
    full = root / p
    require(full.is_file() and not full.is_symlink()
            and full.resolve().is_relative_to(root.resolve()), "missing or indirect file: " + name)
    return full


def hashes(root, names):
    return {name: sha(locate(root, name)) for name in names}


def available_ram():
    row = next(line for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))
    return int(row.split()[1]) * 1024 / 2**30


def arrays(path, names):
    with np.load(path, allow_pickle=False) as source:
        require(set(source.files) == set(names), "array inventory differs")
        return {name: source[name].copy() for name in names}


def visible(canvas):
    require(canvas.shape == (10, 8) and canvas.dtype == np.int64, "canvas shape/type")
    require(np.all((canvas >= 0) & (canvas < 37)), "canvas vocabulary")
    query = (int(canvas[9, 5]) - 35, int(canvas[9, 6]) - 1)
    require(query[0] in (0, 1) and query[1] in range(4)
            and np.count_nonzero(canvas[9]) == 2, "query/output row")
    facts, positions = {}, {}
    for row in range(9):
        cols = np.flatnonzero(canvas[row])
        if not len(cols):
            continue
        require(len(cols) == 3 and np.array_equal(cols, np.arange(cols[0], cols[0] + 3)),
                "fact must be one contiguous triple")
        event, role, filler = map(int, canvas[row, cols])
        key = (event - 35, role - 1)
        require(key[0] in (0, 1) and key[1] in range(4) and key not in facts, "event/role fact key")
        lower, upper = ((5, 17) if key[1] in (0, 2) else (17, 27) if key[1] == 1 else (27, 35))
        require(lower <= filler < upper, "fact filler category")
        facts[key], positions[key] = filler, (row, int(cols[0]))
    require(len(facts) == 8 and len(set(facts.values())) == 8, "complete distinct-filler inventory")
    return facts, positions, query


def digest_record(pair, condition, canvas):
    text = json.dumps({"pair_id": str(pair), "condition": condition, "canvas": canvas.tolist()},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def transition(base, changed, truth):
    c0, c1 = base == truth, changed == truth
    n, cc = len(truth), int((c0 & c1).sum())
    cw, wc = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    ww, equal = int((~c0 & ~c1).sum()), int((base == changed).sum())
    b, t = cc + cw, cc + wc
    return {"n": n, "base_correct": b, "transformed_correct": t, "CC": cc, "CW": cw,
            "WC": wc, "WW": ww, "prediction_equal": equal, "equal_wrong": equal - cc,
            "base_accuracy": b/n, "transformed_accuracy": t/n, "joint_accuracy": cc/n,
            "agreement_rate": equal/n, "equal_wrong_rate": (equal-cc)/n,
            "retained_given_base_correct": cc/b if b else None,
            "corrupted_given_base_correct": cw/b if b else None,
            "repaired_given_base_wrong": wc/(n-b) if n != b else None,
            "stable_wrong_given_base_wrong": (equal-cc)/(n-b) if n != b else None}


def run(raw, plan_path, output):
    global np
    admission = available_ram()
    require(admission >= 8, "available RAM below 8 GiB before NumPy")
    import numpy as np
    require("torch" not in sys.modules, "Torch must not be imported")
    plan = read(plan_path)
    require(plan["item"] == 13 and plan["status"] == "frozen", "not frozen item13")
    require(plan["evidence_recipe"]["condition_mapping"] == CONDITIONS, "condition mapping differs")
    cfg = plan["evidence_recipe"]["secondary"]
    expected = {"original_source_commit": OLD_SOURCE, "seeds": list(SEEDS), "families": list(FAMILIES),
                "contrast": ["base", "swap_other_agent_patient"], "paired_examples_per_run": 2048,
                "shared_bags": 256, "selected_input_files": 12, "new_model_inference": False,
                "new_confidence_intervals": False}
    require(all(cfg.get(k) == v for k, v in expected.items()), "secondary recipe differs")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    require(head == os.environ.get("GITHUB_SHA"), "current source identity differs")
    implementation = hashes(ROOT, plan["implementation_sha256"])
    require(implementation == plan["implementation_sha256"], "source binding differs")
    spec = plan["inputs"]["distractor"]
    run_ids = [f"{family}_seed{seed}" for family in FAMILIES for seed in SEEDS]
    names = ["archive_manifest.json", "study/final_data/examples.npz"] + [
        f"study/final_predictions/{rid}.npz" for rid in run_ids]
    require(set(names) == set(spec["files_sha256"]), "selected input inventory differs")
    before = hashes(raw, names)
    require(before == spec["files_sha256"], "pinned inputs differ")
    manifest = read(raw / "archive_manifest.json")
    require(manifest["final"] is True and manifest["snapshot_kind"] == "final"
            and manifest["source_commit"] == OLD_SOURCE, "wrong original archive")
    for name in names[1:]:
        entry = manifest["files"][name]
        require(entry["sha256"] == before[name] and entry["bytes"] == (raw/name).stat().st_size,
                "manifest entry differs")
    data_names = ("canvas", "target", "base_target", "role", "query_event", "base_query_event",
                  "group_index", "condition_index", "changed_gold", "group_id", "record_id", "pair_id")
    data = arrays(raw / names[1], data_names)
    n = 12288
    require(data["canvas"].shape == (n, 10, 8), "panel shape differs")
    for name in data_names[1:]:
        require(data[name].shape == (n,), "panel alignment differs")
    for name in ("target", "base_target", "role", "query_event", "base_query_event", "group_index", "condition_index"):
        require(data[name].dtype == np.int64, "integer panel dtype differs")
    require(data["changed_gold"].dtype == np.bool_ and all(data[x].dtype.kind == "U"
            for x in ("group_id", "record_id", "pair_id")), "panel metadata dtype differs")
    require(np.array_equal(data["condition_index"], np.repeat(np.arange(6), 2048)), "condition ordering")
    base, other = np.flatnonzero(data["condition_index"] == 0), np.flatnonzero(data["condition_index"] == 2)
    for name in ("target", "base_target", "role", "query_event", "base_query_event", "group_index", "group_id", "pair_id"):
        require(np.array_equal(data[name][base], data[name][other]), "pair alignment: " + name)
    require(np.array_equal(data["role"][base], np.tile(np.arange(4), 512))
            and np.array_equal(data["query_event"][base], np.tile(np.repeat(np.arange(2), 4), 256))
            and np.array_equal(data["group_index"][base], np.repeat(np.arange(256), 8))
            and len(np.unique(data["pair_id"][base])) == 2048
            and len(np.unique(data["group_id"][base])) == 256, "paired population inventory")
    require(all(len(set(data["group_id"][base][8*g:8*g+8])) == 1 for g in range(256)), "bag identity consistency")
    require(not data["changed_gold"][np.r_[base, other]].any(), "contrast changes gold")
    for b, d in zip(base, other):
        f0, p0, q0 = visible(data["canvas"][b]); f1, p1, q1 = visible(data["canvas"][d])
        bag = {"nouns": sorted(v for (e, r), v in f0.items() if r in (0, 2)),
               "verbs": sorted(v for (e, r), v in f0.items() if r == 1),
               "places": sorted(v for (e, r), v in f0.items() if r == 3)}
        gid = hashlib.sha256(json.dumps(bag, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(gid == str(data["group_id"][b]) and int(gid, 16) % 100 >= 85, "visible bag identity/final bucket")
        expected_facts = dict(f0)
        a, p = (1-q0[0], 0), (1-q0[0], 2)
        expected_facts[a], expected_facts[p] = f0[p], f0[a]
        require(q0 == q1 == (int(data["query_event"][b]), int(data["role"][b]))
                and p0 == p1 and f1 == expected_facts, "not the declared irrelevant-event intervention")
        require(f0[q0] == f1[q1] == int(data["target"][b]) == int(data["base_target"][b]), "visible gold differs")
        for index in (b, d):
            require(data["record_id"][index] == digest_record(data["pair_id"][index],
                    CONDITIONS[int(data["condition_index"][index])], data["canvas"][index]), "record identity")
    rows, base_preds, other_preds = [], [], []
    for rid in run_ids:
        require(available_ram() >= 8, "RAM boundary below floor")
        values = arrays(raw / f"study/final_predictions/{rid}.npz",
                        ("logits", "pred", "nll", "target", "role", "group_index", "condition_index", "record_id"))
        for key in ("target", "role", "group_index", "condition_index", "record_id"):
            require(values[key].dtype == data[key].dtype and np.array_equal(values[key], data[key]), "ordered prediction metadata differs")
        logits, pred, nll = values["logits"], values["pred"], values["nll"]
        require(logits.shape == (n, 37) and logits.dtype == np.float32 and np.isfinite(logits).all(),
                "logit shape/type/finiteness")
        require(pred.shape == nll.shape == (n,) and pred.dtype == np.int64
                and nll.dtype == np.float64 and np.isfinite(nll).all() and (nll >= 0).all()
                and np.array_equal(pred, logits.argmax(1)), "prediction/NLL contract differs")
        require(np.all(logits[:, 0] == -10000), "PAD logit policy differs")
        selected = np.r_[base, other]
        z = logits[selected].astype(np.float64); z -= z.max(1, keepdims=True)
        ce = np.log(np.exp(z).sum(1))-z[np.arange(len(selected)), data["target"][selected]]
        require(np.allclose(ce, nll[selected], atol=1e-10, rtol=1e-12), "selected NLL differs")
        bp, dp, y, role = pred[base], pred[other], data["target"][base], data["role"][base]
        base_preds.append(bp.copy()); other_preds.append(dp.copy())
        family, seed = rid.rsplit("_seed", 1)
        masks = [np.ones(2048, dtype=bool), np.isin(role, [0, 2])] + [role == r for r in range(4)]
        for subset, mask in zip(SUBSETS, masks):
            rows.append({"run_id": rid, "family": family, "seed": int(seed), "subset": subset,
                         **transition(bp[mask], dp[mask], y[mask])})
    metrics = [k for k in rows[0] if k not in ("run_id", "family", "seed", "subset")]
    summaries = []
    for family in FAMILIES:
        for subset in SUBSETS:
            panel = [r for r in rows if r["family"] == family and r["subset"] == subset]
            summary = {}
            for metric in metrics:
                values = [r[metric] for r in panel]
                summary[metric] = {"values_by_seed": dict(zip(map(str, SEEDS), values)),
                                   "n_defined": sum(v is not None for v in values),
                                   "mean": sum(values)/5 if all(v is not None for v in values) else None}
            summaries.append({"family": family, "subset": subset, "n_training_realizations": 5, "metrics": summary})
    np.savez_compressed(output / "paired_outcomes.npz", base_prediction=np.stack(base_preds),
        transformed_prediction=np.stack(other_preds), target=data["target"][base], role=data["role"][base],
        group_index=data["group_index"][base], pair_id=data["pair_id"][base], run_id=np.asarray(run_ids))
    with (output / "transitions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    require(hashes(raw, names) == before and hashes(ROOT, implementation) == implementation, "inputs/source changed")
    report = {"schema_version": 1, "item": 13, "status": "completed",
              "source": {"source_commit": head, "plan_sha256": sha(plan_path), "implementation_sha256": implementation},
              "original_source_commit": OLD_SOURCE, "archive_identity": {"archive_commit": spec["archive_commit"], "path": spec["path"]},
              "input_sha256": before, "condition_mapping": CONDITIONS, "contrast": cfg["contrast"],
              "shared_bags": 256, "pairs_per_run": 2048, "rows": rows, "family_summaries": summaries,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "numerical_threads": 1},
              "available_ram_gib_at_admission": admission, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Secondary descriptive recount of already observed base/irrelevant-event-swap pairs.",
              "limits": ["Baseline contains the other event; no distractor-free or novel-token condition was added.",
                         "Conditional correctness is not high task competence; preserve base denominators and wrong stability.",
                         "Five saved training realizations; shared bags and paired queries are not independent replicates.",
                         "No new confidence intervals, hypothesis test, model inference or checkpoint access.",
                         "NLL is rechecked only for the two selected conditions; all saved logits/argmax are checked.",
                         "Selected-input integrity does not repeat the original full archive or source-ZIP audit."]}
    save(output / "report.json", report)
    payloads = {name: {"bytes": (output/name).stat().st_size, "sha256": sha(output/name)}
                for name in ("report.json", "transitions.csv", "paired_outcomes.npz")}
    save(output / "artifact_manifest.json", {"schema_version": 1, "item": 13, "files": payloads})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args.input, args.plan, args.output)
        print(json.dumps({"status": report["status"], "rows": len(report["rows"])}))
        return 0
    except Exception as error:
        save(args.output / "failure.json", {"item": 13, "status": "failed", "type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
