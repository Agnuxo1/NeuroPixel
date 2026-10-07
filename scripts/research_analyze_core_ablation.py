"""Independently audit and summarize the prespecified item-6 core artifacts.

This program imports neither Torch nor NeuroPixel. It reads saved arrays, verifies
the complete 26-training/34-evaluation graph before opening final predictions,
and recomputes scores and paired seed-level contrasts. It performs no training,
model loading, example generation, selection, or new bootstrap. Run:
    python scripts/research_analyze_core_ablation.py --input CORE --output-dir REPORTS

The audit is independent of the evaluator implementation, not an external
laboratory replication. Two initialization seeds remain two statistical units.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product, combinations
import json
import math
import os
from pathlib import Path
import re
from statistics import NormalDist
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20, 21)
ROLES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
T_CRITICAL_DF1 = 1.0 / math.tan(math.pi * 0.025)
HISTORICAL = {
    "neuropixel/model.py": "564045fff799185e15b921873225d344014a4d4bcae6df6deacaf42f6e4bd701",
    "neuropixel/task.py": "e088d35937040f9924264b227e70e0ba9aaf23ede1b1c6bc9f8d6c78821a156f",
    "neuropixel/research/models.py": "646810a33a8ee97a8e66cafb61b418b3fee544394ef71123229e6ab0342c54ee",
    "neuropixel/research/data.py": "44cdf2b9ff492fe9eab115c8092c61c6d38bda5388babaf148380b89f2a7f94b",
    "neuropixel/research/experiment.py": "0a95e445e0654eb3666b3efc12a7bc594a237142dc8a7a2dc9527fac274f41ad",
    "scripts/research_train.py": "e3a8955a8561cf913a99310c4af3916d738b7f2a8290dc756c47df3d6872c92a",
    "docs/research/protocol.json": "41eea054d2be595469a8ffaa32ae4290ac26ebb1c27a20c9a3586dd4c0630290",
}
UNAVAILABLE = {
    "scripts/research_queue_worker.py": "73759c106c56775462c19c1cba9dfd82b0acac3c38467edf2e4c0af868c7be73"
}
DATASETS = {
    "validation": (2048, 61001, "44873b31d17df696ceef3ff7c5fd2f1878921305d19623443780a443be364e0a"),
    "train": (512, 61006, "f74bb2ae12e99b579eeea485bf6c262805a73f3c086a94421478466fc166b564"),
    "test": (4096, 61002, "23bca94c224b195da94ab1794801fdae131007a77bd594c2167a085227e7dba2"),
}
TRAIN_DAMAGE = {"after_step": 8, "apply_probability": 0.5, "erase_probability": 0.3, "seed": 62001}
EVAL_DAMAGE = {"after_step": 8, "erase_probability": 0.3, "seed": 62002}
LOSS_SCOPE = {
    "mean_training_loss_since_last_log": "Window mean of TOTAL objective over 128 updates.",
    "last_answer_loss": "Answer loss at the final update of that window.",
    "school_loss": "No separate series recorded; not reconstructible by subtraction of these differently supported fields.",
}
LIMITATIONS = [
    "Exploratory item 6; H1 is not reopened or rescued and no best arm is selected.",
    "Two initialization units on one split, fixed sampling streams and environment; no estimate of split-by-initialization or external-laboratory variation.",
    "The t interval has one degree of freedom and an uncheckable distributional assumption at n=2; it is not clipped.",
    "Saved evaluator bootstrap intervals are retained as conditional checkpoint intervals, not recomputed or used as training-run uncertainty.",
    "Checkpoint bytes are hashed, not deserialized; tensor contents and model execution are not independently rerun.",
    "Probe predictions were not saved by the frozen trainer. Probe flags are checked against internally consistent recorded probe counts, not a fresh prediction recount.",
    "Confidence and NLL are checked for finite valid values and NLL means; logits and full probability vectors were not saved, so they cannot be independently reconstructed.",
    "Event chronology verifies the saved record; it is not external evidence that no unlogged access occurred.",
    "Recovered historical scientific files are verified separately from the unavailable operational worker; a new execution HEAD is not the historical source commit.",
    "Lesion outcomes are final T16 robustness endpoints with reinjection available, not a measured recovery curve or autonomous reconstruction.",
    "Trained depth changes compute, locality and optimization; deployment truncation uses the same T16 weights and is a separate intervention.",
]


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def finite(value, label, minimum=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label}: expected finite number")
    if minimum is not None:
        require(value >= minimum, f"{label}: below {minimum}")
    return float(value)


def timestamp(value):
    require(isinstance(value, str), "timestamp is not text")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AuditError(f"invalid timestamp {value!r}") from error
    require(result.tzinfo is not None, "timestamp lacks timezone")
    return result


def same_number(actual, expected, label, atol=1e-12):
    finite(actual, label)
    require(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=atol),
            f"{label}: saved {actual!r}, recomputed {expected!r}")


def expected_inventory():
    """Independent transcription of the approved arm definitions, not a factory import."""
    definitions = []
    for seed, a, b, c in product(SEEDS, (False, True), (0.0, 0.3), (False, True)):
        definitions.append(("factorial", seed, a, b, c, 16, 1024, None))
    for seed, steps in product(SEEDS, (1, 4)):
        definitions.append(("recurrence", seed, True, 0.0, True, steps, 1024, None))
    for seed in SEEDS:
        definitions.append(("damage", seed, True, 0.0, True, 16, 1024, dict(TRAIN_DAMAGE)))
    for seed, school in product(SEEDS, (0.0, 0.3)):
        definitions.append(("optimization", seed, True, school, True, 16, 8192, None))
    configs, cases = [], []
    for panel, seed, tied, school, reinject, steps, updates, damage in definitions:
        school_id = str(float(school)).replace(".", "p")
        run_id = (f"{panel}_tie{int(tied)}_school{school_id}_reinj{int(reinject)}"
                  f"_T{steps}_damage{int(damage is not None)}_s0_i{seed}_u{updates}")
        config = {
            "run_id": run_id, "panel": panel, "phase": "item6_ablation",
            "protocol_id": "NP-SCI-20261006-v1", "family": "neuropixel",
            "vocab": 35, "height": 8, "width": 8, "split_seed": 0,
            "init_seed": seed, "learning_rate": 0.003, "updates": updates,
            "batch_size": 64, "steps": steps, "weight_decay": 0.0001,
            "gradient_clip": 1, "school_weight": school, "train_sample_seed": 9002,
            "update_random_seed": 9001,
            "variant": {"tied": tied, "reinject": reinject, "freeze_pad": False},
            "training_damage": damage,
        }
        configs.append(config)
        cases.append({"evaluation_id": run_id + "__clean", "run_id": run_id,
                      "kind": "clean", "steps_override": None, "damage": None})
        baseline = panel == "factorial" and tied and reinject and school == 0
        if baseline:
            for depth in (1, 4):
                cases.append({"evaluation_id": run_id + f"__deploy_T{depth}", "run_id": run_id,
                              "kind": "deployment_truncation", "steps_override": depth, "damage": None})
        if baseline or panel == "damage":
            cases.append({"evaluation_id": run_id + "__lesion", "run_id": run_id,
                          "kind": "fixed_lesion", "steps_override": None, "damage": dict(EVAL_DAMAGE)})
    return configs, cases


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}
        self.final_arrays_opened = False

    def path(self, relative):
        value = Path(relative)
        require(not value.is_absolute() and ".." not in value.parts, "unsafe relative artifact path")
        path = (self.root / value).resolve()
        require(path.is_relative_to(self.root), "artifact escapes input root")
        require(path.is_file(), f"missing artifact: {value.as_posix()}")
        return path

    def record(self, path, label=None):
        path = Path(path)
        name = label or path.relative_to(self.root).as_posix()
        row = {"sha256": digest(path), "bytes": path.stat().st_size}
        if name in self.inputs:
            require(self.inputs[name] == row, f"input changed during audit: {name}")
        self.inputs[name] = row
        return row

    def json(self, relative):
        path = self.path(relative)
        self.record(path)
        try:
            return json.loads(path.read_text(encoding="utf-8"),
                              parse_constant=lambda x: (_ for _ in ()).throw(AuditError(f"nonfinite JSON constant {x}")))
        except (ValueError, UnicodeDecodeError) as error:
            raise AuditError(f"invalid JSON in {relative}: {error}") from error

    def artifact(self, relative, sha256, record=None):
        path = self.path(relative)
        actual = self.record(path)
        require(actual["sha256"] == sha256, f"hash mismatch: {relative}")
        if record is not None:
            same_number(record["bytes"], actual["bytes"], f"{relative}: byte count", atol=0)
            saved = str(record["path"]).replace("\\", "/")
            suffix = str(relative).replace("\\", "/")
            require(saved == suffix or saved.endswith("/" + suffix),
                    f"{relative}: recorded path has wrong suffix")
        return path

    def arrays(self, relative, sha256, record=None, final=False):
        if final:
            require(self.final_arrays_opened, "final-array gate has not passed")
        path = self.artifact(relative, sha256, record)
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key].copy() for key in archive.files}
        except (ValueError, OSError) as error:
            raise AuditError(f"invalid NPZ {relative}: {error}") from error


def content_digest(arrays):
    """Implement the documented canonical integer-array digest without Torch."""
    value = hashlib.sha256()
    for name in ("canvas", "target", "roles"):
        array = np.ascontiguousarray(arrays[name], dtype="<i8")
        header = json.dumps({"name": name, "shape": list(array.shape), "dtype": "<i8"},
                            sort_keys=True).encode()
        value.update(len(header).to_bytes(8, "little"))
        value.update(header)
        value.update(array.tobytes())
    return value.hexdigest()


def read_dataset(reader, record, split, final=False):
    n, seed, expected = DATASETS[split]
    require({k: record[k] for k in ("split", "split_seed", "sampling_seed", "n")}
            == {"split": split, "split_seed": 0, "sampling_seed": seed, "n": n},
            f"{split}: dataset metadata differs")
    relative = f"datasets/split0_{split}_n{n}_seed{seed}.npz"
    arrays = reader.arrays(relative, record["sha256"], record, final=final)
    require(set(arrays) == {"canvas", "target", "roles"}, f"{split}: dataset keys")
    require(arrays["canvas"].shape == (n, 8, 8), f"{split}: canvas shape")
    for name in arrays:
        require(np.issubdtype(arrays[name].dtype, np.integer), f"{split}: {name} not integer")
    require(arrays["target"].shape == arrays["roles"].shape == (n,), f"{split}: target/role shape")
    require(np.all((arrays["canvas"] >= 0) & (arrays["canvas"] < 35)), f"{split}: invalid token")
    require(np.all((arrays["target"] >= 5) & (arrays["target"] < 35)), f"{split}: invalid target")
    require(np.array_equal(arrays["roles"], np.repeat(np.arange(4), n // 4)),
            f"{split}: frozen role-block ordering/balance differs")
    require(np.array_equal(arrays["canvas"][:, 7, 6] - 1, arrays["roles"]),
            f"{split}: role/query mismatch")
    require(content_digest(arrays) == record["content_sha256"] == expected,
            f"{split}: canonical content digest differs from frozen examples")
    return arrays


def wilson95(k, n):
    z = NormalDist().inv_cdf(0.975)
    p, z2 = k / n, z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def score_arrays(arrays, dataset, label):
    require(set(arrays) == {"prediction", "target", "role", "confidence", "nll"},
            f"{label}: prediction array keys")
    n = len(dataset["target"])
    require(all(a.shape == (n,) for a in arrays.values()), f"{label}: array shapes")
    for key in ("prediction", "target", "role"):
        require(np.issubdtype(arrays[key].dtype, np.integer), f"{label}: {key} not integer")
    require(np.array_equal(arrays["target"], dataset["target"]), f"{label}: target alignment")
    require(np.array_equal(arrays["role"], dataset["roles"]), f"{label}: role alignment")
    require(np.all((arrays["prediction"] >= 0) & (arrays["prediction"] < 35)),
            f"{label}: prediction out of vocabulary")
    for key in ("confidence", "nll"):
        require(np.issubdtype(arrays[key].dtype, np.floating), f"{label}: {key} not floating")
        require(np.isfinite(arrays[key]).all(), f"{label}: nonfinite {key}")
    require(np.all((arrays["confidence"] >= 0) & (arrays["confidence"] <= 1)),
            f"{label}: confidence outside [0,1]")
    require(np.all(arrays["nll"] >= 0), f"{label}: negative NLL")
    require(np.all(arrays["confidence"] >= 1 / 35 - 2e-6), f"{label}: maximum probability below uniform bound")
    true_probability = np.exp(-arrays["nll"].astype(np.float64))
    require(np.all(true_probability <= arrays["confidence"] + 2e-6),
            f"{label}: true-label probability exceeds recorded maximum")
    correct = arrays["prediction"] == arrays["target"]
    require(np.allclose(true_probability[correct], arrays["confidence"][correct], atol=2e-6, rtol=2e-6),
            f"{label}: correct prediction confidence/NLL mismatch")
    per_role = {}
    for role, name in enumerate(ROLES):
        group = correct[dataset["roles"] == role]
        count, total = int(group.sum()), len(group)
        per_role[name] = {"correct": count, "n": total, "accuracy": count / total,
                          "wilson_95": wilson95(count, total)}
    role_means = [per_role[name]["accuracy"] for name in ROLES]
    return {
        "correct": int(correct.sum()), "n": n, "accuracy": float(correct.mean()),
        "macro_all_roles": sum(role_means) / 4,
        "macro_agent_patient_accuracy": (role_means[0] + role_means[2]) / 2,
        "per_role": per_role,
        "cross_entropy": float(np.asarray(arrays["nll"], dtype=np.float64).mean()),
    }


def compare_metrics(saved, recounted, label, intervals=False):
    for key in ("correct", "n"):
        require(type(saved[key]) is int and saved[key] == recounted[key], f"{label}: {key}")
    for key in ("accuracy", "macro_all_roles", "macro_agent_patient_accuracy", "cross_entropy"):
        same_number(saved[key], recounted[key], f"{label}: {key}")
    require(set(saved["per_role"]) == set(ROLES), f"{label}: role names")
    for role in ROLES:
        a, b = saved["per_role"][role], recounted["per_role"][role]
        for key in ("correct", "n"):
            require(type(a[key]) is int and a[key] == b[key], f"{label}: {role} {key}")
        same_number(a["accuracy"], b["accuracy"], f"{label}: {role} accuracy")
        require(len(a["wilson_95"]) == 2, f"{label}: Wilson length")
        for i in (0, 1):
            same_number(a["wilson_95"][i], b["wilson_95"][i], f"{label}: {role} Wilson")
    if intervals:
        for key in ("macro_all_roles_interval_95", "binding_interval_95"):
            pair = saved[key]
            require(len(pair) == 2, f"{label}: {key} length")
            require(0 <= finite(pair[0], key) <= finite(pair[1], key) <= 1,
                    f"{label}: {key} bounds")
        require(saved.get("interval_scope") ==
                "stratified example bootstrap conditional on this checkpoint; not training-run uncertainty",
                f"{label}: conditional interval scope")


def audit_reported_probe(metric):
    """Counts-only consistency check; probe prediction arrays do not exist."""
    require(type(metric["n"]) is int and metric["n"] == 512, "probe sample count")
    role_counts = []
    for name in ROLES:
        row = metric["per_role"][name]
        require(type(row["n"]) is int and row["n"] == 128, "probe role sample count")
        require(type(row["correct"]) is int and 0 <= row["correct"] <= 128, "probe correct count")
        role_counts.append(row["correct"])
    reconstructed = {
        "correct": sum(role_counts), "n": 512, "accuracy": sum(role_counts) / 512,
        "macro_all_roles": sum(role_counts) / 512,
        "macro_agent_patient_accuracy": (role_counts[0] + role_counts[2]) / 256,
        "cross_entropy": finite(metric["cross_entropy"], "probe NLL", 0),
        "per_role": {name: {"correct": k, "n": 128, "accuracy": k / 128, "wilson_95": wilson95(k, 128)}
                     for name, k in zip(ROLES, role_counts)},
    }
    compare_metrics(metric, reconstructed, "reported probe")
    return reconstructed


def summarize_pair(values):
    require(set(values) == set(SEEDS), "paired summary requires exactly init20/init21")
    raw = [finite(values[s], f"seed{s} value") for s in SEEDS]
    mean = sum(raw) / 2
    sd = math.sqrt(sum((x - mean) ** 2 for x in raw))
    half = T_CRITICAL_DF1 * sd / math.sqrt(2)
    return {"seeds": list(SEEDS), "values": raw, "n": 2, "mean": mean,
            "sample_sd": sd, "range": [min(raw), max(raw)], "df": 1,
            "t_critical_95": T_CRITICAL_DF1, "t_interval_95": [mean - half, mean + half],
            "interval_scope": "Exploratory initialization-level t interval; one split, n=2, df=1; not clipped."}


def build_contrasts(rows):
    """Use only independently recounted binding scores and explicit intervention labels."""
    clean, deploy, lesion = {}, {}, {}
    for row in rows:
        c, case = row["config"], row["evaluation_case"]
        score = row["recounted_metrics"]["macro_agent_patient_accuracy"]
        key = (c["panel"], c["init_seed"], int(c["variant"]["tied"]),
               int(c["school_weight"] > 0), int(c["variant"]["reinject"]), c["steps"], c["updates"])
        target = clean if case["kind"] == "clean" else lesion if case["kind"] == "fixed_lesion" else deploy
        if target is deploy:
            key = key + (case["steps_override"],)
        require(key not in target, "duplicate statistical cell")
        target[key] = score
    factorial = {(s, a, b, c): clean[("factorial", s, a, b, c, 16, 1024)]
                 for s, a, b, c in product(SEEDS, (0, 1), (0, 1), (0, 1))}
    effects = []
    def add(panel, name, values, definition):
        effects.append({"panel": panel, "contrast": name, "metric": "macro_agent_patient_accuracy",
                        "definition": definition, **summarize_pair(values)})
    names = ("tying", "school", "reinjection")
    for axes_count in (1, 2, 3):
        for axes in combinations(range(3), axes_count):
            remaining = tuple(k for k in range(3) if k not in axes)
            def contrast(seed, fixed=None):
                total, count = 0.0, 0
                for bits in product((0, 1), repeat=3):
                    if fixed and any(bits[k] != v for k, v in fixed.items()):
                        continue
                    sign = math.prod(1 if bits[k] else -1 for k in axes)
                    total += sign * factorial[(seed,) + bits]
                    count += 1
                # Average over background cells, never over the differenced axes.
                return total / (count / (2 ** len(axes)))
            label = "_by_".join(names[k] for k in axes)
            add("factorial", label, {s: contrast(s) for s in SEEDS},
                f"Raw {axes_count}-factor difference, averaged equally over remaining factors; positive direction 0 to 1.")
            if remaining:
                for fixed_values in product((0, 1), repeat=len(remaining)):
                    fixed = dict(zip(remaining, fixed_values))
                    background = "_".join(f"{names[k]}{fixed[k]}" for k in remaining)
                    add("factorial_conditional", label + "__" + background,
                        {s: contrast(s, fixed) for s in SEEDS},
                        f"Raw {axes_count}-factor difference conditional on {background}.")
    baseline = lambda s: ("factorial", s, 1, 0, 1, 16, 1024)
    for depth in (1, 4):
        add("recurrence_trained", f"trained_T{depth}_minus_trained_T16",
            {s: clean[("recurrence", s, 1, 0, 1, depth, 1024)] - clean[baseline(s)] for s in SEEDS},
            "Separate training runs at the two horizons; changes compute/locality/optimization.")
        add("recurrence_deployment", f"same_T16_weights_eval_T{depth}_minus_eval_T16",
            {s: deploy[baseline(s) + (depth,)] - clean[baseline(s)] for s in SEEDS},
            "Inference truncation on the same T16-trained checkpoint.")
    damage_cells = {}
    for s in SEEDS:
        damaged = ("damage", s, 1, 0, 1, 16, 1024)
        damage_cells[s] = {(0, 0): clean[baseline(s)], (0, 1): lesion[baseline(s)],
                           (1, 0): clean[damaged], (1, 1): lesion[damaged]}
    for l in (0, 1):
        add("damage", f"damage_training_minus_clean_training__eval_lesion{l}",
            {s: damage_cells[s][1, l] - damage_cells[s][0, l] for s in SEEDS},
            "Q(train1,evalL) minus Q(train0,evalL).")
    for z in (0, 1):
        add("damage", f"eval_lesion_minus_clean__training_damage{z}",
            {s: damage_cells[s][z, 1] - damage_cells[s][z, 0] for s in SEEDS},
            "Q(trainZ,eval1) minus Q(trainZ,eval0).")
    add("damage", "training_damage_by_eval_lesion",
        {s: damage_cells[s][1, 1] - damage_cells[s][1, 0] - damage_cells[s][0, 1] + damage_cells[s][0, 0]
         for s in SEEDS}, "(Q11-Q10)-(Q01-Q00); positive means a less adverse lesion effect, not necessarily high absolute accuracy.")
    budget = {}
    for b in (0, 1):
        budget[b] = {s: clean[("optimization", s, 1, b, 1, 16, 8192)]
                     - clean[("factorial", s, 1, b, 1, 16, 1024)] for s in SEEDS}
        add("budget", f"updates8192_minus1024__school{b}", budget[b],
            "Eightfold update/exposure contrast at fixed tying/reinjection/T/LR.")
    add("budget", "budget_by_school", {s: budget[1][s] - budget[0][s] for s in SEEDS},
        "(8192-1024 at school0.3) minus (8192-1024 at school0).")
    add("budget_conditional", "school03_minus0__updates8192",
        {s: clean[("optimization", s, 1, 1, 1, 16, 8192)]
         - clean[("optimization", s, 1, 0, 1, 16, 8192)] for s in SEEDS},
        "Conditional school effect at 8192 updates; the 1024 counterpart is a factorial conditional effect.")
    cell_rows = [
        {"training_damage": z, "evaluation_lesion": l,
         **summarize_pair({s: damage_cells[s][z, l] for s in SEEDS})}
        for z, l in product((0, 1), (0, 1))
    ]
    return effects, cell_rows


def check_source(reader, manifest, source_root):
    source, controller = manifest["source"], manifest["controller_source"]
    require(source["sha256"] == HISTORICAL, "historical seven-file source anchor differs")
    require(source["unavailable_expected_sha256"] == UNAVAILABLE, "missing-worker provenance differs")
    require(source["historical_files_verified"] == 7 and source["historical_files_expected"] == 8,
            "historical recovery scope differs")
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source["git_commit"]) is not None,
            "execution commit is not a Git object ID")
    require(manifest["available_historical_sources_unchanged"] is True, "source unchanged declaration")
    required = {"neuropixel/research/ablations.py", "scripts/research_ablations.py",
                "docs/research/06_protocol_amendment.md", "docs/research/06_recovery_manifest.json"}
    require(required <= set(controller), "new implementation/amendment/recovery hashes missing")
    require(all(re.fullmatch(r"[0-9a-f]{64}", v) for v in controller.values()),
            "controller digest is not SHA256")
    paths = dict(HISTORICAL)
    for relative, value in controller.items():
        require(relative not in paths or paths[relative] == value, "conflicting source hashes")
        paths[relative] = value
    source_root = Path(source_root).resolve()
    for relative, value in paths.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe source path")
        actual = (source_root / path).resolve()
        require(actual.is_relative_to(source_root) and actual.is_file(), f"missing source {relative}")
        require(digest(actual) == value, f"source bytes differ: {relative}")
        reader.record(actual, "source/" + relative)
    recovery_path = source_root / "docs/research/06_recovery_manifest.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    require(recovery["available_historical_sha256"] == HISTORICAL
            and recovery["unavailable_historical_sha256"] == UNAVAILABLE,
            "recovery manifest contradicts historical anchor")
    require(source["recovery_manifest"] ==
            {"path": "docs/research/06_recovery_manifest.json",
             "sha256": controller["docs/research/06_recovery_manifest.json"]},
            "source recovery record hash differs")
    for relative in UNAVAILABLE:
        require(not (source_root / relative).exists(), "source declared unavailable now exists; refreeze provenance")
    # At least one frozen plan must contain this independently transcribed inventory.
    plans = []
    expected_runs, expected_cases = expected_inventory()
    for relative in controller:
        if relative.endswith(".json") and relative not in HISTORICAL:
            candidate = json.loads((source_root / relative).read_text(encoding="utf-8"))
            if (isinstance(candidate, dict) and "planned_runs" in candidate
                    and candidate.get("item") == 6 and candidate.get("status") == "frozen"):
                plans.append((relative, candidate))
    require(len(plans) == 1, "expected exactly one controller-hashed frozen core execution plan")
    relative, plan = plans[0]
    require(plan["item"] == 6 and plan["status"] == "frozen", "core execution plan is not frozen")
    require(plan["planned_runs"] == expected_runs and plan["planned_evaluations"] == expected_cases
            and plan["planned_training_runs"] == 26 and plan["planned_evaluation_cases"] == 34,
            "frozen plan differs from independent inventory")
    require({**plan["implementation_sha256"], relative: controller[relative]} == controller,
            "controller source does not match frozen implementation map plus plan")


def check_curve(training):
    config, curve = training["config"], training["curve"]
    require([row["update"] for row in curve] == list(range(128, config["updates"] + 1, 128)),
            f"{config['run_id']}: loss-curve updates differ")
    previous = 0.0
    for row in curve:
        for key in ("mean_training_loss_since_last_log", "last_answer_loss", "last_unclipped_gradient_norm"):
            finite(row[key], key, 0)
        elapsed = finite(row["elapsed_training_seconds"], "elapsed_training_seconds", 0)
        require(previous <= elapsed <= training["training_seconds"] + 1e-6,
                "curve elapsed time is inconsistent")
        previous = elapsed
    resources = training["resources"]
    require(len(resources) == len(curve) + 1, "resource log length differs from initial+each window")
    for row in resources:
        finite(row["available_ram_gib"], "available_ram_gib", 8)
        timestamp(row["at_utc"])
    return {
        "curve": curve, "observables": dict(LOSS_SCOPE),
        "minimum_recorded_available_ram_gib": min(row["available_ram_gib"] for row in resources),
    }


def audit_graph(reader, source_root):
    """All final-array reads are forbidden until this whole graph has passed."""
    manifest = reader.json("run_manifest.json")
    gate = reader.json("evaluation_gate.json")
    events = reader.json("events.json")
    expected_runs, expected_cases = expected_inventory()
    controller, source = manifest["controller_source"], manifest["source"]
    require(manifest["item"] == 6 and manifest["panel"] == "core", "wrong manifest scope")
    require(manifest["planned_training_runs"] == 26 and manifest["planned_evaluation_cases"] == 34,
            "wrong planned counts")
    with_controller = [dict(c, controller_source=controller) for c in expected_runs]
    require(manifest["planned_runs"] == with_controller, "manifest run order/config differs")
    require(manifest["planned_evaluations"] == expected_cases, "manifest evaluation order/config differs")
    require(manifest["selection_policy"] == "none; all prespecified configurations retained",
            "selection policy differs")
    observables = manifest["training_curve_observables"]
    require(set(observables) == {"mean_training_loss_since_last_log", "last_answer_loss",
                                "school_loss", "prohibited_derivation"},
            "training loss observables declaration missing")
    check_source(reader, manifest, source_root)
    require(gate["complete_training_runs"] == 26, "gate does not cover 26 runs")
    require(gate["source"] == source and gate["controller_source"] == controller, "gate provenance differs")
    require(gate["run_manifest_sha256"] == reader.inputs["run_manifest.json"]["sha256"],
            "gate manifest hash differs")
    run_ids = [c["run_id"] for c in with_controller]
    case_ids = [c["evaluation_id"] for c in expected_cases]
    require(set(gate["training_artifacts"]) == set(run_ids), "gate run identifiers differ")
    require(sorted(p.parent.name for p in reader.root.glob("*/training.json")) == sorted(run_ids),
            "extra/missing training artifact directories")
    require(sorted(p.name for p in (reader.root / "evaluations").iterdir() if p.is_dir()) == sorted(case_ids),
            "extra/missing evaluation directories")
    allowed_events = {"ablation_started", "training_completed", "final_evaluation_authorized",
                      "final_dataset_opened", "final_evaluation_saved", "ablation_completed"}
    require(events and all(e["event"] in allowed_events for e in events), "unknown/missing events")
    times = [timestamp(e["at_utc"]) for e in events]
    require(times == sorted(times), "events are not time-ordered")
    require(events[0]["event"] == "ablation_started" and events[-1]["event"] == "ablation_completed",
            "event boundaries differ")
    authorizations = [i for i, e in enumerate(events) if e["event"] == "final_evaluation_authorized"]
    openings = [i for i, e in enumerate(events) if e["event"] == "final_dataset_opened"]
    require(len(authorizations) == len(openings) == 1, "final authorization/opening must be unique")
    auth, opening = authorizations[0], openings[0]
    require(auth < opening, "final dataset opened before authorization")
    require(events[auth] == {"event": "final_evaluation_authorized", **gate}, "gate/event mismatch")
    require(events[opening]["dataset_content_sha256"] == DATASETS["test"][2], "opening dataset hash")
    require(all(e["event"] in {"ablation_started", "training_completed"} for e in events[:auth]),
            "a final event preceded the gate")
    require(not any(e["event"] in {"training_completed", "ablation_started"} for e in events[auth + 1:]),
            "training/resume occurred after final authorization")
    completions = [e for e in events[:auth] if e["event"] == "training_completed"]
    require(set(e["run"] for e in completions) == set(run_ids), "training completion IDs differ")
    # Repeated development completions may be documented resumes, never extra seed units.
    last_completion = {e["run"]: timestamp(e["at_utc"]) for e in completions}
    final_events = [e for e in events if e["event"] == "final_evaluation_saved"]
    require([e["evaluation"] for e in final_events] == case_ids, "saved evaluation order/IDs differ")
    require(all(timestamp(e["at_utc"]) >= times[opening] for e in final_events),
            "evaluation saved before final dataset opening")
    trainings, evaluations = {}, []
    environment = None
    val_record, probe_record = None, None
    for config in with_controller:
        run_id = config["run_id"]
        training = reader.json(f"{run_id}/training.json")
        init_record = reader.json(f"{run_id}/configuration.json")
        progress = reader.json(f"{run_id}/progress.json")
        artifacts = gate["training_artifacts"][run_id]
        require(reader.inputs[f"{run_id}/training.json"]["sha256"] == artifacts["training_sha256"],
                f"{run_id}: training hash differs from gate")
        require(training["status"] == "completed" and training["final_test_accessed"] is False,
                f"{run_id}: incomplete or premature final access")
        require(training["config"] == config and training["source"] == source, f"{run_id}: config/source")
        if environment is None:
            environment = training["environment"]
        require(training["environment"] == environment, f"{run_id}: environment differs")
        require(init_record == {"schema_version": 1, "status": "started", "config": config,
                               "source": source, "started_at_utc": training["started_at_utc"]},
                f"{run_id}: initial configuration evidence differs")
        start, completed = timestamp(training["started_at_utc"]), timestamp(training["completed_at_utc"])
        require(start <= completed <= last_completion[run_id] <= timestamp(gate["at_utc"]),
                f"{run_id}: training time crosses final gate")
        finite(training["training_seconds"], "training seconds", 0)
        wall = finite(training["total_wall_seconds"], "total wall seconds", 0)
        require(training["training_seconds"] <= wall, f"{run_id}: training exceeds wall duration")
        params = 29824 if config["variant"]["tied"] else 30384
        require(training["parameter_count"] == params == training["resolved_model"]["parameters"],
                f"{run_id}: parameter count")
        model = training["resolved_model"]
        require(model["factory"] == "neuropixel.research.ablations.training_model"
                and model["school_weight"] == config["school_weight"]
                and model["training_damage"] == config["training_damage"],
                f"{run_id}: resolved intervention metadata")
        require(model["school_lens_every"] == (4 if config["school_weight"] else None)
                and model["school_state_times"] == ([3, 7, 11, 15] if config["school_weight"] else []),
                f"{run_id}: school supervision times")
        require(model["primary_supervision"] ==
                ("answer_cross_entropy_plus_occupied_token_lens_cross_entropy"
                 if config["school_weight"] else "answer_cross_entropy_only"),
                f"{run_id}: supervision label")
        require(model["validation_intervention"] == "none; deterministic full firing in eval mode",
                f"{run_id}: validation intervention")
        check_curve(training)
        require(progress == {"status": "training", "config": config, "curve": training["curve"],
                             "resources": training["resources"]}, f"{run_id}: progress/training logs differ")
        require(artifacts["weights_sha256"] == training["weights_sha256"], f"{run_id}: weight hash anchor")
        reader.artifact(f"{run_id}/weights.pt", training["weights_sha256"])
        require(artifacts["validation_predictions_sha256"] == training["validation_predictions"]["sha256"],
                f"{run_id}: validation hash anchor")
        reader.artifact(f"{run_id}/validation_predictions.npz",
                        training["validation_predictions"]["sha256"], training["validation_predictions"])
        for record, old, label in ((training["validation_dataset"], val_record, "validation"),
                                   (training["training_probe_dataset"], probe_record, "probe")):
            if old is not None:
                require(record == old, f"{run_id}: shared {label} record differs")
        val_record, probe_record = training["validation_dataset"], training["training_probe_dataset"]
        probe = audit_reported_probe(training["train_probe"])
        require(type(training["optimization_budget_limited"]) is bool
                and training["optimization_budget_limited"] == (probe["macro_agent_patient_accuracy"] < 0.95),
                f"{run_id}: budget-limited flag differs")
        trainings[run_id] = training
    # The complete development-only training gate has passed before reading final result JSON.
    bundle = reader.json("ablation_records.json")
    require(bundle["status"] == "completed" and bundle["item"] == 6 and bundle["panel"] == "core",
            "core bundle is not completed item6")
    require(bundle["complete_training_runs"] == 26 and bundle["complete_evaluation_cases"] == 34,
            "bundle completeness differs")
    require(bundle["source"] == source and bundle["controller_source"] == controller,
            "bundle provenance differs")
    require(bundle["events"] == events, "bundle/event file mismatch")
    require(bundle["inference_latency_seconds"] is None and bundle["physical_energy_joules"] is None,
            "unexpected physical cost claims")
    require(bundle["selection"] == "none; this exploratory item does not reopen or rescue H1",
            "bundle selection policy differs")
    require(len(bundle["records"]) == 34, "bundle evaluation row count")
    require(bundle["environment"] == environment, "bundle environment differs from paired training")
    same_number(bundle["total_training_seconds"], sum(t["training_seconds"] for t in trainings.values()),
                "total training seconds", atol=1e-9)
    for index, case in enumerate(expected_cases):
        relative = f"evaluations/{case['evaluation_id']}/final_evaluation.json"
        row = reader.json(relative)
        training = trainings[case["run_id"]]
        require(row == bundle["records"][index], f"{relative}: bundle copy differs")
        require(row["status"] == "completed" and row["evaluation_case"] == case,
                f"{relative}: incomplete/mismatched evaluation")
        for key in ("config", "weights_sha256", "parameter_count", "source", "controller_source",
                    "training_seconds", "resolved_model", "train_probe", "optimization_budget_limited"):
            expected = controller if key == "controller_source" else training[key]
            require(row[key] == expected, f"{relative}: {key} differs from training")
        require(row["evaluated_at_utc"] == final_events[index]["at_utc"], f"{relative}: event time differs")
        require(times[opening] <= timestamp(row["evaluated_at_utc"]) <= times[-1],
                f"{relative}: evaluation outside declared final phase")
        if case["damage"] is None:
            require(row["lesion_keep_mask"] is None, f"{relative}: unexpected lesion mask")
        else:
            mask = row["lesion_keep_mask"]
            require(mask["damage"] == EVAL_DAMAGE and mask["shape"] == [4096, 1, 8, 8],
                    f"{relative}: lesion metadata")
            require(mask["ordering"] == "example-major; complete mask generated once, then sliced",
                    f"{relative}: lesion order")
        # Hash the saved prediction bytes now; np.load remains forbidden until the gate audit succeeds.
        reader.artifact(f"evaluations/{case['evaluation_id']}/final_predictions.npz",
                        row["predictions"]["sha256"], row["predictions"])
        evaluations.append(row)
    require(environment["device"] == "cpu" and environment["threads"] == 2,
            "core is not the declared two-thread CPU execution")
    require(environment["deterministic_algorithms"] is True and environment["cudnn_benchmark"] is False,
            "determinism settings differ")
    require(environment["versions"]["torch"].startswith("2.6.0"), "Torch version differs from frozen CPU environment")
    reader.final_arrays_opened = True
    return manifest, gate, bundle, trainings, evaluations, val_record, probe_record


def audit_core(input_dir, source_root=ROOT):
    reader = Reader(input_dir)
    manifest, gate, bundle, trainings, evaluations, val_record, probe_record = audit_graph(reader, source_root)
    validation = read_dataset(reader, val_record, "validation")
    probe = read_dataset(reader, probe_record, "train")
    final = read_dataset(reader, bundle["dataset"], "test", final=True)
    validation_recounts = {}
    for run_id, training in trainings.items():
        arrays = reader.arrays(f"{run_id}/validation_predictions.npz",
                               training["validation_predictions"]["sha256"], training["validation_predictions"])
        metrics = score_arrays(arrays, validation, run_id + " validation")
        compare_metrics(training["validation"], metrics, run_id + " validation")
        validation_recounts[run_id] = metrics
    scored, mask_record, mask_summary = [], None, None
    for row in evaluations:
        evaluation_id = row["evaluation_case"]["evaluation_id"]
        arrays = reader.arrays(f"evaluations/{evaluation_id}/final_predictions.npz",
                               row["predictions"]["sha256"], row["predictions"], final=True)
        recounted = score_arrays(arrays, final, evaluation_id)
        compare_metrics(row["metrics"], recounted, evaluation_id, intervals=True)
        if row["lesion_keep_mask"] is not None:
            current = row["lesion_keep_mask"]
            if mask_record is None:
                mask_record = current
                key = json.dumps(EVAL_DAMAGE, sort_keys=True)
                name = "lesion_keep_" + hashlib.sha256(key.encode()).hexdigest()[:16]
                arrays_mask = reader.arrays(f"datasets/{name}.npz", current["sha256"], current, final=True)
                require(set(arrays_mask) == {"keep"}, "lesion NPZ keys")
                keep = arrays_mask["keep"]
                require(keep.dtype == np.bool_ and keep.shape == (4096, 1, 8, 8),
                        "lesion array is not the complete boolean mask")
                mask_summary = {"record": current, "kept_cells": int(keep.sum()),
                                "total_cells": int(keep.size), "erased_fraction": float(1 - keep.mean()),
                                "scope": "Same saved mask across four evaluations; Torch RNG not rerun."}
            else:
                require(current == mask_record, "lesion evaluations do not share the same mask")
        scored.append({**row, "recounted_metrics": recounted})
    effects, damage_cells = build_contrasts(scored)
    score_groups = {}
    for row in scored:
        c, case = row["config"], row["evaluation_case"]
        # Remove the initialization token only, preserving all scientific factor labels.
        name = case["evaluation_id"].replace(f"_i{c['init_seed']}_", "_iPAIRED_")
        score_groups.setdefault(name, {})[c["init_seed"]] = row
    score_summaries = []
    for name, pair in score_groups.items():
        require(set(pair) == set(SEEDS), f"missing paired evaluation: {name}")
        for metric in ("accuracy", "macro_agent_patient_accuracy", "cross_entropy"):
            score_summaries.append({"evaluation": name, "metric": metric,
                                    **summarize_pair({s: pair[s]["recounted_metrics"][metric] for s in SEEDS})})
    training_rows = []
    for training in trainings.values():
        c = training["config"]
        training_rows.append({
            "run_id": c["run_id"], "config": c, "parameter_count": training["parameter_count"],
            "training_seconds": training["training_seconds"], "total_wall_seconds": training["total_wall_seconds"],
            "optimizer_updates": c["updates"], "minibatch_examples": c["updates"] * c["batch_size"],
            "dense_cell_updates": c["updates"] * c["batch_size"] * c["steps"] * 64,
            "train_probe_reported": training["train_probe"],
            "optimization_budget_limited": training["optimization_budget_limited"],
            "validation_recounted": validation_recounts[c["run_id"]],
            **check_curve(training),
        })
    exposures = {
        "training_runs": 26, "evaluation_cases": 34,
        "optimizer_updates": sum(c["config"]["updates"] for c in training_rows),
        "minibatch_examples": sum(c["minibatch_examples"] for c in training_rows),
        "dense_cell_updates": sum(c["dense_cell_updates"] for c in training_rows),
        "total_training_seconds": bundle["total_training_seconds"],
        "summed_run_total_wall_seconds": sum(c["total_wall_seconds"] for c in training_rows),
        "budget_limited_runs": sum(c["optimization_budget_limited"] for c in training_rows),
        "probe_binding_min": min(c["train_probe_reported"]["macro_agent_patient_accuracy"] for c in training_rows),
        "probe_binding_max": max(c["train_probe_reported"]["macro_agent_patient_accuracy"] for c in training_rows),
        "parameter_counts_by_run": {c["run_id"]: c["parameter_count"] for c in training_rows},
        "physical_energy_joules": None, "inference_latency_seconds": None,
        "scope": "Exposure counts include repeated prefixes, not independent examples or FLOPs; all resource flags retained.",
    }
    require(exposures["optimizer_updates"] == 55296 and exposures["minibatch_examples"] == 3538944,
            "core total exposure differs")
    # Ensure every inspected artifact still has the same bytes at the end.
    for label, record in list(reader.inputs.items()):
        path = (Path(source_root) / label[len("source/"):]) if label.startswith("source/") else reader.root / label
        require(digest(path) == record["sha256"], f"artifact changed during analysis: {label}")
    return {
        "schema_version": 1, "item": 6, "panel": "core", "status": "verified", "issues": [],
        "analysis_kind": "Independent saved-prediction recount, not an external laboratory replication.",
        "analyzer_sha256": digest(__file__), "numpy_version": np.__version__,
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": manifest["source"], "controller_source": manifest["controller_source"],
        "environment": bundle["environment"], "evaluation_gate": gate,
        "inputs": reader.inputs, "input_files_checked": len(reader.inputs),
        "final_predictions_recounted": 34 * 4096,
        "validation_predictions_recounted": 26 * 2048,
        "dataset_content_sha256": {split: definition[2] for split, definition in DATASETS.items()},
        "evaluations": scored, "paired_score_summaries": score_summaries,
        "contrasts": effects, "damage_cells": damage_cells,
        "training_observables": training_rows, "exposure": exposures, "lesion_mask": mask_summary,
        "loss_observables": dict(LOSS_SCOPE), "limitations": list(LIMITATIONS),
        "selection": "none", "H1": "not_reconsidered_in_exploratory_item6",
    }


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        require(path.read_bytes() == data, f"output readback failed: {path.name}")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def csv_text(rows, fields):
    from io import StringIO
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_outputs(report, output_dir):
    output = Path(output_dir)
    atomic_write(output / "core_analysis.json",
                 json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if report["status"] != "verified":
        return
    score_rows = []
    for row in report["evaluations"]:
        c, case, metric = row["config"], row["evaluation_case"], row["recounted_metrics"]
        score_rows.append({
            "evaluation_id": case["evaluation_id"], "run_id": c["run_id"], "init_seed": c["init_seed"],
            "panel": c["panel"], "kind": case["kind"], "tied": c["variant"]["tied"],
            "school_weight": c["school_weight"], "reinject": c["variant"]["reinject"],
            "training_steps": c["steps"],
            "evaluation_steps": case["steps_override"] if case["steps_override"] is not None else c["steps"],
            "training_damage": c["training_damage"] is not None, "evaluation_lesion": case["damage"] is not None,
            "updates": c["updates"], "parameters": row["parameter_count"],
            "accuracy": metric["accuracy"], "binding": metric["macro_agent_patient_accuracy"],
            "cross_entropy": metric["cross_entropy"],
            **{name.lower(): metric["per_role"][name]["accuracy"] for name in ROLES},
            "probe_binding_reported": row["train_probe"]["macro_agent_patient_accuracy"],
            "optimization_budget_limited": row["optimization_budget_limited"],
        })
    atomic_write(output / "core_scores.csv", csv_text(score_rows, list(score_rows[0])))
    contrast_rows = []
    for row in report["contrasts"]:
        contrast_rows.append({
            "panel": row["panel"], "contrast": row["contrast"], "metric": row["metric"],
            "init20": row["values"][0], "init21": row["values"][1], "n": 2, "mean": row["mean"],
            "sample_sd": row["sample_sd"], "minimum": row["range"][0], "maximum": row["range"][1],
            "df": 1, "t95_lower": row["t_interval_95"][0], "t95_upper": row["t_interval_95"][1],
            "definition": row["definition"],
        })
    atomic_write(output / "core_contrasts.csv", csv_text(contrast_rows, list(contrast_rows[0])))
    atomic_write(output / "analysis_output_manifest.json", json.dumps({
        "schema_version": 1, "files": {name: {"sha256": digest(output / name),
                                            "bytes": (output / name).stat().st_size}
                                     for name in ("core_analysis.json", "core_scores.csv", "core_contrasts.csv")}
    }, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Completed core artifact directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for independent reports.")
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Checkout containing the exact recorded source/controller bytes (default script repository).")
    args = parser.parse_args()
    require(args.input.resolve() != args.output_dir.resolve(), "output directory must differ from core artifacts")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
            "output directory must be absent or empty; preserve existing reports")
    try:
        report = audit_core(args.input, args.source_root)
    except (AuditError, KeyError, TypeError, ValueError, OSError) as error:
        report = {
            "schema_version": 1, "item": 6, "panel": "core", "status": "failed",
            "issues": [str(error)], "analyzer_sha256": digest(__file__),
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": "none", "H1": "not_reconsidered_in_exploratory_item6",
            "contrasts": [], "reason": "A complete verified graph is required; missing/nonfinite arms are not imputed.",
        }
    write_outputs(report, args.output_dir)
    print(json.dumps({"status": report["status"], "issues": report["issues"],
                      "output": str(args.output_dir)}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "verified" else 1)


if __name__ == "__main__":
    main()
