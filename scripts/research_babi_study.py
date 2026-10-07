"""Bounded item-10 QA training and final evaluation on pinned input artifacts.

No network download occurs here. The final phase opens only the declared TEST
member of the already pinned archive, after the durable training gate and an
exclusive consumed-access receipt. Torch and NumPy imports are deferred until
the committed plan, scientific source bindings and runtime are checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.research import babi_qa as qa
from research_item10_inputs import recover_input

FAMILIES = ("neuropixel", "relative_transformer")
LR_NAMES = {0.001: "0001", 0.003: "0003"}
DEV_FILES = ("encoder.json", "development_split.json", "encoded_official_train.json",
             "official_train_records.json", "train_audit_summary.json",
             "selected_source.json", "shortcut_models.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def save_json(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb" if exclusive else "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def append_json(path, value):
    with Path(path).open("ab") as stream:
        stream.write(qa.canonical_bytes(value) + b"\n")
        stream.flush()


def relative_path(root, path):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_relative_to(root), "artifact path escapes its declared root")
    return path.relative_to(root).as_posix()


def descriptor(root, path):
    path = Path(path)
    return {"path": relative_path(root, path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_reference(root, record):
    require(isinstance(record, dict) and set(("path", "bytes", "sha256")) <= set(record),
            "artifact reference is incomplete")
    path = (Path(root) / record["path"]).resolve()
    relative_path(root, path)
    require(path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
            "artifact identity differs: " + record["path"])
    return path


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, timeout=30,
                                   stdin=subprocess.DEVNULL).decode("utf-8").strip()


def source_record(plan_path, plan):
    require(not git("status", "--porcelain", "--untracked-files=no"), "tracked source is dirty")
    relative = relative_path(ROOT, plan_path)
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT,
                                        timeout=30, stdin=subprocess.DEVNULL)
    require(committed == Path(plan_path).read_bytes(), "plan bytes differ from committed HEAD")
    bindings = plan["implementation_sha256"]
    require(isinstance(bindings, dict) and bindings, "source binding inventory is empty")
    for path, expected in bindings.items():
        file = (ROOT / path).resolve()
        relative_path(ROOT, file)
        require(file.is_file() and sha(file) == expected, "scientific source drift: " + path)
    return {"source_commit": git("rev-parse", "HEAD"), "plan_sha256": sha(plan_path),
            "implementation_sha256": bindings, "recorded_at_utc": now(),
            "tracked_source_clean": True}


def admit():
    import psutil
    available = psutil.virtual_memory().available / 1024**3
    require(available >= 8.0, "available RAM is below 8 GiB")
    return available


def configure_runtime(plan):
    expected = plan["runtime"]
    require(platform.python_version() == expected["python"], "Python runtime differs")
    actual = {}
    for name, version in expected["packages"].items():
        actual[name] = importlib.metadata.version(name)
        require(actual[name] == version, "package runtime differs: " + name)
    require(expected["threads"] == 2 and expected["interop_threads"] == 1,
            "unexpected declared CPU thread policy")
    import torch
    import numpy as np
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1,
            "actual thread policy differs")
    require(torch.version.cuda is None, "a CPU-only Torch build is required")
    return torch, np, {"python": platform.python_version(), "packages": actual,
                      "threads": torch.get_num_threads(),
                      "interop_threads": torch.get_num_interop_threads(),
                      "cuda_version": torch.version.cuda, "available_ram_gib": admit()}


def validate_recipe(recipe):
    require(recipe["batch_size"] == 32 and recipe["weight_decay"] == 1e-4
            and recipe["gradient_clip"] == 1.0, "optimizer recipe differs")
    p = recipe["preflight"]
    required = {"memorization_seed": 58, "tiny_n": 32, "lr": 0.003,
                "max_updates": 4096, "check_every": 128, "required_streak": 2,
                "target_accuracy": 1.0, "target_ce": 0.05, "pilot_seed": 59,
                "pilot_updates": 1024, "learning_rates": [0.001, 0.003]}
    require(p == required, "preflight recipe differs from the declared bounded design")
    require(recipe["train"] == {"seeds": [60, 61, 62, 63, 64], "updates": 4096},
            "primary training recipe differs")
    require(isinstance(recipe["test_member"], str) and recipe["test_member"]
            and type(recipe["member_byte_cap"]) is int and recipe["member_byte_cap"] > 0,
            "final member identity/cap is missing")
    require(isinstance(recipe["development_fixtures"], list), "fixture exposure inventory is missing")
    for fixture in recipe["development_fixtures"]:
        require(set(fixture) == {"facts", "question"}, "fixtures must be input-only")
        qa.normalized_input(fixture["facts"], fixture["question"])


def load_development(plan, output):
    spec = plan["inputs"]["development"]
    directory = Path(recover_input(spec, output, "development")).resolve()
    for name in DEV_FILES:
        require(name in spec["files_sha256"] and sha(directory / name) == spec["files_sha256"][name],
                "development artifact is unbound or changed: " + name)
    summary = read_json(directory / "train_audit_summary.json")
    require(summary["status"] == "completed" and not summary["test_payload_extracted"],
            "development acquisition is incomplete or consumed TEST")
    require(not summary["encoded_input_gold_conflicts"] and not summary["raw_input_gold_conflicts"],
            "conflicting input labels block neural admission")
    encoder = read_json(directory / "encoder.json")
    split = read_json(directory / "development_split.json")
    parsed = read_json(directory / "official_train_records.json")
    encoded = read_json(directory / "encoded_official_train.json")
    by_record = {r["record_id"]: r for r in parsed["records"]}
    by_encoded = {r["record_id"]: r for r in encoded}
    require(len(by_record) == len(parsed["records"]) == len(by_encoded) == len(encoded),
            "development record inventory differs")
    require(set(split["train_ids"]).isdisjoint(split["validation_ids"])
            and set(split["train_ids"]) | set(split["validation_ids"]) == set(by_record),
            "development split does not cover records once")
    for rid, row in by_encoded.items():
        record = by_record[rid]
        fresh = qa.encode_input([f["text"] for f in record["facts"]], record["question"], encoder)
        require(fresh["canvas"] == row["canvas"] and record["answer"] == row["gold"]
                and record["input_sha256"] == row["input_sha256"],
                "encoded input or separate target differs from frozen raw record")
        require(qa.canonical_sha256(row["canvas"]) == row["encoded_input_sha256"],
                "encoded input digest differs")
        supported = record["answer"] in encoder["token_to_id"] and encoder["token_to_id"][record["answer"]] >= 2
        require(row["supported_gold"] is supported
                and row["gold_id"] == (encoder["token_to_id"][record["answer"]] if supported else None),
                "gold vocabulary mapping differs")
    train = [by_encoded[i] for i in split["train_ids"]]
    validation = [by_encoded[i] for i in split["validation_ids"]]
    require(train and validation and all(r["supported_gold"] for r in train),
            "optimization TRAIN must be nonempty and fully representable")
    require(encoder["height"] <= 8 and encoder["width"] <= 16, "geometry cap differs")
    identity = {"archive_commit": spec["archive_commit"], "path": spec["path"],
                "files_sha256": {name: spec["files_sha256"][name] for name in DEV_FILES},
                "encoder_sha256": spec["files_sha256"]["encoder.json"],
                "train_ids_sha256": qa.canonical_sha256(split["train_ids"]),
                "validation_ids_sha256": qa.canonical_sha256(split["validation_ids"])}
    return {"directory": directory, "encoder": encoder, "split": split, "parsed": parsed,
            "train": train, "validation": validation, "identity": identity,
            "identity_sha256": qa.canonical_sha256(identity),
            "controls": read_json(directory / "shortcut_models.json")}


def model_configuration(family, encoder):
    common = {"family": family, "vocab": len(encoder["vocabulary"]),
              "height": encoder["height"], "width": encoder["width"],
              "out_pos": encoder["out_pos"]}
    if family == "neuropixel":
        common["kwargs"] = {"c_id": 16, "c": 48, "hidden": 128, "steps": 16,
                            "fire_rate": 0.5, "retina": False}
    elif family == "relative_transformer":
        # Fixed prospectively after TRAIN geometry H4/W8/V18 was acquired,
        # before any neural result: 29,562 parameters versus core 29,552.
        # The ff128 draft had 27,482 parameters; historical drafts remain intact.
        common["kwargs"] = {"d": 32, "layers": 2, "heads": 4, "ff": 144}
    else:
        raise ValueError("unknown declared family")
    return common


def model_for(config, torch):
    from neuropixel.model import NeuroPixel
    from neuropixel.research.models import RelativeTransformer
    common = {"vocab": config["vocab"], "out_pos": tuple(config["out_pos"])}
    if config["family"] == "neuropixel":
        model = NeuroPixel(**common, **config["kwargs"])
        require(type(model) is NeuroPixel and not torch.count_nonzero(model.dictionary()[0]).item(),
                "direct corrected core PAD contract is required")
    else:
        model = RelativeTransformer(**common, h=config["height"], w=config["width"], **config["kwargs"])
    return model.cpu()


def atomic_arrays(path, arrays, np):
    path = Path(path)
    require(not path.exists(), "prediction output already exists")
    pending = path.with_name("_pending_" + path.name)
    with pending.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def atomic_tensor(path, value, torch):
    path = Path(path)
    require(not path.exists(), "checkpoint output already exists")
    pending = path.with_name("_pending_" + path.name)
    with pending.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def predict(model, rows, encoder, torch, np):
    size, vocabulary = len(rows), encoder["vocabulary"]
    can_encode = np.asarray([r.get("can_encode", True) for r in rows], dtype=np.bool_)
    supported = np.asarray([r["supported_gold"] for r in rows], dtype=np.bool_)
    logits = np.zeros((size, len(vocabulary)), dtype=np.float32)
    prediction = np.full(size, -1, dtype=np.int64)
    gold_id = np.asarray([r["gold_id"] if r["gold_id"] is not None else -1 for r in rows], dtype=np.int64)
    losses = np.zeros(size, dtype=np.float64)
    nll_supported = can_encode & supported
    positions = np.flatnonzero(can_encode).tolist()
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(positions), 128):
                admit()
                indices = positions[start:start + 128]
                canvas = torch.tensor([rows[i]["canvas"] for i in indices], dtype=torch.long)
                value = model(canvas)["logits"]
                require(value.shape == (len(indices), len(vocabulary)) and torch.isfinite(value).all().item(),
                        "nonfinite or malformed prediction logits")
                logits[indices] = value.cpu().numpy().astype(np.float32, copy=False)
                prediction[indices] = value.argmax(-1).cpu().numpy()
                valid = [j for j, i in enumerate(indices) if supported[i]]
                if valid:
                    labels = torch.tensor([gold_id[indices[j]] for j in valid], dtype=torch.long)
                    nll = -torch.log_softmax(value[valid].to(torch.float64), -1).gather(1, labels[:, None]).squeeze(1)
                    losses[[indices[j] for j in valid]] = nll.cpu().numpy()
    finally:
        model.train(was_training)
    tokens = [vocabulary[i] if i >= 0 else None for i in prediction.tolist()]
    metric = qa.score_answers(tokens, [r["gold"] for r in rows],
                              supported_mask=supported.tolist())
    eligible_losses = [float(losses[i]) for i in range(size) if nll_supported[i]]
    # Keep lexical output coverage separate from encodability and CE eligibility.
    metric.update(cross_entropy=math.fsum(eligible_losses) / len(eligible_losses) if eligible_losses else None,
                  cross_entropy_n=len(eligible_losses), gold_vocabulary_supported_n=int(supported.sum()),
                  gold_unseen_as_train_answer_n=sum(r["gold"] not in encoder["answer_tokens"] for r in rows),
                  can_encode_n=int(can_encode.sum()), cannot_encode_n=int((~can_encode).sum()),
                  nll_supported_n=int(nll_supported.sum()))
    arrays = {"logits": logits, "prediction_id": prediction, "gold_id": gold_id,
              "can_encode": can_encode, "gold_supported": supported, "nll_supported": nll_supported,
              "nll": losses, "record_id": np.asarray([r["record_id"] for r in rows], dtype="<U64")}
    return arrays, metric, tokens


def run_configuration(run_id, family, seed, learning_rate, updates, data, recipe, train_rows, kind):
    return {"run_id": run_id, "kind": kind, "family": family, "init_seed": seed,
            "firing_seed": 110000 + seed, "sample_seed": 100000 + seed,
            "learning_rate": learning_rate, "updates": updates, "batch_size": 32,
            "weight_decay": recipe["weight_decay"], "gradient_clip": recipe["gradient_clip"],
            "model": model_configuration(family, data["encoder"]),
            "recipe_sha256": qa.canonical_sha256(recipe),
            "data_identity_sha256": data["identity_sha256"],
            "training_record_ids": [r["record_id"] for r in train_rows],
            "training_record_ids_sha256": qa.canonical_sha256([r["record_id"] for r in train_rows])}


def train_one(config, train_rows, data, recipe, output, source, torch, np):
    run_dir = output / "study" / "runs" / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", config)
    result = {"schema_version": 1, "item": 10, "run_id": config["run_id"],
              "status": "started", "config": config, "config_sha256": qa.canonical_sha256(config),
              "source": source, "started_at_utc": now(), "final_accessed": False}
    save_json(run_dir / "started.json", result)
    started = time.perf_counter()
    try:
        admit()
        generator = np.random.default_rng(config["sample_seed"])
        indices = generator.integers(len(train_rows), size=(config["updates"], 32), dtype=np.int64)
        with (run_dir / "train_indices.npy").open("xb") as stream:
            np.save(stream, indices, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        result["minibatch_indices"] = descriptor(output, run_dir / "train_indices.npy")
        result["minibatch_indices_content_sha256"] = hashlib.sha256(indices.tobytes(order="C")).hexdigest()
        torch.manual_seed(config["init_seed"])
        model = model_for(config["model"], torch)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.manual_seed(config["firing_seed"])
        rng_initial = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        canvas = torch.tensor([r["canvas"] for r in train_rows], dtype=torch.long)
        targets = torch.tensor([r["gold_id"] for r in train_rows], dtype=torch.long)
        model.train()
        update_seconds, window, streak, checks = [], [], 0, []
        training_started = time.perf_counter()
        completed = 0
        for update, row_indices in enumerate(indices, 1):
            admit()
            tick = time.perf_counter()
            batch = torch.from_numpy(row_indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(canvas[batch])["logits"]
            loss = torch.nn.functional.cross_entropy(logits, targets[batch])
            require(torch.isfinite(loss).item(), "nonfinite answer loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"],
                                                  error_if_nonfinite=True)
            optimizer.step()
            elapsed = time.perf_counter() - tick
            update_seconds.append(elapsed)
            window.append(float(loss.detach()))
            completed = update
            if update % 128 == 0 or update == config["updates"]:
                record = {"update": update, "mean_answer_ce": math.fsum(window) / len(window),
                          "window_updates": len(window), "last_gradient_norm": float(norm),
                          "elapsed_training_seconds": time.perf_counter() - training_started}
                window.clear()
                if config["kind"] == "memorization":
                    probe_arrays, probe, _ = predict(model, train_rows, data["encoder"], torch, np)
                    probe_path = run_dir / f"memorization_check_{update:05d}.npz"
                    atomic_arrays(probe_path, probe_arrays, np)
                    passed = probe["accuracy"] == 1.0 and probe["cross_entropy"] <= 0.05
                    streak = streak + 1 if passed else 0
                    check = {"update": update, "accuracy": probe["accuracy"],
                             "cross_entropy": probe["cross_entropy"], "criterion_pass": passed,
                             "consecutive_passes": streak, "n": len(train_rows),
                             "record_ids_sha256": config["training_record_ids_sha256"],
                             "predictions": descriptor(output, probe_path)}
                    checks.append(check)
                    record["memorization_check"] = check
                append_json(run_dir / "train_log.jsonl", record)
                print(json.dumps({"event": "training_progress", "run_id": config["run_id"], **record}), flush=True)
                if config["kind"] == "memorization" and streak >= 2:
                    break
        training_seconds = time.perf_counter() - training_started
        atomic_tensor(run_dir / "checkpoint.pt", model.state_dict(), torch)
        outputs = {}
        for name, rows in (("validation", data["validation"]), ("train_probe", data["train"])):
            arrays, metrics, _ = predict(model, rows, data["encoder"], torch, np)
            path = run_dir / (name + "_predictions.npz")
            atomic_arrays(path, arrays, np)
            outputs[name] = {"metrics": metrics, "predictions": descriptor(output, path)}
        used = indices[:completed]
        unique_indices = sorted(set(used.reshape(-1).tolist()))
        result.update(status="completed", completed_at_utc=now(), completed_updates=completed,
                      parameter_count=parameter_count, training_seconds=training_seconds,
                      median_update_seconds=float(np.median(update_seconds)),
                      mean_update_seconds=math.fsum(update_seconds) / len(update_seconds),
                      update_seconds=update_seconds, training_log=descriptor(output, run_dir / "train_log.jsonl"),
                      checkpoint=descriptor(output, run_dir / "checkpoint.pt"),
                      config_artifact=descriptor(output, run_dir / "config.json"),
                      unique_training_record_count=len(unique_indices),
                      unique_training_raw_input_count=len({train_rows[i]["input_sha256"] for i in unique_indices}),
                      unique_training_encoded_input_count=len({train_rows[i]["encoded_input_sha256"] for i in unique_indices}),
                      used_minibatch_indices_sha256=hashlib.sha256(used.tobytes(order="C")).hexdigest(),
                      firing_rng_initial_sha256=rng_initial,
                      firing_rng_final_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                      validation=outputs["validation"], train_probe=outputs["train_probe"],
                      memorization_checks=checks,
                      memorization_pass=(streak >= 2) if config["kind"] == "memorization" else None,
                      train_probe_accuracy_below_095=outputs["train_probe"]["metrics"]["accuracy"] < 0.95,
                      elapsed_total_seconds=time.perf_counter() - started)
    except Exception as error:
        result.update(status="failed", completed_at_utc=now(),
                      elapsed_total_seconds=time.perf_counter() - started,
                      error={"type": type(error).__name__, "message": str(error)})
    save_json(run_dir / "run.json", result)
    return result, descriptor(output, run_dir / "run.json")


def selection_from(pilots, data, recipe):
    require(len(pilots) == 4 and all(x["status"] == "completed" for x in pilots),
            "all four completed pilot configurations are required")
    choices = {}
    for family in FAMILIES:
        rows = [r for r in pilots if r["config"]["family"] == family]
        require(sorted(r["config"]["learning_rate"] for r in rows) == [0.001, 0.003],
                "pilot learning-rate inventory differs")
        for row in rows:
            require(row["config"]["data_identity_sha256"] == data["identity_sha256"]
                    and row["config"]["recipe_sha256"] == qa.canonical_sha256(recipe),
                    "pilot data or recipe drift")
            metric = row["validation"]["metrics"]
            require(metric["n"] == len(data["validation"])
                    and metric["accuracy"] is not None and metric["cross_entropy"] is not None,
                    "pilot validation population or CE is incomplete")
        selected = min(rows, key=lambda r: (-r["validation"]["metrics"]["accuracy"],
                                             r["validation"]["metrics"]["cross_entropy"],
                                             r["config"]["learning_rate"]))
        choices[family] = {"run_id": selected["run_id"],
                           "learning_rate": selected["config"]["learning_rate"],
                           "validation": selected["validation"]["metrics"],
                           "checkpoint_sha256": selected["checkpoint"]["sha256"],
                           "config_sha256": selected["config_sha256"]}
    return {"schema_version": 1, "item": 10, "rule": ["validation_accuracy_desc", "validation_ce_asc", "lr_asc"],
            "recipe_sha256": qa.canonical_sha256(recipe), "data_identity_sha256": data["identity_sha256"],
            "choices": choices}


def preflight_specs(data):
    seen, tiny = set(), []
    for row in sorted(data["train"], key=lambda r: r["record_id"]):
        if row["input_sha256"] not in seen:
            seen.add(row["input_sha256"])
            tiny.append(row)
        if len(tiny) == 32:
            break
    require(len(tiny) == 32, "fewer than 32 unique optimization TRAIN inputs")
    specs = []
    for family in FAMILIES:
        specs.append((f"mem_{family}_s58", family, 58, 0.003, 4096, tiny, "memorization"))
    for family in FAMILIES:
        for lr in (0.001, 0.003):
            specs.append((f"pilot_{family}_lr{LR_NAMES[lr]}_s59", family, 59, lr, 1024, data["train"], "pilot"))
    return specs


def preflight(output, data, recipe, source, torch, np):
    specs = preflight_specs(data)
    runs, references = [], []
    for rid, family, seed, lr, updates, rows, kind in specs:
        config = run_configuration(rid, family, seed, lr, updates, data, recipe, rows, kind)
        result, reference = train_one(config, rows, data, recipe, output, source, torch, np)
        runs.append(result)
        references.append(reference)
    completed = all(r["status"] == "completed" for r in runs)
    memory_ok = all(r.get("memorization_pass") for r in runs[:2])
    summary = {"schema_version": 1, "item": 10, "status": "completed" if completed else "failed",
               "completed_at_utc": now(), "runs": references, "ordered_run_ids": [r["run_id"] for r in runs],
               "all_runs_completed": completed, "memorization_pass": memory_ok,
               "admission_passed": completed and memory_ok, "source": source,
               "data_identity": data["identity"], "data_identity_sha256": data["identity_sha256"],
               "recipe_sha256": qa.canonical_sha256(recipe), "final_accessed": False}
    if all(r["status"] == "completed" for r in runs[2:]):
        selection = selection_from(runs[2:], data, recipe)
        save_json(output / "selection.json", selection)
        summary["selection"] = descriptor(output, output / "selection.json")
    save_json(output / "preflight_summary.json", summary)
    require(completed and memory_ok, "preflight admission failed; all six run records were retained")


def _same_metrics(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_same_metrics(left[k], right[k]) for k in left)
    if isinstance(left, float):
        return isinstance(right, (float, int)) and not isinstance(right, bool) and math.isclose(left, right, abs_tol=1e-12, rel_tol=0)
    return left == right


def recount_development_predictions(path, rows, encoder, np):
    """Authenticate saved selection inputs without invoking a model."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    keys = {"logits", "prediction_id", "gold_id", "can_encode", "gold_supported",
            "nll_supported", "nll", "record_id"}
    require(set(arrays) == keys, "development prediction arrays differ")
    n, vocab = len(rows), len(encoder["vocabulary"])
    require(arrays["logits"].dtype == np.float32 and arrays["logits"].shape == (n, vocab)
            and np.isfinite(arrays["logits"]).all(), "malformed saved logits")
    for key in keys - {"logits"}:
        require(arrays[key].shape == (n,), "saved prediction row alignment differs")
    for key in ("can_encode", "gold_supported", "nll_supported"):
        require(arrays[key].dtype == np.bool_, "prediction mask dtype differs")
    for key in ("prediction_id", "gold_id"):
        require(arrays[key].dtype == np.int64, "prediction/target ID dtype differs")
    require(arrays["nll"].dtype == np.float64 and np.isfinite(arrays["nll"]).all(), "saved NLL is malformed")
    supported = np.asarray([r["supported_gold"] for r in rows], dtype=np.bool_)
    gold = np.asarray([r["gold_id"] if r["gold_id"] is not None else -1 for r in rows], dtype=np.int64)
    require(arrays["record_id"].tolist() == [r["record_id"] for r in rows]
            and np.array_equal(arrays["gold_id"], gold)
            and arrays["can_encode"].all()
            and np.array_equal(arrays["gold_supported"], supported)
            and np.array_equal(arrays["nll_supported"], supported),
            "saved population/support differs")
    require(np.array_equal(arrays["prediction_id"], arrays["logits"].argmax(1)), "saved argmax differs")
    fresh_losses = np.zeros(n, dtype=np.float64)
    if supported.any():
        values = arrays["logits"][supported].astype(np.float64)
        shifted = values - values.max(1, keepdims=True)
        fresh_losses[supported] = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(values)), gold[supported]]
    require(np.allclose(arrays["nll"], fresh_losses, atol=1e-12, rtol=0), "saved NLL does not recount")
    metrics = subset_metrics(arrays, rows, {"all": [True] * n}, encoder)["all"]
    return metrics


def load_selection(plan, output, data, recipe, source, np):
    spec = plan["inputs"]["preflight"]
    directory = Path(recover_input(spec, output, "preflight")).resolve()
    for name in ("preflight_summary.json", "selection.json"):
        require(name in spec["files_sha256"] and sha(directory / name) == spec["files_sha256"][name],
                "preflight selection artifact drift")
    summary = read_json(directory / "preflight_summary.json")
    require(summary["status"] == "completed" and summary["admission_passed"],
            "preflight did not pass its memorization/completion admission")
    require(summary["recipe_sha256"] == qa.canonical_sha256(recipe)
            and summary["data_identity_sha256"] == data["identity_sha256"],
            "preflight dependency recipe or data drift")
    runs = [read_json(verify_reference(directory, ref)) for ref in summary["runs"]]
    expected_configs = {
        rid: run_configuration(rid, family, seed, lr, updates, data, recipe, rows, kind)
        for rid, family, seed, lr, updates, rows, kind in preflight_specs(data)
    }
    expected_ids = list(expected_configs)
    require(len(runs) == 6 and [r["run_id"] for r in runs] == expected_ids
            and summary["ordered_run_ids"] == expected_ids, "preflight inventory differs")
    shared_paths = ("neuropixel/model.py", "neuropixel/research/models.py",
                    "neuropixel/research/babi_qa.py", "scripts/research_babi_study.py")
    require(all(summary["source"]["implementation_sha256"][path] == source["implementation_sha256"][path]
                for path in shared_paths), "scientific implementation differs from selected preflight")
    for run in runs:
        require(run["status"] == "completed", "incomplete preflight run")
        expected_config = expected_configs[run["run_id"]]
        expected_sha = qa.canonical_sha256(expected_config)
        require(run["config"] == expected_config
                and qa.canonical_sha256(run["config"]) == expected_sha
                and run["config_sha256"] == expected_sha, "preflight configuration differs from its declared recipe")
        for key in ("checkpoint", "config_artifact", "minibatch_indices", "training_log"):
            verify_reference(directory, run[key])
        saved_config = read_json(verify_reference(directory, run["config_artifact"]))
        require(saved_config == expected_config and qa.canonical_sha256(saved_config) == expected_sha,
                "preflight config artifact differs from the reconstructed configuration")
        completed = run["completed_updates"]
        require(type(completed) is int, "completed update count must be an integer")
        if expected_config["kind"] == "pilot":
            require(completed == 1024, "pilot did not complete exactly 1024 updates")
        else:
            require(128 <= completed <= 4096 and completed % 128 == 0,
                    "memorization update count differs from the declared check schedule")
        for key, rows in (("validation", data["validation"]), ("train_probe", data["train"])):
            prediction_path = verify_reference(directory, run[key]["predictions"])
            recounted = recount_development_predictions(prediction_path, rows, data["encoder"], np)
            require(_same_metrics(recounted, run[key]["metrics"]), "preflight metric does not recount: " + key)
        if run["config"]["kind"] == "memorization":
            by_id = {r["record_id"]: r for r in data["train"]}
            tiny = [by_id[rid] for rid in run["config"]["training_record_ids"]]
            require(len(tiny) == 32 and len(run["memorization_checks"]) >= 2,
                    "memorization check population/inventory differs")
            consecutive = 0
            expected_updates = list(range(128, run["completed_updates"] + 1, 128))
            require([c["update"] for c in run["memorization_checks"]] == expected_updates,
                    "memorization checkpoint schedule differs")
            for check in run["memorization_checks"]:
                probe_path = verify_reference(directory, check["predictions"])
                recounted = recount_development_predictions(probe_path, tiny, data["encoder"], np)
                require(check["n"] == 32
                        and check["record_ids_sha256"] == run["config"]["training_record_ids_sha256"]
                        and math.isclose(check["accuracy"], recounted["accuracy"], abs_tol=1e-12, rel_tol=0)
                        and math.isclose(check["cross_entropy"], recounted["cross_entropy"], abs_tol=1e-12, rel_tol=0),
                        "memorization criterion metrics do not recount")
                passed = check["accuracy"] == 1.0 and check["cross_entropy"] <= 0.05
                consecutive = consecutive + 1 if passed else 0
                require(check["criterion_pass"] is passed and check["consecutive_passes"] == consecutive,
                        "memorization consecutive-check decision differs")
            require(consecutive >= 2 and run["memorization_pass"] is True,
                    "memorization admission did not pass both final checks")
    recomputed = selection_from([r for r in runs if r["config"]["kind"] == "pilot"], data, recipe)
    saved = read_json(directory / "selection.json")
    require(saved == recomputed, "validation-only selection does not authenticate")
    return saved, {"archive_commit": spec["archive_commit"], "path": spec["path"],
                   "selection_sha256": sha(directory / "selection.json"),
                   "preflight_summary_sha256": sha(directory / "preflight_summary.json")}


def primary_configs(data, recipe, selection):
    configs = []
    for seed in range(60, 65):
        for family in FAMILIES:
            configs.append(run_configuration(f"train_{family}_s{seed}", family, seed,
                selection["choices"][family]["learning_rate"], 4096, data, recipe, data["train"], "primary"))
    return configs


def train_primary(output, data, recipe, source, selection, selection_identity, torch, np):
    runs, references = [], []
    for config in primary_configs(data, recipe, selection):
        run, reference = train_one(config, data["train"], data, recipe, output, source, torch, np)
        runs.append(run)
        references.append(reference)
    require(all(r["status"] == "completed" for r in runs), "one or more primary runs failed; records retained")
    for first, second in zip(runs[::2], runs[1::2]):
        require(first["minibatch_indices_content_sha256"] == second["minibatch_indices_content_sha256"],
                "paired input minibatch streams differ")
    manifest = {"schema_version": 1, "item": 10, "status": "completed", "created_at_utc": now(),
                "source_commit": source["source_commit"], "source": source,
                "run_key": output.name, "ordered_run_ids": [r["run_id"] for r in runs],
                "runs": [{"run_id": r["run_id"], "config": r["config"], "config_sha256": r["config_sha256"],
                          "checkpoint": r["checkpoint"], "summary": ref}
                         for r, ref in zip(runs, references)],
                "data_identity": data["identity"], "data_identity_sha256": data["identity_sha256"],
                "selection": selection, "selection_identity": selection_identity,
                "recipe_sha256": qa.canonical_sha256(recipe), "final_accessed": False}
    save_json(output / "training_manifest.json", manifest)


def authenticate_gate(output, data, recipe, source, selection, selection_identity):
    manifest_path, gate_path = output / "training_manifest.json", output / "pre_final_archive_receipt.json"
    manifest, gate = read_json(manifest_path), read_json(gate_path)
    require(manifest["schema_version"] == 1 and manifest["item"] == 10 and manifest["status"] == "completed",
            "training manifest is incomplete")
    require(gate["schema_version"] == 1 and gate["item"] == 10
            and gate["source_commit"] == source["source_commit"] == manifest["source_commit"]
            and gate["run_key"] == output.name == manifest["run_key"]
            and gate["training_manifest_sha256"] == sha(manifest_path),
            "pre-final archive receipt differs from durable training inventory")
    commit = gate["training_archive_commit"]
    require(isinstance(commit, str) and len(commit) == 40
            and all(x in "0123456789abcdef" for x in commit), "invalid archive commit identity")
    require(manifest["recipe_sha256"] == qa.canonical_sha256(recipe)
            and manifest["data_identity_sha256"] == data["identity_sha256"]
            and manifest["selection"] == selection and manifest["selection_identity"] == selection_identity,
            "training selection, data or recipe differs")
    expected = primary_configs(data, recipe, selection)
    require(manifest["ordered_run_ids"] == [c["run_id"] for c in expected]
            and len(manifest["runs"]) == 10, "primary checkpoint inventory differs")
    for item, config in zip(manifest["runs"], expected):
        require(item["run_id"] == config["run_id"] and item["config"] == config
                and item["config_sha256"] == qa.canonical_sha256(config), "primary configuration drift")
        run = read_json(verify_reference(output, item["summary"]))
        verify_reference(output, item["checkpoint"])
        require(run["status"] == "completed" and run["config"] == config
                and run["checkpoint"] == item["checkpoint"] and run["completed_updates"] == 4096
                and run["config_sha256"] == qa.canonical_sha256(config)
                and run["source"]["source_commit"] == source["source_commit"],
                "checkpoint summary does not authenticate its completed run")
        for key in ("config_artifact", "minibatch_indices", "training_log"):
            verify_reference(output, run[key])
        for key in ("validation", "train_probe"):
            verify_reference(output, run[key]["predictions"])
    require(datetime.fromisoformat(gate["at_utc"].replace("Z", "+00:00")) >=
            datetime.fromisoformat(manifest["created_at_utc"].replace("Z", "+00:00")),
            "gate receipt predates inventory")
    return manifest, gate


def final_rows(output, data, recipe, np):
    exposure = read_json(data["directory"] / "selected_source.json")
    archive = (data["directory"] / exposure["archive_filename"]).resolve()
    relative_path(data["directory"], archive)
    require(sha(archive) == exposure["archive_sha256"], "pinned compressed archive drift")
    payload = None
    with tarfile.open(archive, mode="r:gz") as tar:
        count, total = 0, 0
        for member in tar:
            count += 1
            total += member.size
            require(count <= 5000 and 0 <= member.size and total <= 1024**3, "TAR metadata exceeds limits")
            if member.name != recipe["test_member"]:
                continue
            require(payload is None and member.isfile()
                    and 0 < member.size <= recipe["member_byte_cap"], "invalid or duplicate exact TEST member")
            stream = tar.extractfile(member)
            require(stream is not None, "TEST member unreadable")
            with stream:
                payload = stream.read(recipe["member_byte_cap"] + 1)
            require(len(payload) == member.size, "TEST member length differs")
    require(payload is not None, "the declared TEST member is absent")
    final_dir = output / "final_data"
    final_dir.mkdir(exist_ok=False)
    raw = final_dir / "official_test.txt"
    with raw.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    parsed = qa.parse_babi(payload.decode("utf-8", errors="strict"),
        {"dataset": "bAbI original English 1k QA4", "split": "official_test",
         "member": recipe["test_member"], "archive_sha256": exposure["archive_sha256"],
         "sha256": hashlib.sha256(payload).hexdigest()})
    save_json(final_dir / "records.json", parsed)
    train_raw = {r["input_sha256"] for r in data["train"]}
    train_encoded = {r["encoded_input_sha256"] for r in data["train"]}
    all_raw = {r["input_sha256"] for r in data["parsed"]["records"]}
    all_encoded = {r["encoded_input_sha256"] for r in data["train"] + data["validation"]}
    fixture_encoded_overflows = []
    for fixture in recipe["development_fixtures"]:
        normalized = qa.normalized_input(fixture["facts"], fixture["question"])
        all_raw.add(qa.canonical_sha256(normalized))
        try:
            encoded = qa.encode_input(fixture["facts"], fixture["question"], data["encoder"])
            all_encoded.add(qa.canonical_sha256(encoded["canvas"]))
        except ValueError as error:
            fixture_encoded_overflows.append({"input_sha256": qa.canonical_sha256(normalized),
                                             "message": str(error)})
    rows, encoding_issues = [], []
    for record in parsed["records"]:
        row = {"record_id": record["record_id"], "episode_id": record["episode_id"],
               "episode_sha256": record["episode_sha256"], "input_sha256": record["input_sha256"],
               "gold": record["answer"], "can_encode": True}
        try:
            encoded = qa.encode_input([f["text"] for f in record["facts"]], record["question"], data["encoder"])
            row.update(canvas=encoded["canvas"], encoded_input_sha256=qa.canonical_sha256(encoded["canvas"]),
                       unknown_input_tokens=encoded["unknown_input_tokens"])
        except ValueError as error:
            row.update(can_encode=False, canvas=None, encoded_input_sha256=None, unknown_input_tokens=None)
            encoding_issues.append({"record_id": row["record_id"], "message": str(error)})
        value = data["encoder"]["token_to_id"].get(row["gold"], -1)
        row.update(supported_gold=value >= 2, gold_id=value if value >= 2 else None)
        row["novel_vs_optimization_train"] = (row["input_sha256"] not in train_raw
            and (not row["can_encode"] or row["encoded_input_sha256"] not in train_encoded))
        row["strict_novel_vs_full_exposure"] = (row["input_sha256"] not in all_raw
            and (not row["can_encode"] or row["encoded_input_sha256"] not in all_encoded))
        rows.append(row)
    save_json(final_dir / "encoded_records.json", rows)
    save_json(final_dir / "encoding_issues.json", encoding_issues)
    masks = {"all_official": [True] * len(rows),
             "novel_vs_optimization_train": [r["novel_vs_optimization_train"] for r in rows],
             "strict_novel_vs_full_exposure": [r["strict_novel_vs_full_exposure"] for r in rows]}
    before_predictions = {"created_at_utc": now(), "record_count": len(rows),
        "masks": masks, "subset_counts": {k: sum(v) for k, v in masks.items()},
        "subset_unencodable_counts": {k: sum(flag and not row["can_encode"] for flag, row in zip(v, rows))
                                     for k, v in masks.items()},
        "unsupported_gold_count": sum(not r["supported_gold"] for r in rows),
        "unencodable_count": len(encoding_issues), "fixture_encoded_overflows": fixture_encoded_overflows,
        "raw": descriptor(output, raw), "records": descriptor(output, final_dir / "records.json"),
        "encoded_records": descriptor(output, final_dir / "encoded_records.json")}
    save_json(final_dir / "population_manifest.json", before_predictions)
    return rows, parsed["records"], masks, before_predictions


def subset_metrics(arrays, rows, masks, encoder):
    vocabulary = encoder["vocabulary"]
    values = arrays["prediction_id"].tolist()
    predicted = [vocabulary[value] if value >= 0 else None for value in values]
    metrics = {}
    for name, mask in masks.items():
        indices = [i for i, present in enumerate(mask) if present]
        result = qa.score_answers([predicted[i] for i in indices], [rows[i]["gold"] for i in indices],
            supported_mask=[rows[i]["supported_gold"] for i in indices])
        eligible_losses = [float(arrays["nll"][i]) for i in indices if arrays["nll_supported"][i]]
        result.update(cross_entropy=math.fsum(eligible_losses) / len(eligible_losses) if eligible_losses else None,
                      cross_entropy_n=len(eligible_losses), nll_supported_n=len(eligible_losses),
                      gold_vocabulary_supported_n=sum(rows[i]["supported_gold"] for i in indices),
                      can_encode_n=sum(rows[i].get("can_encode", True) for i in indices),
                      cannot_encode_n=sum(not rows[i].get("can_encode", True) for i in indices),
                      gold_unseen_as_train_answer_n=sum(rows[i]["gold"] not in encoder["answer_tokens"] for i in indices))
        metrics[name] = result
    return metrics


def final_evaluation(output, data, recipe, source, selection, selection_identity, torch, np):
    manifest, gate = authenticate_gate(output, data, recipe, source, selection, selection_identity)
    save_json(output / "final_access.json", {"schema_version": 1, "item": 10,
        "status": "consumed", "consumed_at_utc": now(), "source": source,
        "training_manifest_sha256": sha(output / "training_manifest.json"),
        "pre_final_archive_receipt_sha256": sha(output / "pre_final_archive_receipt.json"),
        "training_archive_commit": gate["training_archive_commit"],
        "scope": "All ten frozen checkpoints; no retry after failed final access."})
    rows, raw_records, masks, population = final_rows(output, data, recipe, np)
    final_dir = output / "final_predictions"
    final_dir.mkdir(exist_ok=False)
    outputs = []
    for item in manifest["runs"]:
        admit()
        config = item["config"]
        torch.manual_seed(config["init_seed"])
        model = model_for(config["model"], torch)
        weights = torch.load(verify_reference(output, item["checkpoint"]), map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        arrays, _, _ = predict(model, rows, data["encoder"], torch, np)
        path = final_dir / (item["run_id"] + ".npz")
        atomic_arrays(path, arrays, np)
        result = {"run_id": item["run_id"], "checkpoint": item["checkpoint"],
                  "predictions": descriptor(output, path),
                  "metrics": subset_metrics(arrays, rows, masks, data["encoder"])}
        save_json(final_dir / (item["run_id"] + ".json"), result)
        outputs.append(result)
    control_rows = []
    for record in raw_records:
        facts = [f["text"] for f in record["facts"]]
        predictions = qa.predict_shortcuts(data["controls"], facts, record["question"])
        symbolic = qa.solve_raw(facts, record["question"])
        predictions["symbolic_raw"] = symbolic["answer"]
        control_rows.append({"record_id": record["record_id"], "predictions": predictions,
                             "symbolic_status": symbolic["status"], "symbolic_candidates": symbolic["candidates"]})
    save_json(output / "final_control_predictions.json", control_rows)
    controls = {}
    for subset, mask in masks.items():
        selected = [i for i, flag in enumerate(mask) if flag]
        controls[subset] = {}
        for name in control_rows[0]["predictions"]:
            controls[subset][name] = qa.score_answers(
                [control_rows[i]["predictions"][name] for i in selected], [rows[i]["gold"] for i in selected])
    save_json(output / "final_metrics.json", {"schema_version": 1, "item": 10, "status": "completed",
        "completed_at_utc": now(), "source": source, "population": population,
        "ordered_run_ids": manifest["ordered_run_ids"], "runs": outputs, "controls": controls,
        "control_scope": "Visible raw text controls can solve an input even when the neural grid overflows.",
        "limit": "Conditional external synthetic QA; no checkpoint-9 zero-shot or real-world claim."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "train", "final"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.plan)
    require(plan["schema_version"] == 1 and plan["item"] == 10 and plan["status"] == "frozen",
            "plan is not the frozen item-10 contract")
    require(plan["phase"] == ("preflight" if args.phase == "preflight" else "study"), "plan phase differs")
    require(plan.get("freeze_utc"), "freeze timestamp is missing")
    recipe = plan["recipe"]
    validate_recipe(recipe)
    source = source_record(args.plan.resolve(), plan)
    started_path = output / (args.phase + "_started.json")
    save_json(started_path, {"phase": args.phase, "started_at_utc": now(), "source": source})
    started = time.perf_counter()
    try:
        torch, np, runtime = configure_runtime(plan)
        save_json(output / ("study_" + args.phase + "_environment.json"), runtime)
        data = load_development(plan, output)
        save_json(output / (args.phase + "_data_identity.json"), data["identity"])
        if args.phase == "preflight":
            preflight(output, data, recipe, source, torch, np)
        else:
            selection, identity = load_selection(plan, output, data, recipe, source, np)
            if args.phase == "train":
                train_primary(output, data, recipe, source, selection, identity, torch, np)
            else:
                final_evaluation(output, data, recipe, source, selection, identity, torch, np)
        after = source_record(args.plan.resolve(), plan)
        require(after["source_commit"] == source["source_commit"]
                and after["plan_sha256"] == source["plan_sha256"], "source changed during phase")
        status = {"phase": args.phase, "status": "completed", "completed_at_utc": now(),
                  "wall_seconds": time.perf_counter() - started, "source": source,
                  "source_after": after, "available_ram_gib": admit()}
        save_json(output / ("study_" + args.phase + "_status.json"), status)
        return 0
    except Exception as error:
        save_json(output / ("study_" + args.phase + "_status.json"), {
            "phase": args.phase, "status": "failed", "completed_at_utc": now(),
            "wall_seconds": time.perf_counter() - started, "source": source,
            "error": {"type": type(error).__name__, "message": str(error)}})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
