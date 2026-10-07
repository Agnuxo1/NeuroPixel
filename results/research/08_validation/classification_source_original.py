"""Shared execution, scoring and artifact utilities for the prospective pilot."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
from statistics import NormalDist
import subprocess
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

from neuropixel.task import ROLES
from neuropixel.research.data import ResearchRoleTask, frozen_dataset
from neuropixel.research.models import build_model, model_config

ROOT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def atomic_binary(path):
    """Expose a completed output only after closing and flushing its temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with atomic_binary(path) as stream:
        stream.write(data)
    if Path(path).read_bytes() != data:
        raise OSError(f"JSON artifact did not persist correctly: {path}")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(canvas, target, roles):
    digest = hashlib.sha256()
    for name, tensor in (("canvas", canvas), ("target", target), ("roles", roles)):
        array = np.ascontiguousarray(tensor.cpu().numpy(), dtype="<i8")
        header = json.dumps({"name": name, "shape": list(array.shape), "dtype": "<i8"},
                            sort_keys=True).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_arrays(path, **arrays):
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    with atomic_binary(path) as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as persisted:
        if set(persisted.files) != set(arrays):
            raise OSError("array artifact keys changed during persistence")
        for name, values in arrays.items():
            if not np.array_equal(persisted[name], values, equal_nan=True):
                raise OSError(f"array artifact changed during persistence: {name}")
    return {"path": str(path), "sha256": file_sha256(path), "bytes": Path(path).stat().st_size}


def source_record():
    paths = ["neuropixel/model.py", "neuropixel/task.py", "neuropixel/research/models.py",
             "neuropixel/research/data.py", "neuropixel/research/experiment.py",
             "scripts/research_train.py", "scripts/research_queue_worker.py",
             "docs/research/protocol.json"]
    snapshot = ROOT / "source_snapshot.json"
    if snapshot.exists():
        commit = json.loads(snapshot.read_text())["commit"]
    else:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                             text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
    return {"git_commit": commit, "sha256": {p: file_sha256(ROOT / p) for p in paths}}


def environment_record(device):
    versions = {}
    for package in ("numpy", "torch", "psutil", "pytest", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "device": str(device), "threads": torch.get_num_threads(),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}


class ResourceStop(RuntimeError):
    """The declared resource margin no longer holds."""


def resource_sample(device, minimum_ram_gib=8.0):
    import psutil
    row = {"at_utc": utc_now(), "available_ram_gib": psutil.virtual_memory().available / 2**30}
    if row["available_ram_gib"] < minimum_ram_gib:
        raise ResourceStop(f"available RAM {row['available_ram_gib']:.2f} GiB < {minimum_ram_gib} GiB")
    if device.type == "cuda":
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.free,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits", "--id=0"], text=True, timeout=15).strip()
        temperature, free_mib, utilization, power = [float(value.strip()) for value in raw.split(",")]
        row.update(temperature_c=temperature, free_gpu_gib=free_mib / 1024,
                   utilization_percent=utilization, board_power_w=power)
        if temperature >= 83 or free_mib / 1024 < 3.5:
            raise ResourceStop(f"GPU resource margin failed: {row}")
    return row


def configure_runtime(device_name, threads):
    if not 1 <= threads <= 4:
        raise ValueError("research reservation permits one to four CPU threads")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ResourceStop("CUDA was explicitly requested but is unavailable")
    return device


def wilson(correct, n, confidence=0.95):
    if n <= 0 or not 0 <= correct <= n:
        raise ValueError("invalid binary counts")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, z2 = correct / n, z*z
    center = (p + z2 / (2*n)) / (1 + z2/n)
    half = z * math.sqrt(p*(1-p)/n + z2/(4*n*n)) / (1 + z2/n)
    return [max(0.0, center-half), min(1.0, center+half)]


def classification_metrics(prediction, target, roles, nll=None, *, intervals=False,
                           bootstrap_seed=61003, repetitions=2000):
    prediction, target, roles = [np.asarray(x) for x in (prediction, target, roles)]
    if prediction.shape != target.shape or target.shape != roles.shape or target.ndim != 1:
        raise ValueError("prediction, target and roles must be equal-length vectors")
    correct = prediction == target
    per_role, role_means = {}, []
    for index, role in enumerate(ROLES):
        selected = correct[roles == index]
        if len(selected) == 0:
            raise ValueError(f"role {role} has no examples")
        k, n = int(selected.sum()), len(selected)
        per_role[role] = {"correct": k, "n": n, "accuracy": k/n,
                          "wilson_95": wilson(k, n)}
        role_means.append(k/n)
    result = {"correct": int(correct.sum()), "n": len(correct), "accuracy": float(correct.mean()),
              "macro_all_roles": float(np.mean(role_means)),
              "macro_agent_patient_accuracy": float(np.mean([role_means[0], role_means[2]])),
              "per_role": per_role}
    if nll is not None:
        result["cross_entropy"] = float(np.mean(np.asarray(nll, dtype=np.float64)))
    if intervals:
        rng = np.random.default_rng(bootstrap_seed)
        resampled = []
        for index in range(4):
            values = correct[roles == index].astype(np.float64)
            indices = rng.integers(len(values), size=(repetitions, len(values)))
            resampled.append(values[indices].mean(1))
        result["macro_all_roles_interval_95"] = np.quantile(np.mean(resampled, axis=0), [.025, .975]).tolist()
        result["binding_interval_95"] = np.quantile((resampled[0]+resampled[2])/2, [.025, .975]).tolist()
        result["interval_scope"] = "stratified example bootstrap conditional on this checkpoint; not training-run uncertainty"
    return result


@torch.inference_mode()
def evaluate(model, dataset, device, *, batch_size=256, intervals=False):
    canvas, target, roles = dataset
    was_training = model.training
    model.eval()
    predictions, confidences, losses = [], [], []
    try:
        for start in range(0, len(canvas), batch_size):
            stop = start + batch_size
            logits = model(canvas[start:stop].to(device))["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite evaluation logits")
            probabilities = logits.softmax(-1)
            predictions.append(logits.argmax(-1).cpu())
            confidences.append(probabilities.max(-1).values.cpu())
            losses.append(F.cross_entropy(logits, target[start:stop].to(device), reduction="none").cpu())
    finally:
        model.train(was_training)
    arrays = {"prediction": torch.cat(predictions).numpy(), "target": target.cpu().numpy(),
              "role": roles.cpu().numpy(), "confidence": torch.cat(confidences).numpy(),
              "nll": torch.cat(losses).numpy()}
    return classification_metrics(arrays["prediction"], arrays["target"], arrays["role"],
                                  arrays["nll"], intervals=intervals), arrays


def dataset_artifact(task, split, n, seed, directory):
    dataset = frozen_dataset(task, split, n, seed)
    path = Path(directory) / f"split{task.split_seed}_{split}_n{n}_seed{seed}.npz"
    record = save_arrays(path, canvas=dataset[0].numpy(), target=dataset[1].numpy(), roles=dataset[2].numpy())
    record.update(split=split, split_seed=task.split_seed, sampling_seed=seed, n=n,
                  content_sha256=tensor_digest(*dataset))
    return dataset, record


def model_from_config(config):
    return build_model(config["family"], vocab=config["vocab"], h=config["height"],
                       w=config["width"], steps=config["steps"], **config.get("variant", {}))


def train_one(config, output_dir, device, protocol, *, minimum_ram_gib=8.0):
    """Train without constructing or reading final evaluation examples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "training.json"
    source = source_record()
    environment = environment_record(device)
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if (previous.get("status") == "completed" and previous["config"] == config
                and previous["source"] == source and previous.get("environment") == environment):
            if (file_sha256(output_dir / "weights.pt") == previous["weights_sha256"]
                    and file_sha256(output_dir / "validation_predictions.npz") == previous["validation_predictions"]["sha256"]):
                return previous
        raise RuntimeError(f"existing run differs or is incomplete; preserve and investigate: {output_dir}")
    if any(output_dir.iterdir()):
        raise RuntimeError(f"interrupted run directory already contains evidence; preserve it: {output_dir}")
    started_at, total_start = utc_now(), time.perf_counter()
    curve, resources = [], []
    result = {"schema_version": 1, "status": "started", "config": config,
              "source": source, "started_at_utc": started_at}
    save_json(output_dir / "configuration.json", result)
    try:
        resources.append(resource_sample(device, minimum_ram_gib))
        task = ResearchRoleTask(config["height"], config["width"], seed=config["split_seed"])
        validation, val_record = dataset_artifact(task, "validation", protocol["data"]["validation_n"],
                                                 protocol["data"]["validation_sample_seed"], output_dir.parent / "datasets")
        probe, probe_record = dataset_artifact(task, "train", 512, 61006, output_dir.parent / "datasets")
        torch.manual_seed(config["init_seed"])
        model = model_from_config(config).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        resolved_model = model_config(config["family"], vocab=config["vocab"], h=config["height"],
                                      w=config["width"], steps=config["steps"], **config.get("variant", {}))
        # Initialization is complete. Sampling and cellular firing use different RNGs.
        torch.manual_seed(config["update_random_seed"])
        generator = torch.Generator(device="cpu").manual_seed(config["train_sample_seed"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      betas=(0.9, 0.999), eps=1e-8, weight_decay=config["weight_decay"])
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        training_start = time.perf_counter()
        window_loss = []
        for iteration in range(1, config["updates"] + 1):
            canvas, target = task.sample(config["batch_size"], "train", generator, device="cpu")
            canvas, target = canvas.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            school = config.get("school_weight", 0.0)
            output = model(canvas, lens_every=4) if school else model(canvas)
            answer_loss = F.cross_entropy(output["logits"], target)
            loss = answer_loss
            if school:
                if "lens" not in output:
                    raise ValueError("school supervision requires a model with explicit lens outputs")
                count = output["lens"].shape[1]
                occupied = (canvas != 0).unsqueeze(1).expand(-1, count, -1, -1)
                labels = canvas.unsqueeze(1).expand(-1, count, -1, -1)
                loss = loss + school * F.cross_entropy(output["lens"][occupied], labels[occupied])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {iteration}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            window_loss.append(float(loss.detach()))
            if iteration % 128 == 0 or iteration == config["updates"]:
                record = {"update": iteration, "mean_training_loss_since_last_log": float(np.mean(window_loss)),
                          "last_answer_loss": float(answer_loss.detach()), "last_unclipped_gradient_norm": float(gradient_norm),
                          "elapsed_training_seconds": time.perf_counter()-training_start}
                curve.append(record)
                window_loss.clear()
                resources.append(resource_sample(device, minimum_ram_gib))
                save_json(output_dir / "progress.json", {"status": "training", "config": config,
                                                         "curve": curve, "resources": resources})
                print(json.dumps({"event": "training_progress", "run": output_dir.name, **record}), flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter()-training_start
        val_metrics, val_predictions = evaluate(model, validation, device)
        probe_metrics, _ = evaluate(model, probe, device)
        val_artifact = save_arrays(output_dir / "validation_predictions.npz", **val_predictions)
        with atomic_binary(output_dir / "weights.pt") as stream:
            torch.save(model.state_dict(), stream)
        # Safe deserialization also detects an incomplete checkpoint immediately.
        restored = torch.load(output_dir / "weights.pt", map_location="cpu", weights_only=True)
        expected_state = model.state_dict()
        if (set(restored) != set(expected_state)
                or any(not torch.equal(restored[key], value.detach().cpu()) for key, value in expected_state.items())):
            raise OSError("checkpoint tensors differ after persistence")
        result.update(status="completed", completed_at_utc=utc_now(), parameter_count=parameter_count,
                      resolved_model=resolved_model, training_seconds=training_seconds,
                      total_wall_seconds=time.perf_counter()-total_start, environment=environment,
                      validation=val_metrics, train_probe=probe_metrics,
                      optimization_budget_limited=probe_metrics["macro_agent_patient_accuracy"] < 0.95,
                      validation_dataset=val_record, training_probe_dataset=probe_record,
                      validation_predictions=val_artifact, curve=curve, resources=resources,
                      weights_sha256=file_sha256(output_dir / "weights.pt"),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                      final_test_accessed=False)
        save_json(result_path, result)
        return result
    except BaseException as error:
        status = ("resource_stopped" if isinstance(error, ResourceStop) else
                  "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed")
        result.update(status=status,
                      completed_at_utc=utc_now(), total_wall_seconds=time.perf_counter()-total_start,
                      error_type=type(error).__name__, error=str(error), curve=curve, resources=resources,
                      final_test_accessed=False)
        save_json(result_path, result)
        raise


def select_pilot(runs, families, learning_rates):
    """Freeze LR and reference family using only completed validation results."""
    if len(runs) != len(families)*len(learning_rates) or any(r.get("status") != "completed" for r in runs):
        raise ValueError("all predeclared pilot runs must complete before selection")
    selected = {}
    for family in families:
        eligible = [run for run in runs if run["config"]["family"] == family]
        if sorted(run["config"]["learning_rate"] for run in eligible) != sorted(learning_rates):
            raise ValueError(f"pilot trials missing or duplicated for {family}")
        selected[family] = min(eligible, key=lambda run: (
            -run["validation"]["macro_agent_patient_accuracy"], run["validation"]["cross_entropy"],
            run["config"]["learning_rate"]))
    comparator = min((run for family, run in selected.items() if family != "neuropixel"),
                     key=lambda run: (-run["validation"]["macro_agent_patient_accuracy"],
                                      run["validation"]["cross_entropy"], run["parameter_count"],
                                      run["config"]["family"]))
    return {"frozen_at_utc": utc_now(), "selection_inputs": "validation only; no final test metrics",
            "selected_configs": {family: run["config"] for family, run in selected.items()},
            "selected_validation": {family: run["validation"] for family, run in selected.items()},
            "primary_reference_family": comparator["config"]["family"]}
