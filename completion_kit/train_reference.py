"""Complete CPU reference fitting, exact epoch recovery and DEV-only model sealing."""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import argparse
import json
import time
import uuid
from pathlib import Path
import numpy as np

from .common import read_json, write_json, sha256_file, identity, require_resources, verify_files
from .stable_field import StableField


def load_panel(path: Path, expected_digest: str) -> dict:
    """Authenticate a numeric dataset panel; refuse pickle and malformed labels."""
    if sha256_file(path) != expected_digest:
        raise ValueError("Dataset panel bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    if (result["inputs"].ndim != 4 or result["targets"].shape != (len(result["inputs"]),)
            or result["targets"].dtype.kind not in "iu" or not np.isfinite(result["inputs"]).all()
            or len(result["inputs"]) == 0):
        raise ValueError("Invalid dataset panel")
    return result


def evaluate(model: StableField, panel: dict, batch_size: int = 64, *, steps: int | None = None) -> tuple[dict, dict]:
    """Evaluate a fixed model and preserve logits, predictions and state magnitudes."""
    logits, magnitudes = [], []
    for at in range(0, len(panel["targets"]), batch_size):
        values, state = model.forward(panel["inputs"][at:at+batch_size], steps=steps)
        logits.append(values)
        magnitudes.append(np.max(np.abs(state), axis=(1, 2, 3)))
    logits = np.concatenate(logits)
    target = panel["targets"]
    shifted = logits - logits.max(axis=1, keepdims=True)
    nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(target)), target]
    prediction = logits.argmax(axis=1)
    arrays = dict(logits=logits, prediction=prediction, target=target,
                  nll=nll, maximum_state=np.concatenate(magnitudes), groups=panel["groups"])
    return dict(correct=int(np.sum(prediction == target)), n=len(target),
                accuracy=float(np.mean(prediction == target)), mean_nll=float(nll.mean()),
                maximum_state=float(arrays["maximum_state"].max())), arrays


def fit(plan_path: Path, output: Path) -> None:
    """Run one registered reference fit; resume only a matching epoch checkpoint."""
    require_resources()
    plan = read_json(plan_path)
    plan_digest = sha256_file(plan_path)
    verify_files(Path(plan["source_root"]), plan["source_sha256"])
    manifest_path = Path(plan["dataset_manifest"]).resolve()
    if sha256_file(manifest_path) != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    train = load_panel(base / manifest["panels"]["train"]["path"], manifest["panels"]["train"]["sha256"])
    dev = load_panel(base / manifest["panels"]["dev"]["path"], manifest["panels"]["dev"]["sha256"])
    if set(train["groups"]) & set(dev["groups"]):
        raise ValueError("TRAIN/DEV group leakage")
    if min(plan["epochs"], plan["batch_size"]) < 1 or plan["learning_rate"] <= 0:
        raise ValueError("Invalid fixed optimization budget")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "sealed.json").exists():
        raise FileExistsError("Closed fit exists; replay it rather than training it again")
    model = StableField(manifest["input_channels"], plan["state_channels"], manifest["classes"],
                        steps=plan["steps"], alpha=plan["alpha"], rho=plan["rho"],
                        seed=plan["seed"], constrained=plan["constrained"])
    rng = np.random.default_rng(1000000 + plan["seed"])
    moments = {name: np.zeros_like(value) for name, value in model.parameters.items()}
    velocities = {name: np.zeros_like(value) for name, value in model.parameters.items()}
    epoch0, update, curve, best = 0, 0, [], None
    checkpoint = output / "checkpoint.json"
    elapsed_previous, cpu_previous = 0.0, 0.0
    if checkpoint.exists():
        saved = read_json(checkpoint)
        if saved["plan_sha256"] != plan_digest:
            raise ValueError("Cannot resume a different registration")
        current_path = output / saved["current_model_path"]
        model = StableField.load(current_path)
        if sha256_file(current_path) != saved["current_model_sha256"]:
            raise ValueError("Checkpoint model identity changed")
        moments = {k: np.asarray(v) for k, v in saved["moments"].items()}
        velocities = {k: np.asarray(v) for k, v in saved["velocities"].items()}
        rng.bit_generator.state = saved["order_rng"]
        epoch0, update, curve, best = saved["epoch"], saved["updates"], saved["curve"], saved["best"]
        elapsed_previous, cpu_previous = saved["wall_seconds"], saved["cpu_seconds"]
        if best and sha256_file(output / best["path"]) != best["sha256"]:
            raise ValueError("Selected DEV checkpoint changed")
    start, cpu_start = time.perf_counter(), time.process_time()
    for epoch in range(epoch0, plan["epochs"]):
        order = rng.permutation(len(train["targets"]))
        losses = []
        for at in range(0, len(order), plan["batch_size"]):
            require_resources()
            indices = order[at:at+plan["batch_size"]]
            loss, gradients = model.loss_gradient(train["inputs"][indices], train["targets"][indices])
            norm = np.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
            scale = min(1.0, plan["gradient_clip"] / max(norm, 1e-30))
            update += 1
            for name, parameter in model.parameters.items():
                gradient = gradients[name] * scale
                moments[name] = 0.9 * moments[name] + 0.1 * gradient
                velocities[name] = 0.999 * velocities[name] + 0.001 * gradient**2
                corrected_m = moments[name] / (1 - 0.9**update)
                corrected_v = velocities[name] / (1 - 0.999**update)
                parameter -= plan["learning_rate"] * corrected_m / (np.sqrt(corrected_v) + 1e-8)
            model.project()
            losses.append(loss)
        metrics, _ = evaluate(model, dev, plan["batch_size"])
        row = dict(epoch=epoch+1, updates=update, mean_batch_loss=float(np.mean(losses)), dev=metrics)
        curve.append(row)
        # Immutable model generations preserve the last committed epoch across a crash.
        current_relative = str(Path("epochs") / f"epoch{epoch+1:04d}_{uuid.uuid4().hex}" / "model.json")
        model.save(output / current_relative)
        current_digest = sha256_file(output / current_relative)
        selection_key = [-metrics["correct"], metrics["mean_nll"], epoch+1]
        if best is None or selection_key < best["key"]:
            best = dict(key=selection_key, epoch=epoch+1, dev=metrics,
                        sha256=current_digest, path=current_relative)
        write_json(checkpoint, dict(plan_sha256=plan_digest, epoch=epoch+1, updates=update,
                                    curve=curve, best=best, order_rng=rng.bit_generator.state,
                                    moments={k: v.tolist() for k, v in moments.items()},
                                    velocities={k: v.tolist() for k, v in velocities.items()},
                                    current_model_sha256=current_digest, current_model_path=current_relative,
                                    wall_seconds=elapsed_previous+time.perf_counter()-start,
                                    cpu_seconds=cpu_previous+time.process_time()-cpu_start))
        print(json.dumps(row), flush=True)
    final_checkpoint = read_json(checkpoint)
    selected_model = StableField.load(output / best["path"])
    write_json(output / "sealed.json", dict(status="dev_selected_model_sealed",
               plan_sha256=plan_digest, model_sha256=best["sha256"], model_relative_path=best["path"], selection=best,
               dataset_manifest_sha256=plan["dataset_manifest_sha256"],
               epochs=plan["epochs"], updates=update, seed=plan["seed"],
               certificate=selected_model.certificate(),
               wall_seconds=final_checkpoint["wall_seconds"], cpu_seconds=final_checkpoint["cpu_seconds"],
               scope="New CPU reference candidate; no historical hypothesis is repaired",
               interrupted_epoch_work_is_not_fully_accounted=True), exclusive=True)


def score_cohort(plan_paths: list[Path], fits: list[Path], output: Path) -> None:
    """Seal all registered models before opening TEST and preserve every endpoint."""
    if len(plan_paths) != len(fits) or not fits:
        raise ValueError("Complete cohort required")
    records = []
    for plan_path, fit_folder in zip(plan_paths, fits):
        plan, seal = read_json(plan_path), read_json(fit_folder / "sealed.json")
        if seal["plan_sha256"] != sha256_file(plan_path) or seal["status"] != "dev_selected_model_sealed":
            raise ValueError("Fit seal differs")
        if sha256_file(fit_folder / seal["model_relative_path"]) != seal["model_sha256"]:
            raise ValueError("Selected model changed")
        verify_files(Path(plan["source_root"]), plan["source_sha256"])
        records.append(dict(plan=plan, seal=seal, model_path=str((fit_folder / seal["model_relative_path"]).resolve())))
    identities = {(r["plan"]["seed"], r["plan"]["variant"]) for r in records}
    if len(identities) != len(records):
        raise ValueError("Repeated training realization")
    registrations = {r["plan"]["cohort"] for r in records}
    if len(registrations) != 1:
        raise ValueError("Mixed cohort registrations")
    expected = {(s, v) for s in records[0]["plan"]["cohort_seeds"] for v in records[0]["plan"]["cohort_variants"]}
    if identities != expected or any(r["plan"]["cohort_seeds"] != records[0]["plan"]["cohort_seeds"]
                                    or r["plan"]["cohort_variants"] != records[0]["plan"]["cohort_variants"] for r in records):
        raise ValueError("Incomplete fixed cohort")
    if len({r["plan"]["dataset_manifest_sha256"] for r in records}) != 1:
        raise ValueError("Mixed cohort datasets")
    gate = dict(records=records, registration=list(registrations)[0],
                status="all_registered_models_sealed_before_test")
    if (output / "result.json").exists():
        raise FileExistsError("Closed scored cohort exists; recount instead of scoring again")
    if output.exists():
        if not (output / "cohort_gate.json").is_file() or read_json(output / "cohort_gate.json") != gate:
            raise ValueError("Existing scoring directory belongs to another cohort")
        access = read_json(output / "test_access.json")
        if access["gate_sha256"] != sha256_file(output / "cohort_gate.json"):
            raise ValueError("Prior TEST exposure record differs")
    else:
        output.mkdir(parents=True)
        write_json(output / "cohort_gate.json", gate, exclusive=True)
        write_json(output / "test_access.json", dict(status="consumed_before_test_loading",
                    gate_sha256=sha256_file(output / "cohort_gate.json")), exclusive=True)
    summaries = read_json(output / "partial_scores.json") if (output / "partial_scores.json").is_file() else []
    completed = {row["key"]: row for row in summaries}
    if len(completed) != len(summaries):
        raise ValueError("Repeated prior scoring key")
    try:
        for record in records:
            require_resources()
            plan = record["plan"]
            manifest_path = Path(plan["dataset_manifest"])
            if sha256_file(manifest_path) != plan["dataset_manifest_sha256"]:
                raise ValueError("Scoring manifest changed")
            manifest = read_json(manifest_path)
            model = StableField.load(Path(record["model_path"]))
            for panel_name in ("test_base", "test_long"):
                panel_info = manifest["panels"][panel_name]
                panel = load_panel(manifest_path.parent / panel_info["path"], panel_info["sha256"])
                for depth in plan["diagnostic_depths"]:
                    key = f"seed{plan['seed']}_{plan['variant']}_{panel_name}_T{depth}"
                    artifact = output / (key + ".npz")
                    if key in completed:
                        if sha256_file(artifact) != completed[key]["artifact_sha256"]:
                            raise ValueError("Completed scoring artifact changed")
                        continue
                    if artifact.exists():
                        # An atomically completed array may precede its summary at interruption.
                        with np.load(artifact, allow_pickle=False) as archive:
                            arrays = {name: archive[name].copy() for name in archive.files}
                        if not np.array_equal(arrays["target"], panel["targets"]):
                            raise ValueError("Uncommitted saved endpoint has different targets")
                        metrics = dict(correct=int(np.sum(arrays["prediction"] == arrays["target"])),
                                       n=len(arrays["target"]),
                                       accuracy=float(np.mean(arrays["prediction"] == arrays["target"])),
                                       mean_nll=float(arrays["nll"].mean()),
                                       maximum_state=float(arrays["maximum_state"].max()))
                    else:
                        metrics, arrays = evaluate(model, panel, plan["batch_size"], steps=depth)
                        temporary = output / (key + ".partial.npz")
                        if temporary.exists():
                            # Preserve an incomplete write; it was not a completed endpoint.
                            temporary.rename(output / (key + ".failed_" + uuid.uuid4().hex + ".npz"))
                        np.savez_compressed(temporary, **arrays)
                        os.replace(temporary, artifact)
                    summaries.append(dict(key=key, seed=plan["seed"], variant=plan["variant"],
                                          panel=panel_name, depth=depth,
                                          artifact_sha256=sha256_file(artifact), **metrics))
                    write_json(output / "partial_scores.json", summaries)
        write_json(output / "result.json", dict(status="complete_registered_reference_cohort",
                    scores=summaries, external_replication=False,
                    finite_depth_observations_not_indefinite_memory=True), exclusive=True)
    except Exception as error:
        write_json(output / ("failure_" + uuid.uuid4().hex + ".json"), dict(
                    error_type=type(error).__name__, message=str(error),
                    retained_score_rows=len(summaries), test_exposure_preserved=True), exclusive=True)
        raise


def register(manifest_path: Path, output: Path, seeds: list[int], *, epochs: int = 40,
             state_channels: int = 16) -> None:
    """Freeze a NEW constrained/unconstrained paired development cohort."""
    if output.exists() or len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("New directory and unique seeds required")
    output.mkdir(parents=True)
    variants = ["contractive", "unconstrained"]
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {"completion_kit/" + name: sha256_file(source_root / "completion_kit" / name)
                     for name in ("common.py", "stable_field.py", "train_reference.py", "datasets.py")}
    for seed in seeds:
        for variant in variants:
            plan = dict(schema=1, cohort=output.name, variant=variant, seed=seed,
                        source_root=str(source_root), source_sha256=source_hashes,
                        cohort_seeds=seeds, cohort_variants=variants,
                        dataset_manifest=str(manifest_path.resolve()),
                        dataset_manifest_sha256=sha256_file(manifest_path),
                        state_channels=state_channels, steps=16, alpha=0.5, rho=0.95,
                        constrained=variant == "contractive", epochs=epochs, batch_size=32,
                        learning_rate=0.001, gradient_clip=1.0, diagnostic_depths=[8, 16, 32, 64],
                        primary_panel="test_base", primary_depth=16,
                        selection="DEV correct count, mean NLL, earliest epoch",
                        scope="Development reference comparison, not confirmatory architectural superiority")
            write_json(output / f"seed{seed}_{variant}.json", plan, exclusive=True)


def main() -> None:
    """Register, fit or score a complete fixed CPU-reference cohort."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    registration = subparsers.add_parser("register")
    registration.add_argument("--manifest", type=Path, required=True)
    registration.add_argument("--output", type=Path, required=True)
    registration.add_argument("--seeds", type=int, nargs="+", default=[700, 701, 702, 703, 704])
    registration.add_argument("--epochs", type=int, default=40)
    registration.add_argument("--state-channels", type=int, default=16)
    training = subparsers.add_parser("fit")
    training.add_argument("--plan", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    scoring = subparsers.add_parser("score")
    scoring.add_argument("--plans", type=Path, nargs="+", required=True)
    scoring.add_argument("--fits", type=Path, nargs="+", required=True)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "register":
        register(args.manifest, args.output, args.seeds, epochs=args.epochs, state_channels=args.state_channels)
    elif args.action == "fit":
        fit(args.plan, args.output)
    else:
        score_cohort(args.plans, args.fits, args.output)


if __name__ == "__main__":
    main()
