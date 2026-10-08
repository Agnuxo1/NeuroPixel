"""Count category and visible-filler choices in saved item-4 validation outputs.

This script imports no project models, generators, or training code. It reads
only the frozen validation NPZ and the eight validation prediction NPZ files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
FAMILIES = ("neuropixel", "standard_nca", "convgru", "relative_transformer")
RATES = (0.001, 0.003)
CATEGORY_BOUNDS = ((5, 17), (17, 27), (5, 17), (27, 35))


def file_record(path):
    path = path.resolve()
    label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return {"path": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def diagnose(pilot):
    """Return deterministic counts; do not load checkpoints or final outputs."""
    dataset_path = pilot / "datasets/split0_validation_n2048_seed61001.npz"
    with np.load(dataset_path, allow_pickle=False) as data:
        canvas, target, roles = (data[key] for key in ("canvas", "target", "roles"))
    if (canvas.shape != (2048, 8, 8) or target.shape != (2048,)
            or roles.shape != (2048,) or any(x.dtype.kind not in "iu" for x in (canvas, target, roles))
            or any(np.count_nonzero(roles == role) != 512 for role in range(4))):
        raise ValueError("Expected the frozen 2,048-example, four-role, 8x8 validation dataset")
    flattened = canvas.reshape(len(canvas), -1)
    runs = []
    for family in FAMILIES:
        for rate in RATES:
            run_id = f"{family}_s0_i7_lr{rate}"
            path = pilot / run_id / "validation_predictions.npz"
            with np.load(path, allow_pickle=False) as saved:
                prediction = saved["prediction"]
                if (prediction.shape != target.shape or prediction.dtype.kind not in "iu"
                        or not np.array_equal(saved["target"], target)
                        or not np.array_equal(saved["role"], roles)):
                    raise ValueError(f"Predictions or row alignment differ from validation: {run_id}")
            correct = prediction == target
            in_category = np.zeros(len(target), dtype=bool)
            for role, (lo, hi) in enumerate(CATEGORY_BOUNDS):
                in_category |= (roles == role) & (prediction >= lo) & (prediction < hi)
            visible_filler = ((prediction >= 5) & (prediction < 35)
                              & np.any(flattened == prediction[:, None], axis=1))
            indicators = {"correct_category": in_category, "visible_filler": visible_filler,
                          "visible_filler_of_query_category": visible_filler & in_category,
                          "correct_answer": correct}
            per_role = {}
            for role, name in enumerate(ROLES):
                selected = roles == role
                per_role[name] = {"n": int(selected.sum()), **{
                    key: int(np.count_nonzero(values & selected)) for key, values in indicators.items()}}
            noun_query = (roles == 0) | (roles == 2)
            runs.append({
                "run_id": run_id, "family": family, "learning_rate": rate,
                "split_seed": 0, "init_seed": 7, "predictions": file_record(path),
                "n": len(target), "totals": {key: int(values.sum()) for key, values in indicators.items()},
                "per_role": per_role,
                "binding": {"n": int(noun_query.sum()), "correct": int((correct & noun_query).sum()),
                            "macro_agent_patient_accuracy": float(np.mean([
                                correct[roles == role].mean() for role in (0, 2)]))},
            })
    return {
        "schema_version": 1, "scope": "post-hoc descriptive counts from saved validation predictions only",
        "new_model_inference": False, "final_data_accessed": False,
        "analysis_script": file_record(Path(__file__)), "numpy_version": np.__version__,
        "dataset": {**file_record(dataset_path), "n": len(target), "split_seed": 0, "sampling_seed": 61001},
        "rules": {
            "row_alignment": "Prediction target and role arrays must exactly equal dataset target and roles arrays, in order.",
            "public_vocabulary": {"PAD": [0], "role_tokens": [1, 2, 3, 4],
                                  "noun_ids_inclusive": [5, 16], "verb_ids_inclusive": [17, 26],
                                  "place_ids_inclusive": [27, 34]},
            "role_indices": dict(enumerate(ROLES)),
            "correct_category": "Prediction is a noun for AGENTE/PACIENTE, a verb for ACCION, or a place for LUGAR.",
            "visible_filler": "Prediction is a filler ID (5 through 34) appearing anywhere in that example's canvas.",
            "visible_filler_of_query_category": "Both correct_category and visible_filler are true; this is the visible-candidate count used in the interpretation.",
            "correct_answer": "Saved prediction equals saved target.",
            "binding": "Mean of AGENTE and PACIENTE correct-answer rates; counts pool these two equally sized role blocks.",
            "interpretation_limit": "These counts describe outputs; they do not establish positional invariance, a causal internal mechanism, or uncertainty across training seeds.",
        },
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "runs/research/item4_export/pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "results/research/04_validation_diagnostics.json")
    args = parser.parse_args()
    result = diagnose(args.pilot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": len(result["runs"]),
                      "new_model_inference": False, "final_data_accessed": False}))


if __name__ == "__main__":
    main()
