"""Independent integer/prediction recount for the complete new reference cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .common import read_json, write_json, sha256_file
from .statistics import paired_interval


def analyze(folder: Path) -> dict:
    """Recount every saved endpoint; report the registered primary pair without selection."""
    result = read_json(folder / "result.json")
    gate = read_json(folder / "cohort_gate.json")
    if result["status"] != "complete_registered_reference_cohort" or gate["status"] != "all_registered_models_sealed_before_test":
        raise ValueError("Complete scored cohort and prior seal required")
    records = gate["records"]
    first = records[0]["plan"]
    expected = {(record["plan"]["seed"], record["plan"]["variant"], panel, depth)
                for record in records for panel in ("test_base", "test_long")
                for depth in record["plan"]["diagnostic_depths"]}
    observed, hashes = {}, {}
    for row in result["scores"]:
        key = (row["seed"], row["variant"], row["panel"], row["depth"])
        if key in observed or key not in expected:
            raise ValueError("Unexpected or duplicate scored endpoint")
        path = folder / (row["key"] + ".npz")
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Invalid endpoint path")
        if sha256_file(path) != row["artifact_sha256"]:
            raise ValueError("Scored artifact bytes changed")
        with np.load(path, allow_pickle=False) as archive:
            logits, prediction, target, nll = [archive[name] for name in ("logits", "prediction", "target", "nll")]
            if (logits.ndim != 2 or prediction.shape != target.shape or nll.shape != target.shape
                    or len(logits) != len(target) or len(target) < 1
                    or not np.isfinite(logits).all() or not np.isfinite(nll).all()
                    or not np.array_equal(prediction, logits.argmax(1))):
                raise ValueError("Invalid saved endpoint arrays")
            correct = int(np.sum(prediction == target))
            if correct != row["correct"] or len(target) != row["n"]:
                raise ValueError("Saved integer metric differs")
            shifted = logits - logits.max(axis=1, keepdims=True)
            calculated_nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(target)), target]
            if not np.allclose(nll, calculated_nll, rtol=1e-12, atol=1e-12):
                raise ValueError("Saved NLL differs from full logits")
        observed[key] = row
        hashes[path.name] = sha256_file(path)
    if set(observed) != expected:
        raise ValueError("Missing registered scored endpoint")
    differences, competence = [], []
    for seed in first["cohort_seeds"]:
        candidate = observed[(seed, "contractive", first["primary_panel"], first["primary_depth"])]
        control = observed[(seed, "unconstrained", first["primary_panel"], first["primary_depth"])]
        if candidate["n"] != control["n"]:
            raise ValueError("Unpaired primary population")
        differences.append(100*(candidate["correct"]-control["correct"])/candidate["n"])
        competence.append(candidate["correct"]/candidate["n"] >= 0.9)
    interval = paired_interval(differences) if len(differences) >= 2 else None
    return dict(status="verified_complete_reference_prediction_recount", rows=len(observed),
                primary_difference_pp=interval, all_candidate_base_competence_ge_90=all(competence),
                endpoint_sha256=hashes, gate_sha256=sha256_file(folder / "cohort_gate.json"),
                source_result_sha256=sha256_file(folder / "result.json"),
                scope="New development recipe; shared TEST population; no external replication",
                mathematical_theorem_empirically_improved_neuropixel=False,
                peer_review_completed=False)


def main() -> None:
    """Preserve the recount separately from original scoring artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.folder)
    write_json(args.output, result, exclusive=True)
    print(json.dumps({k: result[k] for k in ("status", "rows", "primary_difference_pp", "all_candidate_base_competence_ge_90")}, indent=2))


if __name__ == "__main__":
    main()
