"""Read-only release audit: actual cohort counts, reported values and missing evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .common import read_json, write_json, sha256_file
from .energy import merge_receipts
from .statistics import paired_interval


def audit(root: Path) -> dict:
    """Recount current evidence; fail closed when closure claims exceed the files."""
    root = root.resolve()
    issues, evidence = [], {}
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    energy_plan_path = root / "docs/research/RENDER11_energy_plan.json"
    original_path = root / "results/research/RENDER11_energy/original_20261009A/receipt.json"
    plan, original = read_json(energy_plan_path), read_json(original_path)
    if sha256_file(energy_plan_path) != original["plan_sha256"]:
        issues.append("Energy plan bytes do not match the original receipt")
    recovery_path = root / "results/research/RENDER11_energy/recovered_20261009B/receipt.json"
    recovery = read_json(recovery_path) if recovery_path.is_file() else None
    try:
        merged = merge_receipts(original, recovery, plan)
        evidence["energy"] = {k: v for k, v in merged.items() if k != "rows"}
        if not merged["complete"]:
            issues.append("Published energy evidence is incomplete")
    except (ValueError, KeyError) as error:
        issues.append("Energy admission failed: " + str(error))
    readme = (root / "README.md").read_text(encoding="utf-8")
    energy_report = (root / "docs/research/RENDER11_results.md").read_text(encoding="utf-8")
    manuscript = (root / "docs/papers/NeuroPixel_manuscript_20261009.md").read_text(encoding="utf-8")
    if "105 rows" in readme and (not recovery_path.is_file() or not evidence.get("energy", {}).get("complete")):
        issues.append("README announces 105 energy rows without an admitted complete published receipt")
    if ("100 of 105" in energy_report or "100/105" in manuscript) and len(original["rows"]) != 100:
        issues.append("Energy narrative says 100 original rows but the stored receipt has " + str(len(original["rows"])))
    if "timing cohort is incomplete" in manuscript and "245 registered rows" in manuscript:
        issues.append("Manuscript contains incompatible incomplete/complete RENDER10 descriptions")
    recovered_timing = root / "results/research/RENDER10_benchmark/recovered_20261009B/independent_summary.json"
    if recovered_timing.is_file():
        timing = read_json(recovered_timing)
        if timing["rows"] != 245 or timing["gates"] != 77 or timing["old_rows_repeated"] != 0:
            issues.append("RENDER10 registered completion counts differ")
        evidence["render10"] = {k: timing[k] for k in ("status", "rows", "gates", "original_rows_preserved", "old_rows_repeated")}
    vis_path = root / "results/research/SPRINT_review/VIS07_summary.json"
    visual = read_json(vis_path)
    rows = [row for row in visual["rows"] if row["condition"] == "base"]
    seeds = sorted({row["seed"] for row in rows})
    differences = []
    for seed in seeds:
        nca = [r for r in rows if r["seed"] == seed and r["family"] == "nca"]
        cnn = [r for r in rows if r["seed"] == seed and r["family"] == "cnn"]
        if len(nca) != 1 or len(cnn) != 1 or nca[0]["n"] != cnn[0]["n"]:
            raise ValueError("Unpaired VIS07 saved count")
        differences.append(100 * (nca[0]["correct"] - cnn[0]["correct"]) / nca[0]["n"])
    interval = paired_interval(differences)
    evidence["vis07_recount"] = interval
    if abs(interval["mean"] - visual["primary"]["difference_pp"]) > 1e-10:
        issues.append("VIS07 saved paired effect differs from its integer recount")
    evidence["source_hashes"] = {str(path.relative_to(root)): sha256_file(path)
                                  for path in (energy_plan_path, original_path, vis_path)}
    return dict(status="issues_found" if issues else "bounded_documentation_checks_passed",
                commit=commit, issues=issues, evidence=evidence,
                exhaustive_publication_manifest_verified=False,
                neural_replay_executed=False, physical_gpu_remeasured=False,
                external_replication=False)


def main() -> None:
    """Write a concrete review receipt and use a nonzero status for unresolved issues."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    write_json(args.output, result, exclusive=True)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["issues"] else 0)


if __name__ == "__main__":
    main()
