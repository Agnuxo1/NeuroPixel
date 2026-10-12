"""Validate energy cohorts and recover only absent registered measurement keys."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

from .common import read_json, write_json, identity, sha256_file, verify_files


def registered_keys(plan: dict) -> set[tuple[str, int, str]]:
    """Derive the complete energy endpoint from its frozen plan."""
    names = [item["name"] for item in plan["cases"]]
    methods = plan["methods"]
    blocks = plan["energy_technical_blocks"]
    if len(set(names)) != len(names) or len(set(methods)) != len(methods) or blocks < 1:
        raise ValueError("Invalid energy registration")
    return {(case, block, method) for case in names for block in range(blocks) for method in methods}


def row_key(row: dict) -> tuple[str, int, str]:
    """Identify one technical measurement without using its numeric result."""
    return row["case"], row["block"], row["method"]


def admit_rows(rows: list[dict], plan: dict) -> dict[tuple[str, int, str], dict]:
    """Check identity, registered scope and the original physical admission guards."""
    expected = registered_keys(plan)
    admitted = {}
    for row in rows:
        key = row_key(row)
        if key not in expected or key in admitted:
            raise ValueError("Unexpected or repeated energy key: " + repr(key))
        numbers = [row[name] for name in (
            "active_seconds", "gpu_gross_energy_j", "gpu_gross_j_per_call",
            "idle_before_w", "idle_after_w", "baseline_adjusted_j_per_call_low",
            "baseline_adjusted_j_per_call_high", "available_ram_gib",
            "temperature_start_c", "temperature_end_c")]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("Nonfinite energy measurement")
        count = row["inference_calls"]
        if (not isinstance(count, int) or count < 1
                or row["active_seconds"] < plan["energy_active_seconds"]
                or row["gpu_gross_energy_j"] <= 0
                or row["available_ram_gib"] < plan["resources"]["actual_min_ram_gib"]
                or max(row["temperature_start_c"], row["temperature_end_c"]) >= 83):
            raise ValueError("Original energy guard failed: " + repr(key))
        gross, duration = row["gpu_gross_energy_j"], row["active_seconds"]
        low = (gross - max(row["idle_before_w"], row["idle_after_w"]) * duration) / count
        high = (gross - min(row["idle_before_w"], row["idle_after_w"]) * duration) / count
        for observed, calculated in (
            (row["gpu_gross_j_per_call"], gross / count),
            (row["baseline_adjusted_j_per_call_low"], low),
            (row["baseline_adjusted_j_per_call_high"], high)):
            if not math.isclose(observed, calculated, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Energy arithmetic differs: " + repr(key))
        # Negative baseline-adjusted estimates are retained, never clipped to zero.
        admitted[key] = row
    return admitted


def merge_receipts(original: dict, recovery: dict | None, plan: dict) -> dict:
    """Preserve original rows and reject overwritten or inconsistent measurements."""
    if len(original["checks"]) != 77 or not all(c["passed"] for c in original["checks"]):
        raise ValueError("Original parity admission missing")
    initial = admit_rows(original["rows"], plan)
    combined = dict(initial)
    if recovery is not None:
        if recovery["plan_sha256"] != original["plan_sha256"]:
            raise ValueError("Recovery plan mismatch")
        if len(recovery["checks"]) != 77 or not all(c["passed"] for c in recovery["checks"]):
            raise ValueError("Recovery parity admission missing")
        later = admit_rows(recovery["rows"], plan)
        for key, row in later.items():
            if key in combined and identity(row) != identity(combined[key]):
                raise ValueError("A completed measurement changed: " + repr(key))
            combined[key] = row
    missing = registered_keys(plan) - set(combined)
    return dict(rows=[combined[key] for key in sorted(combined)],
                original_rows=len(initial), unique_rows=len(combined),
                expected_rows=len(registered_keys(plan)),
                missing=[list(key) for key in sorted(missing)],
                complete=not missing)


def summarize(original: dict, recovery: dict | None, plan: dict) -> dict:
    """Recount a complete admitted board-energy cohort; do not claim wall energy."""
    import statistics
    merged = merge_receipts(original, recovery, plan)
    if not merged["complete"]:
        raise ValueError("Incomplete energy cohort: " + repr(merged["missing"]))
    table = []
    for case in plan["cases"]:
        for method in plan["methods"]:
            rows = [r for r in merged["rows"] if r["case"] == case["name"] and r["method"] == method]
            values = [r["gpu_gross_j_per_call"] for r in rows]
            table.append(dict(case=case["name"], method=method,
                              median_j=statistics.median(values), minimum_j=min(values), maximum_j=max(values),
                              baseline_low_median_j=statistics.median(r["baseline_adjusted_j_per_call_low"] for r in rows),
                              baseline_high_median_j=statistics.median(r["baseline_adjusted_j_per_call_high"] for r in rows),
                              calls=sum(r["inference_calls"] for r in rows)))
    return dict(status="verified_complete_registered_board_energy", table=table,
                unique_rows=merged["unique_rows"], original_rows=merged["original_rows"],
                expected_rows=merged["expected_rows"], technical_blocks_not_independent_replications=True,
                interruption_separates_sessions=True, wall_energy_measured=False)


def recover(root: Path, original_path: Path, output: Path) -> None:
    """Execute the immutable measured loop, seeding completed rows and skipping keys.

    Each invocation uses a fresh output directory. A failed recovery can itself be
    used as the next original receipt because it already retains all prior rows.
    This avoids modifying the frozen benchmark's exclusive-directory behavior.
    """
    root, original_path, output = root.resolve(), original_path.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a new recovery directory; completed evidence is immutable")
    plan_path = root / "docs/research/RENDER11_energy_plan.json"
    plan, original = read_json(plan_path), read_json(original_path)
    if sha256_file(plan_path) != original["plan_sha256"]:
        raise ValueError("Original plan identity differs")
    merged = merge_receipts(original, None, plan)
    verify_files(root, plan["source_sha256"])
    verify_files(root, plan["input_sha256"])
    if merged["complete"]:
        raise ValueError("Cohort is already complete; summarize it without remeasurement")
    original_digest = sha256_file(original_path)
    source_path = root / "scripts/measure_RENDER11_energy.py"
    specification = importlib.util.spec_from_file_location("registered_render11", source_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load registered energy module")
    frozen = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(frozen)
    loop = frozen.ENERGY_LOOP
    case_marker = "        for number,case in enumerate(cases):\n"
    method_marker = "                for method in order:\n"
    if loop.count(case_marker) != 1 or loop.count(method_marker) != 1:
        raise ValueError("Registered loop interface changed")
    seed_line = "        rows=json.loads(Path(" + repr(str(original_path)) + ").read_bytes())['rows']\n"
    loop = loop.replace(case_marker, seed_line + case_marker)
    skip_line = ('                    if any(r["case"]==case["spec"]["name"] and '
                 'r["block"]==block and r["method"]==method for r in rows):continue\n')
    frozen.ENERGY_LOOP = loop.replace(method_marker, method_marker + skip_line)
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(source_path), "--output", str(output)]
        frozen.main()
    finally:
        sys.argv = previous_argv
        if sha256_file(original_path) != original_digest:
            raise RuntimeError("Original receipt bytes were altered")
        if (output / "receipt.json").is_file():
            recovered = read_json(output / "receipt.json")
            admission = merge_receipts(original, recovered, plan)
            write_json(output / "recovery_admission.json", dict(
                **{k: v for k, v in admission.items() if k != "rows"},
                original_receipt_sha256=original_digest,
                recovered_receipt_sha256=sha256_file(output / "receipt.json"),
                remeasured_completed_keys=0,
                scope="Operational recovery, no new scientific replication"))
            if admission["complete"]:
                write_json(output / "independent_summary.json", summarize(original, recovered, plan))


def main() -> None:
    """Inspect without GPU, recover absent keys, or summarize an admitted cohort."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "recover", "summarize"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = read_json(args.root / "docs/research/RENDER11_energy_plan.json")
    original = read_json(args.original)
    recovered = read_json(args.recovery) if args.recovery else None
    if args.action == "recover":
        if args.output is None or args.recovery is not None:
            parser.error("recover requires --output and uses --original as its last partial receipt")
        recover(args.root, args.original, args.output)
        return
    result = (summarize(original, recovered, plan) if args.action == "summarize"
              else {k: v for k, v in merge_receipts(original, recovered, plan).items() if k != "rows"})
    if args.output:
        write_json(args.output, result, exclusive=True)
    import json
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
