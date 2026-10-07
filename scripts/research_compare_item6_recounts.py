"""Compare completed item-6 recounts exactly, excluding declared run metadata.

This post-freeze receipt compares saved reports, not models or predictions. It
does not round numbers or silently accept a numerical discrepancy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform


SOURCE = "08d0d52edd05da6835e71479f3ba4399fcbeabae"
PANELS = {
    "core": ("core_analysis.json", {"analyzed_at_utc", "numpy_version"}),
    "growth": ("06_growth_analysis.json", {"audited_at_utc", "numpy_version"}),
}


def read_report(path, panel):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    data = path.read_bytes()
    obj = json.loads(data, parse_constant=reject, object_pairs_hook=unique)
    if not (obj.get("item") == 6 and obj.get("panel") == panel
            and obj.get("status") == "verified" and obj.get("issues") == []
            and obj.get("source", {}).get("git_commit") == SOURCE):
        raise ValueError(f"Report identity or verification differs: {path}")
    return obj, {"path": str(path), "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}


def compare(left, right, path, differences, counts):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append({"path": path, "kind": "keys",
                                "root_only": sorted(set(left) - set(right)),
                                "worker_only": sorted(set(right) - set(left))})
        for key in sorted(set(left) & set(right)):
            compare(left[key], right[key], path + [key], differences, counts)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append({"path": path, "kind": "length",
                                "root": len(left), "worker": len(right)})
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, path + [index], differences, counts)
    else:
        counts["leaves_compared"] += 1
        numeric = type(left) in (int, float) and type(right) in (int, float)
        counts["numeric_leaves_compared"] += int(numeric)
        if left != right or (not numeric and type(left) is not type(right)):
            entry = {"path": path, "kind": "value", "root": left, "worker": right}
            if numeric:
                entry["absolute_difference"] = abs(left - right)
            differences.append(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-review", required=True, type=Path)
    parser.add_argument("--worker-reports", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an audit receipt")
    rows = []
    for panel, (filename, excluded) in PANELS.items():
        left, left_meta = read_report(args.root_review / panel / filename, panel)
        right, right_meta = read_report(args.worker_reports / panel / filename, panel)
        metadata = {key: {"root": left[key], "worker": right[key]} for key in sorted(excluded)}
        for key in excluded:
            del left[key]
            del right[key]
        differences = []
        counts = {"leaves_compared": 0, "numeric_leaves_compared": 0}
        compare(left, right, [], differences, counts)
        rows.append({"panel": panel, "root_report": left_meta, "worker_report": right_meta,
                     "excluded_top_level_metadata": metadata, **counts,
                     "differences": differences, "status": "exact_match" if not differences else "different"})
    report = {"schema_version": 1, "item": 6, "kind": "post-freeze report comparison",
              "at_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": SOURCE, "python": platform.python_version(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "comparison": "Exact decoded values; no rounding or numerical tolerance. Only the named top-level audit timestamp and NumPy version are excluded and retained explicitly.",
              "scope": "Same frozen analyzer run on the same saved inputs in two environments. This is not a second scientific replication or an independent analyzer implementation.",
              "panels": rows,
              "status": "exact_match" if all(row["status"] == "exact_match" for row in rows) else "different"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": report["status"], "differences": {r["panel"]: len(r["differences"]) for r in rows}}))
    return 0 if report["status"] == "exact_match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
