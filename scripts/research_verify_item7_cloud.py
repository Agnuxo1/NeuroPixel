"""Verify the archived item-7 cloud files and source without executing models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

SOURCE = "0ed43bd5bb1d8bcb166b8e63e909195cbe69f769"
ARCHIVE = "f0aa1f17e08a495e92e7e8084003f357abfe9a8d"
PLAN_SHA = "5071ae0e70d47c37e7eaea0d3f0a2299b6e3d06bf6a68337201f67c6f6ca8542"
EXPORT_SHA = "1fc03501b21b2790109b1f66e64fa0dc76a1bce1fb11f6ccd7602bc38f28918d"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the existing audit")
    checks, issues = {}, []

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            issues.append(name)

    record = {"item": 7, "schema_version": 1,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "expected_source_commit": SOURCE, "expected_archive_commit": ARCHIVE,
              "auditor_sha256": sha(Path(__file__).read_bytes())}
    try:
        def git(root, *arguments):
            return subprocess.check_output(["git", "--no-optional-locks", *arguments], cwd=root,
                                           stderr=subprocess.PIPE, timeout=30).decode().strip()
        check("source_commit", git(args.source_root, "rev-parse", "HEAD") == SOURCE)
        check("archive_commit", git(args.input, "rev-parse", "HEAD") == ARCHIVE)
        check("source_clean", not git(args.source_root, "status", "--porcelain", "--untracked-files=no"))
        check("archive_clean", not git(args.input, "status", "--porcelain", "--untracked-files=no"))
        manifest_bytes = (args.input / "archive_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        check("manifest_source", manifest["source_commit"] == SOURCE)
        paths = [row["path"] for row in manifest["files"]]
        check("manifest_unique_paths", len(paths) == len(set(paths)))
        actual = {p.relative_to(args.input).as_posix() for p in args.input.rglob("*") if p.is_file()}
        check("complete_file_set", actual == set(paths) | {"archive_manifest.json"})
        for row in manifest["files"]:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("manifest path leaves archive")
            data = (args.input / relative).read_bytes()
            check("file:" + row["path"], len(data) == row["bytes"] and sha(data) == row["sha256"])
        plan_bytes = (args.input / "execution_plan.json").read_bytes()
        plan = json.loads(plan_bytes)
        check("plan_anchor", sha(plan_bytes) == PLAN_SHA)
        check("plan_matches_source", plan_bytes == (args.source_root / "docs/research/07_cloud_export_plan.json").read_bytes())
        source_paths = git(args.source_root, "ls-files", "-z").split("\0")
        source_paths = [p for p in source_paths if p]
        source_files = []
        with zipfile.ZipFile(args.input / "source.zip") as archive:
            names = [x.filename for x in archive.infolist() if not x.is_dir()]
            check("source_zip_file_set", len(names) == len(set(names)) and set(names) == set(source_paths))
            for name in sorted(source_paths):
                data = archive.read(name)
                local = (args.source_root / name).read_bytes()
                check("source:" + name, data == local)
                source_files.append({"path": name, "bytes": len(data), "sha256": sha(data)})
        validation = json.loads((args.input / "source_validation.json").read_text())
        check("source_validation", validation["status"] == "verified" and validation["source_commit"] == SOURCE
              and validation["plan_sha256"] == PLAN_SHA and validation["files_sha256"] == plan["implementation_sha256"])
        for name, expected in plan["implementation_sha256"].items():
            check("implementation:" + name, sha((args.source_root / name).read_bytes()) == expected)
        worker = json.loads((args.input / "worker_status.json").read_text())
        check("worker_complete", worker["status"] == "completed" and worker["source_commit"] == SOURCE)
        check("stage_inventory", [x["name"] for x in worker["stages"]] == ["tests", "memberships"])
        previous = worker["started_at_utc"]
        for stage in worker["stages"]:
            persisted = json.loads((args.input / (stage["name"] + "_status.json")).read_text())
            check("stage:" + stage["name"], persisted == stage and stage["return_code"] == 0
                  and "stop_reason" not in stage and 0 <= stage["wall_seconds"] <= 905)
            check("time:" + stage["name"], previous <= stage["started_at_utc"] <= stage["completed_at_utc"])
            previous = stage["completed_at_utc"]
        check("completion_time", previous <= worker["completed_at_utc"] <= manifest["created_at_utc"])
        suites = ET.fromstring((args.input / "tests.xml").read_bytes())
        cases = suites.findall(".//testcase")
        check("junit", len(cases) == 21 and not suites.findall(".//failure")
              and not suites.findall(".//error") and not suites.findall(".//skipped"))
        check("junit_unique_cases", len({(x.get("classname"), x.get("name")) for x in cases}) == 21)
        export_bytes = (args.input / "split_pools.json").read_bytes()
        export = json.loads(export_bytes)
        check("export_anchor", sha(export_bytes) == EXPORT_SHA)
        check("export_runtime", export["runtime"]["torch"] == "2.6.0+cpu"
              and export["runtime"]["python"] == "3.12.8" and export["runtime"]["device"] == "cpu"
              and export["runtime"]["torch_threads"] == export["runtime"]["torch_interop_threads"] == 1)
        check("export_scope", export["status"] == "membership_exported" and not export["task_sample_called"]
              and not export["models_imported"] and export["optimizer_updates"] == 0)
        check("export_seeds", [x["seed"] for x in export["seeds"]] == [0, 1, 2, 101, 202])
        worker_min = min(x["available_ram_gib"] for x in worker["resource_samples"])
        export_min = min(x["available_ram_gib"] for x in export["resource_samples"])
        check("worker_ram", worker_min >= 8 and worker_min == worker["minimum_sampled_available_ram_gib"])
        check("export_ram", export_min >= 8)
        record.update(archive_manifest_sha256=sha(manifest_bytes), files=len(paths),
                      archive_bytes_excluding_manifest=sum(x["bytes"] for x in manifest["files"]),
                      source_zip_files=source_files, tests_passed=len(cases),
                      worker_sample_count=len(worker["resource_samples"]), worker_minimum_sampled_ram_gib=worker_min,
                      exporter_sample_count=len(export["resource_samples"]), exporter_minimum_sampled_ram_gib=export_min,
                      worker_wall_seconds_before_archive=worker["worker_wall_seconds"],
                      limitation="Checksums/source and stage receipts verified; full set relationships are the separate split audit.")
    except Exception as exc:
        issues.append(type(exc).__name__ + ": " + str(exc))
    record.update(status="verified" if not issues else "failed", checks=checks, issues=issues)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "checks": len(checks), "issues": issues,
                      "output_sha256": sha(args.output.read_bytes())}))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
