"""Independently verify item-9 Git archives, source bytes and execution receipts.

This standard-library audit never unpickles a checkpoint, imports a scientific
module, generates data, or runs a model. Statistical recount is a separate audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
import xml.etree.ElementTree as ET
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=root,
                                   stderr=subprocess.PIPE, timeout=60)


def tree_files(root, revision, prefix=None):
    command = ["ls-tree", "-r", "-z", revision]
    if prefix is not None:
        command += ["--", prefix]
    files = {}
    for row in git(root, *command).split(b"\0"):
        if not row:
            continue
        metadata, name = row.split(b"\t", 1)
        mode, kind, identity = metadata.decode().split()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError("unsupported source tree entry: " + name.decode())
        files[name.decode()] = {"mode": mode, "git_blob": identity}
    return files


def read_json(path):
    return json.loads(path.read_bytes())


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-commit", required=True)
    parser.add_argument("--phase", choices=("preflight", "study"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous audit attempts")
    raw, source = args.input.resolve(), args.source_root.resolve()
    checks, issues = {}, []
    report = {"schema_version": 1, "item": 9, "phase": args.phase,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": args.source_commit, "archive_commit": args.archive_commit,
              "auditor_sha256": digest(Path(__file__).read_bytes()),
              "limitations": ["Internal artifact audit, not external replication or custody.",
                              "No checkpoint unpickling or model execution.",
                              "Resource receipts are sampled observations, not peak RSS, energy or continuous CPU utilization.",
                              "Scientific data and saved-prediction recount is separate."]}

    def check(name, condition):
        if name in checks:
            raise ValueError("duplicate check identifier: " + name)
        checks[name] = bool(condition)
        if not condition:
            issues.append(name)

    try:
        check("source_head", git(source, "rev-parse", "HEAD").decode().strip() == args.source_commit)
        check("raw_head", git(raw, "rev-parse", "HEAD").decode().strip() == args.archive_commit)
        check("source_clean", not git(source, "status", "--porcelain"))
        raw_git_root = Path(git(raw, "rev-parse", "--show-toplevel").decode().strip())
        raw_prefix = raw.relative_to(raw_git_root).as_posix()
        raw_tree = tree_files(raw_git_root, args.archive_commit, raw_prefix)
        check("raw_clean", not git(raw, "status", "--porcelain", "--", "."))
        for name, descriptor in raw_tree.items():
            path = raw_git_root / name
            check("raw_git_blob:" + name, path.is_file() and not path.is_symlink()
                  and git_blob(path.read_bytes()) == descriptor["git_blob"])
        manifest_bytes = (raw / "archive_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        check("final_archive", manifest["schema_version"] == 1 and manifest["item"] == 9
              and manifest["final"] is True and manifest["snapshot_kind"] == "final"
              and manifest["attempt_status"] == "completed")
        check("manifest_source", manifest["source_commit"] == args.source_commit)
        check("manifest_run_key", manifest["run_key"] == raw.name)
        files = manifest["files"]
        check("manifest_mapping", isinstance(files, dict) and len(files) > 0)
        actual = {p.relative_to(raw).as_posix() for p in raw.rglob("*") if p.is_file()}
        check("all_raw_files_bound", actual == set(files) | {"archive_manifest.json"})
        check("raw_git_complete", bool(raw_tree) and set(raw_tree)
              == {raw_prefix + "/" + name for name in set(files) | {"archive_manifest.json"}})
        for name, descriptor in files.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
                raise ValueError("invalid manifest path: " + name)
            path = raw / relative
            data = path.read_bytes()
            check("manifest_file:" + name, not path.is_symlink()
                  and len(data) == descriptor["bytes"] and digest(data) == descriptor["sha256"])
        plan_bytes = (raw / "execution_plan.json").read_bytes()
        plan = json.loads(plan_bytes)
        plan_name = f"docs/research/09_{args.phase}_plan.json"
        check("plan_exact_source_copy", plan_bytes == (source / plan_name).read_bytes())
        check("plan_phase", plan["phase"] == args.phase and plan["status"] == "frozen"
              and plan["item"] == 9 and plan["schema_version"] == 1)
        recipe_bytes = (raw / "experiment_recipe.json").read_bytes()
        recipe = json.loads(recipe_bytes)
        check("recipe_identity", digest(recipe_bytes) == plan["recipe_sha256"]
              and recipe_bytes == (source / plan["recipe_path"]).read_bytes())
        source_tree = tree_files(source, args.source_commit)
        source_inventory = []
        with zipfile.ZipFile(raw / "source.zip") as archive:
            entries = [row for row in archive.infolist() if not row.is_dir()]
            names = [row.filename for row in entries]
            check("source_zip_complete", len(names) == len(set(names)) and set(names) == set(source_tree))
            check("source_zip_crc", archive.testzip() is None)
            for name, descriptor in source_tree.items():
                data = archive.read(name)
                check("source_blob:" + name, git_blob(data) == descriptor["git_blob"])
                check("source_worktree:" + name, (source / name).read_bytes() == data)
                source_inventory.append({"path": name, "bytes": len(data), "sha256": digest(data),
                                         "git_blob": descriptor["git_blob"]})
        for name, expected in plan["implementation_sha256"].items():
            check("source_binding:" + name, name in source_tree and digest((source / name).read_bytes()) == expected)
        for phase in ("before", "after"):
            record = read_json(raw / f"source_{phase}.json")
            check("source_guard:" + phase, record["verified"] is True
                  and record["source_commit"] == args.source_commit
                  and record["plan_sha256"] == digest(plan_bytes)
                  and record["recipe_sha256"] == digest(recipe_bytes)
                  and record["implementation_sha256"] == plan["implementation_sha256"]
                  and record["unexpected_untracked_files"] == [] and record["tracked_diff"] == ""
                  and record["all_bound_files_tracked"] is True and record["plan_matches_committed_blob"] is True)
        worker = read_json(raw / "worker_status.json")
        check("worker_complete", worker["status"] == "completed" and worker["source_commit"] == args.source_commit
              and worker["phase"] == args.phase and worker["run_key"] == raw.name
              and worker["resource_policy"] == plan["resource_policy"]
              and worker["paid_compute"] is False and worker["gpu_requested"] is False)
        check("worker_budget", 0 < worker["worker_wall_seconds_before_archive"] <= worker["initial_budget_seconds"]
              <= plan["resource_policy"]["worker_seconds"])
        check("stage_inventory", [row["name"] for row in worker["stages"]] == plan["stages"])
        check("freeze_before_worker", instant(plan["freeze_utc"]) < instant(worker["started_at_utc"]))
        previous = instant(worker["started_at_utc"])
        resource_values = [worker[name]["available_ram_gib"] for name in ("initial_resources", "final_resources")]
        monitors = {}
        for stage in worker["stages"]:
            name = stage["name"]
            check("stage_status:" + name, stage == read_json(raw / f"{name}_status.json")
                  and stage["return_code"] == 0 and stage["status"] == "completed"
                  and "supervision_failure" not in stage and "error_type" not in stage)
            check("stage_log:" + name, digest((raw / f"{name}.log").read_bytes()) == stage["log_sha256"])
            check("stage_time:" + name, previous <= instant(stage["started_at_utc"])
                  <= instant(stage["completed_at_utc"])
                  and stage["wall_seconds_including_shutdown"] <= stage["execution_seconds_available_at_launch"] + 20)
            previous = instant(stage["completed_at_utc"])
            resource_values.append(stage["admission"]["available_ram_gib"])
            monitor = read_json(raw / f"{name}_resource_monitor.json")
            values = [row["available_ram_gib"] for row in monitor["samples"]]
            check("resource_monitor:" + name, bool(values) and monitor["failure"] is None
                  and monitor["interval_seconds"] == 1 and min(values) >= 8
                  and min(values) == monitor["minimum_sampled_available_ram_gib"])
            resource_values.extend(values)
            monitors[name] = {"samples": len(values), "minimum_sampled_available_ram_gib": min(values)}
            environment = read_json(raw / f"{name}_environment.json")
            check("environment_issues:" + name, environment["issues"] == [] and environment["skipped_nodeids"] == [])
            check("environment_boundaries:" + name, len(environment["records"]) == 2)
            for index, row in enumerate(environment["records"]):
                check(f"environment:{name}:{index}", row["versions"] == recipe["runtime"]["versions"]
                      and row["python"] == recipe["runtime"]["python"] and row["torch_cuda_version"] is None
                      and row["torch_threads"] == 2 and row["torch_interop_threads"] == 1
                      and all(value == "2" for value in row["thread_environment"].values()))
            if name == "contract_tests":
                collection = read_json(raw / "test_collection.json")
                ids = collection["nodeids"]
                check("test_collection", collection["count"] == collection["expected_count"] == plan["expected_test_count"]
                      == len(ids) == len(set(ids)) and collection["files"] == plan["test_inventory"])
                boundaries = environment["test_thread_boundaries"]
                check("test_threads", [row["nodeid"] for row in boundaries] == ids
                      and all(row["before"] == row["after"] == [2, 1] for row in boundaries))
                suites = ET.fromstring((raw / "tests.xml").read_bytes())
                cases = suites.findall(".//testcase")
                check("test_junit", len(cases) == len(ids) and not suites.findall(".//failure")
                      and not suites.findall(".//error") and not suites.findall(".//skipped"))
                check("test_unique", len({(row.get("classname"), row.get("name")) for row in cases}) == len(cases))
                expected_cases = []
                for nodeid in ids:
                    file_name, class_name, method_name = nodeid.split("::")
                    expected_cases.append((file_name.removesuffix(".py").replace("/", ".")
                                           + "." + class_name, method_name))
                check("test_junit_identity", [(row.get("classname"), row.get("name")) for row in cases] == expected_cases)
                report["collected_test_cases"] = len(ids)
                report["junit_suite_test_counter"] = sum(int(row.get("tests", "0")) for row in suites.findall("testsuite"))
        check("all_recorded_ram_above_floor", min(resource_values) >= 8)
        check("worker_time_order", previous <= instant(worker["completed_at_utc"]) <= instant(manifest["created_at_utc"]))
        if args.phase == "preflight":
            check("no_final_performance_artifacts", not (raw / "study/final_data").exists()
                  and not (raw / "study/final_access.json").exists() and not (raw / "study/final_predictions").exists())
        report.update(archive_manifest_sha256=digest(manifest_bytes), plan_sha256=digest(plan_bytes),
                      archive_files_excluding_manifest=len(files),
                      archive_bytes_excluding_manifest=sum(row["bytes"] for row in files.values()),
                      source_files=len(source_inventory), source_inventory=source_inventory,
                      worker_wall_seconds_before_archive=worker["worker_wall_seconds_before_archive"],
                      resource_monitors=monitors, sampled_worker_ram_min_gib=min(resource_values))
    except Exception as error:
        issues.append(type(error).__name__ + ": " + str(error))
        report["traceback"] = traceback.format_exc()
    report.update(status="verified" if not issues else "failed", checks=checks, issues=issues)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "issues": issues,
                      "output_sha256": digest(args.output.read_bytes())}))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
