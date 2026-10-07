"""Audit only the final item-6 archive's bytes, source, and operational receipts.

Post-freeze, standard-library-only check. Do not run on an interim snapshot.
Scientific artifact contents are hashed, not interpreted; no model, numerical
runtime, training data, or outcome analyzer is imported or executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile


SOURCE_COMMIT = "08d0d52edd05da6835e71479f3ba4399fcbeabae"
PLAN_SHA256 = "20f4e9b4385c95c5f206ac26702bca96ac4232930ea343fd718c4de5bec6048a"
PLAN_PATH = "docs/research/06_execution_plan.json"
STEP_NAMES = ("source_environment", "tests", "core", "growth", "core_recount", "growth_recount")
POLICY = {"cpu_threads": 2, "interop_threads": 1, "minimum_ram_gib": 8,
          "maximum_seconds": 18000, "archive_interval_seconds": 300,
          "supervisor_interval_seconds": 1, "git_timeout_seconds": 120,
          "final_archive_grace_seconds": 180}
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_stream(stream):
    digest, size = hashlib.sha256(), 0
    while block := stream.read(1024 * 1024):
        digest.update(block)
        size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def digest_file(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), parse_constant=reject, object_pairs_hook=unique)


def relative_name(name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name, "invalid artifact path")
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and path.as_posix() == name
            and not any(part in (".", "..") for part in path.parts), f"noncanonical artifact path: {name}")
    return path


def artifact_path(root, name):
    relative = relative_name(name)
    path = root.joinpath(*relative.parts)
    require(not any(parent.is_symlink() for parent in (path, *path.parents) if parent != root and root in parent.parents),
            f"archive symlink is not permitted: {name}")
    require(path.is_relative_to(root) and path.is_file(), f"missing regular artifact: {name}")
    require(stat.S_ISREG(path.stat().st_mode), f"nonregular artifact: {name}")
    return path


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, "timestamp must be UTC")
    return parsed


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "expected a finite numerical value")
    return value


def readonly_git(source, *args):
    # Disabling optional locks prevents status from refreshing the source index.
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=source,
                                   env=env, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   timeout=30)


def check_archive_files(archive, manifest, receipt):
    listed = manifest["files"]
    require(isinstance(listed, dict) and listed and "archive_manifest.json" not in listed, "invalid manifest file map")
    actual = set()
    for base, directories, filenames in os.walk(archive, followlinks=False):
        for name in directories + filenames:
            path = Path(base) / name
            require(not path.is_symlink(), f"unlisted or prohibited symlink: {path.relative_to(archive)}")
        actual.update((Path(base) / name).relative_to(archive).as_posix() for name in filenames)
    expected = set(listed) | {"archive_manifest.json"}
    receipt["observed"]["archive_inventory"] = {
        "listed_files": len(listed), "actual_files_excluding_manifest": len(actual - {"archive_manifest.json"}),
        "unlisted_files": sorted(actual - expected), "missing_files": sorted(expected - actual),
    }
    for name, record in sorted(listed.items()):
        try:
            require(type(record["bytes"]) is int and record["bytes"] >= 0 and HEX.fullmatch(record["sha256"]),
                    f"invalid manifest descriptor: {name}")
            observed = digest_file(artifact_path(archive, name))
            receipt["archive_files"][name] = observed
            require(observed == {"bytes": record["bytes"], "sha256": record["sha256"]}, f"length/hash differs: {name}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            receipt["issues"].append(f"archive file {name}: {error}")
    require(actual == expected, "archive has missing or unlisted files")


def check_source_zip(archive, source, receipt):
    head = readonly_git(source, "rev-parse", "HEAD").decode().strip()
    clean = not readonly_git(source, "status", "--porcelain", "--untracked-files=no").strip()
    receipt["observed"]["source_checkout"] = {"head": head, "tracked_worktree_and_index_clean": clean}
    require(head == SOURCE_COMMIT and clean, "source checkout must be the clean frozen 08d0 commit")
    tracked = {}
    for entry in readonly_git(source, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, _, stage = header.decode("ascii").split()
        name = raw_name.decode("utf-8")
        relative_name(name)
        require(stage == "0" and mode in ("100644", "100755", "120000") and name not in tracked,
                f"unsupported or duplicated source entry: {name}")
        tracked[name] = mode
    require(tracked, "frozen source has no tracked files")
    plan_file = source / PLAN_PATH
    plan_hash = digest_file(plan_file)
    receipt["input_hashes"]["frozen_source_plan"] = plan_hash
    require(plan_hash["sha256"] == PLAN_SHA256, "source plan hash differs from the frozen plan")
    plan = read_json(plan_file)
    require(plan["item"] == 6 and plan["status"] == "frozen", "source plan is not frozen item 6")
    receipt["observed"]["frozen_plan_limits"] = plan["resources"]
    zip_path = artifact_path(archive, "source_commit.zip")
    receipt["input_hashes"]["source_commit.zip"] = digest_file(zip_path)
    with zipfile.ZipFile(zip_path) as bundle:
        infos = bundle.infolist()
        require(len({info.filename for info in infos}) == len(infos), "duplicate ZIP entries")
        files = {info.filename: info for info in infos if not info.is_dir()}
        for info in infos:
            relative_name(info.filename[:-1] if info.is_dir() else info.filename)
        require(set(files) == set(tracked), "source ZIP file inventory differs from tracked frozen source")
        require(bundle.comment == SOURCE_COMMIT.encode("ascii"), "source ZIP commit comment differs")
        for name, mode in sorted(tracked.items()):
            path = source / name
            if mode == "120000":
                require(path.is_symlink(), f"source symlink mode differs: {name}")
                data = os.fsencode(os.readlink(path))
                original = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            else:
                require(path.is_file() and not path.is_symlink(), f"source file mode differs: {name}")
                original = digest_file(path)
            with bundle.open(files[name]) as stream:
                archived = digest_stream(stream)
            receipt["source_zip_files"][name] = {"source_mode": mode, "source": original, "zip": archived}
            if original != archived:
                receipt["issues"].append(f"source ZIP bytes differ from clean frozen source: {name}")
    require(readonly_git(source, "rev-parse", "HEAD").decode().strip() == head
            and not readonly_git(source, "status", "--porcelain", "--untracked-files=no").strip(),
            "source checkout changed during verification")
    receipt["observed"]["source_zip_tracked_files"] = len(tracked)


def check_worker(archive, manifest, receipt):
    path = artifact_path(archive, "worker_status.json")
    receipt["input_hashes"]["worker_status.json"] = digest_file(path)
    worker = read_json(path)
    require(worker["item"] == 6 and worker["status"] == "completed", "worker did not complete item 6")
    require(worker["source"]["git_commit"] == SOURCE_COMMIT and worker["execution_plan_sha256"] == PLAN_SHA256,
            "worker source commit or plan hash differs")
    require(worker["run_key"] == manifest["run_key"] and worker["paid_compute"] is False,
            "worker attempt or paid-compute declaration differs")
    require(worker["archive_branch"] == "research/scientific-validation-2026-10-07-cloud-results", "archive branch differs")
    require(worker["resource_policy"] == POLICY, "worker configured limits differ from the frozen limits")
    require(not any(key in worker for key in ("error", "shutdown_error", "supervision_failure")), "completed worker contains a failure")
    started, ended = utc(worker["started_at_utc"]), utc(worker["completed_at_utc"])
    require(started <= ended <= utc(manifest["at_utc"]), "worker/final archive timestamps are inconsistent")
    wall = finite(worker["wall_seconds"])
    require(0 <= wall <= POLICY["maximum_seconds"], "recorded worker duration exceeds the declared budget")
    steps = worker["steps"]
    require([step["name"] for step in steps] == list(STEP_NAMES), "worker must contain exactly the six declared stages in order")
    previous = started
    observed_steps = []
    for step in steps:
        require(step["status"] == "completed" and type(step["returncode"]) is int and step["returncode"] == 0,
                f"stage did not succeed: {step['name']}")
        begin, end = utc(step["started_at_utc"]), utc(step["completed_at_utc"])
        require(previous <= begin <= end <= ended, f"stage timestamps overlap or exceed worker: {step['name']}")
        previous = end
        name = f"logs/{step['name']}.log"
        require(step["log"] == name and HEX.fullmatch(step["log_sha256"]), "stage log descriptor differs")
        actual = digest_file(artifact_path(archive, name))
        require(actual["sha256"] == step["log_sha256"] == manifest["files"][name]["sha256"], f"stage log hash differs: {name}")
        observed_steps.append({"name": step["name"], "returncode": step["returncode"], "log": name,
                               **actual, "started_at_utc": step["started_at_utc"], "completed_at_utc": step["completed_at_utc"]})
    samples = [worker["initial_resources"], *worker["resources"]]
    require(len(samples) >= 2, "recorded resource samples are missing")
    times, available = [], []
    for sample in samples:
        available.append(finite(sample["available_ram_gib"]))
        times.append(utc(sample["at_utc"]))
        require(sample["thread_limit"] == 2 and type(sample["cpu_count_logical"]) is int
                and sample["cpu_count_logical"] >= 2, "resource sample CPU declaration differs")
    require(times == sorted(times) and times[-1] <= ended, "resource sample timestamps differ")
    receipt["observed"]["worker"] = {"status": worker["status"], "run_key": worker["run_key"],
        "wall_seconds_before_final_archive": wall, "started_at_utc": worker["started_at_utc"],
        "completed_at_utc": worker["completed_at_utc"], "resource_policy": worker["resource_policy"], "steps": observed_steps}
    receipt["observed"]["recorded_resources"] = {"worker_samples": len(samples),
        "minimum_recorded_available_ram_gib": min(available), "maximum_recorded_available_ram_gib": max(available),
        "logical_cpu_counts": sorted({sample["cpu_count_logical"] for sample in samples}),
        "first_sample_utc": samples[0]["at_utc"], "last_sample_utc": samples[-1]["at_utc"],
        "scope": "Worker initial admission and persisted heartbeats only; not continuous RAM, peak RSS, or observed active CPU usage."}
    require(min(available) >= 8, "a recorded available-RAM sample is below 8 GiB")


def check_tests(archive, receipt):
    path = artifact_path(archive, "verification/tests.xml")
    receipt["input_hashes"]["verification/tests.xml"] = digest_file(path)
    root = ET.parse(path).getroot()
    require(root.tag in ("testsuites", "testsuite"), "unexpected JUnit root")
    cases = list(root.iter("testcase"))
    counts = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        flags = [key for key in ("failures", "errors", "skipped")
                 if case.find({"failures": "failure", "errors": "error", "skipped": "skipped"}[key]) is not None]
        require(len(flags) <= 1, "ambiguous JUnit testcase state")
        for key in flags:
            counts[key] += 1
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    receipt["observed"]["tests"] = counts
    require(counts == {"tests": 108, "failures": 0, "errors": 0, "skipped": 1, "passed": 107},
            "JUnit testcase counts differ from the frozen 107-pass/one-skip inventory")
    for suite in root.iter("testsuite"):
        require(not suite.findall("testsuite"), "nested suites are not produced by this frozen pytest command")
        local = suite.findall("testcase")
        for key, tag in (("tests", None), ("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            counted = len(local) if tag is None else sum(case.find(tag) is not None for case in local)
            require(int(suite.attrib[key]) == counted, f"JUnit suite {key} metadata differs from its cases")
    for key in ("tests", "failures", "errors", "skipped"):
        if key in root.attrib:
            require(int(root.attrib[key]) == counts[key], f"JUnit root {key} count differs")


def audit(archive, source):
    receipt = {"schema_version": 1, "item": 6, "audit_kind": "post-freeze final archive integrity audit",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "rejected", "issues": [],
        "expected_source_commit": SOURCE_COMMIT, "expected_plan_sha256": PLAN_SHA256,
        "archive_directory_name": archive.name, "input_hashes": {}, "archive_files": {}, "source_zip_files": {}, "observed": {},
        "auditor_sha256": digest_file(Path(__file__))["sha256"],
        "limits": [
            "Only a final archive is accepted. This is not a scientific prediction recount or a model rerun.",
            "All scientific files are read only as opaque bytes for hash/length verification; their outcomes are not interpreted.",
            "RAM minima concern saved worker samples only. The one-second watchdog does not persist its complete sample series.",
            "Configured thread/deadline limits and recorded stops can be checked; continuous active CPU usage, peak process RSS, and isolated gate-fit duration were not measured.",
            "Recorded worker wall time ends before final archival. This receipt does not independently measure the final archival grace or external GitHub job duration.",
            "Source Git operations are read-only with optional index locks disabled. No refs, tracked files, or archive artifacts are modified.",
        ]}
    try:
        path = artifact_path(archive, "archive_manifest.json")
        receipt["input_hashes"]["archive_manifest.json"] = digest_file(path)
        manifest = read_json(path)
        require(isinstance(manifest, dict), "manifest must be an object")
        receipt["observed"]["manifest"] = {key: manifest.get(key) for key in ("source_commit", "run_key", "snapshot_kind", "at_utc")}
        require(manifest["snapshot_kind"] == "final", "interim archive rejected before further artifact reads")
        require(manifest["schema_version"] == 1 and manifest["source_commit"] == SOURCE_COMMIT,
                "archive source/schema differs")
        require(re.fullmatch(r"\d+-\d+", manifest["run_key"]) and manifest["run_key"] == archive.name, "archive attempt identity differs")
        utc(manifest["at_utc"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        receipt["issues"].append(f"final manifest gate: {type(error).__name__}: {error}")
        return receipt
    for name, check in (
        ("archive inventory", lambda: check_archive_files(archive, manifest, receipt)),
        ("source ZIP", lambda: check_source_zip(archive, source, receipt)),
        ("worker completion/resources", lambda: check_worker(archive, manifest, receipt)),
        ("test XML", lambda: check_tests(archive, receipt)),
    ):
        try:
            check()
        except (OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile, ET.ParseError, subprocess.SubprocessError) as error:
            receipt["issues"].append(f"{name}: {type(error).__name__}: {error}")
    # Ensure the final snapshot's metadata was not replaced while inspecting it.
    try:
        if digest_file(archive / "archive_manifest.json") != receipt["input_hashes"]["archive_manifest.json"]:
            receipt["issues"].append("archive manifest changed during verification")
    except OSError as error:
        receipt["issues"].append(f"archive manifest disappeared during verification: {error}")
    for name, record in receipt["input_hashes"].items():
        if name in receipt["archive_files"] and record != receipt["archive_files"][name]:
            receipt["issues"].append(f"artifact changed between byte verification and receipt parsing: {name}")
    receipt["status"] = "verified" if not receipt["issues"] else "rejected"
    return receipt


def write_receipt(path, receipt):
    data = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication that refuses an existing destination.
    finally:
        os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Final owned attempt directory containing archive_manifest.json")
    parser.add_argument("--source-root", type=Path, required=True, help="Read-only clean frozen 08d0 source checkout")
    parser.add_argument("--output", type=Path, required=True, help="New receipt outside the archive and source checkout")
    args = parser.parse_args()
    archive, source, output = args.archive.resolve(), args.source_root.resolve(), args.output.resolve()
    try:
        require(not output.exists(), "refusing to overwrite the receipt")
        require(not output.is_relative_to(archive) and not output.is_relative_to(source), "receipt must be outside audited directories")
        receipt = audit(archive, source)
        write_receipt(output, receipt)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Archive verification could not finish: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": receipt["status"], "issues": receipt["issues"], "receipt": str(output)}))
    return 0 if receipt["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
