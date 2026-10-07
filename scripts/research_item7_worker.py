"""Bounded CPU validation and membership export for scientific item 7.

No model training or final task/model evaluation is requested. Sampler tests use
small training/validation fixtures; the membership exporter samples no examples.
Source, tests (including failures), the membership export and resource receipts
are preserved on the existing isolated results branch with a normal push.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
RESULTS_BRANCH = "research/scientific-validation-2026-10-07-cloud-results"
PLAN_PATH = ROOT / "docs/research/07_cloud_export_plan.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(*args, cwd=ROOT, timeout=60):
    return subprocess.run(["git", "-c", "pack.threads=1", *args], cwd=cwd, timeout=timeout,
                          stdin=subprocess.DEVNULL, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def sample():
    return {"at_utc": now(), "available_ram_gib": psutil.virtual_memory().available / 2**30}


def stop_owned(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The child exited between poll and the signal.
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def run_stage(name, command, output, samples):
    before = sample()
    samples.append(before)
    if before["available_ram_gib"] < 8:
        raise RuntimeError("stage admission requires at least 8 GiB available RAM")
    record = {"name": name, "command": command, "started_at_utc": now()}
    started = time.monotonic()
    with (output / f"{name}.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                row = sample()
                samples.append(row)
                if row["available_ram_gib"] < 8 or time.monotonic() - started > 900:
                    record["stop_reason"] = "RAM floor or 900-second stage time limit"
                    stop_owned(child)
                    break
                time.sleep(1)
        finally:
            stop_owned(child)
    record.update(completed_at_utc=now(), return_code=child.returncode,
                  wall_seconds=time.monotonic() - started)
    save_new(output / f"{name}_status.json", record)
    return record


def archive(output, source_commit):
    entries = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size,
                "sha256": sha(p)} for p in sorted(output.rglob("*")) if p.is_file()]
    save_new(output / "archive_manifest.json", {"schema_version": 1, "item": 7,
             "created_at_utc": now(), "source_commit": source_commit, "files": entries})
    # Fetch in this worktree and resolve its FETCH_HEAD explicitly.
    git("fetch", "--no-tags", "origin", RESULTS_BRANCH)
    base = git("rev-parse", "FETCH_HEAD")
    worktree = Path(tempfile.mkdtemp(prefix="neuropixel-item7-archive-")) / "tree"
    git("worktree", "add", "--detach", str(worktree), base)
    relative = output.relative_to(ROOT)
    destination = worktree / relative
    if destination.exists():
        raise FileExistsError("this run/attempt already has archived evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output, destination)
    git("config", "user.name", "NeuroPixel scientific worker", cwd=worktree)
    git("config", "user.email", "neuropixel-worker@users.noreply.github.com", cwd=worktree)
    git("add", "--", relative.as_posix(), cwd=worktree)
    git("commit", "-m", f"research: archive item 7 CPU validation {output.name}", cwd=worktree)
    commit = git("rev-parse", "HEAD", cwd=worktree)
    git("push", "origin", f"HEAD:refs/heads/{RESULTS_BRANCH}", cwd=worktree)
    print(json.dumps({"event": "evidence_archived", "source_commit": source_commit,
                      "archive_commit": commit, "path": relative.as_posix()}), flush=True)


def main():
    run_key = os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    output = ROOT / "results/research/07_cloud_runs" / run_key
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_commit = git("rev-parse", "HEAD")
    samples, stages = [sample()], []
    status = {"schema_version": 1, "item": 7, "started_at_utc": now(),
              "source_commit": source_commit, "run_key": run_key,
              "model_training_requested": False, "final_predictions_requested": False,
              "development_sampler_fixtures_requested": True}
    error = None
    try:
        if samples[0]["available_ram_gib"] < 8:
            raise RuntimeError("initial memory admission failed")
        plan = json.loads(PLAN_PATH.read_text())
        if plan["item"] != 7 or plan["status"] != "frozen" or plan["torch"] != "2.6.0+cpu":
            raise ValueError("item-7 cloud plan is not the approved membership recipe")
        plan_relative = PLAN_PATH.relative_to(ROOT).as_posix()
        if git("hash-object", "--", plan_relative) != git("rev-parse", source_commit + ":" + plan_relative):
            raise RuntimeError("execution plan is not the exact committed source-archive version")
        actual = {p: sha(ROOT / p) for p in plan["implementation_sha256"]}
        if actual != plan["implementation_sha256"]:
            raise ValueError("implementation differs from the frozen cloud plan")
        if git("diff", "--name-only", "HEAD", "--", plan_relative, *actual):
            raise RuntimeError("tracked implementation is dirty before validation")
        save_new(output / "source_validation.json", {"status": "verified", "at_utc": now(),
                 "source_commit": source_commit, "plan_sha256": sha(PLAN_PATH), "files_sha256": actual})
        shutil.copyfile(PLAN_PATH, output / "execution_plan.json")
        # Preserve exact source independently of later source-branch edits.
        subprocess.run(["git", "archive", "--format=zip", "--output", str(output / "source.zip"),
                        source_commit], cwd=ROOT, stdin=subprocess.DEVNULL, check=True, timeout=60)
        commands = [
            ("tests", [sys.executable, "-m", "pytest", "-q", "tests/test_research_split_governance.py",
                       "tests/test_research_development_data.py", "--junitxml", str(output / "tests.xml")]),
            ("memberships", [sys.executable, "scripts/research_export_split_pools.py", "--output",
                             str(output / "split_pools.json"), "--source-root", str(ROOT)]),
        ]
        for name, command in commands:
            stages.append(run_stage(name, command, output, samples))
            if "stop_reason" in stages[-1]:
                raise RuntimeError(stages[-1]["stop_reason"])
        if any(stage["return_code"] != 0 for stage in stages):
            raise RuntimeError("at least one validation/export stage failed; preserve both results")
        if {p: sha(ROOT / p) for p in actual} != actual:
            raise RuntimeError("implementation changed during validation")
        status["status"] = "completed"
    except BaseException as exc:
        error = exc
        status.update(status="failed", error_type=type(exc).__name__, message=str(exc)[:500])
    finally:
        status.update(completed_at_utc=now(), worker_wall_seconds=time.monotonic() - started,
                      stages=stages, resource_samples=samples,
                      minimum_sampled_available_ram_gib=min(s["available_ram_gib"] for s in samples))
        save_new(output / "worker_status.json", status)
        archive(output, source_commit)
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
