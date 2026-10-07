"""Execute and preserve the frozen item-6 study on a standard public CPU runner.

Only this study's child processes and output directory are managed. Scientific
source HEAD stays unchanged; periodic archives use a separate detached worktree
and a dedicated results branch. No Actions artifact upload or dependency cache
is required. An incomplete attempt is preserved, never silently resumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BRANCH = "research/scientific-validation-2026-10-07-cloud"
RESULTS_BRANCH = SOURCE_BRANCH + "-results"
REPOSITORY = "Agnuxo1/NeuroPixel"
THREADS = 2
MINIMUM_RAM_GIB = 8.0
MAXIMUM_SECONDS = 18000
ARCHIVE_SECONDS = 300
HEARTBEAT_SECONDS = 60
PLAN = ROOT / "docs/research/06_execution_plan.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if path.read_bytes() != data:
        raise OSError("worker JSON readback failed")


def git(*arguments, cwd=ROOT):
    return subprocess.check_output(["git", "-c", "pack.threads=1", "-c", "index.threads=1",
                                    *map(str, arguments)], cwd=cwd,
                                   text=True, stderr=subprocess.STDOUT).strip()


def admission():
    import psutil

    available = psutil.virtual_memory().available / 2**30
    if available < MINIMUM_RAM_GIB:
        raise RuntimeError(f"available RAM {available:.3f} GiB is below the unchanged 8 GiB floor")
    return {"at_utc": utc_now(), "available_ram_gib": available,
            "cpu_count_logical": psutil.cpu_count(), "thread_limit": THREADS}


def load_module(name, relative):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def check_inventory():
    """Verify frozen source and exact inventories without numerical imports."""
    core = load_module("item6_worker_core", "scripts/research_ablations.py")
    growth = load_module("item6_worker_growth", "scripts/research_growth_ablation.py")
    protocol, plan, _, _ = core.load_execution_plan(PLAN)
    growth.load_execution_plan(PLAN)
    if core.inventory(protocol) != {key: plan[key] for key in core.inventory(protocol)}:
        raise RuntimeError("core inventory differs")
    if growth.operations.growth_inventory() != plan["growth"]:
        raise RuntimeError("growth inventory differs")
    if "torch" in sys.modules or "numpy" in sys.modules:
        raise RuntimeError("inventory verification unexpectedly imported a numerical runtime")
    return plan, core.source_record()


def run_child(panel, output):
    """Establish the inter-op limit before invoking the unchanged CLI contract."""
    admission()
    check_inventory()
    import torch

    torch.set_num_interop_threads(1)
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError("CPU inter-op limit was not applied")
    script = {"core": "scripts/research_ablations.py",
              "growth": "scripts/research_growth_ablation.py"}[panel]
    sys.argv = [str(ROOT / script), "--plan", str(PLAN), "--output", str(output),
                "--device", "cpu", "--threads", str(THREADS),
                "--minimum-ram-gib", str(MINIMUM_RAM_GIB)]
    runpy.run_path(str(ROOT / script), run_name="__main__")


class Archive:
    """Append owned run artifacts on a results branch without changing source HEAD."""

    def __init__(self, output, run_key, source_head):
        self.output, self.run_key, self.source_head = output, run_key, source_head
        self.worktree = Path(tempfile.mkdtemp(prefix="neuropixel-item6-archive-"))
        # git worktree requires a nonexistent or empty destination; this directory is ours.
        remote = git("ls-remote", "--heads", "origin", "refs/heads/" + RESULTS_BRANCH)
        if remote:
            git("fetch", "--no-tags", "origin", "refs/heads/" + RESULTS_BRANCH)
            archive_parent = git("rev-parse", "FETCH_HEAD")
        else:
            archive_parent = source_head
        git("worktree", "add", "--detach", self.worktree, archive_parent)
        self.relative = Path("results/research/06_cloud_runs") / run_key
        self.destination = self.worktree / self.relative
        if self.destination.exists():
            raise RuntimeError("this exact run attempt already has archived evidence")
        self.destination.mkdir(parents=True)
        self.last_commit = None

    def publish(self, final=False):
        if git("rev-parse", "HEAD") != self.source_head:
            raise RuntimeError("source HEAD changed while archiving")
        copied = {}
        for source in sorted(self.output.rglob("*")):
            if not source.is_file() or source.is_symlink() or source.name.endswith(".tmp"):
                continue
            relative = source.relative_to(self.output)
            destination = self.destination / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Each scientific writer uses atomic replacement. Read exactly one
            # complete file version; an interim archive is a snapshot, not a final gate.
            data = source.read_bytes()
            destination.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            if file_sha256(destination) != digest:
                raise OSError("archive copy readback failed")
            copied[relative.as_posix()] = {"sha256": digest, "bytes": len(data)}
        save_json(self.destination / "archive_manifest.json", {
            "schema_version": 1, "at_utc": utc_now(), "run_key": self.run_key,
            "source_commit": self.source_head, "snapshot_kind": "final" if final else "interim",
            "claim": "Each listed file was copied and read back; interim files may reflect different completed write times.",
            "files": copied,
        })
        git("add", "--force", "--", self.relative.as_posix(), cwd=self.worktree)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--exit-code"],
                                 cwd=self.worktree).returncode
        if changed not in (0, 1):
            raise RuntimeError("cannot inspect the archive index")
        if changed == 0:
            return self.last_commit
        body = self.worktree / "item6-archive-message.txt"
        body.write_text(
            f"Preserve item-6 {'final' if final else 'interim'} evidence for {self.run_key}\n\n"
            f"Source commit: {self.source_head}\n"
            "Archive only this owned attempt. Source branch and its running HEAD remain unchanged.\n",
            encoding="utf-8")
        git("-c", "user.name=NeuroPixel research automation",
            "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "commit", "--file", body, cwd=self.worktree)
        commit = git("rev-parse", "HEAD", cwd=self.worktree)
        # Push from the original checkout so the checkout action's scoped
        # credential configuration is used without copying or printing credentials.
        git("push", "origin", f"{commit}:refs/heads/{RESULTS_BRANCH}")
        remote_after = git("ls-remote", "--heads", "origin", "refs/heads/" + RESULTS_BRANCH)
        if remote_after.split()[0] != commit:
            raise RuntimeError("the archive branch did not retain the pushed commit")
        if git("rev-parse", "HEAD") != self.source_head:
            raise RuntimeError("archival mutated the scientific execution HEAD")
        self.last_commit = commit
        print(json.dumps({"event": "evidence_archived", "at_utc": utc_now(),
                          "archive_commit": commit, "files": len(copied), "final": final}), flush=True)
        return commit


def run_study():
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
            or os.environ.get("GITHUB_REF_NAME") != SOURCE_BRANCH):
        raise RuntimeError("this operational worker is restricted to the declared isolated Actions branch")
    run_id, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not re.fullmatch(r"\d+", run_id) or not re.fullmatch(r"\d+", attempt):
        raise RuntimeError("an explicit GitHub run ID and attempt are required")
    run_key = f"{run_id}-{attempt}"
    output = ROOT / "runs/item6" / run_key
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("this attempt directory is not empty; preserve it and use explicit recovery")
    output.mkdir(parents=True, exist_ok=True)
    initial_resources = admission()
    plan, source = check_inventory()
    head = git("rev-parse", "HEAD")
    if head != os.environ.get("GITHUB_SHA") or source["git_commit"] != head:
        raise RuntimeError("workflow source commit and actual execution HEAD differ")
    state = {"schema_version": 1, "item": 6, "status": "running", "run_key": run_key,
             "started_at_utc": utc_now(), "source": source, "execution_plan_sha256": file_sha256(PLAN),
             "initial_resources": initial_resources, "resources": [],
             "resource_policy": {"cpu_threads": 2, "interop_threads": 1, "minimum_ram_gib": 8,
                                 "maximum_seconds": MAXIMUM_SECONDS, "archive_interval_seconds": ARCHIVE_SECONDS},
             "steps": [], "archive_branch": RESULTS_BRANCH, "paid_compute": False}
    save_json(output / "worker_status.json", state)
    # Preserve the exact tracked source tree as well as its Git reference.
    subprocess.run(["git", "archive", "--format=zip", "--output", str(output / "source_commit.zip"), head],
                   cwd=ROOT, check=True)
    archive = Archive(output, run_key, head)
    archive.publish()
    started = time.monotonic()
    last_archive = started
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       NUMEXPR_NUM_THREADS="2", PYTHONHASHSEED="0", PYTHONUTF8="1",
                       PYTHONUNBUFFERED="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8")
    reports = output / "reports"
    verification = output / "verification"
    verification.mkdir()
    commands = [
        ("source_environment", [sys.executable, str(ROOT / "scripts/research_cloud_preflight.py")]),
        ("tests", [sys.executable, "-c",
                   "import sys, torch, pytest\n"
                   "torch.set_num_threads(2)\n"
                   "torch.set_num_interop_threads(1)\n"
                   "raise SystemExit(pytest.main(sys.argv[1:]))",
                   "tests", "-q", "--junitxml", str(verification / "tests.xml")]),
        ("core", [sys.executable, str(Path(__file__)), "--child", "core", "--output", str(output / "core")]),
        ("growth", [sys.executable, str(Path(__file__)), "--child", "growth", "--output", str(output / "growth")]),
        ("core_recount", [sys.executable, str(ROOT / "scripts/research_analyze_core_ablation.py"),
                          "--input", str(output / "core"), "--output-dir", str(reports / "core")]),
        ("growth_recount", [sys.executable, str(ROOT / "scripts/research_analyze_growth_ablation.py"),
                            "--input", str(output / "growth"), "--output-dir", str(reports / "growth")]),
    ]
    process = None
    try:
        for name, command in commands:
            admission()
            check_inventory()
            step = {"name": name, "status": "running", "started_at_utc": utc_now(),
                    "command": command, "log": f"logs/{name}.log"}
            state["steps"].append(step)
            save_json(output / "worker_status.json", state)
            log_path = output / step["log"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"event": "stage_started", "stage": name, "at_utc": step["started_at_utc"]}), flush=True)
            with log_path.open("w", encoding="utf-8") as writer, log_path.open("r", encoding="utf-8") as reader:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=writer,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                last_heartbeat = 0.0
                while True:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    code = process.poll()
                    if code is not None:
                        writer.flush()
                        chunk = reader.read()
                        if chunk:
                            print(chunk, end="", flush=True)
                        break
                    now = time.monotonic()
                    try:
                        resources = admission()
                        if now - started > MAXIMUM_SECONDS:
                            raise RuntimeError("the declared worker wall-time limit was reached")
                    except Exception:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=15)
                        raise
                    if now - last_heartbeat >= HEARTBEAT_SECONDS:
                        state["resources"].append(resources)
                        state["last_heartbeat_at_utc"] = resources["at_utc"]
                        save_json(output / "worker_status.json", state)
                        print(json.dumps({"event": "resource_heartbeat", "stage": name, **resources}), flush=True)
                        last_heartbeat = now
                    if now - last_archive >= ARCHIVE_SECONDS:
                        archive.publish()
                        last_archive = time.monotonic()
                    time.sleep(5)
            process = None
            step.update(status="completed" if code == 0 else "failed",
                        returncode=code, completed_at_utc=utc_now(),
                        log_sha256=file_sha256(log_path))
            save_json(output / "worker_status.json", state)
            archive.publish()
            last_archive = time.monotonic()
            if code:
                raise RuntimeError(f"stage {name} failed with return code {code}; later stages were not started")
        state.update(status="completed", completed_at_utc=utc_now(),
                     wall_seconds=time.monotonic() - started,
                     scientific_report_status="pending root synthesis and review; item 7 remains pending")
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     ended_at_utc=utc_now(), error=f"{type(error).__name__}: {error}",
                     wall_seconds=time.monotonic() - started)
        if state["steps"] and state["steps"][-1]["status"] == "running":
            state["steps"][-1].update(status="interrupted", ended_at_utc=utc_now())
        save_json(output / "worker_status.json", state)
        try:
            archive.publish(final=True)
        except Exception as archival_error:
            print(json.dumps({"event": "archive_failed", "error": str(archival_error),
                              "local_attempt_preserved": str(output)}), flush=True)
        raise
    else:
        save_json(output / "worker_status.json", state)
        archive.publish(final=True)
        print(json.dumps({"event": "study_completed", "run_key": run_key,
                          "archive_branch": RESULTS_BRANCH, "at_utc": utc_now()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("core", "growth"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inventory", action="store_true")
    args = parser.parse_args()
    if args.check_inventory:
        plan, source = check_inventory()
        print(json.dumps({"status": "inventory_verified", "source": source,
                          "core_training_runs": plan["planned_training_runs"],
                          "core_evaluation_cases": plan["planned_evaluation_cases"],
                          "growth_training_stages": plan["growth"]["planned_training_stages"]}), flush=True)
    elif args.child:
        if args.output is None:
            parser.error("--output is required for a child stage")
        run_child(args.child, args.output)
    else:
        run_study()


if __name__ == "__main__":
    main()
