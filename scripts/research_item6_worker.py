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
import threading
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
MONITOR_SECONDS = 1.0
GIT_TIMEOUT_SECONDS = 120.0
FINAL_ARCHIVE_GRACE_SECONDS = 180.0
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


def deadline_timeout(deadline, maximum=GIT_TIMEOUT_SECONDS):
    """Bound one operation by both its own cap and the remaining worker budget."""
    remaining = maximum if deadline is None else min(maximum, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("the declared worker wall-time limit was reached")
    return remaining


def git_result(*arguments, cwd=ROOT, deadline=None, allowed=(0,)):
    """Bound Git and its transport helpers without inherited interactive stdin.

    A temporary file avoids a descendant retaining a captured stdout pipe.
    On timeout the still-owned, unreaped session leader identifies the whole
    Git process group. No process outside that new session is signalled.
    """
    operation_deadline = time.monotonic() + deadline_timeout(deadline)
    command = ["git", "-c", "pack.threads=1", "-c", "index.threads=1", *map(str, arguments)]
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    with tempfile.TemporaryFile() as transcript:
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=transcript,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=deadline_timeout(operation_deadline))
        except BaseException:
            # Do not poll/reap before signalling: if the leader just exited,
            # its unreaped PID still belongs to this owned process group.
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            raise
        transcript.seek(0)
        output = transcript.read().decode("utf-8", errors="replace").strip()
    if process.returncode not in allowed:
        raise subprocess.CalledProcessError(process.returncode, command, output=output)
    return process.returncode, output


def git(*arguments, cwd=ROOT, deadline=None):
    return git_result(*arguments, cwd=cwd, deadline=deadline)[1]


def admission():
    import psutil

    available = psutil.virtual_memory().available / 2**30
    if available < MINIMUM_RAM_GIB:
        raise RuntimeError(f"available RAM {available:.3f} GiB is below the unchanged 8 GiB floor")
    return {"at_utc": utc_now(), "available_ram_gib": available,
            "cpu_count_logical": psutil.cpu_count(), "thread_limit": THREADS}



def stage_admission(deadline):
    """Refuse a new stage if either the resource floor or global deadline fails."""
    deadline_timeout(deadline)
    resources = admission()
    deadline_timeout(deadline)
    return resources


class ChildSupervisor:
    """Watch only one newly created child session while the main thread archives."""

    def __init__(self, process, deadline, interval=MONITOR_SECONDS):
        if not 0 < interval <= 5:
            raise ValueError("the supervision interval must be at most five seconds")
        self.process, self.deadline, self.interval = process, deadline, interval
        self.failure = None
        self.failed = threading.Event()
        self.finished = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._watch, name="item6-owned-child-supervisor",
                                       daemon=True)

    def start(self):
        self.thread.start()
        return self

    def poll(self):
        # All scientific-child wait/poll/signal operations share this lock:
        # neither thread can reap its PID between the other's poll and signal.
        with self.lock:
            return self.process.poll()

    def terminate(self):
        """Interrupt this owned session, with bounded escalation and reaping."""
        with self.lock:
            if self.process.poll() is not None:
                return
            for sig, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
                if self.process.poll() is not None:
                    return
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=grace)
                    return
                except subprocess.TimeoutExpired:
                    continue
            raise RuntimeError("the owned child did not stop after bounded SIGKILL cleanup")

    def _watch(self):
        while not self.finished.is_set():
            if self.poll() is not None:
                return
            try:
                stage_admission(self.deadline)
            except Exception as error:
                self.failure = f"{type(error).__name__}: {error}"
                self.failed.set()
                try:
                    self.terminate()
                except Exception as stop_error:
                    self.failure += f"; cleanup: {type(stop_error).__name__}: {stop_error}"
                return
            self.finished.wait(self.interval)

    def raise_if_failed(self):
        if self.failed.is_set():
            raise RuntimeError("independent child supervision stopped the stage: " + self.failure)

    def close(self):
        self.finished.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise RuntimeError("the owned-child supervisor did not finish")


def launch_child(command, environment, writer, deadline):
    """Apply the deadline immediately before Popen and immediately start supervision."""
    stage_admission(deadline)
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                               stdout=writer, stderr=subprocess.STDOUT, start_new_session=True)
    supervisor = ChildSupervisor(process, deadline)
    try:
        supervisor.start()
    except BaseException:
        supervisor.terminate()
        raise
    return process, supervisor


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

    def __init__(self, output, run_key, source_head, deadline=None):
        self.output, self.run_key, self.source_head = output, run_key, source_head
        self.deadline = deadline
        self.worktree = Path(tempfile.mkdtemp(prefix="neuropixel-item6-archive-"))
        # git worktree requires a nonexistent or empty destination; this directory is ours.
        remote = git("ls-remote", "--heads", "origin", "refs/heads/" + RESULTS_BRANCH, deadline=deadline)
        if remote:
            git("fetch", "--no-tags", "origin", "refs/heads/" + RESULTS_BRANCH, deadline=deadline)
            archive_parent = git("rev-parse", "FETCH_HEAD", deadline=deadline)
        else:
            archive_parent = source_head
        git("worktree", "add", "--detach", self.worktree, archive_parent, deadline=deadline)
        self.relative = Path("results/research/06_cloud_runs") / run_key
        self.destination = self.worktree / self.relative
        if self.destination.exists():
            raise RuntimeError("this exact run attempt already has archived evidence")
        self.destination.mkdir(parents=True)
        self.last_commit = None

    def publish(self, final=False, deadline=None):
        deadline = self.deadline if deadline is None else deadline
        deadline_timeout(deadline)
        if git("rev-parse", "HEAD", deadline=deadline) != self.source_head:
            raise RuntimeError("source HEAD changed while archiving")
        copied = {}
        for source in sorted(self.output.rglob("*")):
            deadline_timeout(deadline)
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
        git("add", "--force", "--", self.relative.as_posix(), cwd=self.worktree, deadline=deadline)
        changed, _ = git_result("diff", "--cached", "--quiet", "--exit-code",
                                cwd=self.worktree, deadline=deadline, allowed=(0, 1))
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
            "commit", "--file", body, cwd=self.worktree, deadline=deadline)
        commit = git("rev-parse", "HEAD", cwd=self.worktree, deadline=deadline)
        # Push from the original checkout so the checkout action's scoped
        # credential configuration is used without copying or printing credentials.
        git("push", "origin", f"{commit}:refs/heads/{RESULTS_BRANCH}", deadline=deadline)
        remote_after = git("ls-remote", "--heads", "origin", "refs/heads/" + RESULTS_BRANCH, deadline=deadline)
        if remote_after.split()[0] != commit:
            raise RuntimeError("the archive branch did not retain the pushed commit")
        if git("rev-parse", "HEAD", deadline=deadline) != self.source_head:
            raise RuntimeError("archival mutated the scientific execution HEAD")
        self.last_commit = commit
        print(json.dumps({"event": "evidence_archived", "at_utc": utc_now(),
                          "archive_commit": commit, "files": len(copied), "final": final}), flush=True)
        return commit


def run_study():
    started = time.monotonic()
    deadline = started + MAXIMUM_SECONDS
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
    initial_resources = stage_admission(deadline)
    plan, source = check_inventory()
    head = git("rev-parse", "HEAD", deadline=deadline)
    if head != os.environ.get("GITHUB_SHA") or source["git_commit"] != head:
        raise RuntimeError("workflow source commit and actual execution HEAD differ")
    state = {"schema_version": 1, "item": 6, "status": "running", "run_key": run_key,
             "started_at_utc": utc_now(), "source": source, "execution_plan_sha256": file_sha256(PLAN),
             "initial_resources": initial_resources, "resources": [],
             "resource_policy": {"cpu_threads": 2, "interop_threads": 1, "minimum_ram_gib": 8,
                                 "maximum_seconds": MAXIMUM_SECONDS, "archive_interval_seconds": ARCHIVE_SECONDS,
                                 "supervisor_interval_seconds": MONITOR_SECONDS,
                                 "git_timeout_seconds": GIT_TIMEOUT_SECONDS,
                                 "final_archive_grace_seconds": FINAL_ARCHIVE_GRACE_SECONDS},
             "steps": [], "archive_branch": RESULTS_BRANCH, "paid_compute": False}
    save_json(output / "worker_status.json", state)
    # Preserve the exact tracked source tree as well as its Git reference.
    git("archive", "--format=zip", "--output", str(output / "source_commit.zip"), head, deadline=deadline)
    archive = Archive(output, run_key, head, deadline=deadline)
    archive.publish()
    last_archive = time.monotonic()
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
    process, supervisor = None, None
    try:
        for name, command in commands:
            stage_admission(deadline)
            check_inventory()
            step = {"name": name, "status": "running", "started_at_utc": utc_now(),
                    "command": command, "log": f"logs/{name}.log"}
            state["steps"].append(step)
            save_json(output / "worker_status.json", state)
            log_path = output / step["log"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"event": "stage_started", "stage": name, "at_utc": step["started_at_utc"]}), flush=True)
            with log_path.open("w", encoding="utf-8") as writer, log_path.open("r", encoding="utf-8") as reader:
                process, supervisor = launch_child(command, environment, writer, deadline)
                last_heartbeat = 0.0
                while True:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    supervisor.raise_if_failed()
                    code = supervisor.poll()
                    supervisor.raise_if_failed()
                    if code is not None:
                        writer.flush()
                        chunk = reader.read()
                        if chunk:
                            print(chunk, end="", flush=True)
                        break
                    now = time.monotonic()
                    resources = stage_admission(deadline)
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
            supervisor.close()
            supervisor.raise_if_failed()
            process, supervisor = None, None
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
        child_stopped = process is None
        if supervisor is not None:
            try:
                supervisor.terminate()
                supervisor.close()
                child_stopped = supervisor.poll() is not None
            except Exception as shutdown_error:
                state["shutdown_error"] = f"{type(shutdown_error).__name__}: {shutdown_error}"
                child_stopped = supervisor.poll() is not None
            if supervisor.failure is not None:
                state["supervision_failure"] = supervisor.failure
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     ended_at_utc=utc_now(), error=f"{type(error).__name__}: {error}",
                     wall_seconds=time.monotonic() - started)
        if state["steps"] and state["steps"][-1]["status"] == "running":
            state["steps"][-1].update(status="interrupted", ended_at_utc=utc_now())
        save_json(output / "worker_status.json", state)
        try:
            if not child_stopped:
                raise RuntimeError("final archival requires the owned scientific child to be stopped")
            archive.publish(final=True, deadline=time.monotonic() + FINAL_ARCHIVE_GRACE_SECONDS)
        except Exception as archival_error:
            print(json.dumps({"event": "archive_failed", "error": str(archival_error),
                              "local_attempt_preserved": str(output)}), flush=True)
        raise
    else:
        save_json(output / "worker_status.json", state)
        archive.publish(final=True, deadline=time.monotonic() + FINAL_ARCHIVE_GRACE_SECONDS)
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
