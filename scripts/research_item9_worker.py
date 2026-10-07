"""Supervise and archive the frozen item-9 CPU phases on an isolated branch.

The supervisor imports no numerical runtime. Its fresh child configures the
recorded Torch runtime before either the two-file contract suite or the study
CLI. Interim snapshots are partial observations; final manifests describe a
stopped attempt, whose success or failure is recorded separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BRANCH = "research/scientific-validation-2026-10-07-cloud"
RESULTS_BRANCH = SOURCE_BRANCH + "-results"
RECIPE_PATH = "docs/research/09_experiment_recipe.json"
TEST_FILES = ["tests/test_complex_binding_data.py", "tests/test_complex_binding_protocol.py"]
VERSIONS = {"torch": "2.6.0+cpu", "numpy": "2.2.6", "scipy": "1.15.1",
            "pytest": "9.1.1", "psutil": "7.2.2", "Pillow": "12.3.0"}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
STAGES = {"preflight": ["contract_tests", "preflight"],
          "study": ["contract_tests", "train", "final"]}


def policy(phase):
    return {"numerical_threads": 2, "interop_threads": 1,
            "minimum_available_ram_gib": 8, "aggregate_active_cpu_threads": 4,
            "git_threads": 1, "supervisor_interval_seconds": 1,
            "heartbeat_interval_seconds": 60, "archive_interval_seconds": 300,
            "git_timeout_seconds": 60, "final_archive_reserve_seconds": 180,
            "contract_test_seconds": 900, "minimum_stage_start_seconds": 60,
            "workflow_minutes": 120 if phase == "preflight" else 240,
            "worker_seconds": 6600 if phase == "preflight" else 13800,
            "stage_seconds": 6600 if phase == "preflight" else 13800}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value, *, exclusive=False):
    """Use exclusive receipts or atomic mutable status, with durable file bytes."""
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if exclusive:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
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
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def remaining(deadline, maximum=60):
    value = min(maximum, deadline - time.monotonic())
    if value <= 0:
        raise TimeoutError("declared execution/archive deadline reached")
    return value


def git(*args, cwd=ROOT, deadline, allowed=(0,)):
    """Bound the owned Git session and transport descendants, with no stdin."""
    timeout = remaining(deadline)
    command = ["git", "-c", "pack.threads=1", "-c", "index.threads=1", *map(str, args)]
    with tempfile.TemporaryFile() as stream:
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            # Before reaping, this PID still identifies our own session leader.
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            raise
        stream.seek(0)
        output = stream.read().decode(errors="replace").strip()
    if code not in allowed:
        raise RuntimeError(f"Git {args[0]} returned {code}: {output[-2000:]}")
    return output


def resource_sample(label):
    return {"at_utc": now(), "phase": label,
            "available_ram_gib": psutil.virtual_memory().available / 2**30}


def admit(deadline, *, minimum_seconds=0):
    if deadline - time.monotonic() < minimum_seconds:
        raise TimeoutError("insufficient declared budget to start another phase")
    remaining(deadline)
    row = resource_sample("admission")
    if row["available_ram_gib"] < 8:
        raise RuntimeError("at least 8 GiB available RAM is required")
    return row


def plan_path(phase):
    return ROOT / f"docs/research/09_{phase}_plan.json"


def load_plan(phase):
    plan = json.loads(plan_path(phase).read_text())
    if (plan.get("schema_version") != 1 or plan.get("item") != 9
            or plan.get("phase") != phase or plan.get("status") != "frozen"
            or plan.get("recipe_path") != RECIPE_PATH or plan.get("stages") != STAGES[phase]
            or plan.get("test_inventory") != TEST_FILES or plan.get("resource_policy") != policy(phase)):
        raise ValueError("plan differs from the declared item-9 execution contract")
    if type(plan.get("expected_test_count")) is not int or plan["expected_test_count"] <= 0:
        raise ValueError("plan must fix a positive collected-test count")
    frozen = datetime.fromisoformat(plan["freeze_utc"])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("freeze timestamp must be a past timezone-aware instant")
    mapping = plan.get("implementation_sha256")
    if not isinstance(mapping, dict):
        raise ValueError("implementation SHA-256 mapping is required")
    required = set(TEST_FILES) | {RECIPE_PATH, "scripts/research_item9_worker.py",
        "scripts/research_complex_binding.py", "requirements/research-cloud.txt", "pytest.ini",
        ".github/workflows/research-item9-preflight.yml", ".github/workflows/research-item9-study.yml"}
    required.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "neuropixel").rglob("*.py"))
    if not required <= set(mapping):
        raise ValueError("plan must bind package sources, controller, worker, workflows, tests and runtime inputs")
    for name, digest in mapping.items():
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("invalid implementation path or SHA-256")
    if plan.get("recipe_sha256") != mapping[RECIPE_PATH] or sha(ROOT / RECIPE_PATH) != plan["recipe_sha256"]:
        raise ValueError("recipe identity differs from its frozen binding")
    return plan


def source_record(plan, phase, commit, output, deadline):
    actual = {name: sha(ROOT / name) for name in plan["implementation_sha256"]}
    tracked = set(git("ls-files", "-z", deadline=deadline).split("\0"))
    untracked = git("ls-files", "--others", "--exclude-standard", "-z", deadline=deadline).split("\0")
    own_prefix = output.relative_to(ROOT).as_posix() + "/"
    unexpected = [name for name in untracked if name and not name.startswith(own_prefix)]
    relative = plan_path(phase).relative_to(ROOT).as_posix()
    current = git("rev-parse", "HEAD", deadline=deadline)
    record = {"at_utc": now(), "source_commit": current, "plan_sha256": sha(plan_path(phase)),
              "recipe_sha256": sha(ROOT / RECIPE_PATH), "implementation_sha256": actual,
              "unexpected_untracked_files": unexpected,
              "tracked_diff": git("diff", "--name-only", "HEAD", "--", deadline=deadline),
              "all_bound_files_tracked": set(actual) <= tracked,
              "plan_matches_committed_blob": git("hash-object", "--", relative, deadline=deadline)
              == git("rev-parse", commit + ":" + relative, deadline=deadline)}
    record["verified"] = (current == commit and actual == plan["implementation_sha256"]
        and record["recipe_sha256"] == plan["recipe_sha256"] and not unexpected
        and not record["tracked_diff"] and record["all_bound_files_tracked"]
        and record["plan_matches_committed_blob"])
    return record


def runtime(torch, label):
    return {"at_utc": now(), "phase": label, "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in VERSIONS},
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_cuda_version": torch.version.cuda, "platform": platform.platform(),
            "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV}}


def child_stage(phase, stage, output):
    plan = load_plan(phase)
    commit = os.environ["GITHUB_SHA"]
    if not source_record(plan, phase, commit, output, time.monotonic() + 60)["verified"]:
        raise RuntimeError("child source differs before numerical imports")
    admit(time.monotonic() + 60)
    for name in THREAD_ENV:
        os.environ[name] = "2"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if platform.python_version() != "3.12.8":
        raise RuntimeError("Python differs from the pinned3.12.8 runtime")
    sys.path.insert(0, str(ROOT))
    import torch
    if ({name: importlib.metadata.version(name) for name in VERSIONS} != VERSIONS
            or str(torch.__version__) != VERSIONS["torch"] or torch.version.cuda is not None):
        save(output / f"{stage}_runtime_rejected.json", runtime(torch, "rejected_versions"), exclusive=True)
        raise RuntimeError("numerical/runtime package versions differ from the CPU freeze")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    admit(time.monotonic() + 60)
    records, boundaries, skips, issues = [runtime(torch, "before_stage")], [], [], []
    code = 1
    try:
        if stage == "contract_tests":
            import pytest

            class Controls:
                def pytest_collection_finish(self, session):
                    ids = [item.nodeid for item in session.items]
                    files = sorted({node.split("::", 1)[0] for node in ids})
                    save(output / "test_collection.json", {"nodeids": ids, "count": len(ids),
                         "expected_count": plan["expected_test_count"], "files": files}, exclusive=True)
                    if (len(ids) != plan["expected_test_count"] or len(ids) != len(set(ids))
                            or files != TEST_FILES):
                        raise pytest.UsageError("collection differs from the frozen two-file inventory")

                @pytest.hookimpl(hookwrapper=True, tryfirst=True)
                def pytest_runtest_protocol(self, item, nextitem):
                    before = (torch.get_num_threads(), torch.get_num_interop_threads())
                    if before != (2, 1):
                        raise RuntimeError("thread controls differ before test")
                    try:
                        yield
                    finally:
                        after = (torch.get_num_threads(), torch.get_num_interop_threads())
                        boundaries.append({"nodeid": item.nodeid, "before": list(before), "after": list(after)})
                        if after != (2, 1):
                            issues.append("thread controls changed during " + item.nodeid)

                def pytest_runtest_logreport(self, report):
                    if report.skipped:
                        skips.append(report.nodeid)
                        issues.append("item-9 contract tests permit no skips: " + report.nodeid)

            code = int(pytest.main([*TEST_FILES, "-q", "--junitxml", str(output / "tests.xml")], plugins=[Controls()]))
        else:
            script = ROOT / "scripts/research_complex_binding.py"
            sys.argv = [str(script), "--phase", stage, "--plan", str(plan_path(phase)),
                        "--output", str(output / "study")]
            try:
                runpy.run_path(str(script), run_name="__main__")
                code = 0
            except SystemExit as exc:
                code = 0 if exc.code is None else exc.code if isinstance(exc.code, int) else 1
    finally:
        records.append(runtime(torch, "after_stage_or_exception"))
        if (torch.get_num_threads(), torch.get_num_interop_threads()) != (2, 1):
            issues.append("stage did not finish with Torch threads2/1")
        save(output / f"{stage}_environment.json", {"records": records,
             "test_thread_boundaries": boundaries, "skipped_nodeids": skips, "issues": issues,
             "limitations": "Thread counts are boundary observations, not continuous utilization."}, exclusive=True)
    return code or int(bool(issues))


class Supervisor:
    """Monitor only one owned session while main may block on snapshot transport."""

    def __init__(self, child, deadline, output, stage):
        self.child, self.deadline, self.output, self.stage = child, deadline, output, stage
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.failure = None
        self.samples = []
        self.thread = threading.Thread(target=self.watch, name="item9-owned-child", daemon=True)

    def poll(self):
        with self.lock:
            return self.child.poll()

    def terminate(self):
        with self.lock:
            if self.child.poll() is not None:
                return
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    self.child.wait(timeout=5)
                    return
                except subprocess.TimeoutExpired:
                    continue
            raise RuntimeError("owned child failed bounded termination")

    def persist(self):
        save(self.output / f"{self.stage}_resource_monitor.json", {"at_utc": now(),
             "interval_seconds": 1, "samples": self.samples, "failure": self.failure,
             "minimum_sampled_available_ram_gib": min((x["available_ram_gib"] for x in self.samples), default=None)})

    def watch(self):
        heartbeat = 0.0
        try:
            while not self.finished.is_set() and self.poll() is None:
                row = resource_sample(self.stage)
                self.samples.append(row)
                if row["available_ram_gib"] < 8 or time.monotonic() >= self.deadline:
                    raise RuntimeError("RAM floor or declared stage/execution deadline reached")
                if time.monotonic() - heartbeat >= 60:
                    self.persist()
                    print(json.dumps({"event": "resource_heartbeat", **row}), flush=True)
                    heartbeat = time.monotonic()
                self.finished.wait(1)
        except BaseException as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            try:
                self.terminate()
            except Exception as stop_error:
                self.failure += f"; termination: {stop_error}"
        finally:
            try:
                self.persist()
            except Exception as exc:
                self.failure = self.failure or f"resource receipt failed: {exc}"

    def close(self):
        self.finished.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            raise RuntimeError("owned child monitor did not stop")


class Archive:
    def __init__(self, output, commit, deadline):
        self.output, self.commit = output, commit
        self.relative = output.relative_to(ROOT)
        self.tree = Path(tempfile.mkdtemp(prefix="neuropixel-item9-results-")) / "tree"
        git("fetch", "--no-tags", "origin", RESULTS_BRANCH, deadline=deadline)
        base = git("rev-parse", "FETCH_HEAD", deadline=deadline)
        git("worktree", "add", "--detach", self.tree, base, deadline=deadline)
        self.destination = self.tree / self.relative
        if self.destination.exists():
            raise FileExistsError("this run/attempt/phase already exists on the results branch")
        self.destination.mkdir(parents=True)
        self.last_files, self.last_commit = {}, None

    def publish(self, *, deadline, final=False, status="running"):
        # A bounded batch keeps the next scheduled archive attempt within300s;
        # independent RAM/deadline/heartbeat supervision continues throughout.
        deadline = min(deadline, time.monotonic() + (180 if final else 120))
        if (self.destination / "archive_manifest.json").exists():
            raise FileExistsError("a final raw archive is immutable")
        if git("rev-parse", "HEAD", deadline=deadline) != self.commit:
            raise RuntimeError("source HEAD changed during archive")
        copied = set()
        for source in sorted(self.output.rglob("*")):
            remaining(deadline)
            if source.is_symlink():
                raise ValueError("owned results must not contain symlinks")
            if (not source.is_file() or source.name.endswith(".tmp")
                    or source.name.startswith("_pending_")):
                continue
            relative = source.relative_to(self.output)
            target = self.destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, target.open("wb") as dst:
                # Bound a live log to its size at open instead of chasing a
                # writer indefinitely. The manifest hashes this copied prefix.
                unread = os.fstat(src.fileno()).st_size
                while unread:
                    remaining(deadline)
                    block = src.read(min(unread, 1024 * 1024))
                    if not block:
                        break
                    dst.write(block)
                    unread -= len(block)
            copied.add(relative.as_posix())
        for path in self.destination.rglob("*"):
            if path.is_file() and path.relative_to(self.destination).as_posix() not in copied:
                path.unlink()
        # Hash the completed COPY, not a later version of a live training log.
        files = {name: {"bytes": (self.destination / name).stat().st_size,
                        "sha256": sha(self.destination / name)} for name in sorted(copied)}
        manifest = {"schema_version": 1, "item": 9, "created_at_utc": now(),
                    "source_commit": self.commit, "run_key": self.output.name,
                    "snapshot_kind": "final" if final else "interim_partial",
                    "attempt_status": status, "final": final, "files": files,
                    "claim": "Hashes cover the completed copied files. Live interim files may reflect different times or incomplete writes; they are not final study evidence."}
        save(self.destination / ("archive_manifest.json" if final else "interim_manifest.json"), manifest, exclusive=True)
        git("add", "--force", "--", self.relative.as_posix(), cwd=self.tree, deadline=deadline)
        message = f"research: archive item9 {'final' if final else 'partial'} {self.output.name}"
        git("-c", "user.name=NeuroPixel research automation",
            "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "commit", "-m", message, cwd=self.tree, deadline=deadline)
        commit = git("rev-parse", "HEAD", cwd=self.tree, deadline=deadline)
        git("push", "origin", f"{commit}:refs/heads/{RESULTS_BRANCH}", deadline=deadline)
        remote = git("ls-remote", "--heads", "origin", "refs/heads/" + RESULTS_BRANCH, deadline=deadline)
        if not remote or remote.split()[0] != commit:
            raise RuntimeError("results-branch push could not be confirmed")
        self.last_files, self.last_commit = files, commit
        print(json.dumps({"event": "evidence_archived", "at_utc": now(), "archive_commit": commit,
                          "snapshot_kind": manifest["snapshot_kind"], "files": len(files)}), flush=True)
        return commit


def run_stage(phase, stage, output, plan, commit, execution_deadline, archive, state):
    limits = policy(phase)
    if not source_record(plan, phase, commit, output, execution_deadline)["verified"]:
        raise RuntimeError("source drift before " + stage)
    cap = limits["contract_test_seconds"] if stage == "contract_tests" else limits["stage_seconds"]
    deadline = min(execution_deadline, time.monotonic() + cap)
    admitted = admit(deadline, minimum_seconds=limits["minimum_stage_start_seconds"])
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
               "--child", stage, "--output", str(output)]
    record = {"name": stage, "status": "running", "command": command,
              "started_at_utc": now(), "admission": admitted, "stage_limit_seconds": cap,
              "execution_seconds_available_at_launch": deadline - time.monotonic()}
    state["stages"].append(record)
    save(output / "worker_status.json", state)
    started, last_archive = time.monotonic(), time.monotonic()
    supervisor = None
    log_path = output / f"{stage}.log"
    try:
        with log_path.open("x", encoding="utf-8") as log:
            admit(deadline, minimum_seconds=limits["minimum_stage_start_seconds"])
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            supervisor = Supervisor(child, deadline, output, stage)
            supervisor.thread.start()
            while supervisor.poll() is None:
                if supervisor.failure:
                    raise RuntimeError(supervisor.failure)
                if time.monotonic() - last_archive >= 300:
                    last_archive = time.monotonic()
                    archive.publish(deadline=execution_deadline)
                time.sleep(0.5)
            if supervisor.failure:
                raise RuntimeError(supervisor.failure)
            record["return_code"] = child.returncode
            record["status"] = "completed" if child.returncode == 0 else "failed"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, message=str(exc)[:2000])
        raise
    finally:
        if supervisor is not None:
            supervisor.terminate()
            supervisor.close()
            record["return_code"] = supervisor.child.returncode
            if supervisor.failure:
                record.update(status="failed", supervision_failure=supervisor.failure)
        record.update(completed_at_utc=now(), wall_seconds_including_shutdown=time.monotonic() - started)
        if log_path.exists():
            record["log_sha256"] = sha(log_path)
        save(output / f"{stage}_status.json", record, exclusive=True)
        save(output / "worker_status.json", state)
    return record


def run_worker(phase):
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != "Agnuxo1/NeuroPixel"
            or os.environ.get("GITHUB_REF_NAME") != SOURCE_BRANCH):
        raise RuntimeError("worker requires the declared isolated public Actions source branch")
    run_id, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not re.fullmatch(r"\d+", run_id) or not re.fullmatch(r"\d+", attempt):
        raise ValueError("numeric GitHub run and attempt IDs are required")
    started = time.monotonic()
    workflow_remaining = float(os.environ["ITEM9_JOB_DEADLINE_UNIX"]) - time.time()
    deadline = started + min(policy(phase)["worker_seconds"], workflow_remaining)
    execution_deadline = deadline - policy(phase)["final_archive_reserve_seconds"]
    output = ROOT / "results/research/09_cloud_runs" / f"{run_id}-{attempt}-{phase}"
    output.mkdir(parents=True, exist_ok=False)
    commit = os.environ["GITHUB_SHA"]
    state = {"schema_version": 1, "item": 9, "phase": phase, "run_key": output.name,
             "source_commit": commit, "started_at_utc": now(), "status": "running", "stages": [],
             "resource_policy": policy(phase), "initial_budget_seconds": deadline - started,
             "results_branch": RESULTS_BRANCH, "paid_compute": False, "gpu_requested": False}
    archive, plan, error = None, None, None
    save(output / "worker_status.json", state)
    try:
        state["initial_resources"] = admit(execution_deadline, minimum_seconds=60)
        shutil.copyfile(plan_path(phase), output / "execution_plan.json")
        shutil.copyfile(ROOT / RECIPE_PATH, output / "experiment_recipe.json")
        git("archive", "--format=zip", "--output", output / "source.zip", commit, deadline=execution_deadline)
        plan = load_plan(phase)
        before = source_record(plan, phase, commit, output, execution_deadline)
        save(output / "source_before.json", before, exclusive=True)
        if not before["verified"]:
            raise RuntimeError("source/recipe/plan is not the clean committed freeze")
        archive = Archive(output, commit, execution_deadline)
        archive.publish(deadline=execution_deadline)
        for stage in STAGES[phase]:
            record = run_stage(phase, stage, output, plan, commit, execution_deadline, archive, state)
            if record["return_code"] != 0 or record["status"] != "completed":
                raise RuntimeError(f"stage {stage} failed; later scientific stages were not started")
            archive_commit = archive.publish(deadline=execution_deadline)
            if stage == "train":
                inventory = output / "study/final_inventory.json"
                identity = sha(inventory)
                if archive.last_files.get("study/final_inventory.json", {}).get("sha256") != identity:
                    raise RuntimeError("final inventory was not durably copied in the confirmed archive")
                gate = json.loads(inventory.read_text())
                if len(gate["entries"]) != 10:
                    raise RuntimeError("durability gate requires all ten declared checkpoints")
                references = [gate["primary_index"]]
                references.extend(entry[key] for entry in gate["entries"] for key in ("summary", "checkpoint"))
                for reference in references:
                    copied = archive.last_files.get("study/" + reference["path"])
                    if copied != {"bytes": reference["bytes"], "sha256": reference["sha256"]}:
                        raise RuntimeError("declared checkpoint/summary/index differs from its archived copy")
                save(output / "study/gate_archive_receipt.json", {
                    "schema_version": 1, "item": 9, "source_commit": commit,
                    "archive_commit": archive_commit, "results_branch": RESULTS_BRANCH,
                    "final_inventory_sha256": identity, "archived_at_utc": now()}, exclusive=True)
        state["status"] = "completed"
    except BaseException as exc:
        error = exc
        state.update(status="failed", error_type=type(exc).__name__, message=str(exc)[:3000])
    finally:
        # run_stage stops and reaps its owned child before returning/raising.
        try:
            if plan is not None:
                after = source_record(plan, phase, commit, output, deadline)
                save(output / "source_after.json", after, exclusive=True)
                if not after["verified"]:
                    raise RuntimeError("source/recipe/plan changed during execution")
        except BaseException as exc:
            error = error or exc
            state.update(status="failed", source_after_error=str(exc)[:2000])
        state["final_resources"] = resource_sample("worker_finished")
        if state["final_resources"]["available_ram_gib"] < 8:
            error = error or RuntimeError("RAM below floor at final observation")
            state.update(status="failed", final_resource_error=str(error))
        state.update(completed_at_utc=now(), worker_wall_seconds_before_archive=time.monotonic() - started)
        save(output / "worker_status.json", state)
        try:
            if archive is None:
                archive = Archive(output, commit, deadline)
            archive.publish(deadline=deadline, final=True, status=state["status"])
        except BaseException as exc:
            error = error or exc
            save(output / "archive_failure.json", {"at_utc": now(), "error": str(exc)[:3000],
                 "local_attempt_preserved": str(output)}, exclusive=True)
            print(json.dumps({"event": "archive_failed", "error": str(exc)[:3000]}), flush=True)
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=tuple(STAGES))
    parser.add_argument("--child", choices=("contract_tests", "preflight", "train", "final"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        if args.child not in STAGES[args.phase] or args.output is None:
            parser.error("child stage and output must match the requested phase")
        return child_stage(args.phase, args.child, args.output.resolve())
    if args.output is not None:
        parser.error("supervisor output is fixed by run/attempt/phase identity")
    return run_worker(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
