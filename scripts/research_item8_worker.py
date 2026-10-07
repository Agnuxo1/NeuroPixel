"""Bounded item-8 CPU fixture validation with preserved failure evidence.

The coordinator never imports Torch. Child stages run only the declared legacy
PAD witness and the complete regression suite, not a scientific training or
final-performance study. Archival uses a separate results worktree.
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
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import psutil


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/08_cloud_validation_plan.json"
RESULTS_BRANCH = "research/scientific-validation-2026-10-07-cloud-results"
STAGES = ["legacy_padding_demonstration", "full_pytest"]
TEST_FILES = ["tests/test_classification_contracts.py", "tests/test_core.py",
              "tests/test_padding_invariant.py", "tests/test_research_ablations.py",
              "tests/test_research_analyze_core_ablation.py", "tests/test_research_analyze_growth_ablation.py",
              "tests/test_research_development_data.py", "tests/test_research_growth_ablation.py",
              "tests/test_research_item6_worker.py", "tests/test_research_item6_worker_supervision.py",
              "tests/test_research_split_governance.py", "tests/test_routing_metrics.py",
              "tests/test_soil_evaluation.py"]
VERSIONS = {"torch": "2.6.0+cpu", "numpy": "2.2.6", "scipy": "1.15.1",
            "pytest": "9.1.1", "psutil": "7.2.2", "Pillow": "12.3.0"}
LIMITS = {"numerical_threads": 1, "interop_threads": 1, "minimum_available_ram_gib": 8,
          "stage_seconds": 900, "workflow_minutes": 35,
          "compatibility_fixture_max_threads": 2, "aggregate_active_cpu_threads": 4}
ALLOWED_SKIPS = ["tests/test_core.py::test_cartilla_layout"]
THREAD_EXCEPTION = "tests/test_core.py::test_safety_refuses_busy_gpu"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


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


def sample(phase):
    return {"at_utc": now(), "phase": phase,
            "available_ram_gib": psutil.virtual_memory().available / 2**30}


def require_ram(row):
    if row["available_ram_gib"] < 8:
        raise RuntimeError("at least 8 GiB available RAM is required")


def stop_owned(child):
    if child.poll() is None:
        for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                pass
        raise RuntimeError("owned child did not stop after bounded termination")


def command_output(command, *, cwd=ROOT, deadline=None):
    """Bound Git and its transport descendants; never wait on inherited pipes."""
    timeout = 60 if deadline is None else min(60, deadline - time.time())
    if timeout <= 0:
        raise TimeoutError("no time remains for the archival/source command")
    with tempfile.TemporaryFile() as stream:
        child = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=stream, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            stop_owned(child)
            raise
        stream.seek(0)
        output = stream.read().decode(errors="replace")
    if code:
        raise RuntimeError(f"command failed ({code}): {command[0]} {command[1:3]}\n{output[-2000:]}")
    return output.strip()


def git(*args, cwd=ROOT, deadline=None):
    return command_output(["git", "-c", "pack.threads=1", *args], cwd=cwd, deadline=deadline)


def load_plan():
    plan = json.loads(PLAN_PATH.read_text())
    if (plan.get("schema_version") != 1 or plan.get("item") != 8 or plan.get("status") != "frozen"
            or plan.get("python") != "3.12.8" or plan.get("torch") != VERSIONS["torch"]
            or plan.get("stages") != STAGES or plan.get("test_inventory") != TEST_FILES
            or plan.get("allowed_skip_nodeids") != ALLOWED_SKIPS or plan.get("resource_limits") != LIMITS
            or plan.get("scientific_training") is not False or plan.get("scientific_final_scoring") is not False
            or plan.get("bounded_optimizer_fixtures") is not True):
        raise ValueError("plan differs from the declared item-8 fixture inventory and resource policy")
    if type(plan.get("expected_test_count")) is not int or plan["expected_test_count"] <= 0:
        raise ValueError("plan must fix a positive collected-test count")
    frozen = datetime.fromisoformat(plan["freeze_utc"])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("freeze timestamp must be a past timezone-aware instant")
    actual_tests = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("test_*.py"))
    if actual_tests != TEST_FILES:
        raise ValueError("current test-file inventory differs from the frozen whole-suite inventory")
    required = set(TEST_FILES) | {"scripts/research_item8_worker.py", ".github/workflows/research-item8.yml",
                                 "requirements/research-cloud.txt", "pytest.ini"}
    mapping = plan.get("implementation_sha256", {})
    if not required <= set(mapping):
        raise ValueError("implementation map must bind the worker, workflow, dependencies and every test file")
    for name, value in mapping.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or len(value) != 64:
            raise ValueError("invalid implementation hash entry")
    return plan


def source_record(plan, commit, deadline):
    actual = {name: sha(ROOT / name) for name in plan["implementation_sha256"]}
    relative = PLAN_PATH.relative_to(ROOT).as_posix()
    record = {"at_utc": now(), "source_commit": git("rev-parse", "HEAD", deadline=deadline),
              "plan_sha256": sha(PLAN_PATH), "implementation_sha256": actual}
    tracked = set(git("ls-files", "-z", deadline=deadline).split("\0"))
    record["all_implementation_files_tracked"] = set(actual) <= tracked
    record["verified"] = (record["source_commit"] == commit and actual == plan["implementation_sha256"]
                          and record["all_implementation_files_tracked"]
                          and git("hash-object", "--", relative, deadline=deadline)
                          == git("rev-parse", commit + ":" + relative, deadline=deadline)
                          and not git("diff", "--name-only", "HEAD", "--", deadline=deadline))
    return record


def runtime_record(torch, phase):
    return {"at_utc": now(), "phase": phase, "python": platform.python_version(),
            "platform": platform.platform(), "versions": {k: importlib.metadata.version(k) for k in VERSIONS},
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_cuda_version": torch.version.cuda, "thread_environment": {k: os.environ.get(k) for k in THREAD_ENV},
            "pytest_plugin_autoload_disabled": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"}


def child_stage(stage, output):
    """Validate source and resources before importing the pinned numerical runtime."""
    plan = load_plan()
    require_ram(sample("child_before_runtime_import"))
    commit = git("rev-parse", "HEAD")
    source = source_record(plan, commit, time.time() + 60)
    if not source["verified"]:
        raise RuntimeError("child source differs before numerical import")
    if platform.python_version() != "3.12.8":
        raise RuntimeError("Python version differs from the frozen runtime")
    for key in THREAD_ENV:
        os.environ[key] = "1"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.path.insert(0, str(ROOT))
    import torch
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    if versions != VERSIONS or str(torch.__version__) != VERSIONS["torch"] or torch.version.cuda is not None:
        save_new(output / f"{stage}_runtime_rejected.json", runtime_record(torch, "rejected_version"))
        raise RuntimeError("installed versions differ from the frozen CPU runtime")
    require_ram(sample("child_after_runtime_import"))
    records, thread_cases, skips, issues = [], [], [], []
    result = 1
    try:
        if stage == "legacy_padding_demonstration":
            # This fresh-process CLI sets both Torch controls itself before its
            # first fixture; setting inter-op twice would violate Torch's API.
            records.append(runtime_record(torch, "imported_before_legacy_cli_initializes_threads"))
            target = ROOT / "tests/test_padding_invariant.py"
            sys.argv = [str(target), "--legacy-demonstration", str(output / "legacy_padding.json")]
            runpy.run_path(str(target), run_name="__main__")
            evidence = json.loads((output / "legacy_padding.json").read_text())
            if (evidence.get("status") != "legacy_defect_reproduced" or evidence.get("fixture_only") is not True
                    or len(evidence.get("routes", [])) != 2):
                raise RuntimeError("legacy fixture did not preserve the two declared defect routes")
            result = 0
        else:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            records.append(runtime_record(torch, "before_pytest_collection"))
            import pytest

            class Controls:
                def pytest_collection_finish(self, session):
                    nodeids = [item.nodeid for item in session.items]
                    files = sorted({item.nodeid.split("::", 1)[0] for item in session.items})
                    save_new(output / "test_collection.json", {"nodeids": nodeids, "count": len(nodeids),
                             "files": files, "expected_count": plan["expected_test_count"]})
                    if len(nodeids) != plan["expected_test_count"] or len(nodeids) != len(set(nodeids)) or files != TEST_FILES:
                        raise pytest.UsageError("collected test inventory differs from the frozen plan")

                @pytest.hookimpl(hookwrapper=True, tryfirst=True)
                def pytest_runtest_protocol(self, item, nextitem):
                    torch.set_num_threads(1)
                    if torch.get_num_interop_threads() != 1:
                        raise RuntimeError("inter-op control drift before test")
                    try:
                        yield
                    finally:
                        intra, inter = torch.get_num_threads(), torch.get_num_interop_threads()
                        exception = item.nodeid == THREAD_EXCEPTION and intra == 2 and inter == 1
                        if (intra, inter) != (1, 1) and not exception:
                            issues.append("unexpected thread controls after " + item.nodeid)
                        torch.set_num_threads(1)
                        thread_cases.append({"nodeid": item.nodeid, "before_intra": 1, "before_interop": 1,
                                             "after_fixture_intra": intra, "after_fixture_interop": inter,
                                             "allowed_safety_fixture_exception": exception, "restored_intra": 1})

                def pytest_runtest_logreport(self, report):
                    if report.skipped:
                        skips.append(report.nodeid)
                        if report.nodeid not in ALLOWED_SKIPS:
                            issues.append("undeclared skipped test: " + report.nodeid)

            result = int(pytest.main(["-q", "tests", "--junitxml", str(output / "tests.xml")], plugins=[Controls()]))
            if issues:
                result = result or 1
    finally:
        records.append(runtime_record(torch, "after_stage_or_exception"))
        if (torch.get_num_threads(), torch.get_num_interop_threads()) != (1, 1):
            issues.append("stage did not finish with Torch threads 1/1")
            result = result or 1
        save_new(output / f"{stage}_environment.json", {"records": records, "test_thread_boundaries": thread_cases,
                 "skipped_nodeids": skips, "issues": issues,
                 "limitations": "Thread observations are boundaries, not continuous utilization. The safety compatibility fixture intentionally requests two intra-op threads; numerical fixtures start at one."})
    return result


def run_stage(name, output, samples, deadline):
    before = sample("before_" + name)
    samples.append(before)
    require_ram(before)
    stage_deadline = min(time.time() + 900, deadline - 180)
    if stage_deadline <= time.time():
        raise TimeoutError("no execution budget remains before the final archival reserve")
    command = [sys.executable, str(Path(__file__).resolve()), "--child", name, "--output", str(output)]
    record = {"name": name, "command": command, "started_at_utc": now(),
              "execution_deadline_unix": stage_deadline, "requested_stage_limit_seconds": 900}
    started = time.monotonic()
    log_path = output / f"{name}.log"
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                row = sample("running_" + name)
                samples.append(row)
                if row["available_ram_gib"] < 8 or time.time() >= stage_deadline:
                    record["stop_reason"] = "RAM floor or stage/job deadline"
                    stop_owned(child)
                    break
                time.sleep(min(1, max(.01, stage_deadline - time.time())))
        finally:
            stop_owned(child)
    record.update(completed_at_utc=now(), return_code=child.returncode,
                  wall_seconds_including_shutdown=time.monotonic() - started, log_sha256=sha(log_path))
    save_new(output / f"{name}_status.json", record)
    return record


def archive(output, source_commit, deadline):
    entries = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
               for p in sorted(output.rglob("*")) if p.is_file()]
    save_new(output / "archive_manifest.json", {"schema_version": 1, "item": 8, "final": True,
             "created_at_utc": now(), "source_commit": source_commit, "files": entries})
    git("fetch", "--no-tags", "origin", RESULTS_BRANCH, deadline=deadline)
    base = git("rev-parse", "FETCH_HEAD", deadline=deadline)
    worktree = Path(tempfile.mkdtemp(prefix="neuropixel-item8-archive-")) / "tree"
    git("worktree", "add", "--detach", str(worktree), base, deadline=deadline)
    relative = output.relative_to(ROOT)
    destination = worktree / relative
    if destination.exists():
        raise FileExistsError("run/attempt evidence already exists on the results branch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output, destination)
    git("config", "user.name", "NeuroPixel scientific worker", cwd=worktree, deadline=deadline)
    git("config", "user.email", "neuropixel-worker@users.noreply.github.com", cwd=worktree, deadline=deadline)
    git("add", "--", relative.as_posix(), cwd=worktree, deadline=deadline)
    git("commit", "-m", f"research: archive item 8 CPU validation {output.name}", cwd=worktree, deadline=deadline)
    commit = git("rev-parse", "HEAD", cwd=worktree, deadline=deadline)
    git("push", "origin", f"HEAD:refs/heads/{RESULTS_BRANCH}", cwd=worktree, deadline=deadline)
    print(json.dumps({"event": "evidence_archived", "source_commit": source_commit,
                      "archive_commit": commit, "path": relative.as_posix()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=STAGES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("--child requires --output")
        return child_stage(args.child, args.output.resolve())
    run_key = os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    deadline = float(os.environ["ITEM8_JOB_DEADLINE_UNIX"])
    output = ROOT / "results/research/08_cloud_runs" / run_key
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    commit = git("rev-parse", "HEAD", deadline=deadline)
    samples, stages, plan, error = [sample("worker_admission")], [], None, None
    status = {"schema_version": 1, "item": 8, "started_at_utc": now(), "source_commit": commit,
              "run_key": run_key, "scientific_training_requested": False,
              "scientific_final_performance_scoring_requested": False,
              "bounded_optimizer_and_sampler_fixtures_requested": True,
              "resource_limits": LIMITS, "job_deadline_unix": deadline}
    try:
        require_ram(samples[0])
        with (output / "execution_plan.json").open("xb") as stream:
            stream.write(PLAN_PATH.read_bytes())
        git("archive", "--format=zip", "--output", str(output / "source.zip"), commit, deadline=deadline)
        plan = load_plan()
        before = source_record(plan, commit, deadline)
        save_new(output / "source_before.json", before)
        if not before["verified"]:
            raise RuntimeError("source or committed plan differs before validation")
        for name in STAGES:
            if not source_record(plan, commit, deadline)["verified"]:
                raise RuntimeError("source drift before " + name)
            stages.append(run_stage(name, output, samples, deadline))
            if "stop_reason" in stages[-1]:
                raise RuntimeError(stages[-1]["stop_reason"])
            # An ordinary nonzero fixture/test result does not suppress the
            # second declared stage. Resource, deadline and source stops do.
        if any(stage["return_code"] != 0 for stage in stages):
            raise RuntimeError("one or both declared validation stages failed; both outputs are preserved")
        status["status"] = "completed"
    except BaseException as exc:
        error = exc
        status.update(status="failed", error_type=type(exc).__name__, message=str(exc)[:1500])
    finally:
        try:
            if plan is not None:
                after = source_record(plan, commit, deadline)
                save_new(output / "source_after.json", after)
                if not after["verified"]:
                    raise RuntimeError("source or plan changed during validation")
        except BaseException as exc:
            error = error or exc
            status.update(status="failed", source_after_error=str(exc)[:1500])
        samples.append(sample("worker_finished_before_archive"))
        if samples[-1]["available_ram_gib"] < 8:
            error = error or RuntimeError("RAM floor was violated at final worker observation")
            status.update(status="failed", final_resource_error=str(error))
        status.update(completed_at_utc=now(), worker_wall_seconds_before_archive=time.monotonic() - started,
                      stages=stages, resource_samples=samples,
                      minimum_sampled_available_ram_gib=min(row["available_ram_gib"] for row in samples))
        save_new(output / "worker_status.json", status)
        archive(output, commit, min(deadline, time.time() + 180))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
