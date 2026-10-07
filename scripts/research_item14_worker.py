"""Bounded item-14 public CPU worker with immutable evidence snapshots.

Low-level Git transport and owned-child supervision reuse the completed item-9
implementation without changing it. This worker fixes item-14 manifests and
loads separately frozen, phase-specific commands. Item 14 uses only the probe
phase for constructed repair, retention and input-availability diagnostics.
The inherited train/final machinery is not admitted by the item-14 protocol.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import tempfile
import time

import research_item9_worker as ops
ROOT = Path(__file__).resolve().parents[1]
RESULTS_BRANCH = ops.RESULTS_BRANCH
git, remaining, save, sha, now = ops.git, ops.remaining, ops.save, ops.sha, ops.now
PHASE_STAGES = {"probe": ["contract_tests", "probe"],
    "preflight": ["contract_tests", "preflight"],
    "study": ["contract_tests", "train", "final"], "audit": ["audit"]}
PHASES = tuple(PHASE_STAGES)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_plan(phase):
    path = ROOT / f"docs/research/14_{phase}_plan.json"
    plan = json.loads(path.read_bytes())
    require(plan.get("schema_version") == 1 and plan.get("item") == 14
            and plan.get("phase") == phase and plan.get("status") == "frozen", "invalid frozen item-14 plan")
    freeze = datetime.fromisoformat(plan["freeze_utc"])
    require(freeze.tzinfo is not None and freeze <= datetime.now(timezone.utc), "invalid freeze timestamp")
    require(plan["runtime"]["threads"] in (1, 2) and plan["runtime"]["interop_threads"] == 1,
            "thread contract exceeds resource policy")
    require(plan["limits"]["worker_seconds"] > 240 and plan["limits"]["archive_reserve_seconds"] == 180,
            "finite worker budget and archival reserve required")
    require([s["name"] for s in plan["stages"]] == PHASE_STAGES[phase], "stage inventory/order differs from phase")
    require(all(type(s["seconds"]) is int and 60 <= s["seconds"] <= plan["limits"]["worker_seconds"]
                for s in plan["stages"]), "invalid stage duration")
    if phase != "audit":
        test = plan["stages"][0]
        require(type(test["expected_count"]) is int and test["expected_count"] > 0
                and test["files"] and len(test["files"]) == len(set(test["files"])), "invalid test inventory")
    mapping = plan["implementation_sha256"]
    require({"scripts/research_item14_worker.py", "scripts/research_item9_worker.py"} <= set(mapping),
            "worker implementation bindings missing")
    for name, digest in mapping.items():
        p = Path(name)
        require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == name, "unsafe source path")
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "invalid source digest")
        require(sha(ROOT / name) == digest, "bound implementation differs: " + name)
    return path, plan


def source_guard(phase, commit, output, deadline):
    path, plan = load_plan(phase)
    own = output.relative_to(ROOT).as_posix() + "/"
    untracked = git("ls-files", "--others", "--exclude-standard", "-z", deadline=deadline).split("\0")
    require(all(not p or p.startswith(own) for p in untracked), "unexpected untracked source")
    require(git("rev-parse", "HEAD", deadline=deadline) == commit, "source HEAD moved")
    require(not git("diff", "--no-ext-diff", "--name-only", "HEAD", deadline=deadline), "tracked source changed")
    require(git("hash-object", path, deadline=deadline) == git("rev-parse", f"{commit}:{path.relative_to(ROOT).as_posix()}", deadline=deadline),
            "plan differs from committed blob")
    tracked = set(git("ls-files", "-z", deadline=deadline).split("\0"))
    require(set(plan["implementation_sha256"]) <= tracked, "bound source is untracked")
    return {"at_utc": now(), "source_commit": commit, "plan_sha256": sha(path),
            "implementation_sha256": plan["implementation_sha256"], "tracked_source_clean": True,
            "all_bindings_verified": True, "unexpected_untracked_files": []}


def runtime_setup(plan):
    runtime = plan["runtime"]
    require(platform.python_version() == runtime["python"], "Python version differs")
    versions = {name: importlib.metadata.version(name) for name in runtime["packages"]}
    require(versions == runtime["packages"], "package versions differ")
    require(all(os.environ.get(k) == str(runtime["threads"]) for k in ops.THREAD_ENV),
            "thread environment differs before imports")
    if "torch" in versions:
        import torch
        require(torch.version.cuda is None and str(torch.__version__) == versions["torch"], "CPU Torch differs")
        torch.set_num_threads(runtime["threads"])
        torch.set_num_interop_threads(runtime["interop_threads"])
    return {"at_utc": now(), "python": platform.python_version(), "packages": versions,
            "platform": platform.platform(), "thread_environment": {k: os.environ.get(k) for k in ops.THREAD_ENV},
            "torch_threads": runtime["threads"] if "torch" in versions else None,
            "torch_interop_threads": runtime["interop_threads"] if "torch" in versions else None}


def run_child(phase, stage_name, output):
    _, plan = load_plan(phase)
    source_guard(phase, os.environ["GITHUB_SHA"], output, time.monotonic() + 60)
    ops.admit(time.monotonic() + 60)
    sys.path.insert(0, str(ROOT))
    before = runtime_setup(plan)
    save(output / f"{stage_name}_environment.json", before, exclusive=True)
    stage = next(s for s in plan["stages"] if s["name"] == stage_name)
    if stage_name == "contract_tests":
        import pytest
        events = {"collected": [], "reports": [], "skips": [], "issues": []}
        class Receipt:
            def pytest_collection_finish(self, session):
                events["collected"] = [x.nodeid for x in session.items]
                save(output / "test_collection.json", {"nodeids": events["collected"],
                     "count": len(events["collected"]), "expected_count": stage["expected_count"]}, exclusive=True)
                if (len(events["collected"]) != stage["expected_count"]
                        or len(events["collected"]) != len(set(events["collected"]))):
                    raise pytest.UsageError("test inventory differs")
            def pytest_runtest_logreport(self, report):
                events["reports"].append({"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome})
                if report.skipped:
                    events["skips"].append(report.nodeid)
            def pytest_sessionfinish(self, session, exitstatus):
                events["exitstatus"] = int(exitstatus)
        code = int(pytest.main([*stage["files"], "-q", "-s", "--junitxml", str(output / "tests.xml")], plugins=[Receipt()]))
        if "torch" in plan["runtime"]["packages"]:
            import torch
            require((torch.get_num_threads(), torch.get_num_interop_threads()) ==
                    (plan["runtime"]["threads"], 1), "Torch thread controls changed during tests")
        save(output / "test_results.json", events, exclusive=True)
        return code or int(bool(events["skips"]))
    script = stage["script"]
    require(script in plan["implementation_sha256"], "unbound scientific stage")
    target = ROOT / script
    sys.argv = [str(target), "--phase", stage_name, "--plan", str(ROOT / f"docs/research/14_{phase}_plan.json"),
                "--output", str(output)]
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as error:
        return 0 if error.code is None else error.code if isinstance(error.code, int) else 1
    return 0


class Archive:
    def __init__(self, output, commit, deadline):
        self.output, self.commit = output, commit
        self.relative = output.relative_to(ROOT)
        self.tree = Path(tempfile.mkdtemp(prefix="neuropixel-item14-results-")) / "tree"
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
        manifest = {"schema_version": 1, "item": 14, "created_at_utc": now(),
                    "source_commit": self.commit, "run_key": self.output.name,
                    "snapshot_kind": "final" if final else "interim_partial",
                    "attempt_status": status, "final": final, "files": files,
                    "claim": "Hashes cover the completed copied files. Live interim files may reflect different times or incomplete writes; they are not final study evidence."}
        save(self.destination / ("archive_manifest.json" if final else "interim_manifest.json"), manifest, exclusive=True)
        git("add", "--force", "--", self.relative.as_posix(), cwd=self.tree, deadline=deadline)
        message = f"research: archive item14 {'final' if final else 'partial'} {self.output.name}"
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



def run_stage(phase, stage, output, plan, commit, execution_end, archive, state):
    name = stage["name"]
    if name == "final":
        require(phase == "study" and len(state["stages"]) == 2
                and all(s["status"] == "completed" for s in state["stages"])
                and state["stages"][-1]["name"] == "train", "final must follow completed tests and training")
        gate = json.loads((output / "pre_final_archive_receipt.json").read_bytes())
        require(gate["source_commit"] == commit and gate["run_key"] == output.name
                and gate["training_archive_commit"] == archive.last_commit
                and gate["training_manifest_sha256"] == sha(output / "training_manifest.json"),
                "final gate does not bind the just-archived training inventory")
        require(archive.last_files["training_manifest.json"]["sha256"] == gate["training_manifest_sha256"],
                "training manifest was not retained before final scoring")
    source_guard(phase, commit, output, execution_end)
    deadline = min(execution_end, time.monotonic() + stage["seconds"])
    admission = ops.admit(deadline, minimum_seconds=60)
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
               "--child", name, "--output", str(output)]
    record = {"name": name, "status": "running", "command": command, "started_at_utc": now(),
              "admission": admission, "stage_limit_seconds": stage["seconds"]}
    state["stages"].append(record)
    save(output / "worker_status.json", state)
    supervisor = None
    started = last_archive = time.monotonic()
    log_path = output / f"{name}.log"
    try:
        with log_path.open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            supervisor = ops.Supervisor(child, deadline, output, name)
            supervisor.thread.start()
            while supervisor.poll() is None:
                if supervisor.failure:
                    raise RuntimeError(supervisor.failure)
                if time.monotonic() - last_archive >= 300:
                    last_archive = time.monotonic()
                    archive.publish(deadline=execution_end)
                time.sleep(0.5)
            require(not supervisor.failure, str(supervisor.failure))
            record["return_code"] = child.returncode
            record["status"] = "completed" if child.returncode == 0 else "failed"
    except BaseException as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
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
        save(output / f"{name}_status.json", record, exclusive=True)
        save(output / "worker_status.json", state)
    require(record["status"] == "completed", "stage failed: " + name)
    return record


def run_worker(phase):
    require(os.environ.get("GITHUB_ACTIONS") == "true"
            and os.environ.get("GITHUB_REPOSITORY") == "Agnuxo1/NeuroPixel"
            and os.environ.get("GITHUB_REF_NAME") == ops.SOURCE_BRANCH,
            "worker requires the explicitly authorized isolated Actions branch")
    require(os.environ.get("GITHUB_RUN_ID", "").isdigit()
            and os.environ.get("GITHUB_RUN_ATTEMPT", "").isdigit(), "invalid run/attempt identifiers")
    started = time.monotonic()
    path, plan = load_plan(phase)
    worker_end = min(started + plan["limits"]["worker_seconds"],
                     started + float(os.environ["ITEM14_JOB_DEADLINE_UNIX"]) - time.time())
    execution_end = worker_end - plan["limits"]["archive_reserve_seconds"]
    commit = os.environ["GITHUB_SHA"]
    key = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-{phase}"
    output = ROOT / "results/research/14_cloud_runs" / key
    output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "item": 14, "phase": phase, "source_commit": commit,
        "run_key": key, "started_at_utc": now(), "status": "running", "stages": [],
        "resource_policy": plan["limits"], "paid_compute": False, "gpu_requested": False}
    archive = None
    save(output / "worker_status.json", state)
    try:
        state["admission"] = ops.admit(execution_end, minimum_seconds=60)
        # Only the child imports Torch; this parent remains a transport/supervision process.
        require(platform.python_version() == plan["runtime"]["python"], "parent Python differs")
        require({k: importlib.metadata.version(k) for k in plan["runtime"]["packages"]} == plan["runtime"]["packages"],
                "installed packages differ")
        save(output / "source_before.json", source_guard(phase, commit, output, execution_end), exclusive=True)
        import shutil
        shutil.copyfile(path, output / "execution_plan.json")
        git("archive", "--format=zip", "--output", output / "source.zip", commit, deadline=execution_end)
        archive = Archive(output, commit, execution_end)
        archive.publish(deadline=execution_end)
        for stage in plan["stages"]:
            run_stage(phase, stage, output, plan, commit, execution_end, archive, state)
            published = archive.publish(deadline=execution_end)
            if stage["name"] == "train":
                require("training_manifest.json" in archive.last_files,
                        "completed training must supply its checkpoint inventory before final")
                manifest_hash = sha(output / "training_manifest.json")
                require(archive.last_files["training_manifest.json"]["sha256"] == manifest_hash,
                        "archived training inventory changed")
                # Internal gate binds the completed training inventory before any final-scoring child.
                save(output / "pre_final_archive_receipt.json", {
                    "schema_version": 1, "item": 14, "at_utc": now(), "source_commit": commit,
                    "training_archive_commit": published, "run_key": key,
                    "training_manifest_sha256": manifest_hash,
                    "claim": "Internal fast-forward Git archive confirmation; not independent custody."}, exclusive=True)
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        try:
            state["source_after"] = source_guard(phase, commit, output, worker_end)
        except Exception as error:
            state["status"] = "failed"
            state["source_guard_error"] = {"type": type(error).__name__, "message": str(error)}
        state["final_resource_observation"] = ops.resource_sample("before_final_archive")
        if state["final_resource_observation"]["available_ram_gib"] < 8:
            state["status"] = "failed"
            state["final_ram_floor_violation"] = True
        state.update(completed_at_utc=now(), wall_seconds_before_final_archive=time.monotonic() - started)
        save(output / "worker_status.json", state)
        try:
            if archive is None:
                archive = Archive(output, commit, worker_end)
            archive.publish(deadline=worker_end, final=True, status=state["status"])
        except Exception as error:
            save(output / "archive_failure.json", {"at_utc": now(), "message": str(error)}, exclusive=True)
            print(json.dumps({"event": "item14_archive_failed", "error": str(error)}), flush=True)
            return 1
    print(json.dumps({"event": "item14_worker_complete", "status": state["status"], "run_key": key,
                     "stage_status": {s["name"]: s["status"] for s in state["stages"]}}), flush=True)
    return int(state["status"] != "completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--child")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        require(args.output is not None, "child output required")
        return run_child(args.phase, args.child, args.output.resolve())
    return run_worker(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
