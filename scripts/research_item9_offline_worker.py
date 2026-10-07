"""Execute the unchanged item-9 offline audits on a bounded public CPU runner.

This is an operational host fallback after the local executor stopped responding.
It never trains a model, regenerates final data, or changes a scientific source.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    if os.environ.get(_name) != "1":
        raise RuntimeError(f"{_name} must be explicitly set to 1 before launch")

import research_item9_worker as ops

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/research/09_offline_plan.json"
S = "83fe135e301f76bc0c74e30c66bb18e067ca5959"
STUDY_KEY = "37593731891-1-study"
RUNTIME = {"numpy": "2.3.5", "scipy": "1.17.0", "matplotlib": "3.10.8", "psutil": "7.2.2"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def source_guard(commit, output, deadline):
    require(ops.git("rev-parse", "HEAD", deadline=deadline) == commit, "driver HEAD moved")
    require(not ops.git("diff", "--no-ext-diff", "--name-only", "HEAD", deadline=deadline),
            "tracked driver source changed")
    require(ops.git("hash-object", PLAN, deadline=deadline) ==
            ops.git("rev-parse", f"{commit}:docs/research/09_offline_plan.json", deadline=deadline),
            "plan differs from its committed bytes")
    own = output.relative_to(ROOT).as_posix() + "/"
    untracked = ops.git("ls-files", "--others", "--exclude-standard", deadline=deadline).splitlines()
    require(all(name.startswith(own) for name in untracked), "unexpected untracked driver files")
    return {"at_utc": ops.now(), "commit": commit, "tracked_source_clean": True,
            "plan_sha256": ops.sha(PLAN), "owned_output_prefix": own,
            "unexpected_untracked_files": []}


def main():
    start = time.monotonic()
    worker_end = start + 1380
    workflow_end = float(os.environ["ITEM9_OFFLINE_DEADLINE_UNIX"])
    worker_end = min(worker_end, start + workflow_end - time.time())
    execution_end = worker_end - 180
    commit = os.environ["GITHUB_SHA"]
    key = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-offline"
    output = ROOT / "results/research/09_offline_runs" / key
    output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "item": 9, "operation": "unchanged offline audit host fallback",
             "run_key": key, "source_commit": commit, "scientific_source_commit": S,
             "started_at_utc": ops.now(), "status": "running", "stages": [],
             "paid_compute": False, "gpu_requested": False,
             "resource_policy": {"numerical_threads": 1, "git_threads": 1,
                 "aggregate_active_cpu_threads": 4, "minimum_available_ram_gib": 8,
                 "supervisor_interval_seconds": 1, "heartbeat_interval_seconds": 60,
                 "archive_interval_seconds": 300, "worker_seconds": 1380,
                 "final_archive_reserve_seconds": 180, "workflow_minutes": 30}}
    archive = None
    ops.save(output / "offline_status.json", state)
    try:
        state["admission"] = ops.admit(execution_end, minimum_seconds=60)
        require(platform.python_version() == "3.12.14", "offline Python differs from frozen runtime")
        versions = {name: importlib.metadata.version(name) for name in RUNTIME}
        require(versions == RUNTIME, "offline dependency versions differ")
        plan = load(PLAN)
        require(plan["schema_version"] == 1 and plan["item"] == 9 and plan["status"] == "frozen",
                "invalid offline plan")
        require(plan["scientific_source_commit"] == S and plan["study_run_key"] == STUDY_KEY,
                "scientific source or study identity differs")
        require(plan["offline_analysis_runtime"] == {"python": "3.12.14", "numpy": "2.3.5", "scipy": "1.17.0"},
                "plan changed the frozen analysis runtime")
        for name in ("final_archive_commit", "observed_remote_head"):
            value = plan[name]
            require(len(value) == 40 and all(c in "0123456789abcdef" for c in value),
                    "invalid raw archive identity")
        for path, digest in plan["operational_file_git_blobs"].items():
            require(ops.git("hash-object", ROOT / path, deadline=execution_end) == digest,
                    f"operational source identity differs: {path}")
        ops.save(output / "source_before.json", source_guard(commit, output, execution_end), exclusive=True)
        shutil.copyfile(PLAN, output / "offline_plan.json")
        ops.save(output / "offline_environment.json", {
            "at_utc": ops.now(), "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": versions, "thread_environment": {name: os.environ[name] for name in ops.THREAD_ENV},
            "frozen_analysis_runtime_preserved": True,
            "host_change": "isolated local executor unavailable; standard public ubuntu-24.04 GitHub runner"},
            exclusive=True)
        ops.git("archive", "--format=zip", "--output", output / "driver_source.zip",
                commit, deadline=execution_end)
        ops.admit(execution_end, minimum_seconds=60)
        ops.git("fetch", "--no-filter", "--no-tags", "origin", ops.RESULTS_BRANCH, deadline=execution_end)
        current_remote = ops.git("ls-remote", "--heads", "origin", "refs/heads/" + ops.RESULTS_BRANCH,
                                 deadline=execution_end).split()[0]
        F, H = plan["final_archive_commit"], plan["observed_remote_head"]
        ops.git("merge-base", "--is-ancestor", F, H, deadline=execution_end)
        ops.git("merge-base", "--is-ancestor", H, current_remote, deadline=execution_end)
        bases = Path(tempfile.mkdtemp(prefix="neuropixel-item9-offline-inputs-"))
        frozen, recovered = bases / "scientific-source", bases / "complete-raw-archive"
        ops.git("worktree", "add", "--detach", frozen, S, deadline=execution_end)
        ops.git("worktree", "add", "--detach", recovered, F, deadline=execution_end)
        raw = recovered / "results/research/09_cloud_runs" / STUDY_KEY
        require(load(raw / "archive_manifest.json")["snapshot_kind"] == "final",
                "input is not the final stopped archive")
        frozen_plan = load(frozen / "docs/research/09_study_plan.json")
        require(ops.sha(frozen / "docs/research/09_study_plan.json") ==
                "7e9089ae013abbc4fa042e30606d5d1329dc8b7e89281fcab080addc832c7792",
                "scientific study plan identity changed")
        require(frozen_plan["offline_analysis_runtime"] == plan["offline_analysis_runtime"],
                "runtime scope differs between plans")
        require(len(frozen_plan["implementation_sha256"]) == 83, "scientific binding count differs")
        for path, digest in frozen_plan["implementation_sha256"].items():
            require(ops.sha(frozen / path) == digest, f"bound scientific source changed: {path}")
        log_source = ROOT / plan["job_log_path"]
        require(ops.git("hash-object", log_source, deadline=execution_end) == plan["job_log_git_blob"],
                "supplied complete Actions log identity differs")
        shutil.copyfile(log_source, output / "study_actions_job.log")
        for relative in plan["remote_receipt_paths"]:
            destination = output / "remote_receipts" / Path(relative).name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        ops.save(output / "recovery_receipt.json", {
            "at_utc": ops.now(), "scientific_source_commit": S, "final_archive_commit": F,
            "root_observed_remote_head": H, "runner_observed_remote_head": current_remote,
            "root_anchor_not_replaced": True, "raw_worktree_head": ops.git("rev-parse", "HEAD", cwd=recovered, deadline=execution_end),
            "frozen_worktree_head": ops.git("rev-parse", "HEAD", cwd=frozen, deadline=execution_end),
            "raw_manifest_sha256": ops.sha(raw / "archive_manifest.json"),
            "complete_decoded_job_log_sha256": ops.sha(output / "study_actions_job.log"),
            "frozen_binding_count": len(frozen_plan["implementation_sha256"])}, exclusive=True)
        archive = ops.Archive(output, commit, execution_end)
        archive.publish(deadline=execution_end)
        last_archive = time.monotonic()
        analysis = output / "scientific_audit_01.json"
        stages = [
            ("full_archive_audit", [sys.executable, str(frozen / "scripts/research_verify_item9_archive.py"),
                "--phase", "study", "--source-commit", S, "--archive-commit", F,
                "--source-root", str(frozen), "--input", str(raw),
                "--output", str(output / "full_archive_audit_01.json")], 600,
                output / "full_archive_audit_01.json", "verified"),
            ("gate_archive_audit", [sys.executable, str(ROOT / "scripts/research_item9_gate_archive_audit.py"),
                "--input", str(raw), "--source-commit", S, "--archive-commit", F, "--remote-head", H,
                "--job-log", str(output / "study_actions_job.log"),
                "--output", str(output / "gate_archive_audit_01.json")], 600,
                output / "gate_archive_audit_01.json", "verified"),
            ("scientific_audit", [sys.executable, str(frozen / "scripts/research_complex_binding_audit.py"),
                "--input", str(raw / "study"), "--recipe", str(frozen / "docs/research/09_experiment_recipe.json"),
                "--source-root", str(frozen), "--output", str(analysis)], 900, analysis, "verified"),
            ("figures", [sys.executable, str(ROOT / "scripts/research_item9_figures.py"),
                "--input", str(analysis), "--output-dir", str(output / "figures"),
                "--run-id", "37593731891"], 300, output / "figures/figure_receipt.json", "rendered")]
        for name, command, cap, receipt, expected_status in stages:
            admission = ops.admit(execution_end, minimum_seconds=60)
            source_guard(commit, output, execution_end)
            boundary = min(execution_end, time.monotonic() + cap)
            record = {"name": name, "started_at_utc": ops.now(), "status": "running",
                      "command": command, "admission": admission, "maximum_seconds": cap}
            state["stages"].append(record)
            ops.save(output / "offline_status.json", state)
            child = supervisor = None
            stage_start = time.monotonic()
            log_path = output / f"{name}.log"
            try:
                with log_path.open("xb") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    supervisor = ops.Supervisor(child, boundary, output, name)
                    supervisor.thread.start()
                    while supervisor.poll() is None:
                        if time.monotonic() - last_archive >= 300:
                            archive.publish(deadline=execution_end)
                            last_archive = time.monotonic()
                        time.sleep(.2)
                    supervisor.close()
                    record["return_code"] = child.returncode
                    record["supervision_failure"] = supervisor.failure
                    require(child.returncode == 0 and supervisor.failure is None,
                            f"{name} failed; preserve the first receipt and log")
                checked = load(receipt)
                require(checked.get("status") == expected_status and not checked.get("issues"),
                        f"{name} receipt did not verify")
                record["status"] = "completed"
                record["receipt_sha256"] = ops.sha(receipt)
                if name == "scientific_audit":
                    # Keep the complete original; this is a lossless display projection
                    # except for an explicitly omitted copy of the already-saved indices.
                    projected = dict(checked)
                    projected["bootstrap"] = dict(checked["bootstrap"])
                    index_rows = projected["bootstrap"].pop("indices")
                    require(len(index_rows) == 2000 and all(len(row) == 256 for row in index_rows),
                            "summary projection received an unexpected bootstrap shape")
                    ops.save(output / "scientific_audit_summary.json", {
                        "schema_version": 1, "item": 9,
                        "full_report": {"path": analysis.name, "bytes": analysis.stat().st_size,
                                        "sha256": ops.sha(analysis)},
                        "projection": "All original top-level data retained; only bootstrap.indices omitted here. Complete indices remain in the unchanged full report.",
                        "omitted_indices_shape": [2000, 256], "analysis": projected}, exclusive=True)
                    with (output / "scientific_audit_01.json.gz").open("xb") as stream:
                        stream.write(gzip.compress(analysis.read_bytes(), mtime=0))
                        stream.flush()
                        os.fsync(stream.fileno())
                    costs = {}
                    for run_id in checked["final_recount"]:
                        trained = load(raw / "study/runs" / run_id / "run.json")
                        evaluated = load(raw / "study/final_predictions" / (run_id + ".json"))
                        costs[run_id] = {
                            "training_summary_sha256": ops.sha(raw / "study/runs" / run_id / "run.json"),
                            "final_summary_sha256": ops.sha(raw / "study/final_predictions" / (run_id + ".json")),
                            "trainable_parameters": trained["configuration"]["trainable_parameters"],
                            "updates_completed": trained["updates_completed"],
                            "learning_rate": trained["learning_rate"],
                            "run_wall_seconds": trained["wall_seconds"],
                            "final_evaluation_wall_seconds": evaluated["evaluation_wall_seconds"]}
                    ops.save(output / "operational_costs.json", {
                        "schema_version": 1, "item": 9, "runs": costs,
                        "scope": "Exact saved values, no new timing. Run wall includes construction/training/checkpoint/development evaluation. Final evaluation wall includes prediction/output/metrics, excludes model load and later summary write. Neither is pure inference latency or energy."},
                        exclusive=True)
            except BaseException as error:
                record["status"] = "failed"
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                if supervisor is not None:
                    supervisor.terminate()
                    supervisor.close()
                elif child is not None and child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                raise
            finally:
                record["completed_at_utc"] = ops.now()
                record["wall_seconds_including_shutdown"] = time.monotonic() - stage_start
                if log_path.exists():
                    record["log_sha256"] = ops.sha(log_path)
                ops.save(output / f"{name}_status.json", record, exclusive=True)
                ops.save(output / "offline_status.json", state)
            archive.publish(deadline=execution_end)
            last_archive = time.monotonic()
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        try:
            state["source_after"] = source_guard(commit, output, worker_end)
        except Exception as error:
            state["status"] = "failed"
            state["source_guard_error"] = {"type": type(error).__name__, "message": str(error)}
        state["completed_at_utc"] = ops.now()
        state["wall_seconds_before_final_archive"] = time.monotonic() - start
        ops.save(output / "offline_status.json", state)
        try:
            if archive is None:
                archive = ops.Archive(output, commit, worker_end)
            archive.publish(deadline=worker_end, final=True, status=state["status"])
        except Exception as error:
            ops.save(output / "archive_failure.json",
                     {"at_utc": ops.now(), "error": str(error), "local_evidence": str(output)}, exclusive=True)
            print(json.dumps({"event": "offline_archive_failed", "error": str(error)}), flush=True)
            return 1
    print(json.dumps({"status": state["status"], "run_key": key, "stages": [
        {"name": row["name"], "status": row["status"]} for row in state["stages"]]}), flush=True)
    return int(state["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
