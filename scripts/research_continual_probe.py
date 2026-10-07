"""Run item12's four bounded components in separate, serial CPU processes.

The enclosing worker may have imported Torch. Saved-output analyses deliberately
run in fresh interpreters, and never import a model or deserialize checkpoints.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_item9_worker as ops
from research_item12_inputs import recover_input

COMPONENTS = ("secondary", "isolation", "secondary_audit", "isolation_audit")
RECIPE = {
    "secondary": {"original_source_commit": "08d0d52edd05da6835e71479f3ba4399fcbeabae",
                  "seeds": [20, 21], "topics": [0, 1, 2], "examples_per_topic": 1024,
                  "examples_per_topic_role": 256, "checkpoints_per_seed": 3,
                  "selected_input_files": 15, "new_model_inference": False,
                  "new_confidence_intervals": False, "FWT": False},
    "isolation": {"seed": 120001, "experts": 8, "reported_sizes": [1, 2, 4, 8],
                  "fixture_examples": 12, "updates_per_expert": 3,
                  "alias_negative_control_updates": 3, "optimizer": "SGD",
                  "learning_rate": 0.01, "model_training_flag": False,
                  "task_competence_assay": False},
    "synthetic_routing": {"examples": 2, "classes": 2, "expert_sizes": [1, 2, 4, 8],
                          "constructed_outputs": True, "argmax_tie": "lowest class index"},
    "components": list(COMPONENTS), "component_limit_seconds": 120,
    "shared_probe_limit_seconds": 540,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    ops.save(path, value)


def read(path):
    return json.loads(Path(path).read_bytes())


def execute(args):
    started = time.monotonic()
    state = {"schema_version": 1, "item": 12, "status": "running",
             "started_at_utc": now(), "components": [],
             "source_commit": os.environ.get("GITHUB_SHA"),
             "plan_sha256": sha(args.plan)}
    status_path = args.output / "continual_probe_status.json"
    require(args.output.is_dir() and not status_path.exists(), "worker output missing or probe already started")
    save(status_path, state)
    try:
        plan = read(args.plan)
        require(args.phase == "probe" and plan["phase"] == "probe" and plan["item"] == 12
                and plan["status"] == "frozen", "wrong phase or frozen plan")
        require(plan["evidence_recipe"] == RECIPE, "evidence recipe differs")
        require(plan["runtime"]["threads"] == plan["runtime"]["interop_threads"] == 1,
                "one numerical/interop thread required")
        require(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "thread environment differs")
        bindings = {name: sha(ROOT / name) for name in plan["implementation_sha256"]}
        require(bindings == plan["implementation_sha256"], "source bindings differ")
        state["implementation_sha256"] = bindings
        deadline = started + RECIPE["shared_probe_limit_seconds"]
        state["admission"] = ops.admit(deadline, minimum_seconds=60)
        raw = recover_input(plan["inputs"]["growth"], args.output, "growth")
        state["input_spec"] = plan["inputs"]["growth"]
        state["input_directory"] = str(raw)
        state["input_sha256"] = {name: sha(raw / name) for name in plan["inputs"]["growth"]["files_sha256"]}
        require(state["input_sha256"] == plan["inputs"]["growth"]["files_sha256"], "recovered inputs differ")
        save(status_path, state)
        definitions = [
            ("secondary", "scripts/research_continual_matrix.py",
             ["--input", str(raw), "--plan", str(args.plan), "--output", str(args.output / "secondary")],
             "secondary/report.json", "completed"),
            ("isolation", "scripts/research_continual_isolation.py",
             ["--output", str(args.output / "isolation")],
             "isolation/isolation_status.json", "completed"),
            ("secondary_audit", "scripts/research_continual_matrix_audit.py",
             ["--input", str(raw), "--plan", str(args.plan),
              "--produced", str(args.output / "secondary"), "--output", str(args.output / "secondary_audit")],
             "secondary_audit/audit.json", "verified"),
            ("isolation_audit", "scripts/research_continual_isolation_audit.py",
             ["--input", str(args.output / "isolation"), "--plan", str(args.plan),
              "--output", str(args.output / "isolation_audit")],
             "isolation_audit/audit.json", "verified"),
        ]
        log_root = args.output / "continual_subprocesses"
        log_root.mkdir(exist_ok=False)
        for name, script, options, report_name, expected_status in definitions:
            require(script in bindings, "unbound component")
            remaining = deadline - time.monotonic()
            require(remaining >= 15, "insufficient shared probe time for next component")
            component_seconds = min(RECIPE["component_limit_seconds"], remaining - 5)
            admission = ops.admit(time.monotonic() + component_seconds, minimum_seconds=10)
            record = {"name": name, "status": "running", "started_at_utc": now(),
                      "admission": admission, "limit_seconds": component_seconds,
                      "command": [sys.executable, str(ROOT / script), *options],
                      "fresh_interpreter": True, "serial": True}
            state["components"].append(record)
            save(status_path, state)
            component_started = time.monotonic()
            process = None
            log_path = log_root / (name + ".log")
            try:
                with log_path.open("x", encoding="utf-8") as log:
                    # Inherit the enclosing owned process group so its supervisor
                    # can terminate this interpreter and any descendants together.
                    process = subprocess.Popen(record["command"], cwd=ROOT, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT)
                    record["pid"] = process.pid
                    code = process.wait(timeout=component_seconds)
                record["return_code"] = code
                require(code == 0, "component returned nonzero: " + name)
                report_path = args.output / report_name
                report = read(report_path)
                require(report["item"] == 12 and report["status"] == expected_status,
                        "component result status differs: " + name)
                record.update(status="completed", result_path=report_name,
                              result_sha256=sha(report_path), result_status=report["status"])
                if name == "secondary_audit":
                    require(report["issues"] == [] and report["checks"] > 0, "secondary audit not clean")
                    record["audit_checks"] = report["checks"]
                if name == "isolation_audit":
                    require(report["issues"] == [] and report["failed_checks"] == 0
                            and report["check_count"] > 0, "isolation audit not clean")
                    record["audit_checks"] = report["check_count"]
            except BaseException as error:
                record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                if process is not None:
                    record["return_code"] = process.returncode
                record.update(completed_at_utc=now(), wall_seconds=time.monotonic() - component_started)
                if log_path.is_file():
                    record["log_sha256"] = sha(log_path)
                save(status_path, state)
        require([r["name"] for r in state["components"]] == list(COMPONENTS)
                and all(r["status"] == "completed" for r in state["components"]), "component inventory incomplete")
        require({name: sha(raw / name) for name in state["input_sha256"]} == state["input_sha256"],
                "raw inputs changed across probe")
        require({name: sha(ROOT / name) for name in bindings} == bindings, "bound source changed across probe")
        secondary, isolation = read(args.output / "secondary/report.json"), read(args.output / "isolation/isolation_report.json")
        require(secondary["source"]["plan_sha256"] == state["plan_sha256"]
                and secondary["source"]["source_commit"] == state["source_commit"],
                "secondary report source differs")
        require(isolation["status"] == "verified" and len(isolation["checks"]) == 10
                and all(c["passed"] for c in isolation["checks"]), "isolation report checks differ")
        state.update(status="completed", input_unchanged=True, source_unchanged=True,
                     full_task_learning_benchmark_executed=False,
                     observed_seeds=[20, 21], new_fixture_optimization_steps=27,
                     contract_optimization_steps_are_separate=True,
                     limitations=[
                         "Previously inspected item6 outputs are reanalysed; no fresh confirmatory final set or FWT baseline.",
                         "Two saved seeds describe the old source commit, including its old PAD implementation.",
                         "The current-source isolation diagnostic establishes no competence on eight different tasks.",
                         "Synthetic routing is a literal counterexample; its numbers are not learned NeuroPixel results.",
                         "Independent programs share the same authorized runner; this is not external laboratory replication."
                     ])
    except BaseException as error:
        state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    state.update(completed_at_utc=now(), wall_seconds=time.monotonic() - started,
                 final_resource_observation=ops.resource_sample("continual_probe_finish"))
    if state["final_resource_observation"]["available_ram_gib"] < 8:
        state.update(status="failed", final_ram_floor_violation=True)
    save(status_path, state)
    print(json.dumps({"status": state["status"],
                      "components": {r["name"]: r["status"] for r in state["components"]}}), flush=True)
    return int(state["status"] != "completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["probe"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
