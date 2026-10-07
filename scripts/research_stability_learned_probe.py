"""Run four separately frozen item15 learned-dynamics components in fresh bounded processes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import research_item9_worker as ops
ROOT = Path(__file__).resolve().parents[1]
def require(ok, message):
    if not ok:
        raise RuntimeError(message)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def execute(args):
    start = time.monotonic()
    status_path = args.output / "stability_learned_status.json"
    state = {"schema_version": 1, "item": 15, "status": "running", "started_at_utc": ops.now(),
             "source_commit": os.environ["GITHUB_SHA"], "plan_sha256": sha(args.plan), "components": []}
    ops.save(status_path, state, exclusive=True)
    try:
        p = json.loads(args.plan.read_bytes())
        require(p["item"] == 15 and p["phase"] == "preflight" and p["status"] == "frozen", "plan identity")
        require(p["evidence_recipe"]["components"] == ["recover", "native", "checkpoint_binding", "audit"]
                and p["evidence_recipe"]["component_limits_seconds"] == {"recover": 120, "native": 240, "checkpoint_binding": 60, "audit": 120}, "component policy")
        bindings = p["implementation_sha256"]
        require({k: sha(ROOT / k) for k in bindings} == bindings, "source bindings differ")
        require(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "thread environment")
        specs = [("recover", "scripts/research_stability_learned_recover.py",
                  ["--output", str(args.output / "inputs")], args.output / "inputs/recovery_status.json", 120),
                 ("native", "scripts/research_stability_learned_native.py",
                  ["--inputs", str(args.output / "inputs"), "--output", str(args.output / "native")],
                  args.output / "native/report.json", 240),
                 ("checkpoint_binding", "scripts/research_stability_learned_bind.py",
                  ["--inputs", str(args.output / "inputs"), "--native", str(args.output / "native"),
                   "--output", str(args.output / "binding")], args.output / "binding/checkpoint_binding.json", 60),
                 ("audit", "scripts/research_stability_learned_audit.py",
                  ["--inputs", str(args.output / "inputs"), "--native", str(args.output / "native"),
                   "--binding", str(args.output / "binding"), "--output", str(args.output / "audit")], args.output / "audit/audit.json", 120)]
        for name, script, extra, result_path, seconds in specs:
            ops.admit(start + 600, minimum_seconds=seconds + 5)
            require(script in bindings, "unbound component")
            command = [sys.executable, str(ROOT / script), "--plan", str(args.plan), *extra]
            rec = {"name": name, "status": "running", "started_at_utc": ops.now(),
                   "command": command, "limit_seconds": seconds}
            state["components"].append(rec); ops.save(status_path, state)
            component_start = time.monotonic()
            process = None
            log_path = args.output / (name + "_component.log")
            try:
                with log_path.open("x", encoding="utf-8") as log:
                    # Inherit the worker-owned process group; outer supervisor
                    # terminates this controller and every child together.
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT)
                    rec["pid"] = process.pid
                    code = process.wait(timeout=seconds)
                rec["return_code"] = code
                require(code == 0, "component failed: " + name)
                result = json.loads(result_path.read_bytes())
                require(result["item"] == 15 and result["status"] == "verified", "component status: " + name)
                if name in ("audit", "checkpoint_binding"):
                    require(result["checks"] > 0 and result["issues"] == [], "independent audit discrepancy")
                elif name == "recover":
                    require(len(result["recovered"]) == 22 and result["binary_files_deserialized"] == 0,
                            "recovery inventory")
                else:
                    require(type(result.get("compatibility_gate_passed")) is bool
                            and type(result.get("extended_panel_executed")) is bool
                            and (result["compatibility_gate_passed"] or not result["extended_panel_executed"]),
                            "conditional learned panel status")
                rec.update(status="completed", result_sha256=sha(result_path))
            except BaseException as error:
                rec.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                if process is not None and process.poll() is None:
                    process.kill(); process.wait(timeout=5)
                if process is not None:
                    rec["return_code"] = process.returncode
                rec.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic()-component_start)
                if log_path.exists():
                    rec["log_sha256"] = sha(log_path)
                ops.save(status_path, state)
        require({k: sha(ROOT / k) for k in bindings} == bindings, "source changed during probe")
        native_result = json.loads((args.output / "native/report.json").read_bytes())
        state.update(status="completed", source_unchanged=True, optimization_steps=0,
                     distinct_historical_checkpoints=5, new_learning_study_executed=False,
                     compatibility_gate_passed=native_result["compatibility_gate_passed"],
                     extended_panel_executed=native_result["extended_panel_executed"])
    except BaseException as error:
        state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    state.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic()-start,
                 final_resource_observation=ops.resource_sample("stability_learned_finish"))
    if state["final_resource_observation"]["available_ram_gib"] < 8:
        state.update(status="failed", final_ram_floor_violation=True)
    ops.save(status_path, state)
    print(json.dumps({"status": state["status"], "components": {r["name"]: r["status"] for r in state["components"]}}), flush=True)
    return int(state["status"] != "completed")
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["preflight"], required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return execute(p.parse_args())
if __name__ == "__main__":
    raise SystemExit(main())
