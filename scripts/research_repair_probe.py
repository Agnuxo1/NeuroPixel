"""Run two separately frozen item14 components in fresh bounded processes."""
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
    status_path = args.output / "repair_probe_status.json"
    state = {"schema_version": 1, "item": 14, "status": "running", "started_at_utc": ops.now(),
             "source_commit": os.environ["GITHUB_SHA"], "plan_sha256": sha(args.plan), "components": []}
    ops.save(status_path, state, exclusive=True)
    try:
        p = json.loads(args.plan.read_bytes())
        require(p["item"] == 14 and p["phase"] == "probe" and p["status"] == "frozen", "plan identity")
        require(p["evidence_recipe"]["components"] == ["native", "audit"]
                and p["evidence_recipe"]["component_limit_seconds"] == 120, "component policy")
        bindings = p["implementation_sha256"]
        require({k: sha(ROOT / k) for k in bindings} == bindings, "source bindings differ")
        require(all(os.environ.get(k) == "1" for k in ops.THREAD_ENV), "thread environment")
        specs = [("native", "scripts/research_repair_native.py", ["--output", str(args.output / "native")],
                  args.output / "native/report.json"),
                 ("audit", "scripts/research_repair_audit.py",
                  ["--native", str(args.output / "native"), "--output", str(args.output / "audit")],
                  args.output / "audit/audit.json")]
        for name, script, extra, result_path in specs:
            ops.admit(start + 540, minimum_seconds=125)
            require(script in bindings, "unbound component")
            command = [sys.executable, str(ROOT / script), "--plan", str(args.plan), *extra]
            rec = {"name": name, "status": "running", "started_at_utc": ops.now(),
                   "command": command, "limit_seconds": 120}
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
                    code = process.wait(timeout=120)
                rec["return_code"] = code
                require(code == 0, "component failed: " + name)
                result = json.loads(result_path.read_bytes())
                require(result["item"] == 14 and result["status"] == "verified", "component status: " + name)
                if name == "audit":
                    require(result["checks"] > 0 and result["issues"] == [], "independent audit discrepancy")
                else:
                    require(len(result["checks"]) == 8 and all(x["passed"] for x in result["checks"]),
                            "native checks inventory")
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
        state.update(status="completed", source_unchanged=True, optimization_steps=0,
                     trained_checkpoints_loaded=0, new_learning_study_executed=False)
    except BaseException as error:
        state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    state.update(completed_at_utc=ops.now(), wall_seconds=time.monotonic()-start,
                 final_resource_observation=ops.resource_sample("repair_probe_finish"))
    if state["final_resource_observation"]["available_ram_gib"] < 8:
        state.update(status="failed", final_ram_floor_violation=True)
    ops.save(status_path, state)
    print(json.dumps({"status": state["status"], "components": {r["name"]: r["status"] for r in state["components"]}}), flush=True)
    return int(state["status"] != "completed")
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["probe"], required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return execute(p.parse_args())
if __name__ == "__main__":
    raise SystemExit(main())
