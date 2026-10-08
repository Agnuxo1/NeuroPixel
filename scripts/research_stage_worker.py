"""Run the item-5 compatibility gate and replication under the shared GPU queue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("GPUQ_HOLDER") != "1":
        raise RuntimeError("This GPU stage requires the shared gpuq reservation")
    root, output = Path(__file__).resolve().parents[1], args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    status_path = output.parent / (output.name + "_worker_status.json")
    if status_path.exists():
        raise RuntimeError("This worker attempt exists; preserve it and review recovery")
    cache = output.parent / "runtime_cache"
    cache.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUTF8="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               TMP=str(cache), TEMP=str(cache), CUDA_CACHE_PATH=str(cache / "cuda"))
    state = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
             "queue_name": env.get("GPUQ_NAME"), "steps": [], "status": "starting"}
    def persist():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, status_path)
    persist()
    commands = [
        ("compatibility_tests", [sys.executable, "-m", "pytest", "-o", "addopts=", "-q"]),
        ("item5_replication", [sys.executable, "scripts/research_replicate.py", "--device", "cuda",
                               "--threads", "4", "--minimum-ram-gib", "8", "--output", str(output)]),
    ]
    for label, command in commands:
        state["status"] = label
        persist()
        print(json.dumps({"event": "worker_step", "step": label, "command": command}), flush=True)
        rc = subprocess.call(command, cwd=root, env=env)
        state["steps"].append({"step": label, "returncode": rc,
                               "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        if rc:
            state["status"] = "failed"
            persist()
            return rc
    state.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
    persist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
