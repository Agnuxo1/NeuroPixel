"""G2.3: repeat the physical render preflights that use the patched PyTorch model as reference.

Runs the unchanged preflight scripts (fixed seeds, weights and inputs, original tolerances, their own
8 GiB guard) into new folders, then compares statuses, checks and every saved array with the original
receipts produced with the unpatched model (sha256 564045ff...). Run through gpuq only.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = "C:/Python313/python.exe"
NEW = ROOT / f"results/research/G2.3_backend_controls_attempt{sys.argv[1] if len(sys.argv) > 1 else 2}"
CASES = {"RENDER08": ("scripts/preflight_RENDER08.py", ROOT / "results/research/RENDER08_preflight/local_gpu_20261009B"),
         "RENDER09": ("scripts/preflight_RENDER09.py", ROOT / "results/research/RENDER09_preflight/original_20261009A")}


def main() -> None:
    NEW.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    report = dict(model_sha256=hashlib.sha256((ROOT / "neuropixel/model.py").read_bytes()).hexdigest(), cases={})
    for name, (script, original) in CASES.items():
        out = NEW / name
        run = subprocess.run([GPU_PYTHON, script, "--output", str(out)], cwd=ROOT, env=env, capture_output=True, text=True,
                             timeout=1800)
        (NEW / f"{name}_stdout.txt").write_text(run.stdout + "\n--- stderr ---\n" + run.stderr, encoding="utf-8")
        row = dict(exit_code=run.returncode)
        if run.returncode == 0:
            new_receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
            old_receipt = json.loads((original / "receipt.json").read_text(encoding="utf-8"))
            new_arrays, old_arrays = np.load(out / "states.npz"), np.load(original / "states.npz")
            keys = sorted(set(new_arrays.files) | set(old_arrays.files))
            diffs = {}
            for key in keys:
                if key in new_arrays.files and key in old_arrays.files and new_arrays[key].shape == old_arrays[key].shape:
                    diffs[key] = float(np.max(np.abs(new_arrays[key].astype(np.float64) - old_arrays[key].astype(np.float64)))) if new_arrays[key].size else 0.0
                else:
                    diffs[key] = None
            row.update(new_status=new_receipt["status"], new_checks=len(new_receipt["checks"]),
                       new_checks_passed=sum(bool(c.get("passed")) for c in new_receipt["checks"]),
                       original_status=old_receipt["status"], original_checks=len(old_receipt["checks"]),
                       arrays_compared=len(keys), arrays_bitwise_identical=sum(1 for v in diffs.values() if v == 0.0),
                       max_abs_difference=max((v for v in diffs.values() if v is not None), default=None),
                       missing_or_reshaped=[k for k, v in diffs.items() if v is None],
                       physical_gpu=new_receipt.get("physical_gpu"), graphics=new_receipt.get("graphics"))
        report["cases"][name] = row
    ok = all(r.get("new_status") == "passed" and r.get("new_checks") == r.get("original_checks") and not r.get("missing_or_reshaped")
             for r in report["cases"].values())
    report["all_passed"] = ok
    (NEW / "G2.3_backend_controls.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1)[:3000])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
