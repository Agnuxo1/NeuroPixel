"""G2.2 runner: 8 GiB admission first, then the Torch contracts and regressions on CPU in the historical GPU Python.

Writes results/research/COMPLETION20261012/G2.2_runtime_receipt.json. Refuses to import Torch below the guard.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = "C:/Python313/python.exe"
ATTEMPT = sys.argv[1] if len(sys.argv) > 1 else "1"
OUT = ROOT / f"results/research/COMPLETION20261012/G2.2_runtime_receipt_attempt{ATTEMPT}.json"
ADMISSION = f"results/research/COMPLETION20261012/G2.2_admission_runtime_attempt{ATTEMPT}.json"


def run(args: list[str], env: dict) -> dict:
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    return dict(command=" ".join(args), exit_code=result.returncode, stdout_tail=result.stdout[-4000:],
                stderr_tail=result.stderr[-4000:])


def main() -> None:
    assert not OUT.exists(), OUT
    admission = subprocess.run([sys.executable, "scripts/completion_resource_admission.py", ADMISSION], cwd=ROOT)
    record = json.loads((ROOT / ADMISSION).read_text(encoding="utf-8"))
    if admission.returncode != 0:
        print(json.dumps(dict(status="refused_resource_guard", admission=record)))
        raise SystemExit(2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8",
               TEMP="D:/PROJECTS/.cognition/neuropixel-completion-20261012/tmp", TMP="D:/PROJECTS/.cognition/neuropixel-completion-20261012/tmp")
    steps = [run([GPU_PYTHON, "-c", "import sys,torch;print(sys.version.split()[0], torch.__version__, torch.get_num_threads())"], env),
             run([GPU_PYTHON, "-m", "unittest", "completion_kit.test_original_forward", "-v"], env),
             run([GPU_PYTHON, "-m", "pytest", "tests/test_g2_runtime_contracts.py", "-v", "-p", "no:cacheprovider"], env),
             run([GPU_PYTHON, "-m", "pytest", "tests/test_core.py", "-q", "-p", "no:cacheprovider"], env)]
    receipt = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(), admission=record, device="CPU (CUDA hidden)",
                   steps=steps, all_passed=all(s["exit_code"] == 0 for s in steps),
                   scope="Runtime contracts of the T0 patch and project regressions in the historical Torch environment on CPU; physical backend checks are G2.3")
    OUT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(all_passed=receipt["all_passed"], exit_codes=[s["exit_code"] for s in steps])))
    raise SystemExit(0 if receipt["all_passed"] else 1)


if __name__ == "__main__":
    main()
