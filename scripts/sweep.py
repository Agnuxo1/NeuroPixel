"""Barrido en GPU con vigilante: varias ejecuciones a la vez, cortadas si peligra el PC.

    python scripts/sweep.py --iters 30000 --seeds 0 1 2
Comprueba la GPU una vez (libre de verdad) y lanza todas las ejecuciones con tope de VRAM.
Cada 30 s vigila: si la temperatura >= --max-temp o la VRAM libre < --min-free, las para.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.safety import gpu_status  # noqa: E402

CONFIGS = {
    "np": ["--model", "neuropixel"],
    "np_lens": ["--model", "neuropixel", "--lens-aux", "0.3"],
    "tf": ["--model", "transformer"],
}


def gpu_temp() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    return float(out.splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--vram-cap", type=float, default=2.0)
    ap.add_argument("--max-temp", type=float, default=85)
    ap.add_argument("--min-free", type=float, default=1.5)
    ap.add_argument("--tag", default="gpu")
    a = ap.parse_args()

    st = gpu_status()
    if st is None or st.util_pct > 30 or st.free_gib < 8:
        raise SystemExit(f"GPU no libre ({st}); no lanzo nada")
    procs = {}
    for cfg in a.configs:
        for s in a.seeds:
            name = f"{a.tag}_{cfg}_s{s}"
            cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), *CONFIGS[cfg], "--iters", str(a.iters),
                   "--batch", str(a.batch), "--seed", str(s), "--device", "cuda", "--force-gpu",
                   "--vram-cap", str(a.vram_cap), "--threads", "2", "--eval-every", "1000", "--name", name]
            log = open(ROOT / "runs" / f"{name}.out", "w", encoding="utf-8")
            procs[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            time.sleep(2)
    print(f"lanzadas {len(procs)} ejecuciones", flush=True)
    t0 = time.time()
    while any(p.poll() is None for p in procs.values()):
        time.sleep(30)
        st, temp = gpu_status(), gpu_temp()
        alive = sum(p.poll() is None for p in procs.values())
        print(f"[{(time.time()-t0)/60:5.1f} min] vivas {alive}  GPU {st.util_pct:.0f} %  "
              f"libre {st.free_gib:.1f} GiB  {temp:.0f} °C", flush=True)
        if temp >= a.max_temp or st.free_gib < a.min_free:
            print("VIGILANTE: parando todo por seguridad", flush=True)
            for p in procs.values():
                if p.poll() is None:
                    p.terminate()
            break
    summary = {}
    for name in procs:
        f = ROOT / "runs" / name / "result.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            summary[name] = {"final": d["final"]["acc_test_new_combos"], "best": d["best_test"],
                             "s": d["seconds"]}
    (ROOT / "runs" / f"{a.tag}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
