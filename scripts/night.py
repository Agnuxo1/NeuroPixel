"""Cola nocturna: hasta K trabajos a la vez, por prioridad, con dependencias y vigilante.

    python scripts/night.py --k 6
Vigilante: no lanza nada nuevo con la GPU >= 84 °C; si llega a 88 °C para todo.
Registro en runs/phase3/logs/ y estado en runs/phase3/night_state.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "runs" / "phase3" / "logs"
P = [sys.executable, str(ROOT / "scripts" / "phase3.py")]

# (nombre, argumentos, dependencias)  — orden = prioridad (largos e importantes primero)
JOBS = [
    ("scale_np230k", ["scale", "--arg", "np230k", "--vram-cap", "4"], []),
    ("scale_np105k", ["scale", "--arg", "np105k"], []),
    ("far_neuropixel", ["far", "--arg", "neuropixel", "--vram-cap", "6"], []),
    ("memory", ["memory"], []),
    ("stable_reposo", ["stable", "--arg", "reposo"], []),
    ("far_tf_big", ["far", "--arg", "tf_big"], []),
    ("stable_fijo16", ["stable", "--arg", "fijo16"], []),
    ("far_tf_small", ["far", "--arg", "tf_small"], []),
    ("scale_np30k", ["scale", "--arg", "np30k"], []),
    ("scale_np14k", ["scale", "--arg", "np14k"], []),
    ("scale_np5k", ["scale", "--arg", "np5k"], []),
    ("scale_tf800k", ["scale", "--arg", "tf800k"], []),
    ("scale_tf170k", ["scale", "--arg", "tf170k"], []),
    ("scale_tf44k", ["scale", "--arg", "tf44k"], []),
    ("scale_tf12k", ["scale", "--arg", "tf12k"], []),
    ("grow_grow", ["grow", "--arg", "grow", "--iters", "12000"], []),
    ("grow_seq", ["grow", "--arg", "seq", "--iters", "12000"], []),
    ("energy_1", ["energy", "--arg", "1.0", "--iters", "15000"], []),
    ("energy_3", ["energy", "--arg", "3.0", "--iters", "15000"], []),
    ("energy_0.3", ["energy", "--arg", "0.3", "--iters", "15000"], []),
    ("dream", ["dream", "--iters", "15000"], []),
    ("newword", ["newword"], ["stable_reposo"]),
]


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    t, u, m = (float(x) for x in out.splitlines()[0].split(","))
    return t, u, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in JOBS if not a.only or j[0] in a.only]
    pending, running, done, failed = list(jobs), {}, set(), set()
    t0 = time.time()
    state_f = ROOT / "runs" / "phase3" / "night_state.json"
    while pending or running:
        for name, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                (done if rc == 0 else failed).add(name)
                del running[name]
                print(time.strftime("%H:%M:%S"), "terminó", name, "rc", rc, flush=True)
        temp, util, mem = gpu()
        if temp >= 88:
            print("VIGILANTE: 88 °C, paro todo", flush=True)
            for p in running.values():
                p.terminate()
            break
        while pending and len(running) < a.k and temp < 84:
            ready = [j for j in pending if all(d in done for d in j[2])]
            blocked = [j for j in pending if any(d in failed for d in j[2])]
            for j in blocked:
                pending.remove(j)
                failed.add(j[0])
            if not ready:
                break
            name, args, _ = ready[0]
            pending.remove(ready[0])
            log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
            running[name] = subprocess.Popen(P + args, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            print(time.strftime("%H:%M:%S"), "lanzado", name, flush=True)
            time.sleep(5)
        state = {"min": round((time.time() - t0) / 60, 1), "temp": temp, "util": util, "mem_mib": mem,
                 "corriendo": list(running), "hechos": sorted(done), "fallidos": sorted(failed),
                 "pendientes": [j[0] for j in pending]}
        state_f.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
        time.sleep(30)
    print("FIN NOCHE", json.dumps({"hechos": sorted(done), "fallidos": sorted(failed)}), flush=True)


if __name__ == "__main__":
    main()
