"""Aplica la regla preregistrada de docs/research/PREREGISTRO_CONSENSO_FILAMENTOS_20261010.md.

Uso: python docs/research/analyze_consenso_prereg.py [carpeta_runs]

Lee solo el bloque `heldout` (test externo) de runs/<run>/result.json. La linea RESULT
de los logs es de validacion y no se usa.
"""
import json
import statistics as st
import sys
from pathlib import Path

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/PROJECTS/196_NeuroPixel/kaggle/filament/runs")
PAIRS = {  # semilla: (baseline, consenso)
    0: ("cv0_base", "cv0_cons"),
    1: ("cv0_base_s1_rerun1", "cv0_cons_s1"),
    2: ("cv0_base_s2", "cv0_cons_s2"),
}


def heldout_pq(run):
    p = RUNS / run / "result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["heldout"]["PQ"]


def main():
    deltas = []
    print(f"{'semilla':>7} {'base':>8} {'cons':>8} {'delta':>8}")
    for seed, (base, cons) in PAIRS.items():
        pb, pc = heldout_pq(base), heldout_pq(cons)
        d = None if pb is None or pc is None else pc - pb
        if d is not None:
            deltas.append(d)
        fmt = lambda v: "pend." if v is None else f"{v:.4f}"
        print(f"{seed:>7} {fmt(pb):>8} {fmt(pc):>8} {fmt(d):>8}")
    if len(deltas) < 3:
        print(f"\nINCOMPLETO: {len(deltas)} de 3 pares. La regla no se aplica todavia.")
        return
    mean = st.mean(deltas)
    print(f"\nDelta medio = {mean:+.4f}; todos > 0: {all(d > 0 for d in deltas)}")
    if mean >= 0.010 and all(d > 0 for d in deltas):
        veredicto = "APOYO PRELIMINAR FUERTE: ampliar semillas. No se declara mejora con tres semillas."
    elif mean < 0.005 or (any(d < 0 for d in deltas) and mean < 0.010):
        veredicto = "SIN APOYO: no hay evidencia de mejora del consenso en este protocolo."
    else:
        veredicto = ("NO CONCLUYENTE: zona intermedia, o delta medio >= 0,010 con algun delta <= 0 "
                     "(caso que la regla preregistrada no cubre explicitamente).")
    print(veredicto)


if __name__ == "__main__":
    main()
