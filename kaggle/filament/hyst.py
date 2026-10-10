"""FIL-005: instancias por histéresis (semillas p>hi que crecen por p>lo) frente a umbral simple, en validación.
Aviso: ajustar aquí es optimista (mismo split); la cifra honesta sale de los pliegues de FIL-003."""
import json, sys, itertools
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fil

def instances_hyst(prob, hi=0.6, lo=0.4, min_area=120, close=0):
    lab, n = ndi.label(prob > lo, structure=np.ones((3, 3)))
    if n == 0: return lab
    has_seed = ndi.maximum(prob > hi, lab, index=np.arange(1, n + 1)).astype(bool)
    area = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool); keep[1:] = has_seed & (area >= min_area)
    m = keep[lab]
    if close: m = ndi.binary_closing(m, iterations=close)
    lab, _ = ndi.label(m, structure=np.ones((3, 3)))
    return lab

def pq(probs, labs, meta, va, f):
    S = TP = FP = FN = 0
    cache = {}
    for k in va:
        i = meta["ann"][k]["img"]
        if i not in cache: cache[i] = f(probs[str(i)].astype(np.float32))
        s, tp, fp, fn = fil.pq_counts(np.asarray(labs[k]), cache[i]); S += s; TP += tp; FP += fp; FN += fn
    return round(S / max(1e-9, TP + .5 * FP + .5 * FN), 4), TP, FP, FN

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "ens_ms_learned_cont"
    probs = np.load(HERE / "runs" / run / "val_probs.npz")
    imgs, labs, meta = fil.load_cache(); _, va = fil.split(meta)
    res = {"simple_0.6_120": pq(probs, labs, meta, va, lambda p: fil.instances(p, 0.6, 120, 0))}
    for hi, lo, ma in itertools.product((0.6, 0.7, 0.8), (0.3, 0.4, 0.5), (60, 120)):
        res[f"hyst_{hi}_{lo}_{ma}"] = pq(probs, labs, meta, va, lambda p: instances_hyst(p, hi, lo, ma))
    for k, v in sorted(res.items(), key=lambda kv: -kv[1][0])[:8]: print(k, v)
    (HERE / "runs" / run / "hyst.json").write_text(json.dumps(res, indent=1))
