"""FIL-005: los FN 'perdidos': ¿pequeños? ¿los marcan también otros anotadores de la misma imagen?"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fil
run = "ens_ms_learned_cont"
pp = json.loads((HERE / "runs" / run / "val.json").read_text())
probs = np.load(HERE / "runs" / run / "val_probs.npz")
imgs, labs, meta = fil.load_cache()
_, va = fil.split(meta)
by_img = {}
for k in va: by_img.setdefault(meta["ann"][k]["img"], []).append(k)
lost, found = [], []
for k in va:
    i = meta["ann"][k]["img"]; gt = np.asarray(labs[k]); p = probs[str(i)].astype(np.float32)
    pr = fil.instances(p, pp["thr"], pp["min_area"], pp["close"]) > 0
    others = [np.asarray(labs[j]) > 0 for j in by_img[i] if j != k]
    for g in [x for x in np.unique(gt) if x]:
        m = gt == g; a = int(m.sum()); cov = (pr & m).sum() / a
        agree = np.mean([(o & m).sum() / a > 0.3 for o in others]) if others else np.nan
        rec = (a, float(p[m].mean()), agree)
        (lost if cov < 0.2 else found).append(rec)
L, F = np.array(lost, float), np.array(found, float)
def s(X): return {"n": len(X), "area_mediana": float(np.median(X[:, 0])), "p_media": round(float(np.mean(X[:, 1])), 3),
                  "acuerdo_otros_anot": round(float(np.nanmean(X[:, 2])), 3), "con_otros_anot": int(np.sum(~np.isnan(X[:, 2])))}
print(json.dumps({"perdidos": s(L), "resto": s(F),
                  "perdidos_p>0.3": int((L[:, 1] > 0.3).sum()), "perdidos_p>0.45": int((L[:, 1] > 0.45).sum())}, indent=1))
