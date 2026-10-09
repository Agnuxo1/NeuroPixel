"""Diagnóstico de FN/FP por tipo (FIL-005): ¿perdidos, fragmentados, fusionados o mal delimitados?"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fil

run = sys.argv[1] if len(sys.argv) > 1 else "ens_ms_learned_cont"
pp = json.loads((HERE / "runs" / run / "val.json").read_text())
probs = np.load(HERE / "runs" / run / "val_probs.npz")
imgs, labs, meta = fil.load_cache()
_, va = fil.split(meta)
fn_t, fp_t = Counter(), Counter()
for k in va:
    gt = np.asarray(labs[k]); pr = fil.instances(probs[str(meta["ann"][k]["img"])].astype(np.float32), pp["thr"], pp["min_area"], pp["close"])
    g_ids = [i for i in np.unique(gt) if i]; p_ids = [i for i in np.unique(pr) if i]
    inter = np.zeros((gt.max() + 1, pr.max() + 1)); np.add.at(inter, (gt.ravel(), pr.ravel()), 1)
    ga, pa = inter.sum(1), inter.sum(0)
    iou = inter / np.maximum(1, ga[:, None] + pa[None, :] - inter)
    for g in g_ids:
        if (iou[g, 1:] > 0.5).any(): continue
        ov = [p for p in p_ids if inter[g, p] > 0.1 * ga[g]]
        if inter[g, 1:].sum() < 0.2 * ga[g]: fn_t["perdido (<20 % cubierto)"] += 1
        elif len(ov) >= 2: fn_t["fragmentado (>=2 trozos)"] += 1
        elif ov and any((inter[[x for x in g_ids if x != g], p] > 0.1 * pa[p]).any() for p in ov): fn_t["fusionado con otro GT"] += 1
        else: fn_t["mal delimitado (1 trozo, IoU<=0,5)"] += 1
    for p in p_ids:
        if (iou[1:, p] > 0.5).any(): continue
        if inter[0, p] > 0.8 * pa[p]: fp_t["espurio (>80 % fondo)"] += 1
        elif sum(inter[g, p] > 0.1 * ga[g] for g in g_ids) >= 2: fp_t["fusion de varios GT"] += 1
        else: fp_t["trozo o mal delimitado"] += 1
print(json.dumps({"run": run, "postproc": pp, "FN": dict(fn_t), "FP": dict(fp_t)}, ensure_ascii=False, indent=1))
