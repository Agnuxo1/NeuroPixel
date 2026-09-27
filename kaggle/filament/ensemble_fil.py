"""Combina varios lienzos (con o sin TTA), ajusta el posprocesado en validación y genera el envío.

    python ensemble_fil.py --runs np_ret_v2 np_pure_v2 --tta --val      # mide y guarda la mejor configuración
    python ensemble_fil.py --runs np_ret_v2 np_pure_v2 --tta --test     # escribe runs/ens_<...>/submission.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import fil  # noqa: E402
from neuropixel.model import NeuroPixel  # noqa: E402
from neuropixel.safety import choose_device  # noqa: E402
from predict_fil import prob_1024  # noqa: E402


def load(run, dev):
    res = json.loads((HERE / "runs" / run / "result.json").read_text(encoding="utf-8"))
    a = res["args"]
    m = NeuroPixel(len(fil.VOCAB), (0, 0), c=a["c"], hidden=a["hidden"], retina=not a["no_retina"]).to(dev)
    m.load_state_dict(torch.load(HERE / "runs" / run / "best.pt", map_location=dev))
    return m.eval(), a["steps"]


def ens_prob(models, img_u8, dev, tta):
    return sum(prob_1024(m, img_u8, st, dev, tta) for m, st in models) / len(models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    dev = choose_device("cuda", threads=4, force_gpu=True, vram_cap_gib=6)
    models = [load(r, dev) for r in a.runs]
    name = "ens_" + "+".join(a.runs) + ("_tta" if a.tta else "")
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.val:
        imgs, labs, meta = fil.load_cache()
        _, va = fil.split(meta)
        probs = {}
        for k in va:
            i = meta["ann"][k]["img"]
            if i not in probs:
                probs[i] = ens_prob(models, np.ascontiguousarray(imgs[i]), dev, a.tta)[0, 0].cpu().numpy()
        np.savez_compressed(out / "val_probs.npz", **{str(k): v.astype(np.float16) for k, v in probs.items()})
        best = None
        for thr in (0.6, 0.7, 0.75, 0.8, 0.85):          # zona buena ya conocida
            for mina in (120, 160):
                for close in (0, 2):
                    S = TP = FP = FN = 0
                    for k in va:
                        s, tp, fp, fn = fil.pq_counts(np.asarray(labs[k]), fil.instances(probs[meta["ann"][k]["img"]], thr, mina, close))
                        S, TP, FP, FN = S + s, TP + tp, FP + fp, FN + fn
                    pq = S / max(1e-9, TP + 0.5 * FP + 0.5 * FN)
                    if best is None or pq > best["PQ"]:
                        best = {"PQ": round(pq, 4), "thr": thr, "min_area": mina, "close": close, "TP": TP, "FP": FP, "FN": FN}
        best["segundos"] = round(time.time() - t0)
        (out / "val.json").write_text(json.dumps(best, indent=1), encoding="utf-8")
        print("VAL", name, json.dumps(best), flush=True)
    if a.test:
        pp = json.loads((out / "val.json").read_text(encoding="utf-8"))
        rows = []
        for f in sorted((fil.DATA / "test" / "test_images").glob("*.jp*g")):
            im = np.asarray(Image.open(f).convert("L").resize((fil.RES, fil.RES), Image.BILINEAR))
            p2 = F.interpolate(ens_prob(models, im, dev, a.tta), size=(2048, 2048), mode="bilinear",
                               align_corners=False)[0, 0].cpu().numpy()
            lab = fil.instances(p2, pp["thr"], pp["min_area"] * 4, pp["close"] * 2)
            rows += [(f"{f.stem}_{n}", fil.rle_counts(lab == n)) for n in range(1, lab.max() + 1)]
        with open(out / "submission.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["filament_id", "segmentation_rle"])
            w.writerows(rows)
        print("TEST", name, len(rows), "filamentos", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
