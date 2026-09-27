"""Genera submission.csv para el test (180 imágenes) con el mejor modelo y su posprocesado.

    python predict_fil.py --run np_ret [--tta]
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


@torch.no_grad()
def prob_1024(model, img_u8, steps, dev, tta):
    x = fil.norm_img(torch.from_numpy(img_u8)[None].to(dev))
    views = [(0, False)] + ([(1, False), (2, True), (3, True)] if tta else [])
    acc = 0
    for r, fl in views:
        v = torch.rot90(x, r, (2, 3))
        v = v.flip(3) if fl else v
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            p = fil.seg_forward(model, v, steps, ckpt=False).float().softmax(1)[:, 1:2]
        p = p.flip(3) if fl else p
        acc = acc + torch.rot90(p, -r, (2, 3))
    return acc / len(views)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="np_ret")
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()
    dev = choose_device("cuda", threads=4, vram_cap_gib=8)
    rd = HERE / "runs" / a.run
    res = json.loads((rd / "result.json").read_text(encoding="utf-8"))
    args, pp = res["args"], res["best_postproc"]
    model = NeuroPixel(len(fil.VOCAB), (0, 0), c=args["c"], hidden=args["hidden"], retina=not args["no_retina"]).to(dev)
    model.load_state_dict(torch.load(rd / "best.pt", map_location=dev))
    model.eval()
    files = sorted((fil.DATA / "test" / "test_images").glob("*.jp*g"))
    rows, t0 = [], time.time()
    for f in files:
        im = np.asarray(Image.open(f).convert("L").resize((fil.RES, fil.RES), Image.BILINEAR))
        p = prob_1024(model, im, args["steps"], dev, a.tta)
        p2 = F.interpolate(p, size=(2048, 2048), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        lab = fil.instances(p2, pp["thr"], pp["min_area"] * 4, pp["close"] * 2)
        for n in range(1, lab.max() + 1):
            rows.append((f"{f.stem}_{n}", fil.rle_counts(lab == n)))
    out = rd / ("submission_tta.csv" if a.tta else "submission.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)
    print(json.dumps({"archivo": str(out), "imagenes": len(files), "filamentos": len(rows),
                      "por_imagen": round(len(rows) / len(files), 2), "segundos": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
