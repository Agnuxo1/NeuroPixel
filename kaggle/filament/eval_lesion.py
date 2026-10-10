"""Evaluación de lesión preregistrada (PREREGISTRO_REPARACION_FILAMENTOS_20261010.md).

    python eval_lesion.py --runs cv0_base cv0_base_s1_rerun1 cv0_base_s2

Por run: carga best.pt + best_ms.pt (los mismos que train_fil.py usa en su evaluación final de heldout),
aplica el post-proceso (thr, min_area, close) de result.json["heldout"] SIN reajustar, evalúa el test con
24 pasos limpio y con lesión (paso 12, keep ~ Bernoulli(0,7) por píxel, semilla 1000+índice de imagen) y
escribe runs/<name>/lesion_eval.json. Ninguna etiqueta del test se usa para elegir nada.
Importar train_fil no ejecuta nada (todo está bajo main()).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import fil  # noqa: E402
import multiscale  # noqa: E402
import split_protocol  # noqa: E402
import train_fil  # noqa: E402
from neuropixel.model import NeuroPixel  # noqa: E402
from neuropixel.safety import choose_device  # noqa: E402

STEPS, DAMAGE_T, KEEP_P, SEED0 = 24, 12, 0.7, 1000
MANIFEST = HERE / "work" / "cv-protocol-20260930" / "fold-0.json"


def keep_mask(img_id, dev):
    """Máscara de supervivencia [1,1,RES,RES]; depende solo del índice de imagen (no del lote ni del run)."""
    g = torch.Generator(device="cpu").manual_seed(SEED0 + int(img_id))
    return (torch.rand(1, 1, fil.RES, fil.RES, generator=g) < KEEP_P).float().to(dev)


def pq_from_probs(probs, labs, meta, idx, cfg):
    """Mismo cálculo de PQ que train_fil.evaluate (suma de s/tp/fp/fn sobre las anotaciones)."""
    thr, mina, close = cfg
    S = TP = FP = FN = 0
    for k in idx:
        pr = fil.instances(probs[meta["ann"][k]["img"]], thr, mina, close)
        s, tp, fp, fn = fil.pq_counts(np.asarray(labs[k]), pr)
        S, TP, FP, FN = S + s, TP + tp, FP + fp, FN + fn
    return S / max(1e-9, TP + 0.5 * FP + 0.5 * FN)


def eval_run(name, imgs, labs, meta, test, dev):
    out_dir = HERE / "runs" / name
    res = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    ho, args = res["heldout"], res["args"]
    cfg = (ho["thr"], ho["min_area"], ho["close"])
    assert args.get("ms") == "learned" and not args.get("unet"), f"{name}: no es un run --ms learned"
    model = NeuroPixel(len(fil.VOCAB), (0, 0), c=args["c"], hidden=args["hidden"],
                       retina=not args["no_retina"], fire_rate=args["fire_rate"]).to(dev)
    ms = multiscale.MultiScale("learned", c=args["c"], hidden=args["hidden"], scale=args["scale"]).to(dev)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev))
    ms.load_state_dict(torch.load(out_dir / "best_ms.pt", map_location=dev))
    model.eval(); ms.eval()
    sf = args["steps_fine"]
    clean_fwd = lambda m, x, st, ckpt=True, damage=None: multiscale.ms_forward(m, ms, x, st, sf, ckpt, damage)

    # limpio: la propia función de train_fil (mismo PQ, mismo post-proceso fijado)
    clean, _ = train_fil.evaluate(model, imgs, labs, meta, test, STEPS, dev, [cfg], fwd=clean_fwd)
    pq_clean = clean[cfg]["PQ"]
    model.eval()

    # lesión: damage=(12, keep) como en entrenamiento; la máscara se elige por imagen vía closure
    cur, keeps, keeps_c = {}, [], []

    def les_fwd(m, x, st, ckpt=True, damage=None):
        keep = keep_mask(cur["i"], dev)
        keeps.append(keep.mean().item())
        keeps_c.append(F.interpolate(keep, size=(fil.RES // args["scale"],) * 2, mode="nearest").mean().item())
        return multiscale.ms_forward(m, ms, x, st, sf, ckpt, (DAMAGE_T, keep))

    probs = {}
    for k in test:
        i = meta["ann"][k]["img"]
        if i not in probs:
            cur["i"] = i
            probs[i] = train_fil.predict_prob(model, imgs[i], STEPS, dev, les_fwd)
    pq_les = pq_from_probs(probs, labs, meta, test, cfg)
    pq_les_r = round(pq_les, 4)

    pq_res = ho["PQ"]
    if abs(pq_clean - pq_res) > 1e-4:
        print(f"AVISO {name}: PQ limpio recomputado {pq_clean} != heldout.PQ de result.json {pq_res}", flush=True)
    out = {"name": name, "pq_clean_recomputado": pq_clean, "pq_clean_result": pq_res,
           "pq_lesion": pq_les_r, "ri": round(pq_les / max(1e-9, pq_clean), 4),
           "keep_fraction_media": round(float(np.mean(keeps)), 4),
           "keep_fraction_media_rejilla_gruesa": round(float(np.mean(keeps_c)), 4),
           "n_test": len(test), "n_imagenes_test": len(probs),
           "thr": cfg[0], "min_area": cfg[1], "close": cfg[2]}
    (out_dir / "lesion_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--vram-cap", type=float, default=12)
    ap.add_argument("--force-gpu", action="store_true")
    a = ap.parse_args()
    dev = choose_device("cuda", threads=4, vram_cap_gib=a.vram_cap, force_gpu=a.force_gpu)
    imgs, labs, meta = fil.load_cache()          # sin np.asarray: mmap, como --lowmem
    protocol = split_protocol.load(meta, MANIFEST)
    test = protocol["indices"]["test"]
    with torch.no_grad():
        for name in a.runs:
            eval_run(name, imgs, labs, meta, test, dev)


if __name__ == "__main__":
    main()
