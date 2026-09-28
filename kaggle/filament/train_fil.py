"""Entrena NeuroPixel (retina + lienzo) para segmentar filamentos; valida con PQ por imagen única.

    python train_fil.py --iters 8000 --name np_ret
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import fil  # noqa: E402
import multiscale  # noqa: E402
from neuropixel.model import NeuroPixel, n_params  # noqa: E402
from neuropixel.safety import choose_device  # noqa: E402


def sample_batch(imgs, labs, meta, idx, B, crop, pos_frac, rng, dev, by_img=None):
    xs, ys = [], []
    for _ in range(B):
        k = idx[rng.integers(len(idx))]
        lab = labs[k]
        img = imgs[meta["ann"][k]["img"]]
        if rng.random() < pos_frac:
            yy, xx = np.nonzero(lab)
            j = rng.integers(len(yy))
            cy, cx = yy[j] + rng.integers(-crop // 3, crop // 3), xx[j] + rng.integers(-crop // 3, crop // 3)
        else:
            cy, cx = rng.integers(crop // 2, fil.RES - crop // 2, size=2)
        y0 = int(np.clip(cy - crop // 2, 0, fil.RES - crop))
        x0 = int(np.clip(cx - crop // 2, 0, fil.RES - crop))
        a, b = img[y0:y0 + crop, x0:x0 + crop], (lab[y0:y0 + crop, x0:x0 + crop] > 0)
        if by_img is not None:                      # consenso suave: media de los anotadores de la imagen
            b = np.mean([labs[j][y0:y0 + crop, x0:x0 + crop] > 0 for j in by_img[meta["ann"][k]["img"]]], 0)
        r = rng.integers(4)
        a, b = np.rot90(a, r), np.rot90(b, r)
        if rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        xs.append(np.ascontiguousarray(a))
        ys.append(np.ascontiguousarray(b))
    x = torch.from_numpy(np.stack(xs)).to(dev)
    y = torch.from_numpy(np.stack(ys)).to(dev)
    y = y.float() if by_img is not None else y.long()
    return fil.norm_img(x), y


@torch.no_grad()
def predict_prob(model, img_u8, steps, dev, fwd=None):
    x = fil.norm_img(torch.from_numpy(np.ascontiguousarray(img_u8))[None].to(dev))
    amp_dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=amp_dtype, enabled=dev.type == "cuda"):
        lg = (fwd or fil.seg_forward)(model, x, steps, ckpt=False)
    return lg.float().softmax(1)[0, 1].cpu().numpy()


def evaluate(model, imgs, labs, meta, idx, steps, dev, grid=None, fwd=None):
    """Probabilidades por imagen única y PQ global (todas las anotaciones de validación)."""
    model.eval()
    probs = {}
    for k in idx:
        i = meta["ann"][k]["img"]
        if i not in probs:
            probs[i] = predict_prob(model, imgs[i], steps, dev, fwd)
    grid = grid or [(0.5, 20, 0)]
    out = {}
    for thr, mina, close in grid:
        S = TP = FP = FN = 0
        dices = []
        for k in idx:
            pr = fil.instances(probs[meta["ann"][k]["img"]], thr, mina, close)
            s, tp, fp, fn = fil.pq_counts(np.asarray(labs[k]), pr)
            S, TP, FP, FN = S + s, TP + tp, FP + fp, FN + fn
            dices.append(fil.dice(np.asarray(labs[k]), pr))
        pq = S / max(1e-9, TP + 0.5 * FP + 0.5 * FN)
        out[(thr, mina, close)] = {"PQ": round(pq, 4), "Dice": round(float(np.mean(dices)), 4),
                                   "TP": TP, "FP": FP, "FN": FN}
    model.train()
    return out, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--c", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--no-retina", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-frac", type=float, default=0.7)
    ap.add_argument("--w-pos", type=float, default=8.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--vram-cap", type=float, default=10)
    ap.add_argument("--fire-rate", type=float, default=1.0, help="1.0 = entrenar como se predice")
    ap.add_argument("--force-gpu", action="store_true")
    ap.add_argument("--steps-max", type=int, default=0, help=">steps activa el entrenamiento de reposo")
    ap.add_argument("--damage-p", type=float, default=0.5)
    ap.add_argument("--init", default=None, help="continuar desde runs/<init>/best.pt")
    ap.add_argument("--ms", choices=multiscale.MODES, default=None, help="lienzo multiescala (FIL-002)")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--steps-fine", type=int, default=6, help="pasos del lienzo fino (--ms learned)")
    ap.add_argument("--consensus", action="store_true", help="objetivo = media de anotadores (FIL-005)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filters", action="store_true", help="retina con [limbo, sato, DoG] (FIL-007)")
    ap.add_argument("--sdo", action="store_true", help="retina con [Halfa, AIA 304, HMI] reales (FIL-008)")
    ap.add_argument("--name", default="np_ret")
    a = ap.parse_args()
    dev = choose_device("cuda", threads=4, vram_cap_gib=a.vram_cap, force_gpu=a.force_gpu)
    torch.manual_seed(a.seed)
    imgs, labs, meta = fil.load_cache()
    if a.sdo:                                            # canales solares reales coetaneos (mmap)
        imgs, labs = np.load(fil.CACHE / "imgs_sdo.npy", mmap_mode="r"), np.asarray(labs)
    elif a.filters:                                      # 3 canales filtrados, leidos de disco (mmap)
        imgs, labs = np.load(fil.CACHE / "imgs3.npy", mmap_mode="r"), np.asarray(labs)
    else:
        imgs, labs = np.asarray(imgs), np.asarray(labs)  # a RAM (≈2 GB)
    tr, va = fil.split(meta)
    model = NeuroPixel(len(fil.VOCAB), (0, 0), c=a.c, hidden=a.hidden, retina=not a.no_retina,
                       fire_rate=a.fire_rate).to(dev)
    if a.init:
        model.load_state_dict(torch.load(HERE / "runs" / a.init / "best.pt", map_location=dev))
    ms, fwd = None, fil.seg_forward
    params = list(model.parameters())
    if a.ms:
        ms = multiscale.MultiScale(a.ms, c=a.c, hidden=a.hidden, scale=a.scale).to(dev)
        params += list(ms.parameters())
        fwd = lambda m, x, st, ckpt=True, damage=None: multiscale.ms_forward(m, ms, x, st, a.steps_fine, ckpt, damage)
        if a.init and (HERE / "runs" / a.init / "best_ms.pt").exists():
            ms.load_state_dict(torch.load(HERE / "runs" / a.init / "best_ms.pt", map_location=dev))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.05)
    out_dir = HERE / "runs" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"args": vars(a), "params": n_params(model) + (multiscale.n_extra(ms) if ms else 0), "train": len(tr), "val": len(va)}
    print(json.dumps(info), flush=True)
    rng = np.random.default_rng(a.seed)
    by_img = None
    if a.consensus:
        by_img = {}
        for k in tr:
            by_img.setdefault(meta["ann"][k]["img"], []).append(k)
        first = {}
        for k in tr:                                # una entrada por imagen: muestreo uniforme por imagen
            first.setdefault(meta["ann"][k]["img"], k)
        tr = list(first.values())
    wts = torch.tensor([1.0, a.w_pos], device=dev)
    t0, best, log = time.time(), -1, []
    for it in range(1, a.iters + 1):
        x, y = sample_batch(imgs, labs, meta, tr, a.batch, a.crop, a.pos_frac, rng, dev, by_img)
        steps, dmg = a.steps, None
        if a.steps_max > a.steps:                               # reposo: pasos variables + daño
            steps = int(rng.integers(a.steps, a.steps_max + 1))
            if rng.random() < a.damage_p:
                keep = (torch.rand(x.shape[0], 1, *x.shape[2:], device=dev) >= 0.3).float()
                dmg = (int(rng.integers(2, steps)), keep)
        amp_dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast("cuda", dtype=amp_dtype, enabled=dev.type == "cuda"):
            lg = fwd(model, x, steps, damage=dmg)
        lg = lg.float()
        p = lg.softmax(1)[:, 1]
        yf = y.float()
        dice_l = 1 - (2 * (p * yf).sum() + 1) / (p.sum() + yf.sum() + 1)
        if by_img is not None:
            lp = lg.log_softmax(1)
            ce = -(wts[1] * yf * lp[:, 1] + wts[0] * (1 - yf) * lp[:, 0]).sum() / (wts[1] * yf + wts[0] * (1 - yf)).sum()
        else:
            ce = F.cross_entropy(lg, y, weight=wts)
        loss = ce + dice_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if it % a.eval_every == 0 or it == a.iters:
            res, _ = evaluate(model, imgs, labs, meta, va, a.steps, dev, fwd=fwd)
            if ms:
                ms.train()
            r = next(iter(res.values()))
            rec = {"it": it, "loss": round(loss.item(), 4), **r, "s": round(time.time() - t0)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if r["PQ"] > best:
                best = r["PQ"]
                torch.save(model.state_dict(), out_dir / "best.pt")
                if ms:
                    torch.save(ms.state_dict(), out_dir / "best_ms.pt")
    # ajuste del posprocesado en validación con el mejor modelo
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev))
    if ms:
        ms.load_state_dict(torch.load(out_dir / "best_ms.pt", map_location=dev))
    grid = [(t, m, c) for t in (0.6, 0.75, 0.85, 0.95) for m in (60, 120, 200) for c in (0, 2)]
    res, _ = evaluate(model, imgs, labs, meta, va, a.steps, dev, grid, fwd=fwd)
    bestcfg = max(res, key=lambda k: res[k]["PQ"])
    final = {**info, "log": log, "best_postproc": {"thr": bestcfg[0], "min_area": bestcfg[1], "close": bestcfg[2],
                                                    **res[bestcfg]}, "seconds": round(time.time() - t0)}
    (out_dir / "result.json").write_text(json.dumps(final, indent=1), encoding="utf-8")
    print("RESULT", json.dumps(final["best_postproc"]), flush=True)


if __name__ == "__main__":
    main()
