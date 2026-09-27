"""Digit Recognizer con el lienzo puro (L1): cada píxel vota el dígito con el diccionario.

La imagen 28x28 entra como píxeles de cámara (sin retina). Tras T pasos locales, la lente del
diccionario se aplica a todos los píxeles y la respuesta es el voto medio (tipo 'self-classifying').
Entrenamiento de reposo: pasos variables + daño aleatorio. Pensado para CPU.

    python train_digits.py --iters 3000 --threads 6
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from neuropixel.model import NeuroPixel, n_params  # noqa: E402
from neuropixel.safety import choose_device  # noqa: E402

VOCAB = ["<vacío>"] + [f"DIGITO_{d}" for d in range(10)]


def load():
    tr = np.loadtxt(HERE / "data" / "train.csv", delimiter=",", skiprows=1, dtype=np.uint8)
    te = np.loadtxt(HERE / "data" / "test.csv", delimiter=",", skiprows=1, dtype=np.uint8)
    y, x = tr[:, 0].astype(np.int64), tr[:, 1:].reshape(-1, 28, 28)
    return x, y, te.reshape(-1, 28, 28)


def to_rgb(x):
    x = torch.as_tensor(x).float() / 127.5 - 1
    return x.unsqueeze(1).expand(-1, 3, -1, -1)


def vote(model, rgb, steps, damage=None):
    """Lienzo puro sobre la imagen; logits de los 10 dígitos por píxel y su media."""
    ids = torch.cat([rgb, rgb.new_zeros(rgb.shape[0], model.embed.embedding_dim - 3, *rgb.shape[2:])], 1)
    s = model.seed(ids)
    for t in range(1, steps + 1):
        h = torch.cat([s, model.perceive(s), ids], 1)
        ds = model.f2(F.relu(model.f1(h)))
        if model.training and model.fire_rate < 1:
            ds = ds * (torch.rand_like(ds[:, :1]) < model.fire_rate)
        s = s + ds
        if damage is not None and t == damage[0]:
            s = s * damage[1]
    lg = model.lens_logits(s)[..., 1:]                  # B,H,W,10 (sin <vacío>)
    return lg.mean((1, 2)), lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--steps-max", type=int, default=20)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--name", default="np_pure_cpu")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--aug", type=int, default=0, help="desplazamiento aleatorio máximo en píxeles")
    a = ap.parse_args()
    dev = choose_device(a.device, threads=a.threads, force_gpu=a.device == "cuda", vram_cap_gib=8)
    torch.manual_seed(0)
    x, y, xt = load()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    va, tr = perm[:4000], perm[4000:]
    model = NeuroPixel(len(VOCAB), (0, 0), c=a.c, hidden=a.hidden, fire_rate=1.0).to(dev)
    tg = lambda arr: to_rgb(arr).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.05)
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"args": vars(a), "params": n_params(model)}), flush=True)

    def evaluate(idx, steps):
        model.eval()
        with torch.no_grad():
            pr = torch.cat([vote(model, tg(x[idx[i:i + 500]]), steps)[0].argmax(-1).cpu() for i in range(0, len(idx), 500)])
        model.train()
        return (pr.numpy() == y[idx]).mean()

    t0, log, best = time.time(), [], 0
    for it in range(1, a.iters + 1):
        b = rng.choice(tr, a.batch, replace=False)
        xb_ = x[b]
        if a.aug:
            dy, dx = rng.integers(-a.aug, a.aug + 1, size=2)
            xb_ = np.roll(xb_, (int(dy), int(dx)), axis=(1, 2))
        rgb = tg(xb_)
        steps = int(rng.integers(a.steps, a.steps_max + 1))
        dmg = None
        if rng.random() < 0.5:
            dmg = (int(rng.integers(2, steps)), (torch.rand(len(b), 1, 28, 28, device=dev) >= 0.3).float())
        mean_lg, lg = vote(model, rgb, steps, dmg)
        yb = torch.as_tensor(y[b]).to(dev)
        # escuela: cada píxel debe decir el dígito, más el voto global
        loss = F.cross_entropy(mean_lg, yb) + 0.3 * F.cross_entropy(lg.flatten(0, 2), yb.repeat_interleave(28 * 28))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if it % 250 == 0 or it == a.iters:
            acc = evaluate(va, a.steps + 4)
            rec = {"it": it, "loss": round(loss.item(), 4), "val_acc": round(float(acc), 4), "s": round(time.time() - t0)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if acc > best:
                best = acc
                torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    robust = {}
    model.eval()
    with torch.no_grad():                                # autorreparación: borrar el 50 % a mitad de pensar
        xb = tg(x[va[:2000]])
        keep = (torch.rand(2000, 1, 28, 28, device=dev) >= 0.5).float()
        for k in (16, 24, 32):
            robust[f"pasos{k}"] = round(float((vote(model, xb, k)[0].argmax(-1).cpu().numpy() == y[va[:2000]]).mean()), 4)
            robust[f"daño50_pasos{k}"] = round(float((vote(model, xb, k, (6, keep))[0].argmax(-1).cpu().numpy() == y[va[:2000]]).mean()), 4)
        pred = torch.cat([vote(model, tg(xt[i:i + 500]), a.steps + 4)[0].argmax(-1).cpu() for i in range(0, len(xt), 500)])
    with open(out / "submission.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ImageId", "Label"])
        w.writerows([(i + 1, int(p)) for i, p in enumerate(pred)])
    res = {"args": vars(a), "params": n_params(model), "log": log, "best_val_acc": round(float(best), 4),
           "robustez": robust, "seconds": round(time.time() - t0)}
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("RESULT", json.dumps({"best_val_acc": res["best_val_acc"], **robust}), flush=True)


if __name__ == "__main__":
    main()
