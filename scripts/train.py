"""Entrena NeuroPixel (o la referencia) en la tarea de papeles, de principio a fin.

Por defecto va en CPU con 4 hilos y prioridad baja para no molestar a otros
entrenamientos. Ejemplo:
    python scripts/train.py --model neuropixel --iters 2000
    python scripts/train.py --model transformer --iters 2000
    python scripts/train.py --device cuda ...   (solo si la GPU está libre)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.model import NeuroPixel, TinyTransformer, n_params  # noqa: E402
from neuropixel.safety import choose_device  # noqa: E402
from neuropixel.task import RoleTask  # noqa: E402


@torch.no_grad()
def evaluate(model, task, device, split, n=1024, seed=123):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    canvas, target = task.sample(n, split, g)
    out = model(canvas.to(device))
    model.train()
    return (out["logits"].argmax(-1).cpu() == target).float().mean().item()


def save_trace(model, task, device, path: Path, seed=7):
    """Guarda la evolución del lienzo como tira de imágenes (el 'escáner')."""
    from PIL import Image
    g = torch.Generator().manual_seed(seed)
    canvas, target = task.sample(1, "test", g)
    model.eval()
    with torch.no_grad():
        out = model(canvas.to(device), trace=True)
    model.train()
    fr = out["frames"][0].cpu()                      # T+1,C,H,W
    rgb = fr[:, :3]
    lo, hi = rgb.amin(dim=(0, 2, 3), keepdim=True), rgb.amax(dim=(0, 2, 3), keepdim=True)
    rgb = ((rgb - lo) / (hi - lo + 1e-8) * 255).byte().permute(0, 2, 3, 1).numpy()
    act = fr.abs().mean(1)                           # actividad por píxel
    act = (act / (act.max() + 1e-8) * 255).byte().numpy()
    scale, pad = 24, 4
    t, h, w = act.shape
    strip = np.full((2 * h * scale + pad, t * (w * scale + pad), 3), 40, np.uint8)
    for i in range(t):
        x0 = i * (w * scale + pad)
        strip[: h * scale, x0: x0 + w * scale] = rgb[i].repeat(scale, 0).repeat(scale, 1)
        a = act[i].repeat(scale, 0).repeat(scale, 1)
        strip[h * scale + pad:, x0: x0 + w * scale] = np.stack([a, a // 2, 255 - a], -1) * (a[..., None] > 0)
    Image.fromarray(strip).save(path)
    pred = out["logits"].argmax(-1).item()
    return {"target": task.v.tokens[target.item()], "pred": task.v.tokens[pred]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["neuropixel", "transformer"], default="neuropixel")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--steps", type=int, default=16, help="pasos de dinámica del lienzo")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--activity-l1", type=float, default=0.0, help="penaliza actividad (energía)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force-gpu", action="store_true")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = choose_device(a.device, threads=a.threads, force_gpu=a.force_gpu)
    task = RoleTask(a.size, a.size, seed=a.seed)
    if a.model == "neuropixel":
        model = NeuroPixel(len(task.v), task.out_pos, steps=a.steps)
    else:
        model = TinyTransformer(len(task.v), a.size, a.size, task.out_pos)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.1)
    name = a.name or f"{a.model}_s{a.size}_t{a.steps}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"args": vars(a), "device": str(device), "params": n_params(model),
            "train_triples": len(task.train_triples), "test_triples": len(task.test_triples)}
    print(json.dumps(info, ensure_ascii=False))
    log = open(out_dir / "log.jsonl", "w", encoding="utf-8")
    g = torch.Generator().manual_seed(a.seed + 1)
    t0, best = time.time(), 0.0
    for it in range(1, a.iters + 1):
        canvas, target = task.sample(a.batch, "train", g)
        out = model(canvas.to(device))
        loss = F.cross_entropy(out["logits"], target.to(device)) + a.activity_l1 * out["activity"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % a.eval_every == 0 or it == a.iters:
            rec = {"it": it, "loss": round(loss.item(), 4),
                   "acc_train": round(evaluate(model, task, device, "train"), 4),
                   "acc_test_new_combos": round(evaluate(model, task, device, "test"), 4),
                   "activity": round(out["activity"].item(), 5), "s": round(time.time() - t0, 1)}
            best = max(best, rec["acc_test_new_combos"])
            print(json.dumps(rec))
            log.write(json.dumps(rec) + "\n")
            log.flush()
    final = {**info, "final": rec, "best_test": best, "seconds": round(time.time() - t0, 1),
             "chance_approx": round(1 / 12, 4)}
    if a.model == "neuropixel":
        final["trace"] = save_trace(model, task, device, out_dir / "trace.png")
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "result.json").write_text(json.dumps(final, indent=1, ensure_ascii=False), encoding="utf-8")
    print("RESULT", json.dumps({k: final[k] for k in ("params", "best_test", "seconds")}))


if __name__ == "__main__":
    main()
