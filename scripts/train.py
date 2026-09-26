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


@torch.no_grad()
def evaluate_camera(model, task, device, n=2000, seed=321):
    """Transferencia sin ejemplos: el LUGAR llega solo como píxel de cámara (color real con
    ruido) y se pregunta por él. Solo lugares con color típico."""
    from neuropixel.task import GROUNDED_RGB, PLACES
    pool = [task.v.idx[p] for p in PLACES if p in GROUNDED_RGB]
    g = torch.Generator().manual_seed(seed)
    c, y, rgb, cam = task.sample_camera(n, "test", g, "cpu", cam_roles={3: 1.0},
                                        query_role=3, place_pool=pool)
    model.eval()
    pred = model(c.to(device), rgb=rgb.to(device), cam=cam.to(device))["logits"].argmax(-1).cpu()
    model.train()
    names = task.v.tokens
    conf = {}
    for t_, p_ in zip(y.tolist(), pred.tolist()):
        conf.setdefault(names[t_], {}).setdefault(names[p_], 0)
        conf[names[t_]][names[p_]] += 1
    return {"acc": round((pred == y).float().mean().item(), 4), "chance_places": round(1 / len(PLACES), 4),
            "confusion": conf}


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
    ap.add_argument("--lens-aux", type=float, default=0.0,
                    help="peso de la pérdida del diccionario en todos los píxeles con dato")
    ap.add_argument("--grounded", action="store_true",
                    help="diccionario anclado: color real fijo en 3 canales de los conceptos con color típico")
    ap.add_argument("--cam-animals", type=float, default=0.0,
                    help="curso 1: prob. de dar agente/paciente con color típico como píxel de cámara")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force-gpu", action="store_true")
    ap.add_argument("--vram-cap", type=float, default=3.0, help="tope de memoria de GPU en GiB")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = choose_device(a.device, threads=a.threads, force_gpu=a.force_gpu, vram_cap_gib=a.vram_cap)
    task = RoleTask(a.size, a.size, seed=a.seed)
    if a.model == "neuropixel":
        model = NeuroPixel(len(task.v), task.out_pos, steps=a.steps,
                           grounded=task.v.grounded() if a.grounded else None)
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
    g = torch.Generator(device=device).manual_seed(a.seed + 1)
    t0, best = time.time(), 0.0
    for it in range(1, a.iters + 1):
        use_lens = a.lens_aux > 0 and a.model == "neuropixel"
        if a.cam_animals > 0:
            canvas, target, rgb, cam = task.sample_camera(a.batch, "train", g, device,
                                                          cam_roles={0: a.cam_animals, 2: a.cam_animals})
            out = model(canvas, lens_every=4 if use_lens else 0, rgb=rgb, cam=cam)
        else:
            canvas, target = task.sample(a.batch, "train", g, device)
            out = model(canvas, lens_every=4 if use_lens else 0)
        loss = F.cross_entropy(out["logits"], target.to(device)) + a.activity_l1 * out["activity"]
        if use_lens:  # escuela: cada píxel con dato debe seguir 'diciendo' su palabra
            L = out["lens"].shape[1]
            mask = (canvas != 0).unsqueeze(1).expand(-1, L, -1, -1)
            tgt = canvas.unsqueeze(1).expand(-1, L, -1, -1)
            loss = loss + a.lens_aux * F.cross_entropy(out["lens"][mask], tgt[mask])
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
        final["zeroshot_camera_place"] = evaluate_camera(model, task, device)
        print("ZEROSHOT", json.dumps({k: v for k, v in final["zeroshot_camera_place"].items() if k != "confusion"}))
        final["trace"] = save_trace(model, task, device, out_dir / "trace.png")
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "result.json").write_text(json.dumps(final, indent=1, ensure_ascii=False), encoding="utf-8")
    print("RESULT", json.dumps({k: final[k] for k in ("params", "best_test", "seconds")}))


if __name__ == "__main__":
    main()
