"""Curso 2, cartilla: imagen + palabra en el mismo lienzo, en multitarea con los papeles.

    python scripts/train_cartilla.py --device cuda --iters 20000
    python scripts/train_cartilla.py --model cnn --device cuda --iters 20000   (referencia)
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
from neuropixel.cartilla import CIFAR_WORDS, MODES, CartillaTask, TinyCNN, full_vocab  # noqa: E402
from neuropixel.model import NeuroPixel, n_params  # noqa: E402
from neuropixel.safety import choose_device  # noqa: E402
from neuropixel.task import RoleTask  # noqa: E402


def run_np(model, b, steps, amp):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        return model(b["canvas"], rgb=b["rgb"], cam=b["cam"], out_pos=CartillaTask.out_pos_static,
                     steps=steps)


@torch.no_grad()
def eval_cartilla(model, task, steps, amp, n=2000, cnn=False):
    model.eval()
    res = {}
    idx = torch.arange(n, device=task.dev)
    for mode in (["nombrar"] if cnn else MODES):
        g = torch.Generator(device=task.dev).manual_seed(99)
        b = task.sample(n, "test", g, mode=mode, idx=idx)
        if cnn:
            pred = task.word_ids[model(b["rgb"][:, :, :32, :32]).argmax(-1)]
        else:
            pred = torch.cat([run_np(model, {k: v[i:i + 500] for k, v in b.items()}, steps, amp)["logits"]
                              .argmax(-1) for i in range(0, n, 500)])
        res[mode] = round((pred == b["target"]).float().mean().item(), 4)
    model.train()
    return res


@torch.no_grad()
def eval_roles(model, rtask, device):
    model.eval()
    c, y = rtask.sample(1024, "test", torch.Generator().manual_seed(123))
    acc = (model(c.to(device))["logits"].argmax(-1).cpu() == y).float().mean().item()
    model.train()
    return round(acc, 4)


@torch.no_grad()
def save_scan(model, task, steps, path: Path, n=6):
    """Imagen real a la izquierda; a la derecha, qué palabra 'dice' cada píxel al final."""
    from PIL import Image, ImageDraw, ImageFont
    model.eval()
    g = torch.Generator(device=task.dev).manual_seed(5)
    b = task.sample(n, "test", g, mode="nombrar")
    out = model(b["canvas"], rgb=b["rgb"], cam=b["cam"], out_pos=task.out_pos, steps=steps)
    words = model.lens_logits(out["state"][:, :, :32, :32])
    word_logits = words[..., task.word_ids]                      # solo las 10 palabras
    lens_cls = word_logits.argmax(-1).cpu().numpy()               # B,32,32
    pred = out["logits"].argmax(-1).cpu()
    model.train()
    palette = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212]],
                       np.uint8)
    s, pad = 6, 10
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    img = Image.new("RGB", (n * (2 * 32 * s + 3 * pad), 32 * s + 60 + 22), (20, 20, 24))
    d = ImageDraw.Draw(img)
    names = task.v.tokens
    for i in range(n):
        x0 = i * (2 * 32 * s + 3 * pad) + pad
        im = ((b["rgb"][i, :, :32, :32].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        img.paste(Image.fromarray(im).resize((32 * s, 32 * s), Image.NEAREST), (x0, 30))
        lm = palette[lens_cls[i]]
        img.paste(Image.fromarray(lm).resize((32 * s, 32 * s), Image.NEAREST), (x0 + 32 * s + pad, 30))
        ok = "(bien)" if pred[i].item() == b["target"][i].item() else "(mal)"
        d.text((x0, 8), f"es: {names[b['target'][i]]}   dice: {names[pred[i]]} {ok}", fill=(235, 235, 235), font=font)
        top = np.bincount(lens_cls[i].ravel(), minlength=10).argmax()
        d.text((x0 + 32 * s + pad, 32 * s + 34), f"píxeles dicen sobre todo: {CIFAR_WORDS[top]}",
               fill=(200, 200, 200), font=font)
    for k, w in enumerate(CIFAR_WORDS):  # leyenda
        d.rectangle((pad + k * 90, 32 * s + 58, pad + k * 90 + 12, 32 * s + 70), fill=tuple(int(v) for v in palette[k]))
        d.text((pad + k * 90 + 16, 32 * s + 57), w, fill=(220, 220, 220), font=font)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["neuropixel", "cnn"], default="neuropixel")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48, help="lote de imágenes")
    ap.add_argument("--batch-role", type=int, default=256, help="lote de la tarea de papeles (0 = sin)")
    ap.add_argument("--steps", type=int, default=32, help="pasos del lienzo de la cartilla")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lens-img", type=float, default=0.3, help="escuela: la imagen dice su palabra")
    ap.add_argument("--lens-role", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vram-cap", type=float, default=6.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--force-gpu", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = choose_device(a.device, threads=a.threads, force_gpu=a.force_gpu, vram_cap_gib=a.vram_cap)
    amp = device.type == "cuda"
    vocab = full_vocab()
    task = CartillaTask(vocab, ROOT / "data" / "cifar10.npz", device)
    CartillaTask.out_pos_static = task.out_pos
    rtask = RoleTask(8, 8, seed=a.seed)
    if a.model == "neuropixel":
        model = NeuroPixel(len(vocab), rtask.out_pos, hidden=a.hidden, steps=16)
    else:
        model = TinyCNN()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.05)
    name = a.name or f"cartilla_{a.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"args": vars(a), "device": str(device), "params": n_params(model), "vocab": len(vocab)}
    print(json.dumps(info, ensure_ascii=False), flush=True)
    log = open(out_dir / "log.jsonl", "w", encoding="utf-8")
    g = torch.Generator(device=device).manual_seed(a.seed + 1)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        b = task.sample(a.batch, "train", g)
        if a.model == "cnn":
            loss = F.cross_entropy(model(b["rgb"][:, :, :32, :32]), b["cls"])
        else:
            out = run_np(model, b, a.steps, amp)
            loss = F.cross_entropy(out["logits"].float(), b["target"])
            if a.lens_img > 0:  # escuela: al final, cada píxel de la imagen dice su palabra
                lz = model.lens_logits(out["state"][:, :, :32, :32].float())
                loss = loss + a.lens_img * F.cross_entropy(lz.flatten(0, 2), b["word"].repeat_interleave(32 * 32))
            if a.batch_role > 0:  # multitarea: la tarea de papeles que ya funciona
                c, y = rtask.sample(a.batch_role, "train", g, device)
                ro = model(c, lens_every=4 if a.lens_role > 0 else 0)
                loss = loss + F.cross_entropy(ro["logits"], y)
                if a.lens_role > 0:
                    L = ro["lens"].shape[1]
                    mask = (c != 0).unsqueeze(1).expand(-1, L, -1, -1)
                    tgt = c.unsqueeze(1).expand(-1, L, -1, -1)
                    loss = loss + a.lens_role * F.cross_entropy(ro["lens"][mask], tgt[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % a.eval_every == 0 or it == a.iters:
            rec = {"it": it, "loss": round(loss.item(), 4),
                   **eval_cartilla(model, task, a.steps, amp, cnn=a.model == "cnn"),
                   "s": round(time.time() - t0, 1)}
            if a.model == "neuropixel" and a.batch_role > 0:
                rec["papeles"] = eval_roles(model, rtask, device)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
    final = {**info, "final": rec, "seconds": round(time.time() - t0, 1)}
    final["test_full_nombrar"] = eval_cartilla(model, task, a.steps, amp, n=10000, cnn=a.model == "cnn")["nombrar"]
    if a.model == "neuropixel":
        save_scan(model, task, a.steps, out_dir / "scan_cartilla.png")
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "result.json").write_text(json.dumps(final, indent=1, ensure_ascii=False), encoding="utf-8")
    print("RESULT", json.dumps({"params": info["params"], "nombrar_10k": final["test_full_nombrar"],
                                "seconds": final["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
