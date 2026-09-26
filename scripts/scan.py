"""Escanea una red entrenada: palabra leída por el diccionario en cada píxel y paso.

    python scripts/scan.py runs/np_v1 --n 3
Genera runs/<run>/scan_<i>.png y scan_<i>.json (entrada, palabras por paso, respuesta).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.scanner import lens, render  # noqa: E402
from neuropixel.task import ROLES  # noqa: E402
from scripts.eval_roles import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--every", type=int, default=4, help="cada cuántos pasos dibujar")
    a = ap.parse_args()
    torch.set_num_threads(2)
    run = Path(a.run)
    model, task = load(run)
    if not hasattr(model, "read"):
        raise SystemExit("El escáner es para modelos NeuroPixel")
    g = torch.Generator().manual_seed(a.seed)
    canvas, target = task.sample(a.n, "test", g)
    with torch.no_grad():
        out = model(canvas, trace=True)
    toks = task.v.tokens
    for i in range(a.n):
        word, conf = lens(model, out["frames"][i])
        T = word.shape[0]
        steps = sorted(set(list(range(0, T, a.every)) + [T - 1]))
        q = toks[int(canvas[i, task.query_pos[0], task.query_pos[1]])]
        pred = toks[int(out["logits"][i].argmax())]
        pairs = []
        for r in range(task.h - 1):
            for c in range(task.w - 1):
                if toks[int(canvas[i, r, c])] in ROLES:
                    pairs.append(f"{toks[int(canvas[i, r, c])]}={toks[int(canvas[i, r, c + 1])]}")
        title = (f"Frase: {', '.join(pairs)}   |   Pregunta: {q}?   Respuesta: {pred}   "
                 f"(correcta: {toks[int(target[i])]})")
        img = render(word, conf, canvas[i], toks, steps, task.query_pos, task.out_pos, title)
        img.save(run / f"scan_{i}.png")
        dump = {"frase": pairs, "pregunta": q, "respuesta": pred, "correcta": toks[int(target[i])],
                "pasos": {int(t): [[toks[int(w)] for w in row] for row in word[t]] for t in steps}}
        (run / f"scan_{i}.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
        print(title)


if __name__ == "__main__":
    main()
