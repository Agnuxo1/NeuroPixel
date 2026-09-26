"""Escáner del diccionario: traduce a palabra el estado de CADA píxel en CADA paso.

Aplica al lienzo entero la misma lectura que el píxel de salida (read + diccionario).
Ojo: la lectura solo se entrena en el píxel de salida; en el resto de píxeles es una
'lente' interpretativa (qué palabra estaría diciendo esa zona), no una prueba causal.
"""
from __future__ import annotations

import colorsys

import numpy as np
import torch

from .task import NOUNS, PLACES, ROLES, VERBS


@torch.no_grad()
def lens(model, frames: torch.Tensor):
    """frames: T,C,H,W -> (palabra [T,H,W], confianza [T,H,W])."""
    h = model.read(frames.permute(0, 2, 3, 1))                 # T,H,W,c_id
    logits = h @ model.embed.weight.T
    logits[..., 0] = -1e4
    p = logits.softmax(-1)
    conf, word = p.max(-1)
    return word, conf


def token_color(tok: str, vocab_tokens: list[str]) -> tuple[int, int, int]:
    """Color fijo y legible por categoría: papeles gris, sustantivos azul,
    acciones naranja, lugares verde; el tono varía dentro de cada categoría."""
    for group, hue in ((NOUNS, 0.60), (VERBS, 0.08), (PLACES, 0.33)):
        if tok in group:
            k = group.index(tok) / max(1, len(group) - 1)
            r, g, b = colorsys.hls_to_rgb(hue + 0.08 * (k - 0.5), 0.35 + 0.3 * k, 0.8)
            return int(r * 255), int(g * 255), int(b * 255)
    if tok in ROLES:
        v = 150 + 30 * ROLES.index(tok)
        return v, v, v
    return 0, 0, 0


def render(word, conf, canvas, vocab_tokens, steps, query_pos, out_pos, title="", cell=70):
    """Devuelve una imagen PIL con una rejilla por paso; cada celda = palabra leída."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 11)
        big = ImageFont.truetype("arialbd.ttf", 16)
    except OSError:
        font = big = ImageFont.load_default()
    T, H, W = word.shape
    gap, head, foot = 14, 30, 34
    img = Image.new("RGB", (len(steps) * (W * cell + gap) + gap, H * cell + head + foot), (20, 20, 24))
    d = ImageDraw.Draw(img)
    for i, t in enumerate(steps):
        x0 = gap + i * (W * cell + gap)
        d.text((x0, 6), f"paso {t}", fill=(230, 230, 230), font=big)
        for r in range(H):
            for c in range(W):
                tok = vocab_tokens[int(word[t, r, c])]
                a = float(conf[t, r, c])
                base = np.array(token_color(tok, vocab_tokens), float)
                col = tuple(int(v) for v in base * (0.15 + 0.85 * a))
                box = (x0 + c * cell, head + r * cell, x0 + (c + 1) * cell - 2, head + (r + 1) * cell - 2)
                d.rectangle(box, fill=col)
                if canvas[r, c] != 0:
                    d.rectangle(box, outline=(255, 255, 255), width=2)   # dato de entrada
                if (r, c) in (tuple(query_pos), tuple(out_pos)):
                    d.rectangle(box, outline=(255, 220, 0), width=3)     # consulta / salida
                txt = (255, 255, 255) if sum(col) < 380 else (0, 0, 0)
                d.text((box[0] + 4, box[1] + 4), tok[:9], fill=txt, font=font)
                d.text((box[0] + 4, box[1] + 18), f"{a:.2f}", fill=txt, font=font)
    d.text((gap, H * cell + head + 8), title, fill=(230, 230, 230), font=big)
    return img
