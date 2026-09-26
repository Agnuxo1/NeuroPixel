"""Piezas de la fase 3 ('lienzo vivo'): memoria en flujo, composición lejana, referencias."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NeuroPixel, TinyTransformer
from .task import ROLES, RoleTask


# ------------------------------------------------------------------ memoria en flujo
def np_stream(model: NeuroPixel, frames, steps, out_pos=None):
    """Ejecuta el lienzo sobre una secuencia de fotogramas. Cada fotograma solo es visible
    mientras dura: lo anterior tiene que sobrevivir en el ESTADO (memoria sin contexto)."""
    s = None
    for canvas, st in zip(frames, steps):
        ids = F.embedding(canvas, model.dictionary()).permute(0, 3, 1, 2)
        new = model.seed(ids) * (canvas != 0).unsqueeze(1)
        s = new if s is None else s + new
        for _ in range(st):
            h = torch.cat([s, model.perceive(s), ids], 1)
            ds = model.f2(F.relu(model.f1(h)))
            if model.training and model.fire_rate < 1:
                ds = ds * (torch.rand_like(ds[:, :1]) < model.fire_rate)
            s = s + ds
    r, c = out_pos or model.out_pos
    logits = model.read(s[:, :, r, c]) @ model.dictionary().T
    logits[:, 0] = -1e4
    return {"logits": logits, "state": s}


class FrameGRU(nn.Module):
    """Referencia recurrente clásica para memoria: resume cada fotograma y lo pasa a una GRU."""

    def __init__(self, vocab: int, h: int, w: int, e: int = 8, d: int = 48, hid: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab, e)
        self.proj = nn.Linear(h * w * e, d)
        self.gru = nn.GRU(d, hid, batch_first=True)
        self.head = nn.Linear(hid, vocab)

    def forward_frames(self, frames):
        x = torch.stack([F.relu(self.proj(self.emb(f).flatten(1))) for f in frames], 1)
        _, hN = self.gru(x)
        logits = self.head(hN[-1])
        logits[:, 0] = -1e4
        return {"logits": logits}


class MemoryTask:
    """Los 4 hechos [PAPEL][RELLENO] llegan de uno en uno; luego 'delay' fotogramas vacíos;
    al final, solo la pregunta. Hay que recordar sin ver."""

    def __init__(self, task: RoleTask):
        self.t = task

    def sample(self, batch, split, g, device, delay):
        c, y, m = self.t.sample(batch, split, g, device, meta=True)
        B = c.shape[0]
        bi = torch.arange(B, device=c.device)
        frames = []
        for k in range(len(ROLES)):
            f = torch.zeros_like(c)
            r, col = m["rows"][:, k], m["cols"][:, k]
            f[bi, r, col] = c[bi, r, col]
            f[bi, r, col + 1] = c[bi, r, col + 1]
            frames.append(f)
        frames += [torch.zeros_like(c) for _ in range(delay)]
        q = torch.zeros_like(c)
        qr, qc = self.t.query_pos
        q[:, qr, qc] = c[:, qr, qc]
        frames.append(q)
        return frames, y


def memory_steps(delay, fact=3, gap=3, query=10):
    return [fact] * len(ROLES) + [gap] * delay + [query]


# ------------------------------------------------------------------ composición lejana
class RoleTaskFar(RoleTask):
    """Como RoleTask, pero el relleno NO está junto a su papel: misma fila, a 2 o más
    columnas de distancia. La ligadura ya no es 'mi vecino', es 'mi fila'."""

    def sample(self, batch, split="train", generator=None, device="cpu", query_role=None,
               place_pool=None, meta=False):
        c, y, m = super().sample(batch, split, generator, device, query_role, place_pool, meta=True)
        dev = c.device
        g = generator if generator is not None and generator.device == dev else None
        B, R = m["rows"].shape
        bi = torch.arange(B, device=dev).unsqueeze(1).expand(-1, R)
        rows, cols = m["rows"], m["cols"]
        fill_vals = c[bi, rows, cols + 1]
        role_vals = c[bi, rows, cols]
        c[bi, rows, cols] = 0
        c[bi, rows, cols + 1] = 0
        c0 = torch.randint(0, 3, (B, R), generator=g, device=dev)                  # papel a la izquierda
        span = (self.w - 1) - (c0 + 2)                                                 # huecos posibles
        gap = 2 + (torch.rand(B, R, generator=g, device=dev) * (span + 1)).long().clamp(max=span)
        c[bi, rows, c0] = role_vals
        c[bi, rows, c0 + gap] = fill_vals
        qr, qc = self.query_pos
        if meta:
            return c, y, {"rows": rows, "cols": c0, "fill_cols": c0 + gap, "fillers": m["fillers"]}
        return c, y


# ------------------------------------------------------------------ ampliar diccionario (palabra nueva)
def expand_vocab(model, new: int = 1):
    """Copia del modelo con 'new' filas nuevas; devuelve (modelo, parámetros entrenables, id nuevo)."""
    m = copy.deepcopy(model)
    dev = next(m.parameters()).device
    if isinstance(m, NeuroPixel):
        old = m.embed.weight.data
        V = old.shape[0] + new
        emb = nn.Embedding(V, old.shape[1], padding_idx=0).to(dev)
        with torch.no_grad():
            emb.weight[: old.shape[0]] = old
            nouns = old[1:].mean(0)
            emb.weight[old.shape[0]:] = nouns + 0.1 * old[1:].std() * torch.randn(new, old.shape[1], device=dev)
        m.embed = emb
        m.register_buffer("g_rgb", torch.zeros(V, 3, device=dev), persistent=False)
        m.register_buffer("g_mask", torch.zeros(V, 1, dtype=torch.bool, device=dev), persistent=False)
        train = [m.embed.weight]
    else:
        old_t, old_h, old_b = m.tok.weight.data, m.head.weight.data, m.head.bias.data
        V = old_t.shape[0] + new
        d = old_t.shape[1]
        m.tok = nn.Embedding(V, d).to(dev)
        m.head = nn.Linear(d, V).to(dev)
        with torch.no_grad():
            m.tok.weight[: old_t.shape[0]] = old_t
            m.tok.weight[old_t.shape[0]:] = old_t[1:].mean(0)
            m.head.weight[: old_h.shape[0]] = old_h
            m.head.weight[old_h.shape[0]:] = old_h[1:].mean(0)
            m.head.bias[: old_b.shape[0]] = old_b
            m.head.bias[old_b.shape[0]:] = old_b.mean()
        train = [m.tok.weight, m.head.weight, m.head.bias]
    for p in m.parameters():
        p.requires_grad_(False)
    rows = torch.zeros(V, device=dev)
    rows[V - new:] = 1
    for p in train:
        p.requires_grad_(True)
        p.register_hook(lambda gr, rows=rows: gr * (rows[:, None] if gr.dim() == 2 else rows))
    return m, train, V - new
