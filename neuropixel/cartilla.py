"""Curso 2, 'cartilla': imagen real con su palabra debajo, como enseñar a leer a un niño.

Lienzo (H=36, W=32):
  filas 0-31  imagen CIFAR-10 32x32 como píxeles de cámara (solo canales perceptivos)
  fila 33     palabra escrita debajo (token del diccionario), o vacío
  fila 35     píxel de pregunta (¿QUÉ? / ¿ES?) y píxel de salida a su derecha
Modos por muestra: 'cartilla' (imagen + palabra correcta + ¿QUÉ?), 'nombrar' (imagen sola
+ ¿QUÉ?) y 'verificar' (imagen + palabra correcta o falsa + ¿ES? -> SÍ/NO).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .task import NOUNS, PLACES, ROLES, VERBS, Vocab

CIFAR_WORDS = ["avión", "coche", "pájaro", "gato", "ciervo", "perro", "rana", "caballo",
               "barco", "camión"]
SPECIAL = ["¿QUÉ?", "¿ES?", "SÍ", "NO"]
MODES = ["cartilla", "nombrar", "verificar"]


def full_vocab() -> Vocab:
    """Vocabulario común: el de la tarea de papeles (mismos IDs) + cartilla + especiales."""
    base = ["<vacío>"] + ROLES + NOUNS + VERBS + PLACES
    extra = [w for w in CIFAR_WORDS + SPECIAL if w not in base]
    return Vocab(base + extra)


class CartillaTask:
    H, W = 36, 32

    def __init__(self, vocab: Vocab, data_path: str | Path, device="cpu"):
        d = np.load(data_path)
        self.v = vocab
        dev = torch.device(device)
        # imágenes en [-1, 1], ya en el dispositivo (CIFAR entero cabe: 50k*3*32*32 floats16)
        self.x = {"train": torch.from_numpy(d["x_train"]).to(dev),
                  "test": torch.from_numpy(d["x_test"]).to(dev)}
        self.y = {"train": torch.from_numpy(d["y_train"]).to(dev),
                  "test": torch.from_numpy(d["y_test"]).to(dev)}
        self.word_ids = torch.tensor(vocab.ids(CIFAR_WORDS), device=dev)
        self.q_what, self.q_is, self.yes, self.no = vocab.ids(SPECIAL)
        self.word_pos = (33, 14)
        self.query_pos = (35, 30)
        self.out_pos = (35, 31)
        self.dev = dev

    def sample(self, batch: int, split: str = "train", generator: torch.Generator | None = None,
               probs=(0.3, 0.4, 0.3), mode: str | None = None, idx: torch.Tensor | None = None):
        """Devuelve dict con canvas, rgb, cam, target, mode (0/1/2) y clase de la imagen."""
        g = generator if generator is not None and generator.device == self.dev else None
        dev, H, W = self.dev, self.H, self.W
        n = len(self.y[split])
        if idx is None:
            idx = torch.randint(n, (batch,), generator=g, device=dev)
        batch = len(idx)
        img = self.x[split][idx].float() / 127.5 - 1                  # B,3,32,32
        cls = self.y[split][idx]
        if mode is None:
            m = torch.multinomial(torch.tensor(probs, device=dev), batch, True, generator=g)
        else:
            m = torch.full((batch,), MODES.index(mode), device=dev)
        canvas = torch.zeros(batch, H, W, dtype=torch.long, device=dev)
        rgb = torch.zeros(batch, 3, H, W, device=dev)
        cam = torch.zeros(batch, H, W, dtype=torch.bool, device=dev)
        rgb[:, :, :32, :32] = img
        cam[:, :32, :32] = True
        word = self.word_ids[cls]
        # verificar: 50 % palabra falsa (otra clase)
        wrong = (m == 2) & (torch.rand(batch, generator=g, device=dev) < 0.5)
        shift = torch.randint(1, 10, (batch,), generator=g, device=dev)
        shown = torch.where(wrong, self.word_ids[(cls + shift) % 10], word)
        wr, wc = self.word_pos
        canvas[:, wr, wc] = torch.where(m == 1, torch.zeros_like(shown), shown)
        qr, qc = self.query_pos
        canvas[:, qr, qc] = torch.where(m == 2, torch.full_like(word, self.q_is),
                                        torch.full_like(word, self.q_what))
        target = torch.where(m == 2, torch.where(wrong, torch.full_like(word, self.no),
                                                 torch.full_like(word, self.yes)), word)
        return {"canvas": canvas, "rgb": rgb, "cam": cam, "target": target, "mode": m,
                "word": word, "cls": cls}


class TinyCNN(torch.nn.Module):
    """Referencia para 'nombrar': CNN pequeña con parámetros comparables."""

    def __init__(self, n_cls: int = 10, w: int = 24):
        super().__init__()
        c = torch.nn
        self.net = c.Sequential(
            c.Conv2d(3, w, 3, padding=1), c.ReLU(), c.MaxPool2d(2),
            c.Conv2d(w, 2 * w, 3, padding=1), c.ReLU(), c.MaxPool2d(2),
            c.Conv2d(2 * w, 2 * w, 3, padding=1), c.ReLU(), c.AdaptiveAvgPool2d(1),
            c.Flatten(), c.Linear(2 * w, n_cls))

    def forward(self, x):
        return self.net(x)
