"""Tarea 1: asignación de papeles ("quién hizo qué a quién y dónde").

En el lienzo negro se escriben parejas de píxeles horizontales [PAPEL][RELLENO] en filas
y columnas aleatorias. En la esquina inferior derecha se escribe un píxel de consulta con
un PAPEL; la red debe hacer aparecer en el píxel de salida (a su derecha) el RELLENO
correspondiente. Como las parejas cambian de sitio, no basta con memorizar posiciones:
la información debe viajar por el lienzo y ligarse papel-relleno.

Generalización: una parte de las tripletas (agente, acción, paciente) nunca se ve en
entrenamiento; en test solo aparecen esas combinaciones nuevas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

ROLES = ["AGENTE", "ACCION", "PACIENTE", "LUGAR"]
NOUNS = ["perro", "gato", "niña", "robot", "médico", "zorro", "abuelo", "pájaro",
         "pintora", "lobo", "cocinero", "caballo"]
VERBS = ["muerde", "empuja", "dibuja", "persigue", "saluda", "cura", "lava", "mira",
         "abraza", "llama"]
PLACES = ["casa", "bosque", "playa", "calle", "escuela", "río", "mercado", "huerto"]


# Color perceptivo típico de cada concepto (idea de Fran: el diccionario anclado a lo que
# se ve). Solo para conceptos con un color característico; las acciones, los papeles y
# las personas no llevan color (evita estereotipos y colores inventados).
GROUNDED_RGB = {
    "perro": (139, 90, 43), "gato": (128, 128, 128), "robot": (170, 170, 180),
    "zorro": (200, 90, 30), "lobo": (110, 110, 120), "caballo": (120, 70, 40),
    "bosque": (34, 100, 34), "playa": (238, 214, 175), "calle": (90, 90, 90),
    "río": (60, 110, 160), "huerto": (80, 140, 60),
}


@dataclass
class Vocab:
    tokens: list[str] = field(default_factory=lambda: ["<vacío>"] + ROLES + NOUNS + VERBS + PLACES)

    def __post_init__(self):
        self.idx = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def ids(self, words):
        return [self.idx[w] for w in words]

    def grounded(self):
        """(rgb [V,3] en [-1,1], máscara [V]) con el color perceptivo de cada token."""
        rgb = torch.zeros(len(self), 3)
        mask = torch.zeros(len(self), dtype=torch.bool)
        for t, c in GROUNDED_RGB.items():
            rgb[self.idx[t]] = torch.tensor(c) / 127.5 - 1
            mask[self.idx[t]] = True
        return rgb, mask


class RoleTask:
    def __init__(self, h: int = 8, w: int = 8, heldout_frac: float = 0.2, seed: int = 0):
        assert h >= 5 and w >= 4
        self.h, self.w, self.v = h, w, Vocab()
        g = torch.Generator().manual_seed(seed)
        triples = [(a, b, c) for a in range(len(NOUNS)) for b in range(len(VERBS))
                   for c in range(len(NOUNS)) if a != c]
        perm = torch.randperm(len(triples), generator=g).tolist()
        n_test = int(len(triples) * heldout_frac)
        self.test_triples = [triples[i] for i in perm[:n_test]]
        self.train_triples = [triples[i] for i in perm[n_test:]]
        self.role_ids = torch.tensor(self.v.ids(ROLES))
        self.query_pos = (h - 1, w - 2)
        self.out_pos = (h - 1, w - 1)

    def sample(self, batch: int, split: str = "train", generator: torch.Generator | None = None):
        """Devuelve (lienzo [B,H,W] de IDs, objetivo [B])."""
        trip = self.train_triples if split == "train" else self.test_triples
        g = generator
        canvas = torch.zeros(batch, self.h, self.w, dtype=torch.long)
        target = torch.zeros(batch, dtype=torch.long)
        ti = torch.randint(len(trip), (batch,), generator=g)
        places = torch.randint(len(PLACES), (batch,), generator=g)
        qrole = torch.randint(len(ROLES), (batch,), generator=g)
        for b in range(batch):
            a, vb, p = trip[ti[b]]
            fillers = [self.v.idx[NOUNS[a]], self.v.idx[VERBS[vb]], self.v.idx[NOUNS[p]],
                       self.v.idx[PLACES[places[b]]]]
            rows = torch.randperm(self.h - 1, generator=g)[: len(ROLES)]
            for k, r in enumerate(rows.tolist()):
                c = int(torch.randint(self.w - 1, (1,), generator=g))
                canvas[b, r, c] = self.role_ids[k]
                canvas[b, r, c + 1] = fillers[k]
            canvas[b, self.query_pos[0], self.query_pos[1]] = self.role_ids[qrole[b]]
            target[b] = fillers[qrole[b]]
        return canvas, target
