"""NeuroPixel: una red cuyo estado entero vive en un lienzo de píxeles.

- Diccionario E (vocab x c_id): cada token tiene un 'color' aprendido de c_id canales.
  El token 0 (vacío) es negro y se queda en cero: sin dato no hay actividad.
- Identidad y activación separadas: el color de entrada se reinyecta en cada paso
  (memoria persistente) y la actividad evoluciona en c canales de estado.
- Dinámica local tipo autómata celular neuronal: cada píxel solo ve sus 8 vecinos.
- Salida: el estado del píxel de salida se traduce con el MISMO diccionario (pesos
  atados), es decir, 'leemos el color' que ha emergido.

El baseline es un transformer pequeño que ve todo el lienzo de golpe (atención global).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Retina(nn.Module):
    """L2 híbrida: 'ojo' que convierte cada píxel de cámara en un color rico (c_id canales)
    mirando su entorno (convoluciones dilatadas, campo ~31 px, sin reducir resolución).
    Su salida vive en el MISMO espacio que el diccionario de palabras."""

    def __init__(self, c_id: int = 16, w: int = 32):
        super().__init__()
        layers, cin = [], 3
        for d in (1, 2, 4, 8):
            layers += [nn.Conv2d(cin, w, 3, padding=d, dilation=d), nn.ReLU()]
            cin = w
        layers.append(nn.Conv2d(w, c_id, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, rgb):
        return self.net(rgb)


class NeuroPixel(nn.Module):
    def __init__(self, vocab: int, out_pos: tuple[int, int], c_id: int = 16, c: int = 48,
                 hidden: int = 128, steps: int = 16, fire_rate: float = 0.5,
                 grounded: tuple[torch.Tensor, torch.Tensor] | None = None, retina: bool = False):
        super().__init__()
        self.retina = Retina(c_id) if retina else None
        self.out_pos, self.steps, self.fire_rate = out_pos, steps, fire_rate
        self.embed = nn.Embedding(vocab, c_id, padding_idx=0)
        # Diccionario anclado: los 3 primeros canales de los conceptos con color típico
        # quedan FIJOS a ese color real; el resto de canales se aprende.
        rgb, mask = grounded if grounded is not None else (torch.zeros(vocab, 3), torch.zeros(vocab, dtype=torch.bool))
        self.register_buffer("g_rgb", rgb.float(), persistent=False)
        self.register_buffer("g_mask", mask.unsqueeze(-1), persistent=False)
        self.seed = nn.Conv2d(c_id, c, 1)
        self.perceive = nn.Conv2d(c, 2 * c, 3, padding=1, groups=c, bias=False)
        self.f1 = nn.Conv2d(3 * c + c_id, hidden, 1)
        self.f2 = nn.Conv2d(hidden, c, 1)
        nn.init.zeros_(self.f2.weight)
        nn.init.zeros_(self.f2.bias)
        self.read = nn.Linear(c, c_id)

    def dictionary(self) -> torch.Tensor:
        """Tabla completa vocab x c_id (con los colores anclados si los hay)."""
        w = self.embed.weight
        return torch.cat([torch.where(self.g_mask, self.g_rgb, w[:, :3]), w[:, 3:]], 1)

    def lens_logits(self, s: torch.Tensor) -> torch.Tensor:
        """Diccionario aplicado a todos los píxeles: B,C,H,W -> B,H,W,vocab."""
        return self.read(s.permute(0, 2, 3, 1)) @ self.dictionary().T

    def forward(self, canvas: torch.Tensor, trace: bool = False, lens_every: int = 0,
                rgb: torch.Tensor | None = None, cam: torch.Tensor | None = None,
                out_pos: tuple[int, int] | None = None, steps: int | None = None, hook=None):
        ids = F.embedding(canvas, self.dictionary()).permute(0, 3, 1, 2)  # B,c_id,H,W (color)
        present = canvas != 0
        if rgb is not None:  # píxeles de cámara: solo el color percibido, sin palabra
            cm = cam.unsqueeze(1)
            if self.retina is not None:   # L2: la retina da el color completo del píxel
                ids = torch.where(cm, self.retina(rgb), ids)
            else:                          # L1: solo el color percibido, en 3 canales
                ids = torch.cat([torch.where(cm, rgb, ids[:, :3]), ids[:, 3:] * ~cm], 1)
            present = present | cam
        s = self.seed(ids) * present.unsqueeze(1)               # activa solo píxeles con dato
        frames, act, lens = [s.detach()] if trace else None, [], []
        for t in range(1, (steps or self.steps) + 1):
            if lens_every and t % lens_every == 0:
                lens.append(self.lens_logits(s))
            h = torch.cat([s, self.perceive(s), ids], 1)
            ds = self.f2(F.relu(self.f1(h)))
            if self.training and self.fire_rate < 1:
                ds = ds * (torch.rand_like(ds[:, :1]) < self.fire_rate)
            s = s + ds
            if hook is not None:          # p. ej. daño a mitad de pensar (autorreparación)
                s = hook(t, s)
            act.append(ds.abs().mean())
            if trace:
                frames.append(s.detach())
        r, c = out_pos or self.out_pos
        logits = self.read(s[:, :, r, c]) @ self.dictionary().T  # traducir con el diccionario
        logits[:, 0] = -1e4                                      # 'vacío' nunca es respuesta
        out = {"logits": logits, "activity": torch.stack(act).mean(), "state": s}
        if lens:
            out["lens"] = torch.stack(lens, 1)                    # B,L,H,W,vocab
        if trace:
            out["frames"] = torch.stack(frames, 1)               # B,T+1,C,H,W
        return out


class TinyTransformer(nn.Module):
    """Referencia fuerte: ve los H*W píxeles como secuencia con posición 2D aprendida."""

    def __init__(self, vocab: int, h: int, w: int, out_pos: tuple[int, int], d: int = 48,
                 layers: int = 2, heads: int = 4, ff: int = 96):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(h * w, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out_idx = out_pos[0] * w + out_pos[1]
        self.head = nn.Linear(d, vocab)

    def forward(self, canvas: torch.Tensor, trace: bool = False, lens_every: int = 0):
        x = self.tok(canvas.flatten(1)) + self.pos
        x = self.norm(self.enc(x))
        logits = self.head(x[:, self.out_idx])
        logits[:, 0] = -1e4
        return {"logits": logits, "activity": torch.zeros(())}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
