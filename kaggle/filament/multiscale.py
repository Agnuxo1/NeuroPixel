"""Lienzo multiescala para filamentos: pensar a 1/s de resolución y reconstruir a resolución completa.

Idea (Fran, "DLSS"): el lienzo razona en una rejilla gruesa (s=4: cada paso cubre 4 px reales, el
campo de visión crece x4 con el mismo número de pasos y el coste por paso baja x16) y la salida se
reconstruye a resolución completa. Tres reconstrucciones comparables:

- ``bilinear``: probabilidades gruesas -> interpolación bilineal (referencia mínima).
- ``fsr``: probabilidades gruesas -> FSR 1 portado (EASU: interpolación Lanczos-2 adaptada a la
  dirección del borde, analizada en la luminancia de la imagen reducida; RCAS: nitidez adaptativa
  con límite de halo). Sin parámetros; diferenciable, así que se entrena de extremo a extremo.
  Nota: AMD recomienda EASU hasta x2-x3; aquí se usa a x4 como comparación, no como su punto óptimo.
- ``learned``: el ESTADO grueso se sube a resolución completa y un lienzo fino (regla propia,
  pocos pasos, ve la retina a resolución completa) lo refina: superresolución aprendida.

Todas devuelven logits [B,2,H,W] (FONDO, FILAMENTO) compatibles con ``fil.evaluate``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MODES = ("bilinear", "fsr", "learned")


# ------------------------------------------------------------------ FSR 1 (EASU + RCAS)
def _lanczos2_approx(d2, lob):
    """Aproximación polinómica de Lanczos-2 de FSR (d2 = distancia^2 ya recortada)."""
    base = (25.0 / 16.0) * (0.4 * d2 - 1.0) ** 2 - (25.0 / 16.0 - 1.0)
    return base * (lob * d2 - 1.0) ** 2


def easu(low, luma, scale):
    """EASU de FSR 1 sobre ``low`` [B,C,h,w] usando la dirección de borde de ``luma`` [B,1,h,w].

    Por cada píxel de salida: 12 muestras (4x4 sin esquinas) alrededor de su posición en la
    rejilla gruesa, núcleo Lanczos-2 girado según el borde y estirado a lo largo de él, y
    deringing (recorte al mín/máx de las 4 muestras más cercanas).
    """
    B, C, h, w = low.shape
    H, W = h * scale, w * scale
    dev, dt = low.device, low.dtype
    # dirección y fuerza del borde por píxel grueso (diferencias centrales, como el cuadrante de FSR)
    lp = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    gx = lp[:, :, 1:-1, 2:] - lp[:, :, 1:-1, :-2]
    gy = lp[:, :, 2:, 1:-1] - lp[:, :, :-2, 1:-1]
    rng = (F.max_pool2d(lp, 3, 1) + F.max_pool2d(-lp, 3, 1)).clamp_min(1e-4)   # máx - mín local
    strength = ((gx.abs() + gy.abs()) / (2 * rng)).clamp(0, 1) ** 2
    # posición de cada píxel de salida en coordenadas gruesas
    oy = (torch.arange(H, device=dev, dtype=dt) + 0.5) / scale - 0.5
    ox = (torch.arange(W, device=dev, dtype=dt) + 0.5) / scale - 0.5
    by, bx = oy.floor(), ox.floor()
    fy, fx = (oy - by)[:, None], (ox - bx)[None, :]
    by, bx = by.long(), bx.long()
    # dirección/longitud interpoladas bilinealmente en la posición (FSR acumula el cuadrante 2x2)
    ana = torch.cat([gx, gy, strength], 1)
    ana = F.interpolate(ana, size=(H, W), mode="bilinear", align_corners=False)
    dx, dy, ln = ana[:, 0:1], ana[:, 1:2], ana[:, 2:3]
    n = torch.sqrt(dx * dx + dy * dy)
    flat = n < 1e-6
    dx = torch.where(flat, torch.ones_like(dx), dx / n.clamp_min(1e-6))
    dy = torch.where(flat, torch.zeros_like(dy), dy / n.clamp_min(1e-6))
    stretch = (dx * dx + dy * dy) / torch.maximum(dx.abs(), dy.abs()).clamp_min(1e-6)
    len2x = 1.0 + (stretch - 1.0) * ln
    len2y = 1.0 - 0.5 * ln
    lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * ln
    clp = 1.0 / lob
    pad = F.pad(low, (1, 2, 1, 2), mode="replicate")       # taps -1..2 sin salirse
    acc = torch.zeros(B, C, H, W, device=dev, dtype=dt)
    wsum = torch.zeros(B, 1, H, W, device=dev, dtype=dt)
    near = []
    for i in (-1, 0, 1, 2):
        for j in (-1, 0, 1, 2):
            if i in (-1, 2) and j in (-1, 2):
                continue                                   # EASU usa 12 taps (sin esquinas)
            tap = pad[:, :, (by + i + 1)[:, None], (bx + j + 1)[None, :]]     # B,C,H,W
            vy, vx = (i - fy)[None, None], (j - fx)[None, None]
            rx = (vx * dx + vy * dy) * len2x                  # a lo largo del borde
            ry = (-vx * dy + vy * dx) * len2y                 # a través del borde
            d2 = torch.minimum(rx * rx + ry * ry, clp)
            wt = _lanczos2_approx(d2, lob)
            acc = acc + wt * tap
            wsum = wsum + wt
            if i in (0, 1) and j in (0, 1):
                near.append(tap)
    out = acc / wsum.where(wsum.abs() > 1e-6, torch.full_like(wsum, 1e-6))
    near = torch.stack(near, 0)
    return torch.minimum(torch.maximum(out, near.amin(0)), near.amax(0))   # deringing


def rcas(x, sharpness=0.2):
    """RCAS de FSR 1 sobre datos en [0,1] [B,C,H,W]: nitidez con lóbulo limitado para no crear halos."""
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    b, d = p[:, :, :-2, 1:-1], p[:, :, 1:-1, :-2]
    f, hh = p[:, :, 1:-1, 2:], p[:, :, 2:, 1:-1]
    e = x
    mn4 = torch.minimum(torch.minimum(b, d), torch.minimum(f, hh))
    mx4 = torch.maximum(torch.maximum(b, d), torch.maximum(f, hh))
    hit_min = torch.minimum(mn4, e) / (4.0 * mx4).clamp_min(1e-6)
    hit_max = (1.0 - torch.maximum(mx4, e)) / (4.0 * mn4 - 4.0).clamp_max(-1e-6)
    lobe = torch.maximum(-hit_min, hit_max).clamp(-0.1875, 0.0) * (2.0 ** -sharpness)
    return (lobe * (b + d + f + hh) + e) / (4.0 * lobe + 1.0)


# ------------------------------------------------------------------ lienzo multiescala
class MultiScale(nn.Module):
    """Piezas nuevas alrededor de un ``NeuroPixel`` (cuya regla se usa en la rejilla gruesa)."""

    def __init__(self, mode, c_id=16, c=48, hidden=128, scale=4):
        super().__init__()
        assert mode in MODES
        self.mode, self.scale = mode, scale
        # bajada aprendida: los s*s colores de la retina de cada bloque -> un color grueso
        self.down = nn.Conv2d(c_id * scale * scale, c_id, 1)
        if mode == "learned":                              # lienzo fino: regla propia, mismo diccionario
            self.seed_f = nn.Conv2d(c_id, c, 1)
            self.perceive_f = nn.Conv2d(c, 2 * c, 3, padding=1, groups=c, bias=False)
            self.f1_f = nn.Conv2d(3 * c + c_id, hidden, 1)
            self.f2_f = nn.Conv2d(hidden, c, 1)
            nn.init.zeros_(self.f2_f.weight)
            nn.init.zeros_(self.f2_f.bias)


def _run(s, ids, perceive, f1, f2, steps, ckpt, damage=None, collect=None, got=None):
    def step(s):
        return s + f2(F.relu(f1(torch.cat([s, perceive(s), ids], 1))))
    for t in range(1, steps + 1):
        s = checkpoint(step, s, use_reentrant=False) if ckpt else step(s)
        if damage is not None and t == damage[0]:
            s = s * F.interpolate(damage[1], size=s.shape[-2:], mode="nearest")
        if collect and t in collect:                                  # P7: estados intermedios para pérdida auxiliar
            got.append(s)
    return s


def ms_forward(model, ms, rgb, steps, steps_fine=6, ckpt=True, damage=None, aux_steps=None):
    """rgb [B,3,H,W] (H,W múltiplos de s) -> logits [B,2,H,W]."""
    train = model.training and ckpt
    s_ = ms.scale
    ids = model.retina(rgb)                                          # B,c_id,H,W
    ids_c = ms.down(F.pixel_unshuffle(ids, s_))                       # B,c_id,H/s,W/s
    got = []
    st = _run(model.seed(ids_c), ids_c, model.perceive, model.f1, model.f2, steps, train, damage, aux_steps, got)
    w = model.dictionary()[1:3]
    read = lambda z: torch.einsum("bchw,kc->bkhw", model.read(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), w)
    if ms.mode == "learned":
        up = F.interpolate(st, scale_factor=s_, mode="bilinear", align_corners=False)
        sf = _run(up + ms.seed_f(ids), ids, ms.perceive_f, ms.f1_f, ms.f2_f, steps_fine, train)
        if aux_steps:
            return read(sf), [read(z) for z in got]                   # logits finos + logits gruesos intermedios
        return read(sf)
    p = read(st).float().softmax(1)                                   # B,2,h,w
    if ms.mode == "bilinear":
        p1 = F.interpolate(p[:, 1:2], scale_factor=s_, mode="bilinear", align_corners=False)
    else:
        luma = F.avg_pool2d(rgb[:, :1].float(), s_)                   # EASU analiza la imagen reducida
        p1 = rcas(easu(p[:, 1:2], luma, s_).clamp(0, 1))
    p1 = p1.clamp(1e-6, 1 - 1e-6)
    return torch.cat([torch.log1p(-p1), torch.log(p1)], 1)            # softmax(logits) = (1-p, p)


def n_extra(ms):
    return sum(p.numel() for p in ms.parameters())


if __name__ == "__main__":                                           # prueba de humo en CPU (segundos)
    torch.manual_seed(0)
    x = torch.linspace(0, 1, 16)
    low = (x[None, :] > 0.5).float().expand(16, 16)[None, None].clone()
    up = easu(low, low, 4)
    assert up.shape == (1, 1, 64, 64) and up.min() >= 0 and up.max() <= 1 + 1e-6
    sh = rcas(up)
    assert sh.shape == up.shape and torch.isfinite(sh).all()
    const = torch.full((1, 3, 8, 8), 0.3)
    assert torch.allclose(easu(const, const[:, :1], 4), torch.full((1, 3, 32, 32), 0.3), atol=1e-5)
    print("easu/rcas ok; borde vertical conservado:", round(float(up[0, 0, 32, 20]), 3), round(float(up[0, 0, 32, 44]), 3))


class SmallUNet(nn.Module):
    """P5 (RES-001, control honesto): U-Net de ~91 k parámetros, 3 niveles, mismos datos/pérdida que el lienzo."""

    def __init__(self, w=14, cin=3, ncls=2):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.ReLU())
        self.e1, self.e2, self.e3, self.b = c(cin, w), c(w, 2 * w), c(2 * w, 4 * w), c(4 * w, 4 * w)
        self.d3, self.d2, self.d1 = c(8 * w, 2 * w), c(4 * w, w), c(2 * w, w)
        self.out = nn.Conv2d(w, ncls, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2)); b = self.b(F.max_pool2d(e3, 2))
        up = lambda z: F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([up(b), e3], 1)); d2 = self.d2(torch.cat([up(d3), e2], 1)); d1 = self.d1(torch.cat([up(d2), e1], 1))
        return self.out(d1)
