"""Representaciones holográficas reducidas (Plate, 1995) con torch.fft.

ligar(a, b)   = a ⊛ b  (convolución circular: no conmutativa en el papel que codifica)
desligar(c,a) = c ⊛ a⁺ (involución de a: inversa aproximada)
agrupar       = suma
Diferenciable, así que puede entrenarse dentro de NeuroPixel.
"""
from __future__ import annotations

import torch


def random_vectors(n: int, d: int, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(n, d, generator=generator) / d**0.5


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=d)


def involution(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[..., :1], a[..., 1:].flip(-1)], -1)


def unbind(c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return bind(c, involution(a))


def cleanup(x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """Índice del vector del diccionario más parecido (coseno)."""
    x = torch.nn.functional.normalize(x, dim=-1)
    m = torch.nn.functional.normalize(memory, dim=-1)
    return (x @ m.T).argmax(-1)
