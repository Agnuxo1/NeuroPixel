"""Fase 0: un token = un píxel RGB24 (ID de 24 bits repartido en R, G, B).

El negro RGB(0,0,0) queda reservado como 'vacío' (ID 0 = PAD).
"""
from __future__ import annotations

import io
import math

import numpy as np

MAX_ID = 2**24 - 1


def ids_to_rgb(ids) -> np.ndarray:
    a = np.asarray(ids, dtype=np.uint32)
    if a.size and int(a.max()) > MAX_ID:
        raise ValueError("ID fuera de 24 bits")
    return np.stack([(a >> 16) & 255, (a >> 8) & 255, a & 255], -1).astype(np.uint8)


def rgb_to_ids(rgb: np.ndarray) -> np.ndarray:
    r = rgb.astype(np.uint32)
    return (r[..., 0] << 16) | (r[..., 1] << 8) | r[..., 2]


def pack_image(ids, width: int | None = None) -> np.ndarray:
    """Coloca la secuencia de IDs en una imagen HxWx3; relleno con negro (PAD)."""
    ids = np.asarray(ids, dtype=np.uint32)
    w = width or math.ceil(math.sqrt(max(1, ids.size)))
    h = math.ceil(ids.size / w)
    flat = np.zeros(h * w, np.uint32)
    flat[: ids.size] = ids
    return ids_to_rgb(flat.reshape(h, w))


def to_png(img: np.ndarray) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(img, "RGB").save(b, "PNG", optimize=True)
    return b.getvalue()


def from_png(data: bytes, n: int) -> np.ndarray:
    from PIL import Image
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return rgb_to_ids(img).reshape(-1)[:n]
