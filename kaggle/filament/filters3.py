"""FIL-007: tres canales para la retina = [limbo corregido, sato (crestas oscuras), DoG (à trous)], uint8.

    python filters3.py          # construye cache/imgs3.npy (N,1024,1024,3) en D:
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage import filters as skf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fil  # noqa: E402
from filters_bench import limb, norm01  # noqa: E402


def three(img_u8):
    """Imagen gris uint8 -> HxWx3 uint8 con los tres filtros normalizados a [0,255]."""
    Ln = norm01(limb(img_u8))
    sato = skf.sato(Ln, sigmas=(2, 4, 6), black_ridges=True)
    dog = cv2.GaussianBlur(Ln, (0, 0), 8) - cv2.GaussianBlur(Ln, (0, 0), 2)
    return np.stack([(Ln * 255), norm01(sato) * 255, norm01(dog) * 255], -1).astype(np.uint8)


if __name__ == "__main__":
    cv2.setNumThreads(4)
    imgs, _, _ = fil.load_cache()
    out = np.lib.format.open_memmap(fil.CACHE / "imgs3.npy", mode="w+", dtype=np.uint8, shape=(len(imgs), fil.RES, fil.RES, 3))
    t0 = time.time()
    for i in range(len(imgs)):
        out[i] = three(np.asarray(imgs[i]))
        if i % 50 == 0:
            print(i, round(time.time() - t0), flush=True)
    out.flush()
    print("hecho", len(imgs), round(time.time() - t0), "s")
