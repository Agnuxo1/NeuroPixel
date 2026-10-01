"""P10 (RES-001): canal de líneas de inversión de polaridad (PIL) del magnetograma HMI, como pista física suave.

Los PNG de Helioviewer son 8 bit (sin gauss calibrados), así que se usa el SIGNO del campo respecto a la mediana del disco, suavizado, y la PIL es la zona
donde hay polaridad positiva y negativa a menos de `r` píxeles (receta tipo Surya: umbral, filtro de regiones pequeñas, dilatación, intersección).
Canales: [Halfa, PIL suave (distancia a la PIL), |campo| con signo]. Si no hay HMI cercano (< 1 h), el canal PIL queda neutro (0).
    python pil_channels.py     # cache/imgs_pil.npy (N,1024,1024,3) uint8
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fil  # noqa: E402
import sdo_channels  # noqa: E402
from filters_bench import disk_geometry  # noqa: E402


def pil3(stem, ha_u8, r=12, thr=0.12, min_px=100):
    cy, cx, rr, disk = disk_geometry(ha_u8)
    out = np.zeros((fil.RES, fil.RES, 3), np.uint8)
    out[..., 0] = ha_u8
    out[..., 2] = 127
    hmi = sdo_channels._align(stem, "hmi", cy, cx, rr)
    if hmi is None:
        return out
    h = hmi - np.median(hmi[disk])
    sc = np.percentile(np.abs(h[disk]), 99.5) + 1e-6
    h = np.clip(h / sc, -1, 1)
    out[..., 2] = (h * 127 + 127).astype(np.uint8)
    s = cv2.GaussianBlur(h, (0, 0), 3)
    pos, neg = (s > thr).astype(np.uint8), (s < -thr).astype(np.uint8)
    for m in (pos, neg):                                                  # filtro de regiones pequeñas
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < min_px])
        m[small] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    pil = (cv2.dilate(pos, k) & cv2.dilate(neg, k)).astype(np.uint8)
    dist = cv2.distanceTransform(1 - pil, cv2.DIST_L2, 5)               # distancia a la PIL
    soft = np.exp(-(dist / (r * 1.5)) ** 2) * disk
    out[..., 1] = (soft * 255).astype(np.uint8)
    return out


if __name__ == "__main__":
    cv2.setNumThreads(4)
    imgs, _, meta = fil.load_cache()
    out = np.lib.format.open_memmap(fil.CACHE / "imgs_pil.npy", mode="w+", dtype=np.uint8, shape=(len(imgs), fil.RES, fil.RES, 3))
    t0 = time.time()
    for i, fname in enumerate(meta["files"]):
        out[i] = pil3(Path(fname).stem, np.asarray(imgs[i]))
        if i % 100 == 0:
            print(i, round(time.time() - t0), flush=True)
    out.flush()
    print("hecho", len(imgs), round(time.time() - t0), "s")
