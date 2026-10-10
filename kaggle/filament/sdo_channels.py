"""FIL-008: retina con canales solares REALES coetaneos: [Halfa GONG, He II 30,4 nm SDO/AIA, magnetograma SDO/HMI], uint8.

Alineado por geometria del encabezado (centro CRPIX, radio RSUN_OBS/CDELT) al disco ajustado en Halfa.
Si la imagen SDO dista mas de MAX_DT s del instante Halfa, el canal queda neutro (127).
    python sdo_channels.py      # construye cache/imgs_sdo.npy (train) en D:
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fil  # noqa: E402
from filters_bench import disk_geometry  # noqa: E402

SDO = HERE / "sdo"
MAX_DT = 3600
META = {}
for line in (SDO / "meta.jsonl").read_text().splitlines():
    r = json.loads(line)
    META[(r["stem"], r["src"])] = r


def _dt(r):
    f = lambda s: datetime.fromisoformat(s.replace("Z", "").split(".")[0])
    try:
        return abs((f(r["DATE-OBS"]) - f(r["ha_date"])).total_seconds())
    except Exception:
        return 1e9


def _align(stem, src, cy, cx, rr):
    r = META.get((stem, src))
    if r is None or _dt(r) > MAX_DT or r.get("CRPIX1") is None:
        return None
    a = np.asarray(Image.open(SDO / "png" / f"{r['key']}.png").convert("L")).astype(np.float32)
    f = r["factor"]
    c1, c2 = (float(r["CRPIX1"]) - 0.5) / f - 0.5, (float(r["CRPIX2"]) - 0.5) / f - 0.5
    rad = float(r["RSUN_OBS"]) / float(r["CDELT1"]) / f
    s = rr / rad
    M = np.array([[s, 0, cx - s * c1], [0, s, cy - s * c2]], np.float32)
    return cv2.warpAffine(a, M, (fil.RES, fil.RES), flags=cv2.INTER_AREA)


def sdo3(stem, ha_u8):
    """Halfa 1024 uint8 -> HxWx3 uint8 [Halfa, AIA 304 (log), HMI con signo (127 = 0 G)]."""
    cy, cx, rr, disk = disk_geometry(ha_u8)
    out = np.full((fil.RES, fil.RES, 3), 127, np.uint8)
    out[..., 0] = ha_u8
    aia = _align(stem, "aia304", cy, cx, rr)
    if aia is not None:
        la = np.log1p(aia)
        lo, hi = np.percentile(la[disk], (1, 99.7))
        out[..., 1] = (np.clip((la - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    hmi = _align(stem, "hmi", cy, cx, rr)
    if hmi is not None:
        h = hmi - np.median(hmi[disk])
        sc = np.percentile(np.abs(h[disk]), 99.5) + 1e-6
        out[..., 2] = (np.clip(h / sc, -1, 1) * 127 + 127).astype(np.uint8)
    return out


if __name__ == "__main__":
    imgs, _, meta = fil.load_cache()
    out = np.lib.format.open_memmap(fil.CACHE / "imgs_sdo.npy", mode="w+", dtype=np.uint8, shape=(len(imgs), fil.RES, fil.RES, 3))
    t0, miss = time.time(), 0
    for i, fname in enumerate(meta["files"]):
        stem = Path(fname).stem
        out[i] = sdo3(stem, np.asarray(imgs[i]))
        miss += int((out[i, ..., 1] == 127).all()) + int((out[i, ..., 2] == 127).all())
        if i % 100 == 0:
            print(i, round(time.time() - t0), flush=True)
    out.flush()
    print("hecho", len(imgs), "canales ausentes", miss, round(time.time() - t0), "s")
