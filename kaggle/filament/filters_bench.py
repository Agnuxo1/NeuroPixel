"""FIL-007: banco de filtros de otros campos para los canales 2-3 de la retina (hoy: 3 copias iguales).

Mide, en imágenes de entrenamiento, cuánto separa cada filtro los píxeles de filamento del resto del
disco solar (AUC; 0,5 = azar, 1 = perfecto), en total y solo para filamentos pequeños (los que perdemos).
CPU, sin GPU. Uso: python filters_bench.py [n_imagenes]

Filtros (campo de origen):
- raw_inv            imagen invertida (línea base: filamento = oscuro)
- limb               corrección del oscurecimiento del limbo (astronomía solar: I / perfil radial mediano)
- clahe              contraste local adaptativo (radiología / microscopía) sobre limb
- mgn                Multi-scale Gaussian Normalization (Morgan & Druckmüller 2014, imágenes SDO/AIA)
- tophat             top-hat negro morfológico (microscopía, detección de estructuras oscuras finas)
- dog                diferencia de gaussianas / wavelet 'à trous' de escala media (radioastronomía, starlet)
- sato / frangi / meijering   filtros de crestas por hessiana (angiografía y microscopía de neuritas)
- gabor              máximo de un banco de Gabor orientado (análisis de texturas / escáneres de huellas)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage import filters as skf
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fil  # noqa: E402


def disk_geometry(img):
    m = img > max(10, np.percentile(img, 30) * 0.5)
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    r = np.sqrt(m.sum() / np.pi)
    return cy, cx, r, m


def limb(img):
    cy, cx, r, m = disk_geometry(img)
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.clip((rr / r * 100).astype(int), 0, 150)
    prof = np.array([np.median(img[(bins == b) & m]) if ((bins == b) & m).any() else 1 for b in range(151)])
    prof = np.maximum(cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (1, 9), 2).ravel(), 1)
    out = img.astype(np.float32) / prof[bins]
    return np.where(m, out, 1.0)


def norm01(x):
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def mgn(x, sigmas=(1.25, 2.5, 5, 10, 20, 40), k=0.7):
    acc = np.zeros_like(x)
    for s in sigmas:
        mu = cv2.GaussianBlur(x, (0, 0), s)
        sd = np.sqrt(np.maximum(cv2.GaussianBlur((x - mu) ** 2, (0, 0), s), 1e-8))
        acc += np.arctan(k * (x - mu) / sd)
    return acc / len(sigmas)


def compute(img):
    L = limb(img)
    Ln = norm01(L)
    F = {"raw_inv": -img.astype(np.float32), "limb": -Ln}
    F["clahe"] = -cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16)).apply((Ln * 255).astype(np.uint8)).astype(np.float32)
    F["mgn"] = -mgn(Ln)
    F["tophat"] = cv2.morphologyEx((Ln * 255).astype(np.uint8), cv2.MORPH_BLACKHAT,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(np.float32)
    F["dog"] = cv2.GaussianBlur(Ln, (0, 0), 8) - cv2.GaussianBlur(Ln, (0, 0), 2)
    F["sato"] = skf.sato(Ln, sigmas=(2, 4, 6), black_ridges=True)
    F["frangi"] = skf.frangi(Ln, sigmas=(2, 4, 6), black_ridges=True)
    F["meijering"] = skf.meijering(Ln, sigmas=(2, 4), black_ridges=True)
    g = [cv2.filter2D(Ln, -1, cv2.getGaborKernel((31, 31), 4, th, 12, 0.4, 0)) for th in np.arange(8) * np.pi / 8]
    F["gabor"] = -np.min(np.stack(g), 0)
    return F


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cv2.setNumThreads(4)
    imgs, labs, meta = fil.load_cache()
    tr, _ = fil.split(meta)
    by = {}
    for k in tr:
        by.setdefault(meta["ann"][k]["img"], []).append(k)
    rng = np.random.default_rng(0)
    pick = rng.choice(sorted(by), n, replace=False)
    ys, ys_small, scores = [], [], {}
    t0 = time.time()
    for i in pick:
        img = np.asarray(imgs[i])
        pos = np.any([np.asarray(labs[k]) > 0 for k in by[i]], 0)
        small = np.zeros_like(pos)
        for k in by[i]:
            lab = np.asarray(labs[k])
            ids, cnt = np.unique(lab[lab > 0], return_counts=True)
            small |= np.isin(lab, ids[cnt < 250])
        _, _, _, disk = disk_geometry(img)
        cand = np.flatnonzero(disk.ravel())
        sel = np.concatenate([np.flatnonzero(pos.ravel()),
                              rng.choice(cand, min(len(cand), 20000), replace=False)])
        F = compute(img)
        ys.append(pos.ravel()[sel])
        ys_small.append(small.ravel()[sel] | ~pos.ravel()[sel])     # pequeños frente a fondo
        for name, f in F.items():
            scores.setdefault(name, []).append(f.ravel()[sel])
    y = np.concatenate(ys)
    ysm = np.concatenate(ys_small)
    out = {}
    for name, s in scores.items():
        s = np.concatenate(s)
        out[name] = {"AUC": round(roc_auc_score(y, s), 4),
                     "AUC_pequenos": round(roc_auc_score((y & ysm)[ysm], s[ysm]), 4)}
    out = dict(sorted(out.items(), key=lambda kv: -kv[1]["AUC"]))
    res = {"imagenes": n, "segundos": round(time.time() - t0), "filtros": out}
    (HERE / "runs" / "filters_bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(f"{k:10s} AUC {v['AUC']:.4f}  pequeños {v['AUC_pequenos']:.4f}")
    print("segundos", res["segundos"])


if __name__ == "__main__":
    main()
