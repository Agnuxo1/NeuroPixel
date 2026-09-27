"""Solar Filament Segmentation 2026 con NeuroPixel: datos, modelo por píxel, PQ y envío.

El lienzo recibe la imagen H-alfa como píxeles de cámara (línea L2: retina + lienzo) y cada
píxel "dice" con el diccionario si es FONDO o FILAMENTO. Las instancias salen de las
componentes conexas de la máscara.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "MAGFiLO_1.0_Kaggle_2026"
CACHE = HERE / "cache"
RES = 1024                      # resolución de trabajo (el envío es a 2048)
VOCAB = ["<vacío>", "FONDO", "FILAMENTO"]


# ------------------------------------------------------------------ datos
def build_cache():
    """Imágenes a 1024 (uint8) y mapas de instancias por anotador (uint8), en D:."""
    from PIL import Image
    from pycocotools import mask as mu
    CACHE.mkdir(exist_ok=True)
    d = json.load(open(DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    files = sorted({i["file_name"] for i in d["images"]})
    fidx = {f: k for k, f in enumerate(files)}
    imgs = np.zeros((len(files), RES, RES), np.uint8)
    for f, k in fidx.items():
        im = Image.open(DATA / "train" / "train_images" / f).convert("L").resize((RES, RES), Image.BILINEAR)
        imgs[k] = np.asarray(im)
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    ids = [i for i in d["images"] if i["id"] in anns]
    labs = np.zeros((len(ids), RES, RES), np.uint8)
    meta = []
    for j, im in enumerate(ids):
        for n, a in enumerate(anns[im["id"]], 1):
            rle = mu.frPyObjects(a["segmentation"], im["height"], im["width"])
            m = mu.decode(mu.merge(rle))
            small = F.interpolate(torch.from_numpy(m)[None, None].float(), size=(RES, RES), mode="area")[0, 0] > 0.3
            labs[j][small.numpy() & (labs[j] == 0)] = n
        meta.append({"id": im["id"], "file": im["file_name"], "img": fidx[im["file_name"]]})
    np.save(CACHE / "imgs.npy", imgs)
    np.save(CACHE / "labs.npy", labs)
    (CACHE / "meta.json").write_text(json.dumps({"files": files, "ann": meta}), encoding="utf-8")
    return imgs, labs, meta


def load_cache():
    imgs = np.load(CACHE / "imgs.npy", mmap_mode="r")
    labs = np.load(CACHE / "labs.npy", mmap_mode="r")
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    return imgs, labs, meta


def split(meta, val_frac=0.15, seed=0):
    """Validación por imagen ÚNICA (los anotadores de una imagen no se reparten)."""
    files = sorted({m["file"] for m in meta["ann"]})
    rng = np.random.default_rng(seed)
    val = set(rng.choice(files, int(len(files) * val_frac), replace=False))
    tr = [k for k, m in enumerate(meta["ann"]) if m["file"] not in val]
    va = [k for k, m in enumerate(meta["ann"]) if m["file"] in val]
    return tr, va


def norm_img(x):
    """uint8 -> [-1,1], 3 canales iguales (la retina espera 3)."""
    x = x.float() / 127.5 - 1
    return x.unsqueeze(1).expand(-1, 3, -1, -1)


# ------------------------------------------------------------------ modelo
def seg_forward(model, rgb, steps, ckpt=True, damage=None):
    """Lienzo NeuroPixel sobre la imagen; devuelve logits por píxel [B,2,H,W] (FONDO, FILAMENTO)."""
    ids = model.retina(rgb) if model.retina is not None else torch.cat(
        [rgb, rgb.new_zeros(rgb.shape[0], model.embed.embedding_dim - 3, *rgb.shape[2:])], 1)
    s = model.seed(ids)

    def step(s):
        h = torch.cat([s, model.perceive(s), ids], 1)
        ds = model.f2(F.relu(model.f1(h)))
        if model.training and model.fire_rate < 1:
            ds = ds * (torch.rand_like(ds[:, :1]) < model.fire_rate)
        return s + ds
    for t in range(1, steps + 1):
        s = checkpoint(step, s, use_reentrant=False) if (ckpt and model.training) else step(s)
        if damage is not None and t == damage[0]:          # reposo: daño a mitad de pensar
            s = s * damage[1]
    w = model.dictionary()[1:3]                                    # colores de FONDO y FILAMENTO
    return torch.einsum("bchw,kc->bkhw", model.read(s.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), w)


# ------------------------------------------------------------------ instancias y PQ
def instances(prob, thr=0.5, min_area=20, close=0):
    """Máscara de probabilidad -> mapa de instancias (componentes conexas)."""
    from scipy import ndimage as ndi
    m = prob > thr
    if close:
        m = ndi.binary_closing(m, iterations=close)
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n:
        areas = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = areas >= min_area
        lab = np.where(keep[lab], lab, 0)
        _, lab = np.unique(lab, return_inverse=True)
        lab = lab.reshape(m.shape)
    return lab


def pq_counts(gt, pr):
    """Suma de IoU de TP, nº TP, FP, FN para una imagen (emparejado IoU>0,5)."""
    g_ids = [i for i in np.unique(gt) if i]
    p_ids = [i for i in np.unique(pr) if i]
    iou_sum, tp, matched_p = 0.0, 0, set()
    if g_ids and p_ids:
        inter = np.zeros((max(g_ids) + 1, max(p_ids) + 1))
        np.add.at(inter, (gt.ravel(), pr.ravel()), 1)
        ga = inter.sum(1)
        pa = inter.sum(0)
        for g in g_ids:
            for p in p_ids:
                if inter[g, p] == 0 or p in matched_p:
                    continue
                iou = inter[g, p] / (ga[g] + pa[p] - inter[g, p])
                if iou > 0.5:
                    iou_sum += iou
                    tp += 1
                    matched_p.add(p)
                    break
    fp = len(p_ids) - len(matched_p)
    fn = len(g_ids) - tp
    return iou_sum, tp, fp, fn


def dice(gt, pr):
    a, b = gt > 0, pr > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s


# ------------------------------------------------------------------ envío
def rle_counts(mask2048):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask2048.astype(np.uint8)))
    return r["counts"].decode("ascii")
