"""Soil Grain Size con NeuroPixel: cada píxel vota el tamaño de grano; el histograma de votos es la curva.

Diccionario de 11 palabras de tamaño (los 11 diámetros de DIN EN ISO 14688-1). Las fotos se
reescalan a una escala común (px/mm) con la tabla ppm por teléfono. Cada píxel del lienzo da una
distribución sobre los 11 tramos; la media de todos los píxeles de la foto es la predicción, y la
de todas las fotos de una muestra, su curva. Validación dejando fuera muestras enteras.

    python soil.py --cv          # validación cruzada (EMD)
    python soil.py --submit      # entrena con todo y escribe submission.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from neuropixel.model import NeuroPixel, n_params  # noqa: E402
from neuropixel.safety import choose_device  # noqa: E402

D = HERE / "data"
SIZES = [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0]
LOGW = np.diff(np.log10(SIZES))                        # pesos logarítmicos de los 10 tramos
VOCAB = ["<vacío>"] + [f"GRANO_{s}" for s in SIZES]
PPM_TARGET = 6.0                                        # px/mm tras reescalar


def ppm_table():
    t = {}
    for r in csv.DictReader(open(D / "ppm_updated.csv", encoding="utf-8")):
        t[r["phone"].strip()] = float(r["ppm"])
    return t


def norm(x):
    """Clave normalizada: minúsculas, umlauts -> ae/oe/ue, solo letras y números."""
    x = x.lower().replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss")
    return "".join(ch for ch in x if ch.isalnum())


def split_name(stem, table):
    """'Motorola_Edge_60_fusion_H374_01' -> ('Motorola Edge 60 Fusion', 'h374');
    'iPhone14_HPC_Audorfring (1)' -> ('iPhone 14', 'hpcaudorfring')."""
    base = stem.rsplit(" (", 1)[0] if stem.endswith(")") else stem.rsplit("_", 1)[0]
    nb = norm(base)
    for ph in sorted(table, key=lambda k: -len(norm(k))):
        if nb.startswith(norm(ph)):
            return ph, nb[len(norm(ph)):]
    return None, nb


def load_photos(folder, table):
    """Devuelve [(muestra_normalizada, array uint8 HxWx3 reescalado a PPM_TARGET)]."""
    out = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        ph, sample = split_name(f.stem, table)
        ppm = table.get(ph, 15.0)
        im = Image.open(f).convert("RGB")
        sc = PPM_TARGET / ppm
        im = im.resize((max(64, int(im.width * sc)), max(64, int(im.height * sc))), Image.BILINEAR)
        out.append((sample, np.asarray(im)))
    return out


def labels():
    lab = {}
    for r in csv.DictReader(open(D / "Training_labels_updated.csv", encoding="utf-8")):
        lab[norm(r["sample_id"])] = np.array([float(r[k]) for k in list(r)[1:]])
    return lab


def cdf_to_pdf(cdf):
    pdf = np.diff(np.concatenate([[0.0], cdf]))
    return np.clip(pdf, 0, None) / max(1e-9, np.clip(pdf, 0, None).sum())


def emd(cdf_true, cdf_pred):
    """EMD logarítmico entre curvas acumuladas (en %), sobre los 10 tramos."""
    d = np.abs(np.asarray(cdf_true) - np.asarray(cdf_pred))
    return float(np.sum(0.5 * (d[:-1] + d[1:]) * LOGW))


def canvas_votes(model, rgb, steps):
    """Distribución media sobre los 11 tamaños que vota el lienzo (retina + pasos locales)."""
    ids = model.retina(rgb)
    s = model.seed(ids)
    for _ in range(steps):
        h = torch.cat([s, model.perceive(s), ids], 1)
        s = s + model.f2(F.relu(model.f1(h)))
    p = model.lens_logits(s)[..., 1:].softmax(-1)      # B,H,W,11
    return p.mean((1, 2)), p


def to_t(arrs, dev):
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
    return x


def train(photos, labs, iters, dev, crop=128, batch=16, steps=16, seed=0, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = NeuroPixel(len(VOCAB), (0, 0), c=48, hidden=128, retina=True, fire_rate=1.0).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.05)
    items = [(s, a) for s, a in photos if s in labs]
    lw = torch.tensor(LOGW, dtype=torch.float32, device=dev)
    for it in range(iters):
        xs, ys = [], []
        for _ in range(batch):
            s, a = items[rng.integers(len(items))]
            h, w = a.shape[:2]
            c = min(crop, h, w)
            y0, x0 = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
            patch = a[y0:y0 + c, x0:x0 + c]
            patch = np.rot90(patch, rng.integers(4))
            if rng.random() < 0.5:                      # espejo: 8 variantes por recorte (idea de Fran)
                patch = patch[:, ::-1]
            xs.append(np.ascontiguousarray(patch if c == crop else np.asarray(Image.fromarray(patch).resize((crop, crop)))))
            ys.append(labs[s])
        x = to_t(xs, dev)
        cdf_t = torch.tensor(np.stack(ys), dtype=torch.float32, device=dev) / 100
        mean_p, p = canvas_votes(m, x, steps)
        cdf_p = mean_p.cumsum(-1)
        d = (cdf_p - cdf_t).abs()
        loss = (0.5 * (d[:, :-1] + d[:, 1:]) * lw).sum(-1).mean()
        pdf_t = torch.diff(torch.cat([torch.zeros_like(cdf_t[:, :1]), cdf_t], 1), dim=1).clamp_min(0)
        pdf_t = pdf_t / pdf_t.sum(-1, keepdim=True).clamp_min(1e-6)
        loss = loss + 0.1 * -(pdf_t[:, None, None, :] * (p + 1e-8).log()).sum(-1).mean()   # escuela
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if log is not None and (it + 1) % max(1, iters // 5) == 0:
            log.append({"it": it + 1, "loss": round(loss.item(), 4)})
    return m.eval()


@torch.no_grad()
def predict_sample(m, arrs, dev, steps=16, tile=256):
    """Curva (en %) de una muestra: media de las teselas de todas sus fotos."""
    ps = []
    for a in arrs:
        h, w = a.shape[:2]
        for y0 in range(0, max(1, h - tile + 1), tile):
            for x0 in range(0, max(1, w - tile + 1), tile):
                t_ = a[y0:y0 + tile, x0:x0 + tile]
                for v in (t_, t_[:, ::-1]):             # promedio con la tesela reflejada
                    ps.append(canvas_votes(m, to_t([np.ascontiguousarray(v)], dev), steps)[0][0])
    return (torch.stack(ps).mean(0).cumsum(-1) * 100).clamp(0, 100).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=6)
    a = ap.parse_args()
    dev = choose_device("cuda", threads=4, force_gpu=True, vram_cap_gib=6)
    table = ppm_table()
    labs = labels()
    tr = load_photos(D / "Training-All_Photos_updated" / "Training-All_Photos_updated", table)
    samples = sorted({s for s, _ in tr if s in labs})
    print(json.dumps({"fotos": len(tr), "muestras_etiquetadas": len(samples),
                      "sin_etiqueta": sorted({s for s, _ in tr} - set(labs))[:5]}), flush=True)
    out = HERE / "runs" / "np_votes"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"iters": a.iters}
    if a.cv:
        rng = np.random.default_rng(0)
        perm = rng.permutation(samples)
        folds = np.array_split(perm, a.folds)
        errs, base = [], []
        mean_cdf = np.mean([labs[s] for s in samples], 0)
        for k, fo in enumerate(folds):
            trs = [(s, x) for s, x in tr if s not in set(fo)]
            m = train(trs, labs, a.iters, dev, seed=k)
            for s in fo:
                pred = predict_sample(m, [x for ss, x in tr if ss == s], dev)
                errs.append(emd(labs[s], pred))
                base.append(emd(labs[s], mean_cdf))
            print(json.dumps({"fold": k, "EMD_np": round(float(np.mean(errs)), 3), "EMD_media": round(float(np.mean(base)), 3)}), flush=True)
        res.update({"cv_EMD_neuropixel": round(float(np.mean(errs)), 3), "cv_EMD_curva_media": round(float(np.mean(base)), 3),
                    "cv_EMD_trivial_9pct": round(float(np.mean([emd(labs[s], np.cumsum([100 / 11] * 11)) for s in samples])), 3)})
    if a.submit:
        m = train(tr, labs, a.iters, dev, seed=0)
        te = load_photos(D / "Test_All_Photos" / "Test_All_Photos", table)
        sub = list(csv.DictReader(open(D / "sample_submission.csv", encoding="utf-8")))
        with open(out / "submission.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(list(sub[0].keys()))
            for r in sub:
                sid = r["sample_id"].strip()
                arrs = [x for s, x in te if s == norm(sid)]
                pred = predict_sample(m, arrs, dev) if arrs else np.cumsum([100 / 11] * 11)
                pred = np.maximum.accumulate(pred)
                pred[-1] = 100.0
                w.writerow([sid] + [round(float(v), 4) for v in pred])
        res["test_muestras"] = len(sub)
        res["test_sin_fotos"] = [r["sample_id"] for r in sub if not any(s == norm(r["sample_id"]) for s, _ in te)]
    res["segundos"] = round(time.time() - t0)
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
