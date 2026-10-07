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
import hashlib
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
from kaggle.soil.evaluation import (  # noqa: E402
    SIZES, LOGW, METRIC_NAME, CDF_ENDPOINT_ATOL, cdf_to_pdf, emd,
    normalize_sample_id, unique_normalized_ids, validate_cdf, make_folds,
    train_mean_cdf, score_sample, summarize_scores,
)

D = HERE / "data"
VOCAB = ["<vacío>"] + [f"GRANO_{s}" for s in SIZES]
PPM_TARGET = 6.0                                        # px/mm tras reescalar


def ppm_table():
    with open(D / "ppm_updated.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    unique_normalized_ids([r["phone"].strip() for r in rows])
    table = {r["phone"].strip(): float(r["ppm"]) for r in rows}
    if any(not np.isfinite(value) or value <= 0 for value in table.values()):
        raise ValueError("ppm values must be positive and finite")
    return table


def norm(x):
    """Existing normalization, now rejecting empty keys; labels reject collisions."""
    return normalize_sample_id(x)


def split_name(stem, table):
    """'Motorola_Edge_60_fusion_H374_01' -> ('Motorola Edge 60 Fusion', 'h374');
    'iPhone14_HPC_Audorfring (1)' -> ('iPhone 14', 'hpcaudorfring')."""
    base = stem.rsplit(" (", 1)[0] if stem.endswith(")") else stem.rsplit("_", 1)[0]
    nb = norm(base)
    for ph in sorted(table, key=lambda k: -len(norm(k))):
        if nb.startswith(norm(ph)):
            return ph, nb[len(norm(ph)):]
    return None, nb


def load_photos(folder, table, metadata=None):
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
        if metadata is not None:
            metadata.append({"file": f.name, "phone": ph, "sample_id": sample,
                             "ppm": ppm, "ppm_source": "table" if ph in table else "fallback_15"})
    return out


def labels(raw_ids=None):
    with open(D / "Training_labels_updated.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    keys = unique_normalized_ids([r["sample_id"] for r in rows])
    lab = {norm(r["sample_id"]): validate_cdf([float(r[k]) for k in list(r)[1:]],
                                           name=f"label {r['sample_id']}") for r in rows}
    if raw_ids is not None:
        raw_ids.update(keys)
    return lab


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
    raw_ids, photo_metadata = {}, []
    labs = labels(raw_ids)
    tr = load_photos(D / "Training-All_Photos_updated" / "Training-All_Photos_updated", table, photo_metadata)
    samples = sorted({s for s, _ in tr if s in labs})
    print(json.dumps({"fotos": len(tr), "muestras_etiquetadas": len(samples),
                      "sin_etiqueta": sorted({s for s, _ in tr} - set(labs)),
                      "labels_without_photos": sorted(set(labs) - {s for s, _ in tr})}), flush=True)
    out = HERE / "runs" / "np_votes"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"iters": a.iters}
    if a.cv:
        folds = make_folds(samples, a.folds, seed=0)
        records, fold_reports = [], []
        # Unique raw evidence survives the legacy result.json summary's later replacement.
        evidence_path = out / f"cv_evaluation_{time.time_ns()}.json"
        with evidence_path.open("x", encoding="utf-8") as fh:
            fh.write("{}\n")
        evidence = {"schema_version": 1, "status": "running", "iters": a.iters,
                    "metric": METRIC_NAME, "cdf_endpoint_atol_pp": CDF_ENDPOINT_ATOL,
                    "group_unit": "normalized_sample_id", "split_seed": 0,
                    "sample_ids": samples, "raw_label_ids": raw_ids,
                    "validation_folds": folds, "photo_metadata": photo_metadata,
                    "numpy_version": np.__version__, "torch_version": torch.__version__,
                    "source_sha256": {str(path.relative_to(HERE.parents[1])):
                                      hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (Path(__file__).resolve(), HERE / "evaluation.py")},
                    "folds": fold_reports, "per_sample": records}
        for k, fo in enumerate(folds):
            train_ids = [s for s in samples if s not in set(fo)]
            train_labs = {s: labs[s] for s in train_ids}
            mean_cdf = train_mean_cdf(train_labs, train_ids, fo)
            trs = [(s, x) for s, x in tr if s in train_labs]
            m = train(trs, train_labs, a.iters, dev, seed=k)
            fold_records = []
            for sample in fo:
                pred = predict_sample(m, [x for sid, x in tr if sid == sample], dev)
                row = score_sample(sample, k, labs[sample], pred, mean_cdf)
                row["raw_sample_id"] = raw_ids[sample]
                row["photos"] = [r for r in photo_metadata if r["sample_id"] == sample]
                fold_records.append(row)
            records.extend(fold_records)
            fold_metrics = summarize_scores(fold_records)
            pooled = summarize_scores(records)
            fold_reports.append({"fold": k, "train_ids": train_ids, "validation_ids": fo,
                                 "n_train_samples": len(train_ids), "train_mean_cdf": mean_cdf.tolist(),
                                 "metrics": fold_metrics})
            evidence.update({"pooled_metrics": pooled,
                             "status": "complete" if k + 1 == len(folds) else "running"})
            temporary = evidence_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(evidence_path)
            print(json.dumps({"fold": k, "fold_metrics": fold_metrics,
                              "pooled_to_date_metrics": pooled}, allow_nan=False), flush=True)
        res.update({"cv_EMD_neuropixel": pooled["means"]["model"],
                    "cv_EMD_curva_media": pooled["means"]["train_mean"],
                    "cv_EMD_trivial_9pct": pooled["means"]["uniform"],
                    "cv_metric": METRIC_NAME, "cv_n_samples": len(records),
                    "cv_evidence": evidence_path.name})
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
                pred = validate_cdf(pred, name=f"submission {sid}")
                pred[-1] = 100.0
                w.writerow([sid] + [round(float(v), 4) for v in pred])
        res["test_muestras"] = len(sub)
        res["test_sin_fotos"] = [r["sample_id"] for r in sub if not any(s == norm(r["sample_id"]) for s, _ in te)]
    res["segundos"] = round(time.time() - t0)
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
