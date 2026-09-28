"""DOC-001: GIF de zoom desde el disco completo hasta un filamento, Halfa GONG (2048 nativo) junto a He II 30,4 nm
SDO/AIA (4096 nativo) sincronizados; al final aparece el contorno que detecta NeuroPixel.
Uso: python make_zoom_gif.py [stem] (requiere prob_<stem>.npy y los JP2 de prueba)."""
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
H = Path(__file__).resolve().parent; F = H.parents[1] / "kaggle" / "filament"
sys.path.insert(0, str(F)); import fil; from filters_bench import disk_geometry
stem = sys.argv[1] if len(sys.argv) > 1 else "20110120105534Ch"
ha = np.asarray(Image.open(F / f"data/MAGFiLO_1.0_Kaggle_2026/test/test_images/{stem}.jpeg").convert("L")).astype(np.float32)   # 2048
N = ha.shape[0]
cy, cx, r, disk = disk_geometry(ha.astype(np.uint8))
aia = np.asarray(Image.open(F / "sdo/test_dl/s13.jp2").convert("L")).astype(np.float32)                                  # 4096
s = r / (975.15 / 0.6)                                          # AIA -> pixeles de Halfa 2048
aia2 = cv2.warpAffine(np.log1p(aia), np.array([[s, 0, cx - s * 2047.5], [0, s, cy - s * 2047.5]], np.float32), (N, N), flags=cv2.INTER_AREA)
p = np.load(H / f"prob_{stem}.npy").astype(np.float32)
lab = cv2.resize(fil.instances(p, 0.6, 120, 0).astype(np.uint8), (N, N), interpolation=cv2.INTER_NEAREST)
ids, cnt = np.unique(lab[lab > 0], return_counts=True); tgt = ids[cnt.argmax()]
ty, tx = [float(v.mean()) for v in np.nonzero(lab == tgt)]
def nd(x):
    a, b = np.percentile(x[disk], (1, 99.7)); return np.clip((x - a) / max(b - a, 1e-6), 0, 1)
def cmap(x, st):
    st = np.array(st, np.float32); xs = np.linspace(0, 1, len(st))
    return np.stack([np.interp(x, xs, st[:, k]) for k in range(3)], -1).astype(np.uint8)
A = cmap(nd(ha), [(0, 0, 0), (90, 5, 0), (190, 45, 5), (245, 130, 40), (255, 225, 160)])
B = cmap(nd(aia2), [(0, 0, 0), (80, 10, 0), (180, 60, 10), (240, 140, 60), (255, 230, 180)])
edge = cv2.morphologyEx((lab > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
S = 320
try: font = ImageFont.truetype("arial.ttf", 13)
except Exception: font = ImageFont.load_default()
frames = []
def frame(w, alpha, text):
    y0 = int(np.clip(cy + (ty - cy) * (1 - (w - 200) / (N - 200)) - w / 2, 0, N - w)) if w < N else 0
    x0 = int(np.clip(cx + (tx - cx) * (1 - (w - 200) / (N - 200)) - w / 2, 0, N - w)) if w < N else 0
    panels = []
    for im in (A, B):
        c = im[y0:y0 + w, x0:x0 + w].astype(np.float32)
        if alpha > 0:
            e = edge[y0:y0 + w, x0:x0 + w]
            c[e] = (1 - alpha) * c[e] + alpha * np.array([60, 240, 255])
        panels.append(cv2.resize(c.astype(np.uint8), (S, S), interpolation=cv2.INTER_AREA if w > S else cv2.INTER_CUBIC))
    fr = Image.fromarray(np.concatenate([panels[0], np.zeros((S, 4, 3), np.uint8), panels[1]], 1))
    d = ImageDraw.Draw(fr); d.rectangle([0, S - 22, 2 * S + 4, S], fill=(0, 0, 0))
    d.text((6, S - 18), text, fill=(235, 235, 235), font=font)
    d.text((6, 4), "Hα 656,3 nm · GONG", fill=(255, 220, 180), font=font); d.text((S + 10, 4), "He II 30,4 nm · SDO/AIA", fill=(255, 220, 180), font=font)
    frames.append(fr)
for w in np.geomspace(N, 200, 55):
    km = w / N * 1.39e6 * N / (2 * r)                 # km del campo de vision (diametro solar 1,39e6 km)
    frame(int(w), 0, f"campo de visión ≈ {km:,.0f} km".replace(",", "."))
for a in np.linspace(0, 1, 12):
    frame(200, a, "NeuroPixel: filamento detectado (contorno cian)")
frames += [frames[-1]] * 20
out = H / "zoom_filamento.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
print(out, round(out.stat().st_size / 1e6, 2), "MB", "objetivo", round(ty), round(tx))
