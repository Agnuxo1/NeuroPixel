"""DOC-001: GIF 'escaner' que recorre los canales de una misma fecha (Halfa GONG, retina en falso color, AIA 304, HMI).
Uso: python make_scanner_gif.py [stem] (por defecto 20110120105534Ch, imagen de test con SDO de prueba)."""
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
F = Path(__file__).resolve().parents[2] / "kaggle" / "filament"
sys.path.insert(0, str(F)); from filters_bench import disk_geometry, limb
from skimage import filters as skf
stem = sys.argv[1] if len(sys.argv) > 1 else "20110120105534Ch"
ha = np.asarray(Image.open(F / f"data/MAGFiLO_1.0_Kaggle_2026/test/test_images/{stem}.jpeg").convert("L").resize((1024, 1024))).astype(np.float32)
ry, rx, rr, disk = disk_geometry(ha.astype(np.uint8))
def align(src):
    a = np.asarray(Image.open(src).convert("L")).astype(np.float32); c = 2047.5; s = rr / (975.15 / 0.6)
    return cv2.warpAffine(a, np.array([[s, 0, rx - s * c], [0, s, ry - s * c]], np.float32), (1024, 1024), flags=cv2.INTER_AREA)
aia, hmi = align(F / "sdo/test_dl/s13.jp2"), align(F / "sdo/test_dl/s19.jp2")
def nd(x, lo=1, hi=99.7):                       # normaliza con los pixeles del disco
    a, b = np.percentile(x[disk], (lo, hi)); return np.clip((x - a) / max(b - a, 1e-6), 0, 1)
def cmap(x, stops):
    xs = np.linspace(0, 1, len(stops)); st = np.array(stops, np.float32)
    return np.stack([np.interp(x, xs, st[:, k]) for k in range(3)], -1).astype(np.uint8)
L = nd(limb(ha.astype(np.uint8)))
sato = nd(skf.sato(L, sigmas=(2, 4, 6), black_ridges=True), 1, 99.5)
dog = nd(cv2.GaussianBlur(L, (0, 0), 8) - cv2.GaussianBlur(L, (0, 0), 2), 1, 99.5)
red = [(0, 0, 0), (90, 5, 0), (190, 45, 5), (245, 130, 40), (255, 225, 160)]
ha_rgb = cmap(nd(ha), red)
aia_rgb = cmap(nd(np.log1p(aia)), [(0, 0, 0), (80, 10, 0), (180, 60, 10), (240, 140, 60), (255, 230, 180)])
hm = hmi - np.median(hmi[disk]); hm = np.clip(hm / (np.percentile(np.abs(hm[disk]), 99.5) + 1e-6), -1, 1) * 0.5 + 0.5
hmi_rgb = cmap(hm, [(30, 60, 200), (100, 100, 110), (128, 128, 128), (150, 150, 140), (255, 235, 60)])
f3 = (np.stack([L, sato, dog], -1) * 255).astype(np.uint8)
seq = [("Hα 656,3 nm · GONG (imagen del concurso)", ha_rgb), ("Retina: limbo · crestas (Sato) · à trous (DoG)", f3),
       ("He II 30,4 nm · SDO/AIA · mismo instante", aia_rgb), ("Magnetograma LOS Fe I 617,3 nm · SDO/HMI", hmi_rgb),
       ("Hα 656,3 nm · GONG", ha_rgb)]
S = 384
try: font = ImageFont.truetype("arial.ttf", 14)
except Exception: font = ImageFont.load_default()
def prep(z):
    o = z.copy(); o[~disk] = 0; return cv2.resize(o, (S, S), interpolation=cv2.INTER_AREA)
frames = []
for (na, A), (nb, B) in zip(seq[:-1], seq[1:]):
    A, B = prep(A), prep(B)
    for t in np.linspace(0, 1, 20):
        x = int(t * S); fr = A.copy(); fr[:, :x] = B[:, :x]
        glow = np.exp(-((np.arange(S) - x) / 5.0) ** 2)[None, :, None]
        fr = np.clip(fr + glow * np.array([170, 220, 255])[None, None, :], 0, 255).astype(np.uint8)
        im = Image.fromarray(fr); d = ImageDraw.Draw(im)
        d.rectangle([0, S - 26, S, S], fill=(0, 0, 0)); d.text((8, S - 21), nb if t > 0.5 else na, fill=(235, 235, 235), font=font)
        frames.append(im)
    frames += [frames[-1]] * 10
out = Path(__file__).resolve().parent / "escaner_canales.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)
print(out, round(out.stat().st_size / 1e6, 2), "MB")
