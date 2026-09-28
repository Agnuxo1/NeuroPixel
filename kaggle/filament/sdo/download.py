"""FIL-008: descarga SDO/AIA 304 (sourceId 13) y HMI magnetograma (19) coetaneos de cada imagen GONG H-alfa
desde la API publica de Helioviewer; guarda PNG 1024 (reducido x4) + geometria del encabezado JP2.
Reanudable: salta lo ya descargado. Uso: python download.py"""
import io, json, re, sys, time, urllib.request
from pathlib import Path
from PIL import Image
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "MAGFiLO_1.0_Kaggle_2026"
OUT = HERE / "png"; OUT.mkdir(exist_ok=True)
META = HERE / "meta.jsonl"
done = {json.loads(l)["key"] for l in META.read_text().splitlines()} if META.exists() else set()
names = sorted({p.stem for p in (DATA / "train" / "train_images").glob("*.jp*g")} |
               {p.stem for p in (DATA / "test" / "test_images").glob("*.jp*g")})
print(len(names), "imagenes GONG;", len(done), "ya hechas", flush=True)
for n, stem in enumerate(names):
    t = stem[:14]
    date = f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[8:10]}:{t[10:12]}:{t[12:14]}Z"
    for src, tag in ((13, "aia304"), (19, "hmi")):
        key = f"{stem}_{tag}"
        if key in done:
            continue
        url = f"https://api.helioviewer.org/v2/getJP2Image/?date={date}&sourceId={src}"
        for intento in range(3):
            try:
                b = urllib.request.urlopen(url, timeout=120).read()
                x = b[b.find(b"<?xml"):b.find(b"<?xml") + 60000].decode("latin1")
                g = {k: (lambda m: m.group(1) if m else None)(re.search(rf"<{k}>([^<]*)</{k}>", x))
                     for k in ("CRPIX1", "CRPIX2", "CDELT1", "RSUN_OBS", "CROTA2", "DATE-OBS", "NAXIS1")}
                im = Image.open(io.BytesIO(b)).convert("L")
                f = int(g["NAXIS1"] or im.width) // 1024
                im.resize((im.width // f, im.height // f), Image.BOX).save(OUT / f"{key}.png")
                with open(META, "a") as fh:
                    fh.write(json.dumps({"key": key, "stem": stem, "src": tag, "ha_date": date, "factor": f, **g}) + "\n")
                break
            except Exception as e:
                print("fallo", key, intento, str(e)[:120], flush=True)
                time.sleep(10 * (intento + 1))
        time.sleep(0.3)
    if n % 50 == 0:
        print(n, time.strftime("%H:%M:%S"), flush=True)
print("hecho", flush=True)
