"""Genera las figuras del README (en inglés) a partir del modelo entrenado y de results/.

    python scripts/make_readme_figures.py
Salida: docs/img/banner.png, thinking.gif, scaling_en.png, capabilities_en.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.model import NeuroPixel  # noqa: E402
from neuropixel.task import NOUNS, PLACES, ROLES, VERBS, RoleTask, Vocab  # noqa: E402

IMG = ROOT / "docs" / "img"
RES = ROOT / "results" / "phase3"
NP_C, BASE_C, INK, MUTED, SURF, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b68", "#fcfcfb", "#e6e6e3"
EN = {"AGENTE": "AGENT", "ACCION": "ACTION", "PACIENTE": "PATIENT", "LUGAR": "PLACE", "<vacío>": "",
      "perro": "dog", "gato": "cat", "niña": "girl", "robot": "robot", "médico": "doctor", "zorro": "fox",
      "abuelo": "grandpa", "pájaro": "bird", "pintora": "painter", "lobo": "wolf", "cocinero": "cook",
      "caballo": "horse", "muerde": "bites", "empuja": "pushes", "dibuja": "draws", "persigue": "chases",
      "saluda": "greets", "cura": "heals", "lava": "washes", "mira": "watches", "abraza": "hugs",
      "llama": "calls", "casa": "house", "bosque": "forest", "playa": "beach", "calle": "street",
      "escuela": "school", "río": "river", "mercado": "market", "huerto": "garden"}


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_model():
    task = RoleTask(8, 8, seed=0)
    m = NeuroPixel(len(Vocab()), task.out_pos)
    m.load_state_dict(torch.load(ROOT / "runs" / "phase3" / "stable_reposo_s0.pt", map_location="cpu"))
    return m.eval(), task


def state_rgb(frames):
    """Estado (T,C,H,W) -> colores por PCA común; negro donde no hay actividad."""
    T, C, H, W = frames.shape
    x = frames.permute(0, 2, 3, 1).reshape(-1, C)
    norm = x.norm(dim=1)
    xc = x - x.mean(0)
    _, _, v = torch.pca_lowrank(xc, q=3)
    p = xc @ v[:, :3]
    lo, hi = p.quantile(0.02, 0), p.quantile(0.98, 0)
    rgb = ((p - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    rgb = 0.15 + 0.85 * rgb
    act = (norm / norm.quantile(0.9)).clamp(0, 1).unsqueeze(1)
    return (rgb * act).reshape(T, H, W, 3).numpy()


def draw_canvas(img, x0, y0, rgb, cell, gap=4, radius=6):
    glow = Image.new("RGB", img.size, (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(img)
    H, W, _ = rgb.shape
    for r in range(H):
        for c in range(W):
            col = tuple(int(255 * v) for v in rgb[r, c])
            box = (x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell - gap, y0 + (r + 1) * cell - gap)
            gd.rounded_rectangle(box, radius, fill=col)
            d.rounded_rectangle(box, radius, fill=tuple(max(12, v) for v in col))
    return glow


def banner(model, task):
    g = torch.Generator().manual_seed(11)
    c, _ = task.sample(1, "test", g)
    with torch.no_grad():
        fr = model(c, trace=True, steps=24)["frames"][0]
    rgb = state_rgb(fr)
    W_, H_ = 1600, 440
    img = Image.new("RGB", (W_, H_), (11, 14, 20))
    d = ImageDraw.Draw(img)
    for i in range(0, W_, 40):  # retícula tenue
        d.line([(i, 0), (i, H_)], fill=(18, 22, 30))
    for j in range(0, H_, 40):
        d.line([(0, j), (W_, j)], fill=(18, 22, 30))
    steps = [0, 2, 5, 10, 24]
    cell, x = 30, 640
    glow_all = Image.new("RGB", img.size, (0, 0, 0))
    for k, t in enumerate(steps):
        gl = draw_canvas(img, x + k * 190, 110, rgb[t], cell // 1 - 10 if False else 21, gap=3, radius=4)
        glow_all = Image.fromarray(np.maximum(np.array(glow_all), np.array(gl)))
        d.text((x + k * 190 + 50, 300), f"step {t}", fill=(150, 160, 175), font=font(18))
        if k < len(steps) - 1:
            d.text((x + k * 190 + 172, 180), "›", fill=(90, 100, 120), font=font(36, True))
    glow_all = glow_all.filter(ImageFilter.GaussianBlur(9))
    img = Image.fromarray(np.clip(np.array(img).astype(int) + (np.array(glow_all) * 0.55).astype(int), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.text((70, 110), "NeuroPixel", fill=(240, 244, 250), font=font(84, True))
    d.text((74, 215), "A neural network that thinks in pixels", fill=(170, 200, 240), font=font(30))
    d.text((74, 268), "learned colour dictionary  ·  local cellular dynamics", fill=(130, 140, 155), font=font(20))
    d.text((74, 296), "self-repairing  ·  growing canvases  ·  readable thoughts", fill=(130, 140, 155), font=font(20))
    d.text((640, 350), "real canvas state of a trained model (PCA of 48 channels → RGB); black = no activity",
           fill=(95, 105, 120), font=font(15))
    img.save(IMG / "banner.png", optimize=True)


def cat_color(tok):
    import colorsys
    for group, hue in ((NOUNS, 0.60), (VERBS, 0.08), (PLACES, 0.33)):
        if tok in group:
            k = group.index(tok) / max(1, len(group) - 1)
            r, g, b = colorsys.hls_to_rgb(hue + 0.06 * (k - 0.5), 0.42 + 0.18 * k, 0.75)
            return np.array([r, g, b]) * 255
    if tok in ROLES:
        return np.array([185, 185, 190.0])
    return np.array([30, 30, 34.0])


def thinking_gif(model, task):
    g = torch.Generator().manual_seed(11)
    c, y = task.sample(1, "test", g)
    names = task.v.tokens
    T = 24
    with torch.no_grad():
        out = model(c, trace=True, steps=T)
        p = model.lens_logits(out["frames"][0]).softmax(-1)          # T+1,H,W,V
    conf, word = p.max(-1)
    pairs = []
    for r in range(7):
        for col in range(7):
            tk = names[int(c[0, r, col])]
            if tk in ROLES:
                pairs.append((EN[tk], EN[names[int(c[0, r, col + 1])]]))
    q = EN[names[int(c[0, task.query_pos[0], task.query_pos[1]])]]
    pred = names[int(out["logits"][0].argmax())]
    cell, pad, side = 70, 24, 330
    Wd, Hd = side + 8 * cell + 2 * pad, 8 * cell + 2 * pad + 60
    frames = []
    for t in list(range(T + 1)) + [T] * 8:
        im = Image.new("RGB", (Wd, Hd), (11, 14, 20))
        d = ImageDraw.Draw(im)
        d.text((pad, pad), "Input sentence", fill=(150, 160, 175), font=font(18, True))
        for i, (role, w) in enumerate(pairs):
            d.text((pad, pad + 36 + i * 30), f"{role:<8} = {w}", fill=(225, 230, 238), font=font(20))
        d.text((pad, pad + 176), f"Question:  {q}?", fill=(255, 214, 90), font=font(20, True))
        d.text((pad, pad + 230), f"Thinking step {t}/{T}", fill=(150, 160, 175), font=font(18))
        d.rectangle((pad, pad + 262, pad + 260, pad + 272), outline=(60, 70, 85))
        d.rectangle((pad, pad + 262, pad + int(260 * t / T), pad + 272), fill=(42, 120, 214))
        if t == T:
            ok = pred == names[int(y[0])]
            d.text((pad, pad + 300), f"Answer:  {EN[pred]}", fill=(120, 220, 140) if ok else (240, 120, 100), font=font(24, True))
            d.text((pad, pad + 336), "read from the output pixel", fill=(130, 140, 155), font=font(15))
            d.text((pad, pad + 356), "through the same dictionary", fill=(130, 140, 155), font=font(15))
        x0, y0 = side, pad
        for r in range(8):
            for col in range(8):
                tk = names[int(word[t, r, col])]
                a = float(conf[t, r, col])
                colr = tuple(int(v) for v in cat_color(tk) * (0.18 + 0.82 * a))
                box = (x0 + col * cell, y0 + r * cell, x0 + (col + 1) * cell - 4, y0 + (r + 1) * cell - 4)
                d.rounded_rectangle(box, 7, fill=colr)
                if c[0, r, col] != 0:
                    d.rounded_rectangle(box, 7, outline=(235, 240, 248), width=2)
                if (r, col) in (task.query_pos, task.out_pos):
                    d.rounded_rectangle(box, 7, outline=(255, 214, 90), width=3)
                lab = EN.get(tk, tk)
                txt = (15, 15, 18) if sum(colr) > 420 else (235, 238, 244)
                d.text((box[0] + 6, box[1] + 8), lab[:8], fill=txt, font=font(13, True))
                d.text((box[0] + 6, box[1] + 28), f"{a:.2f}", fill=txt, font=font(12))
        d.text((side, Hd - 44), "Each pixel shows the word the dictionary reads from its state (and confidence).",
               fill=(130, 140, 155), font=font(15))
        d.text((side, Hd - 24), "white = input pixels   yellow = question / output pixel   blue = nouns  orange = verbs  green = places",
               fill=(110, 120, 135), font=font(13))
        frames.append(im)
    frames[0].save(IMG / "thinking.gif", save_all=True, append_images=frames[1:], duration=[260] * (T + 1) + [260] * 8,
                   loop=0, optimize=True)


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=10)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c5")
    ax.tick_params(colors=MUTED, labelsize=9)


def J(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def scaling_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {}
    for f in RES.glob("scale_*.json"):
        base = f.stem.replace("scale_", "").split("_s")[0]
        groups.setdefault(base, []).append(json.loads(f.read_text(encoding="utf-8")))
    rows = {}
    for base, rs in groups.items():
        r = {"params": rs[0]["params"]}
        for k in ("combos_nuevas", "daño50"):
            r[k] = np.mean([x[k] for x in rs])
        ok = [x for x in rs if "palabra_nueva_k5_min" in x]
        if ok:
            r["new"] = np.mean([x["palabra_nueva_k5"] for x in ok])
            r["old"] = np.mean([x["viejo_tras_palabra"] for x in ok])
        rows[base] = r
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURF)
    panels = [("combos_nuevas", "Accuracy on unseen combinations"), ("daño50", "Accuracy with 50 % of the state erased"),
              ("old", "Old knowledge kept after learning a new word")]
    for ax, (k, title) in zip(axes, panels):
        style(ax, title)
        for pref, col, lab in (("np", NP_C, "NeuroPixel"), ("tf", BASE_C, "Transformer")):
            pts = sorted((r["params"], r[k]) for b, r in rows.items() if b.startswith(pref) and k in r)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col, lw=2.2, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2, label=lab)
            ax.annotate(lab, (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", color=INK, fontsize=9.5,
                        va="center")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.03)
        ax.set_xlim(3.5e3, 3e6)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("parameters (log scale)", color=MUTED, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.suptitle("Scaling: the transformer plateaus at 54 % from 13 k to 811 k parameters; NeuroPixel climbs to 99.7 %",
                 color=INK, fontsize=13, x=0.012, ha="left", y=1.0)
    fig.tight_layout()
    fig.savefig(IMG / "scaling_en.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def capabilities_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), facecolor=SURF)
    pct = matplotlib.ticker.PercentFormatter(1.0)
    # (a) reposo
    ax = axes[0, 0]
    style(ax, "Rest-state training: stable thinking of any length")
    st_f, st_r = J("stable_fijo16")["fijo16_s0"], J("stable_reposo")["reposo_s0"]
    steps = [8, 16, 24, 32, 48, 64]
    for d, col, lab in ((st_r, NP_C, "rest-state (random steps + damage)"), (st_f, BASE_C, "fixed 16 steps")):
        ys = [d[f"pasos{s}"] for s in steps]
        ax.plot(steps, ys, color=col, lw=2.2, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2, label=lab)
    ax.set_xlabel("thinking steps at test time", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(pct)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    ax.annotate("50 % of the state erased at step 8\n→ recovers to 99.4 %", (16, st_r["daño50_t8_pasos16"]),
                xytext=(36, 0.80), color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    # (b) memoria
    ax = axes[0, 1]
    style(ax, "Persistent memory: facts seen one by one, asked later")
    mem = J("memory")
    ds = [0, 4, 8, 12, 16, 24, 32]
    for key, col, lab in (("neuropixel", NP_C, "NeuroPixel (30 k)"), ("gru", BASE_C, "GRU (49 k)")):
        ax.plot(ds, [mem[key][f"retraso{d}"] for d in ds], color=col, lw=2.2, marker="o", ms=8,
                markeredgecolor=SURF, markeredgewidth=2, label=lab)
    ax.axvspan(0, 8, color="#eef3fb", zorder=0)
    ax.text(0.4, 0.06, "trained range", color=MUTED, fontsize=9)
    ax.set_xlabel("blank frames between facts and question", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(pct)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 0.12))
    # (c) crecer
    ax = axes[1, 0]
    style(ax, "Growing canvases: continual learning without forgetting")
    gn, gs = J("grow_novedad"), J("grow_seq")
    xs = np.arange(3)
    a = [gn[f"tema{i}_escaner"] for i in range(3)]
    b = [gs[f"tema{i}_final"] for i in range(3)]
    ax.bar(xs - 0.2, a, 0.38, color=NP_C, label="new canvas on novelty + scanner routing", edgecolor=SURF, linewidth=2)
    ax.bar(xs + 0.2, b, 0.38, color=BASE_C, label="one canvas, trained sequentially", edgecolor=SURF, linewidth=2)
    for i in range(3):
        ax.text(xs[i] - 0.2, a[i] + 0.02, f"{a[i]:.0%}", ha="center", color=INK, fontsize=9)
        ax.text(xs[i] + 0.2, b[i] + 0.02, f"{b[i]:.0%}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(xs, ["topic 1", "topic 2", "topic 3"])
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(pct)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=1)
    # (d) coste
    ax = axes[1, 1]
    style(ax, "Cost per answer vs a local LLM (same 200 questions, CPU)")
    llm = J("llm")
    npr, q = llm["neuropixel"], llm["qwen2_494m_cpu"]
    labels = ["NeuroPixel\n30 k params", "Qwen2\n494 M params"]
    vals = [npr["flops_por_respuesta"], q["flops_por_respuesta_aprox"]]
    ax.barh([1, 0], vals, color=[NP_C, BASE_C], height=0.5, edgecolor=SURF, linewidth=2)
    ax.set_xscale("log")
    ax.set_yticks([1, 0], labels)
    ax.set_xlabel("operations per answer (log scale)", color=MUTED, fontsize=9.5)
    ax.text(vals[0] * 1.3, 1, f"{vals[0] / 1e6:.0f} MFLOP · {npr['cpu_latencia_1_ms']:.1f} ms · {npr['cpu_acierto']:.1%} correct",
            va="center", color=INK, fontsize=9.5)
    ax.text(vals[1] / 1.3, 0, f"{vals[1] / 1e9:.0f} GFLOP · {q['ms_por_respuesta']:.0f} ms · {q['acierto']:.0%} correct",
            va="center", ha="right", color="white", fontsize=9.5)
    ax.set_xlim(1e6, 1e12)
    fig.tight_layout(h_pad=3)
    fig.savefig(IMG / "capabilities_en.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, task = load_model()
    banner(model, task)
    thinking_gif(model, task)
    scaling_chart()
    capabilities_chart()
    print("figuras en", IMG)


if __name__ == "__main__":
    main()
