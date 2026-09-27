"""Informe de la fase 3: junta runs/phase3/*.json en INFORME_FASE3.md y dibuja las curvas de escala.

    python scripts/report_phase3.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "runs" / "phase3"
NP_C, TF_C, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b68", "#fcfcfb"


def J(name):
    f = D / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def scale_plot(np_rows, tf_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("combos_nuevas", "Combinaciones nuevas"), ("daño50", "Con el 50 % del estado borrado"),
               ("palabra_nueva_k5", "Palabra nueva con 5 ejemplos")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURF)
    for ax, (k, title) in zip(axes, metrics):
        ax.set_facecolor(SURF)
        for rows, col, lab in ((np_rows, NP_C, "NeuroPixel"), (tf_rows, TF_C, "Transformer")):
            pts = sorted((r["params"], r[k]) for r in rows if k in r)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2, label=lab)
            ax.annotate(lab, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=INK,
                        fontsize=9, va="center")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("parámetros (escala logarítmica)", color=MUTED, fontsize=9)
        ax.grid(True, color="#e6e6e3", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c9c5")
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("acierto", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Curvas de escala: capacidad frente a tamaño (tarea de papeles, combinaciones no vistas)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(D / "escala.png", dpi=130)
    plt.close(fig)


def main():
    L = ["# NeuroPixel · Fase 3 «lienzo vivo» — informe", ""]
    # 9 escala
    np_rows, tf_rows = [], []
    for f in sorted(D.glob("scale_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        r["modelo"] = f.stem.replace("scale_", "")
        (np_rows if r["modelo"].startswith("np") else tf_rows).append(r)
    if np_rows or tf_rows:
        cols = ["modelo", "params", "combos_nuevas", "daño50", "palabra_nueva_k5", "viejo_tras_palabra", "segundos"]
        L += ["## 9 · Curvas de escala", "", table(sorted(np_rows, key=lambda r: r["params"]) +
                                                   sorted(tf_rows, key=lambda r: r["params"]), cols), "",
              "![escala](escala.png)", ""]
        try:
            scale_plot(np_rows, tf_rows)
        except Exception as e:  # noqa: BLE001
            L.append(f"(gráfica no generada: {e})")
    # 5 composición lejana
    far = [dict(modelo=f.stem.replace("far_", ""), **json.loads(f.read_text(encoding="utf-8")))
           for f in sorted(D.glob("far_*.json"))]
    if far:
        L += ["## 5 · Composición lejana (papel y relleno separados, lienzo 12×12)", "",
              table(far, ["modelo", "params", "combos_nuevas", "entrenamiento", "AGENTE", "ACCION", "PACIENTE",
                          "LUGAR", "segundos"]), ""]
    # 3 memoria
    mem = J("memory")
    if mem:
        cols = ["modelo", "params"] + [k for k in next(iter(mem.values())) if k.startswith("retraso")]
        L += ["## 3 · Memoria persistente (hechos de uno en uno, pregunta tras N fotogramas vacíos; entrenado hasta 8)", "",
              table([dict(modelo=k, **v) for k, v in mem.items()], cols), ""]
    # 1 estabilidad
    st = {**(J("stable_fijo16") or {}), **(J("stable_reposo") or {})}
    if st:
        cols = ["config"] + [k for k in next(iter(st.values())) if k != "segundos"]
        L += ["## 1 · Estado de reposo (acierto según pasos de pensamiento; daño del 50 % en el paso 8)", "",
              table([dict(config=k, **v) for k, v in st.items()], cols), ""]
    # 4 crecimiento
    gg, gs = J("grow_grow"), J("grow_seq")
    if gg or gs:
        L += ["## 4 · Crecimiento automático por resonancia (temas 0,1,2 y repaso del 0)", ""]
        if gg:
            L += [f"- Lienzos creados: **{gg['lienzos']}** (esperado 3; el repaso no debe crear otro)"]
            L += [f"  - tema {r['tema']}: resonancias {r['resonancias']} umbral {r['umbral']} → {r['accion']}" for r in gg["registro"]]
            L += [f"- Acierto final por tema (enrutado por escáner): " + ", ".join(f"tema {i}: {gg.get(f'tema{i}_escaner')}" for i in range(3))]
        if gs:
            L += [f"- Un solo lienzo en secuencia (olvido): " + ", ".join(f"tema {i}: {gs.get(f'tema{i}_final')}" for i in range(3))]
        L.append("")
    # 6 energía
    en = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(D.glob("energy_*.json"))]
    if en:
        L += ["## 6 · Energía (penalizar actividad)", "",
              table(sorted(en, key=lambda r: r["lambda"]), ["lambda", "acierto", "pixeles_activos_final", "actualizaciones_activas"]), ""]
    # 2 palabra nueva
    nw = J("newword")
    if nw:
        L += ["## 2 · Palabra nueva sin olvido", ""]
        for base, r in nw.items():
            L += [f"**{base}** (viejo antes: {r['viejo_antes']})", ""]
            rows = []
            for meth in ("simple", "norma", "repaso", "norma+repaso"):
                rows.append({"método": meth, **{f"k{k}_{w}": r.get(f"{meth}_k{k}_{w}") for k in (1, 5) for w in ("nueva", "viejo")}})
            L += [table(rows, ["método", "k1_nueva", "k1_viejo", "k5_nueva", "k5_viejo"]), ""]
    # 7 LLM
    llm = J("llm")
    if llm:
        L += ["## 7 · Coste por respuesta frente a un LLM local", "", "```", json.dumps(llm, indent=1, ensure_ascii=False), "```", ""]
    dr = J("dream")
    if dr:
        L += ["## 8 · Imaginación (rellenar un hueco con algo plausible)", "", "```", json.dumps(dr, indent=1, ensure_ascii=False), "```", ""]
    (D / "INFORME_FASE3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("informe:", D / "INFORME_FASE3.md")


if __name__ == "__main__":
    main()
