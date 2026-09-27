"""Registra un entrenamiento en el historial (kaggle/historial.jsonl + kaggle/HISTORIAL.md).

    python kaggle/log_run.py --comp filament --run np_ret_v2 --linea L2 \
        --debiles "..." --cambiar "..." --decision "..." [--kaggle 0.31] [--envio submission.csv] [--tag filament-v2]

Rellena solo: configuración, parámetros, tiempo, validación, tendencia de la curva y commit.
El análisis (puntos débiles, qué cambiar, decisión) lo escribe quien entrena.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kaggle"


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def trend(log, key="PQ"):
    """¿Mejoraría con más entrenamiento? según las últimas validaciones."""
    v = [r[key] for r in log if key in r]
    if len(v) < 3:
        return "pocas validaciones para saberlo"
    best_i = max(range(len(v)), key=lambda i: v[i])
    if best_i == len(v) - 1 and v[-1] > v[-2]:
        return "sí: seguía subiendo en la última validación"
    if best_i >= len(v) - 2:
        return "quizá: cerca de la meseta"
    return f"no: el mejor punto fue la validación {best_i + 1} de {len(v)} (sobreajuste o inestabilidad)"


def render(entries):
    L = ["# Historial de entrenamientos y envíos", "",
         "Una fila por entrenamiento. Para volver a una versión: `git checkout <commit>` (o la etiqueta).", "",
         "| # | Fecha | Concurso | Ejecución | Línea | Params | Tiempo | Validación | Kaggle | ¿Más entrenamiento? | Commit / etiqueta |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(entries, 1):
        L.append(f"| {i} | {e['fecha']} | {e['concurso']} | {e['ejecucion']} | {e['linea']} | {e['params']} | "
                 f"{e['tiempo_min']} min | {e['validacion']} | {e.get('kaggle') or '—'} | {e['tendencia']} | "
                 f"{e['commit']}{' / ' + e['tag'] if e.get('tag') else ''} |")
    L.append("")
    for i, e in enumerate(entries, 1):
        L += [f"## {i}. {e['concurso']} · {e['ejecucion']} ({e['fecha']})", "",
              f"- **Qué se hizo:** {e.get('que', '')}",
              f"- **Configuración:** `{json.dumps(e['config'], ensure_ascii=False)}`",
              f"- **Tiempo:** {e['tiempo_min']} min en {e.get('hardware', 'RTX 3090 local')}",
              f"- **Validación local:** {e['validacion']} · **Kaggle:** {e.get('kaggle') or 'sin enviar'}",
              f"- **¿Mejoraría con más entrenamiento?** {e['tendencia']}",
              f"- **Puntos débiles:** {e.get('debiles', '')}",
              f"- **Cambiar / quitar / mejorar:** {e.get('cambiar', '')}",
              f"- **Decisión:** {e.get('decision', '')}",
              f"- **Volver atrás:** `git checkout {e['commit']}` · envío: `{e.get('envio') or '—'}`", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--linea", default="L2")
    ap.add_argument("--que", default="")
    ap.add_argument("--debiles", default="")
    ap.add_argument("--cambiar", default="")
    ap.add_argument("--decision", default="")
    ap.add_argument("--kaggle", default=None)
    ap.add_argument("--envio", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--hardware", default="RTX 3090 local")
    a = ap.parse_args()
    res = json.loads((K / a.comp / "runs" / a.run / "result.json").read_text(encoding="utf-8"))
    bp = res.get("best_postproc", {})
    val = (f"PQ {bp.get('PQ')} · Dice {bp.get('Dice')}" if bp.get("PQ") is not None
           else f"acierto {res.get('best_val_acc')}")
    e = {"fecha": dt.date.today().isoformat(), "concurso": a.comp, "ejecucion": a.run, "linea": a.linea,
         "params": res.get("params"), "tiempo_min": round(res.get("seconds", 0) / 60),
         "config": res.get("args"), "validacion": val, "posprocesado": {k: bp.get(k) for k in ("thr", "min_area", "close")},
         "tendencia": trend(res.get("log", [])), "kaggle": a.kaggle, "envio": a.envio, "tag": a.tag,
         "commit": git("rev-parse", "--short", "HEAD"), "que": a.que, "debiles": a.debiles,
         "cambiar": a.cambiar, "decision": a.decision, "hardware": a.hardware}
    if a.tag:
        git("tag", "-f", a.tag)
    f = K / "historial.jsonl"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    entries = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    (K / "HISTORIAL.md").write_text(render(entries), encoding="utf-8")
    print("registrado", a.run, "->", K / "HISTORIAL.md")


if __name__ == "__main__":
    main()
