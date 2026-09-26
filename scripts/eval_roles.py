"""Acierto por papel en combinaciones no vistas y diagnóstico del problema de ligadura.

    python scripts/eval_roles.py runs/np_v1 runs/tf_v1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.model import NeuroPixel, TinyTransformer  # noqa: E402
from neuropixel.task import ROLES, RoleTask  # noqa: E402


def load(run: Path):
    a = json.loads((run / "result.json").read_text(encoding="utf-8"))["args"]
    t = RoleTask(a["size"], a["size"], seed=a["seed"])
    if a["model"] == "neuropixel":
        m = NeuroPixel(len(t.v), t.out_pos, steps=a["steps"])
    else:
        m = TinyTransformer(len(t.v), a["size"], a["size"], t.out_pos)
    m.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return m.eval(), t


@torch.no_grad()
def report(run: Path, n: int = 2000):
    m, t = load(run)
    c, y = t.sample(n, "test", torch.Generator().manual_seed(5))
    p = m(c)["logits"].argmax(-1)
    q = c[:, t.query_pos[0], t.query_pos[1]]
    res = {"run": run.name, "total": round((p == y).float().mean().item(), 3)}
    for k, r in enumerate(ROLES):
        s = q == t.role_ids[k]
        res[r] = round((p[s] == y[s]).float().mean().item(), 3)
    swaps = 0
    errs = 0
    for k in (0, 2):  # AGENTE <-> PACIENTE: ¿elige el sustantivo del otro papel?
        for b in ((q == t.role_ids[k]) & (p != y)).nonzero().flatten().tolist():
            r_, c_ = (c[b, :-1] == t.role_ids[2 - k]).nonzero()[0].tolist()
            swaps += int(p[b] == c[b, r_, c_ + 1])
            errs += 1
    res["errores_agente_paciente_por_intercambio"] = f"{swaps}/{errs}"
    return res


if __name__ == "__main__":
    torch.set_num_threads(2)
    for r in sys.argv[1:]:
        print(json.dumps(report(Path(r)), ensure_ascii=False))
