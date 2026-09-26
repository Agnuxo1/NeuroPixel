"""Batería nocturna de capacidades 'biológicas' (ordenadas de más corta a más larga).

  T1 autorreparación   : dañar el estado a mitad de pensar (minutos, sin entrenar)
  T2 palabra nueva     : aprender 'delfín' con 1, 5 o 20 ejemplos, solo el diccionario (~15 min)
  T3 aprendizaje continuo + lienzos que crecen (idea de Fran): A -> B, olvido y enrutador (~45 min)

Siempre se compara con el transformer del mismo tamaño y protocolo.
    python scripts/battery.py all --wait-for ret_np_s0 ret_np_s1 ret_np_noschool_s0
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.model import NeuroPixel, TinyTransformer  # noqa: E402
from neuropixel.safety import choose_device  # noqa: E402
from neuropixel.task import NOUNS, RoleTask, Vocab  # noqa: E402
from scripts.eval_roles import load  # noqa: E402

OUT = ROOT / "runs" / "battery"
DEV = None


def acc(model, c, y, **kw):
    with torch.no_grad():
        return round((model(c, **kw)["logits"].argmax(-1) == y).float().mean().item(), 4)


def test_set(task, n=2000, seed=5):
    c, y = task.sample(n, "test", torch.Generator().manual_seed(seed))
    return c.to(DEV), y.to(DEV)


# ----------------------------------------------------------------- T1 autorreparación
def t1_damage():
    res = {}
    fracs = [0.0, 0.1, 0.3, 0.5]
    for run in ["gpu30k_np_s0", "gpu30k_np_s1", "gpu30k_np_s2", "gpu30k_np_lens_s0",
                "gpu30k_tf_s0", "gpu30k_tf_s1", "gpu30k_tf_s2"]:
        m, t = load(ROOT / "runs" / run)
        m.to(DEV).eval()
        c, y = test_set(t)
        r = {}
        for f in fracs:
            g = torch.Generator(device=DEV).manual_seed(1)
            if isinstance(m, NeuroPixel):
                for td in (4, 8):
                    mask = (torch.rand(c.shape[0], 1, *c.shape[1:], generator=g, device=DEV) >= f).float()

                    def hook(tt, s, td=td, mask=mask):
                        return s * mask if tt == td else s
                    r[f"f{f}_t{td}"] = acc(m, c, y, hook=hook)
                    r[f"f{f}_t{td}_+8pasos"] = acc(m, c, y, hook=hook, steps=24)
            else:
                mask = (torch.rand(c.shape[0], c.shape[1] * c.shape[2], 1, generator=g, device=DEV) >= f).float()
                h = m.enc.layers[0].register_forward_hook(lambda mod, i, o, mask=mask: o * mask)
                r[f"f{f}_capa1"] = acc(m, c, y)
                h.remove()
        res[run] = r
        print("T1", run, json.dumps(r), flush=True)
    return res


# ----------------------------------------------------------------- T2 palabra nueva
def expand(model, new: int):
    """Copia el modelo con 'new' filas nuevas de diccionario; solo esas filas aprenden."""
    old = model.embed.num_embeddings if isinstance(model, NeuroPixel) else model.tok.num_embeddings
    V = old + new
    if isinstance(model, NeuroPixel):
        m2 = NeuroPixel(V, model.out_pos, steps=model.steps).to(DEV)
        sd = model.state_dict()
        w = torch.randn(V, sd["embed.weight"].shape[1], device=DEV) * sd["embed.weight"][1:].std()
        w[:old] = sd["embed.weight"]
        sd["embed.weight"] = w
        m2.load_state_dict(sd)
        params = [m2.embed.weight]
    else:
        h, w_ = int(model.pos.shape[0] ** 0.5), int(model.pos.shape[0] ** 0.5)
        m2 = TinyTransformer(V, h, w_, (h - 1, w_ - 1)).to(DEV)
        sd = model.state_dict()
        for k in ("tok.weight", "head.weight"):
            w = torch.randn(V, sd[k].shape[1], device=DEV) * sd[k].std()
            w[:old] = sd[k]
            sd[k] = w
        b = torch.zeros(V, device=DEV)
        b[:old] = sd["head.bias"]
        b[old:] = sd["head.bias"].mean()
        sd["head.bias"] = b
        m2.load_state_dict(sd)
        params = [m2.tok.weight, m2.head.weight, m2.head.bias]
    for p in m2.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    rows = torch.zeros(V, 1, device=DEV)
    rows[old:] = 1
    for p in params:
        p.register_hook(lambda g, rows=rows: g * (rows if g.dim() == 2 else rows.squeeze(1)))
    return m2, params, old


def with_new_word(task, n, split, new_id, seed, ask_new=True):
    c, y, m = task.sample(n, split, torch.Generator().manual_seed(seed), meta=True)
    role = torch.where(torch.rand(n, generator=torch.Generator().manual_seed(seed + 1)) < 0.5, 0, 2)
    b = torch.arange(n)
    c[b, m["rows"][b, role], m["cols"][b, role] + 1] = new_id
    if ask_new:
        qpos = task.query_pos
        c[b, qpos[0], qpos[1]] = task.role_ids[role]
        y = torch.full((n,), new_id)
    return c.to(DEV), y.to(DEV)


def t2_new_word():
    res = {}
    for run in ["gpu30k_np_s0", "gpu30k_np_lens_s0", "gpu30k_tf_s0"]:
        base, t = load(ROOT / "runs" / run)
        base.to(DEV).eval()
        c_old, y_old = test_set(t)
        r = {"antes_viejo": acc(base, c_old, y_old)}
        for k in (1, 5, 20):
            m, params, new_id = expand(base, 1)
            m.eval()                      # sin disparo aleatorio: solo cambia el diccionario
            ck, yk = with_new_word(t, k, "train", new_id, seed=100 + k)
            opt = torch.optim.Adam(params, lr=1e-2)
            for _ in range(300):
                loss = F.cross_entropy(m(ck)["logits"], yk)
                opt.zero_grad()
                loss.backward()
                opt.step()
            cn, yn = with_new_word(t, 1000, "test", new_id, seed=7)
            r[f"k{k}_nueva"] = acc(m, cn, yn)
            r[f"k{k}_viejo"] = acc(m, c_old, y_old)
        res[run] = r
        print("T2", run, json.dumps(r), flush=True)
    return res


# ----------------------------------------------------------------- T3 continuo + lienzos que crecen
def train_role(model, task, iters, lens_aux=0.0, batch=512, lr=2e-3, seed=0):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
    g = torch.Generator(device=DEV).manual_seed(seed)
    np_ = isinstance(model, NeuroPixel)
    for _ in range(iters):
        c, y = task.sample(batch, "train", g, DEV)
        out = model(c, lens_every=4 if (np_ and lens_aux) else 0)
        loss = F.cross_entropy(out["logits"], y)
        if np_ and lens_aux:
            L = out["lens"].shape[1]
            mk = (c != 0).unsqueeze(1).expand(-1, L, -1, -1)
            loss = loss + lens_aux * F.cross_entropy(out["lens"][mk], c.unsqueeze(1).expand(-1, L, -1, -1)[mk])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    return model


def routed(models, c):
    """Enrutador por confianza: responde el lienzo más seguro de su respuesta."""
    with torch.no_grad():
        probs = torch.stack([m(c)["logits"].softmax(-1) for m in models])     # M,B,V
        conf, pred = probs.max(-1)
        pick = conf.argmax(0)
        return pred.gather(0, pick.unsqueeze(0)).squeeze(0), pick


def t3_continual(iters):
    A, B = list(range(6)), list(range(6, 12))
    tA, tB = RoleTask(8, 8, nouns_allowed=A, seed=0), RoleTask(8, 8, nouns_allowed=B, seed=0)
    V = len(Vocab())
    cA, yA = test_set(tA)
    cB, yB = test_set(tB)
    res = {}
    for kind in ("neuropixel", "transformer"):
        t0 = time.time()
        mk = (lambda: NeuroPixel(V, tA.out_pos)) if kind == "neuropixel" else (lambda: TinyTransformer(V, 8, 8, tA.out_pos))
        torch.manual_seed(0)
        mA = train_role(mk().to(DEV), tA, iters, lens_aux=0.3 if kind == "neuropixel" else 0)
        r = {"A_tras_A": acc(mA, cA, yA), "B_tras_A": acc(mA, cB, yB)}
        mB = train_role(copy.deepcopy(mA), tB, iters, lens_aux=0.3 if kind == "neuropixel" else 0, seed=1)
        r["secuencial_A_tras_B"] = acc(mB, cA, yA)          # olvido
        r["secuencial_B_tras_B"] = acc(mB, cB, yB)
        for name, (c, y) in {"A": (cA, yA), "B": (cB, yB)}.items():
            pred, pick = routed([mA, mB], c)
            r[f"crecer_{name}"] = round((pred == y).float().mean().item(), 4)
            r[f"crecer_{name}_elige_lienzo_correcto"] = round((pick == (0 if name == "A" else 1)).float().mean().item(), 4)
        r["segundos"] = round(time.time() - t0, 1)
        res[kind] = r
        print("T3", kind, json.dumps(r), flush=True)
    return res


def summary(res):
    lines = ["# Batería nocturna NeuroPixel", ""]
    if "t1" in res:
        lines += ["## T1 Autorreparación (acierto con combinaciones nuevas)", ""]
        keys = sorted({k for v in res["t1"].values() for k in v})
        lines += ["| ejecución | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
        for run, v in res["t1"].items():
            lines.append(f"| {run} | " + " | ".join(str(v.get(k, "")) for k in keys) + " |")
        lines.append("")
    if "t2" in res:
        lines += ["## T2 Palabra nueva ('delfín'), solo diccionario", ""]
        keys = list(next(iter(res["t2"].values())))
        lines += ["| ejecución | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
        for run, v in res["t2"].items():
            lines.append(f"| {run} | " + " | ".join(str(v[k]) for k in keys) + " |")
        lines.append("")
    if "t3" in res:
        lines += ["## T3 Aprendizaje continuo y lienzos que crecen", ""]
        keys = list(next(iter(res["t3"].values())))
        lines += ["| modelo | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
        for run, v in res["t3"].items():
            lines.append(f"| {run} | " + " | ".join(str(v[k]) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["t1", "t2", "t3", "all"])
    ap.add_argument("--wait-for", nargs="*", default=[])
    ap.add_argument("--t3-iters", type=int, default=15000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    for run in a.wait_for:  # esperar a que acabe lo anterior (p. ej. la retina)
        while not (ROOT / "runs" / run / "result.json").exists():
            time.sleep(60)
    DEV = choose_device(a.device, threads=4, vram_cap_gib=6)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    f = OUT / "results.json"
    if f.exists():
        res = json.loads(f.read_text(encoding="utf-8"))
    steps = ["t1", "t2", "t3"] if a.which == "all" else [a.which]
    for s in steps:
        t0 = time.time()
        res[s] = {"t1": t1_damage, "t2": t2_new_word}[s]() if s != "t3" else t3_continual(a.t3_iters)
        res[s + "_segundos"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        (OUT / "RESUMEN.md").write_text(summary(res), encoding="utf-8")
    print("FIN BATERÍA", flush=True)


if __name__ == "__main__":
    main()
