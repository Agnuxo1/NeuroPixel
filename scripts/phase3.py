"""Fase 3, 'lienzo vivo': una prueba por subcomando; cada una guarda runs/phase3/<prueba>.json.

  stable   1 estado de reposo: pasos variables + daño en entrenamiento
  newword  2 palabra nueva sin olvido (norma acotada / repaso)
  memory   3 memoria persistente en flujo (vs GRU)
  grow     4 crecimiento automático por resonancia a lo largo de temas
  far      5 composición lejana (vs transformers pequeño y 7x)
  energy   6 energía: penalizar actividad
  scale    9 curvas de escala (NP 5k..230k vs TF 12k..800k)
  llm      7 coste por respuesta frente a Qwen2.5-7B local
  dream    8 imaginación (completar parejas sin verlas)
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
from neuropixel.model import NeuroPixel, TinyTransformer, n_params  # noqa: E402
from neuropixel.phase3 import (FrameGRU, MemoryTask, RoleTaskFar, expand_vocab,  # noqa: E402
                               memory_steps, np_stream)
from neuropixel.safety import choose_device  # noqa: E402
from neuropixel.task import ROLES, RoleTask, Vocab  # noqa: E402

OUT = ROOT / "runs" / "phase3"
DEV = None
V = len(Vocab())


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print("GUARDADO", name, flush=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@torch.no_grad()
def acc_of(model, c, y, **kw):
    model.eval()
    out = []
    for i in range(0, len(c), 4000):   # un solo bloque para n<=4000 (los ganchos de daño usan el lote entero)
        out.append(model(c[i:i + 4000], **kw)["logits"].argmax(-1) == y[i:i + 4000])
    return round(torch.cat(out).float().mean().item(), 4)


def tset(task, n=2000, seed=5):
    c, y = task.sample(n, "test", torch.Generator().manual_seed(seed))
    return c.to(DEV), y.to(DEV)


def train(model, task, iters, *, lens=0.3, batch=512, lr=2e-3, seed=0, steps_range=None,
          damage_p=0.0, damage_frac=0.3, act_l1=0.0, every=0, probe=None):
    """Bucle común. steps_range=(a,b) sortea pasos; damage_p = prob. de daño en entrenamiento."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
    g = torch.Generator(device=DEV).manual_seed(seed)
    isnp = isinstance(model, NeuroPixel)
    curve = []
    for it in range(1, iters + 1):
        c, y = task.sample(batch, "train", g, DEV)
        kw = {}
        if isnp:
            if steps_range:
                kw["steps"] = int(torch.randint(steps_range[0], steps_range[1] + 1, (1,), generator=g, device=DEV))
            if damage_p and torch.rand(1, generator=g, device=DEV).item() < damage_p:
                td = int(torch.randint(2, kw.get("steps", model.steps), (1,), generator=g, device=DEV))
                mask = (torch.rand(c.shape[0], 1, *c.shape[1:], generator=g, device=DEV) >= damage_frac).float()
                kw["hook"] = lambda t, s, td=td, mask=mask: s * mask if t == td else s
            if lens:
                kw["lens_every"] = 4
        out = model(c, **kw)
        loss = F.cross_entropy(out["logits"], y)
        if isnp and lens and "lens" in out:
            L = out["lens"].shape[1]
            mk = (c != 0).unsqueeze(1).expand(-1, L, -1, -1)
            loss = loss + lens * F.cross_entropy(out["lens"][mk], c.unsqueeze(1).expand(-1, L, -1, -1)[mk])
        if act_l1:
            loss = loss + act_l1 * out["activity"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if every and it % every == 0 and probe is not None:
            curve.append({"it": it, "acc": acc_of(model, *probe)})
            model.train()
    model.eval()
    return curve


def damage_hook(td, frac, shape, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    mask = (torch.rand(shape[0], 1, *shape[1:], generator=g, device=DEV) >= frac).float()
    return lambda t, s: s * mask if t == td else s


# =================================================================== 1 estable
def t_stable(seeds, iters, only=None):
    task = RoleTask(8, 8, seed=0)
    c, y = tset(task)
    res = {}
    cfgs = {"fijo16": {}, "reposo": {"steps_range": (12, 32), "damage_p": 0.5}}
    for name, cfg in cfgs.items():
        if only and name != only:
            continue
        for s in seeds:
            torch.manual_seed(s)
            m = NeuroPixel(V, task.out_pos).to(DEV)
            t0 = time.time()
            train(m, task, iters, seed=s, **cfg)
            r = {f"pasos{k}": acc_of(m, c, y, steps=k) for k in (8, 16, 24, 32, 48, 64)}
            for k in (16, 32, 48):
                r[f"daño50_t8_pasos{k}"] = acc_of(m, c, y, steps=k, hook=damage_hook(8, 0.5, c.shape))
            r["segundos"] = round(time.time() - t0)
            torch.save(m.state_dict(), OUT / f"stable_{name}_s{s}.pt")
            res[f"{name}_s{s}"] = r
            log("stable", name, s, r)
            save(f"stable_{name}", res)
    return res


# =================================================================== 2 palabra nueva sin olvido
def t_newword():
    task = RoleTask(8, 8, seed=0)
    c_old, y_old = tset(task)
    bases = {"np_lens30k": ROOT / "runs" / "gpu30k_np_lens_s0" / "model.pt"}
    for f in sorted(OUT.glob("stable_reposo_s*.pt"))[:1]:
        bases["np_reposo"] = f
    res = {}

    def with_new(n, split, new_id, seed):
        cc, yy, m = task.sample(n, split, torch.Generator().manual_seed(seed), meta=True)
        role = torch.where(torch.rand(n, generator=torch.Generator().manual_seed(seed + 1)) < 0.5, 0, 2)
        b = torch.arange(n)
        cc[b, m["rows"][b, role], m["cols"][b, role] + 1] = new_id
        cc[b, task.query_pos[0], task.query_pos[1]] = task.role_ids[role]
        return cc.to(DEV), torch.full((n,), new_id, device=DEV)

    for bname, path in bases.items():
        base = NeuroPixel(V, task.out_pos).to(DEV)
        base.load_state_dict(torch.load(path, map_location=DEV))
        base.eval()
        r = {"viejo_antes": acc_of(base, c_old, y_old)}
        for method in ("simple", "norma", "repaso", "norma+repaso"):
            for k in (1, 5):
                m, params, nid = expand_vocab(base, 1)
                ck, yk = with_new(k, "train", nid, 100 + k)
                opt = torch.optim.Adam(params, lr=1e-2)
                target_norm = base.embed.weight[1:].norm(dim=1).mean()
                g = torch.Generator().manual_seed(9)
                for _ in range(300):
                    cc, yy = ck, yk
                    if "repaso" in method:  # repaso breve de frases viejas (como el sueño)
                        co, yo = task.sample(16, "train", g)
                        cc, yy = torch.cat([ck, co.to(DEV)]), torch.cat([yk, yo.to(DEV)])
                    loss = F.cross_entropy(m(cc)["logits"], yy)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if "norma" in method:
                        with torch.no_grad():
                            w = m.embed.weight[nid]
                            w.mul_(target_norm / w.norm().clamp_min(1e-6))
                cn, yn = with_new(1000, "test", nid, 7)
                r[f"{method}_k{k}_nueva"] = acc_of(m, cn, yn)
                r[f"{method}_k{k}_viejo"] = acc_of(m, c_old, y_old)
        res[bname] = r
        log("newword", bname, r)
        save("newword", res)
    return res


# =================================================================== 3 memoria persistente
def t_memory(iters, seed=0):
    task = RoleTask(8, 8, seed=seed)
    mt = MemoryTask(task)
    res = {}
    for kind in ("neuropixel", "gru"):
        torch.manual_seed(seed)
        m = (NeuroPixel(V, task.out_pos) if kind == "neuropixel" else FrameGRU(V, 8, 8)).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.1)
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        t0 = time.time()
        m.train()
        for it in range(iters):
            d = int(torch.randint(0, 9, (1,), generator=g, device=DEV))
            frames, y = mt.sample(256, "train", g, DEV, d)
            out = np_stream(m, frames, memory_steps(d)) if kind == "neuropixel" else m.forward_frames(frames)
            loss = F.cross_entropy(out["logits"], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
        m.eval()
        r = {"params": n_params(m), "segundos": round(time.time() - t0)}
        with torch.no_grad():
            for d in (0, 4, 8, 12, 16, 24, 32):
                frames, y = mt.sample(1000, "test", torch.Generator(device=DEV).manual_seed(3), DEV, d)
                out = np_stream(m, frames, memory_steps(d)) if kind == "neuropixel" else m.forward_frames(frames)
                r[f"retraso{d}"] = round((out["logits"].argmax(-1) == y).float().mean().item(), 4)
        res[kind] = r
        log("memory", kind, r)
        save("memory", res)
    return res


# =================================================================== 4 crecimiento automático
def scanner_score(model, c):
    with torch.no_grad():
        p = model.lens_logits(model(c)["state"]).softmax(-1)
        own = p.gather(-1, c.unsqueeze(-1)).squeeze(-1)
        mk = (c != 0).float()
        return ((own * mk).sum((1, 2)) / mk.sum((1, 2)))


def t_grow(iters, mode):
    groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    seq = [0, 1, 2, 0]                                   # el último tema es un repaso
    tasks = [RoleTask(8, 8, nouns_allowed=gp, seed=0) for gp in groups]
    tests = [tset(t) for t in tasks]
    log_ = []
    if mode == "grow":
        canv, tau = [], None
        for step, ti in enumerate(seq):
            probe = tasks[ti].sample(512, "train", torch.Generator().manual_seed(50 + step))[0].to(DEV)
            scores = [round(scanner_score(m, probe).mean().item(), 4) for m in canv]
            if not canv or max(scores) < tau:
                m = copy.deepcopy(canv[max(range(len(canv)), key=lambda i: scores[i])]) if canv else NeuroPixel(V, tasks[0].out_pos).to(DEV)
                train(m, tasks[ti], iters, seed=step)
                canv.append(m)
                action = "lienzo nuevo"
            else:
                best = max(range(len(canv)), key=lambda i: scores[i])
                train(canv[best], tasks[ti], iters // 5, seed=step, lr=5e-4)
                action = f"repasa lienzo {best}"
            if tau is None:
                self_s = scanner_score(canv[0], probe).mean().item()
                tau = 0.75 * self_s
            log_.append({"tema": ti, "resonancias": scores, "umbral": round(tau, 4), "accion": action})
            log("grow", log_[-1])
        res = {"lienzos": len(canv), "registro": log_}
        for ti, (c, y) in enumerate(tests):
            sc = torch.stack([scanner_score(m, c) for m in canv])
            with torch.no_grad():
                preds = torch.stack([m(c)["logits"].argmax(-1) for m in canv])
            pick = sc.argmax(0)
            res[f"tema{ti}_escaner"] = round((preds.gather(0, pick[None])[0] == y).float().mean().item(), 4)
    else:  # un solo lienzo, aprendiendo en secuencia (referencia de olvido)
        m = NeuroPixel(V, tasks[0].out_pos).to(DEV)
        res = {"lienzos": 1}
        for step, ti in enumerate(seq):
            train(m, tasks[ti], iters if step < 3 else iters // 5, seed=step, lr=2e-3 if step < 3 else 5e-4)
            res[f"tras_paso{step}"] = [acc_of(m, *tests[k]) for k in range(3)]
            log("secuencial", step, res[f"tras_paso{step}"])
        for ti in range(3):
            res[f"tema{ti}_final"] = res["tras_paso3"][ti]
    save(f"grow_{mode}", res)
    return res


# =================================================================== 5 composición lejana
def t_far(kind, iters, seed=0):
    task = RoleTaskFar(12, 12, seed=seed)
    c, y = tset(task)
    torch.manual_seed(seed)
    if kind == "neuropixel":
        m = NeuroPixel(V, task.out_pos, steps=24)
    elif kind == "tf_small":
        m = TinyTransformer(V, 12, 12, task.out_pos)
    else:
        m = TinyTransformer(V, 12, 12, task.out_pos, d=96, layers=4, ff=192)
    m.to(DEV)
    t0 = time.time()
    curve = train(m, task, iters, seed=seed, lens=0.3 if kind == "neuropixel" else 0,
                  every=iters // 6, probe=(c, y))
    ctr, ytr = task.sample(2000, "train", torch.Generator().manual_seed(8))
    r = {"params": n_params(m), "combos_nuevas": acc_of(m, c, y),
         "entrenamiento": acc_of(m, ctr.to(DEV), ytr.to(DEV)), "curva": curve, "segundos": round(time.time() - t0)}
    q = c[:, task.query_pos[0], task.query_pos[1]]
    with torch.no_grad():
        p = torch.cat([m(c[i:i + 1000])["logits"].argmax(-1) for i in range(0, len(c), 1000)])
    for k, role in enumerate(ROLES):
        s = q == task.role_ids.to(DEV)[k]
        r[role] = round((p[s] == y[s]).float().mean().item(), 4)
    save(f"far_{kind}_s{seed}", r)
    log("far", kind, r)
    return r


# =================================================================== 6 energía
def t_energy(lam, iters, seed=0):
    task = RoleTask(8, 8, seed=seed)
    c, y = tset(task)
    torch.manual_seed(seed)
    m = NeuroPixel(V, task.out_pos).to(DEV)
    train(m, task, iters, seed=seed, act_l1=lam)
    stats = {"upd": [], "prev": None}

    def hook(t, s):
        if stats["prev"] is not None:
            stats["upd"].append(((s - stats["prev"]).abs().amax(1) > 0.01).float().mean().item())
        stats["prev"] = s
        return s
    with torch.no_grad():
        out = m(c[:1000], hook=hook)
    fin = (out["state"].abs().amax(1) > 0.05).float().mean().item()
    r = {"lambda": lam, "acierto": acc_of(m, c, y), "pixeles_activos_final": round(fin, 4),
         "actualizaciones_activas": round(sum(stats["upd"]) / max(1, len(stats["upd"])), 4)}
    save(f"energy_l{lam}", r)
    log("energy", r)
    return r


# =================================================================== 9 escala
NP_SIZES = {"np5k": (16, 48), "np14k": (32, 96), "np30k": (48, 128), "np105k": (96, 256), "np230k": (144, 384)}
TF_SIZES = {"tf12k": (24, 2, 48), "tf44k": (48, 2, 96), "tf170k": (96, 2, 192), "tf800k": (128, 4, 512)}


def t_scale(name, iters, seed=0):
    task = RoleTask(8, 8, seed=seed)
    c, y = tset(task)
    torch.manual_seed(seed)
    if name in NP_SIZES:
        cc, hh = NP_SIZES[name]
        m = NeuroPixel(V, task.out_pos, c=cc, hidden=hh)
    else:
        d, L, ff = TF_SIZES[name]
        m = TinyTransformer(V, 8, 8, task.out_pos, d=d, layers=L, ff=ff)
    m.to(DEV)
    t0 = time.time()
    train(m, task, iters, seed=seed, lens=0.3 if name in NP_SIZES else 0)
    r = {"params": n_params(m), "combos_nuevas": acc_of(m, c, y), "segundos": round(time.time() - t0)}
    if name in NP_SIZES:
        r["daño50"] = acc_of(m, c, y, hook=damage_hook(4, 0.5, c.shape))
    else:
        g = torch.Generator(device=DEV).manual_seed(1)
        mask = (torch.rand(c.shape[0], 64, 1, generator=g, device=DEV) >= 0.5).float()
        # daño equivalente: borrar la mitad de posiciones tras la primera capa
        h = m.enc.layers[0].register_forward_hook(lambda mod, i, o: o * mask[: o.shape[0]])
        accs = []
        with torch.no_grad():
            for i in range(0, len(c), 1000):
                mask_i = mask[i:i + 1000]
                h.remove()
                h = m.enc.layers[0].register_forward_hook(lambda mod, inp, o, mk=mask_i: o * mk)
                accs.append(m(c[i:i + 1000])["logits"].argmax(-1) == y[i:i + 1000])
        h.remove()
        r["daño50"] = round(torch.cat(accs).float().mean().item(), 4)
    # palabra nueva con 5 ejemplos (solo diccionario)
    mm, params, nid = expand_vocab(m, 1)
    cc_, yy_, meta = task.sample(5, "train", torch.Generator().manual_seed(105), meta=True)
    b = torch.arange(5)
    role = torch.tensor([0, 2, 0, 2, 0])
    cc_[b, meta["rows"][b, role], meta["cols"][b, role] + 1] = nid
    cc_[b, task.query_pos[0], task.query_pos[1]] = task.role_ids[role]
    ck, yk = cc_.to(DEV), torch.full((5,), nid, device=DEV)
    opt = torch.optim.Adam(params, lr=1e-2)
    mm.eval()
    for _ in range(300):
        loss = F.cross_entropy(mm(ck)["logits"], yk)
        opt.zero_grad()
        loss.backward()
        opt.step()
    cn, yn, meta = task.sample(1000, "test", torch.Generator().manual_seed(7), meta=True)
    b = torch.arange(1000)
    role = torch.where(torch.rand(1000, generator=torch.Generator().manual_seed(8)) < 0.5, 0, 2)
    cn[b, meta["rows"][b, role], meta["cols"][b, role] + 1] = nid
    cn[b, task.query_pos[0], task.query_pos[1]] = task.role_ids[role]
    r["palabra_nueva_k5"] = acc_of(mm, cn.to(DEV), torch.full((1000,), nid, device=DEV))
    r["viejo_tras_palabra"] = acc_of(mm, c, y)
    save(f"scale_{name}", r)
    log("scale", name, r)
    return r


# =================================================================== 7 coste frente a LLM
def t_llm(n=200):
    task = RoleTask(8, 8, seed=0)
    c, y = task.sample(n, "test", torch.Generator().manual_seed(5))
    names = task.v.tokens
    res = {}
    # NeuroPixel: latencia y rendimiento
    m = NeuroPixel(V, task.out_pos).to(DEV)
    m.load_state_dict(torch.load(ROOT / "runs" / "gpu30k_np_lens_s0" / "model.pt", map_location=DEV))
    m.eval()
    cg, yg = c.to(DEV), y.to(DEV)
    with torch.no_grad():
        m(cg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            p = m(cg)["logits"].argmax(-1)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
        t0 = time.perf_counter()
        for i in range(20):
            m(cg[i:i + 1])
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / 20
    flops_np = 2 * 16 * 64 * (48 * 2 * 9 + (3 * 48 + 16) * 128 + 128 * 48)
    res["neuropixel"] = {"params": n_params(m), "acierto": round((p == yg).float().mean().item(), 4),
                         "ms_por_respuesta_en_lote": round(dt / n * 1000, 5), "latencia_1_ms": round(lat * 1000, 2),
                         "flops_por_respuesta": flops_np}
    log("llm np", res["neuropixel"])
    # LLM local: la misma información, en texto
    try:
        from llama_cpp import Llama
        path = "E:/workspace/arc3-models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=512, verbose=False)
        ok, t_tot, toks = 0, 0.0, 0
        for b in range(n):
            pairs = []
            for r_ in range(7):
                for col in range(7):
                    tk = names[int(c[b, r_, col])]
                    if tk in ROLES:
                        pairs.append(f"{tk}={names[int(c[b, r_, col + 1])]}")
            q = names[int(c[b, task.query_pos[0], task.query_pos[1]])]
            prompt = (f"<|im_start|>user\nHechos: {', '.join(pairs)}.\nResponde solo con una palabra: "
                      f"¿cuál es el {q}?<|im_end|>\n<|im_start|>assistant\n")
            t0 = time.perf_counter()
            o = llm(prompt, max_tokens=6, temperature=0)
            t_tot += time.perf_counter() - t0
            toks += o["usage"]["total_tokens"]
            ans = o["choices"][0]["text"].strip().lower().strip(".").split()
            ok += int(bool(ans) and ans[0] == names[int(y[b])].lower())
        res["qwen2.5_7b_q4"] = {"params": 7.6e9, "acierto": round(ok / n, 4),
                               "ms_por_respuesta": round(t_tot / n * 1000, 1),
                               "tokens_por_respuesta": round(toks / n, 1),
                               "flops_por_respuesta_aprox": round(2 * 7.6e9 * toks / n)}
    except Exception as e:  # noqa: BLE001
        res["qwen2.5_7b_q4"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    log("llm", res.get("qwen2.5_7b_q4"))
    save("llm", res)
    return res


# =================================================================== 8 imaginación
def t_dream(iters, seed=0):
    """Completar parejas: se oculta el relleno de una pareja (solo queda su papel) y la lente
    del diccionario debe proponer en ese píxel un relleno PLAUSIBLE (de la categoría correcta)."""
    from neuropixel.task import NOUNS, PLACES, VERBS
    task = RoleTask(8, 8, seed=seed)
    cat = {0: NOUNS, 1: VERBS, 2: NOUNS, 3: PLACES}
    cat_ids = {k: torch.tensor(task.v.ids(v), device=DEV) for k, v in cat.items()}
    torch.manual_seed(seed)
    m = NeuroPixel(V, task.out_pos).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.1)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)

    def batch(n, split, gen):
        c, y, meta = task.sample(n, split, gen, DEV, meta=True)
        k = torch.randint(0, 4, (n,), generator=gen if gen.device == c.device else None, device=DEV)
        b = torch.arange(n, device=DEV)
        r, col = meta["rows"][b, k], meta["cols"][b, k] + 1
        hidden = c[b, r, col].clone()
        c[b, r, col] = 0
        return c, y, b, r, col, hidden, k

    m.train()
    for it in range(iters):
        c, y, b, r, col, hidden, k = batch(512, "train", g)
        out = m(c)
        lz = m.lens_logits(out["state"])[b, r, col]              # lo que 'imagina' en el hueco
        loss = F.cross_entropy(out["logits"], y) + F.cross_entropy(lz, hidden)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    m.eval()
    gt = torch.Generator(device=DEV).manual_seed(3)
    with torch.no_grad():
        c, y, b, r, col, hidden, k = batch(2000, "test", gt)
        lz = m.lens_logits(m(c)["state"])[b, r, col]
        pred = lz.argmax(-1)
        plaus = torch.stack([(pred[i] == cat_ids[int(k[i])]).any() for i in range(len(pred))]).float().mean().item()
        exact = (pred == hidden).float().mean().item()
        probs = lz.softmax(-1)
        samples = torch.multinomial(probs, 5, replacement=True)
        div = torch.tensor([len(set(s.tolist())) for s in samples]).float().mean().item()
    r_ = {"plausible_categoria_correcta": round(plaus, 4), "exacto": round(exact, 4),
          "azar_exacto_aprox": round(1 / 10.5, 3), "diversidad_5_muestras": round(div, 2)}
    save("dream", r_)
    log("dream", r_)
    return r_


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("--arg", default=None)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--vram-cap", type=float, default=3.0)
    a = ap.parse_args()
    DEV = choose_device("cuda", threads=2, force_gpu=True, vram_cap_gib=a.vram_cap)
    OUT.mkdir(parents=True, exist_ok=True)
    t = a.test
    if t == "stable":
        t_stable(a.seeds, a.iters, a.arg)
    elif t == "newword":
        t_newword()
    elif t == "memory":
        t_memory(a.iters, a.seeds[0])
    elif t == "grow":
        t_grow(a.iters, a.arg)
    elif t == "far":
        t_far(a.arg, a.iters, a.seeds[0])
    elif t == "energy":
        t_energy(float(a.arg), a.iters)
    elif t == "scale":
        t_scale(a.arg, a.iters)
    elif t == "llm":
        t_llm(int(a.arg or 200))
    elif t == "dream":
        t_dream(a.iters)
    print("FIN", t, flush=True)


if __name__ == "__main__":
    main()
