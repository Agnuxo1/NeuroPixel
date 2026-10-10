"""P1 (RES-001): filtro de objetos por puntuacion media + umbral por tamano, calibrado en una mitad de imagenes y evaluado en la otra.
CPU, sin reentrenar, sobre val_probs.npz. Comparado con la base (thr 0,6 / area 120 / cierre 0) con bootstrap emparejado por imagen."""
import sys, json, itertools, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, '.'); import fil
run = sys.argv[1] if len(sys.argv) > 1 else 'ens_ms_learned_cont+ms_learned_cons_cont'
imgs, labs, meta = fil.load_cache(); _, va = fil.split(meta)
z = np.load(f'runs/{run}/val_probs.npz')
img = np.array([meta['ann'][k]['img'] for k in va]); ui = np.unique(img)
P = {u: z[str(u)].astype(np.float32) for u in ui}

def rule(p, lo, s_small, s_big, a_cut, min_area=120):
    lab, n = ndi.label(p > lo, structure=np.ones((3, 3)))
    if n == 0: return lab
    ids = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(lab), lab, ids); score = ndi.mean(p, lab, ids)
    keep = np.zeros(n + 1, bool)
    keep[1:] = (area >= min_area) & (score >= np.where(area < a_cut, s_small, s_big))
    out, _ = ndi.label(keep[lab], structure=np.ones((3, 3)))
    return out

def counts(fn):
    cache = {u: fn(P[u]) for u in ui}
    return np.array([fil.pq_counts(np.asarray(labs[k]), cache[meta['ann'][k]['img']]) for k in va])
def pq(a): return a[:, 0].sum() / (a[:, 1].sum() + .5 * a[:, 2].sum() + .5 * a[:, 3].sum())

base = counts(lambda p: fil.instances(p, 0.6, 120, 0))
grid = list(itertools.product((0.4, 0.5, 0.6), (0.6, 0.7, 0.8, 0.9), (0.5, 0.6, 0.7, 0.8), (250, 400)))
G = {g: counts(lambda p, g=g: rule(p, g[0], g[1], g[2], g[3])) for g in grid}
print('base PQ', round(pq(base), 4), '| mejor config de la rejilla en todo val (optimista):', max(round(pq(a), 4) for a in G.values()))
rng = np.random.default_rng(0); d_out, d_in = [], []
for r in range(200):
    perm = rng.permutation(ui); A = set(perm[:len(ui) // 2])
    ia = np.array([j for j in range(len(va)) if img[j] in A]); ib = np.array([j for j in range(len(va)) if img[j] not in A])
    sa = {g: pq(G[g][ia]) for g in grid}; best = max(sa, key=sa.get)
    d_out.append(pq(G[best][ib]) - pq(base[ib])); d_in.append(sa[best] - pq(base[ia]))
d_out = np.array(d_out)
print('Delta PQ (calibrado en A, evaluado en B) medio', d_out.mean().round(4), 'IC95 de particiones', np.percentile(d_out, [2.5, 97.5]).round(4), 'P(>0)', (d_out > 0).mean().round(3))
print('Delta en la mitad de calibracion (optimista)', np.mean(d_in).round(4))
bg = max(G, key=lambda g: pq(G[g])); print('config global (lo, s_small, s_big, a_cut):', bg, '-> PQ', round(pq(G[bg]), 4), 'TP/FP/FN', G[bg][:, 1].sum(), G[bg][:, 2].sum(), G[bg][:, 3].sum(), '| base', base[:, 1].sum(), base[:, 2].sum(), base[:, 3].sum())
json.dump({'run': run, 'delta_out_mean': float(d_out.mean()), 'p_pos': float((d_out > 0).mean()), 'best_global': list(map(float, bg)), 'pq_best_global': float(pq(G[bg])), 'pq_base': float(pq(base))}, open(f'runs/{run}/p1_object_filter.json', 'w'), indent=1)
