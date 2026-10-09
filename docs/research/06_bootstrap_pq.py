import sys, json, numpy as np
sys.path.insert(0,'.')
import fil
imgs, labs, meta = fil.load_cache()
_, va = fil.split(meta)
runs = {
 'M1': ('ens_ms_learned_cont', dict(thr=0.6,mina=120,close=0)),
 'M2': ('ens_ms_learned_cont+ms_learned_cons_cont', dict(thr=0.6,mina=120,close=0)),
 'M4': ('ens_ms_learned_cont+ms_learned_cons_cont+ms_learned_s1+ms_learned_s2', dict(thr=0.6,mina=120,close=2)),
}
cnt = {}
for name,(d,c) in runs.items():
    z = np.load(f'runs/{d}/val_probs.npz')
    rows=[]
    for k in va:
        p = z[str(meta['ann'][k]['img'])].astype(np.float32)
        rows.append(fil.pq_counts(np.asarray(labs[k]), fil.instances(p,c['thr'],c['mina'],c['close'])))
    cnt[name]=np.array(rows)
img = np.array([meta['ann'][k]['img'] for k in va])
ui = np.unique(img); print('anotaciones',len(va),'imagenes',len(ui))
def pq(a): return a[:,0].sum()/(a[:,1].sum()+.5*a[:,2].sum()+.5*a[:,3].sum())
for n,a in cnt.items(): print(n, round(pq(a),4))
rng=np.random.default_rng(0); B=2000
idx_by={u:np.where(img==u)[0] for u in ui}
boots=[]
for b in range(B):
    s=rng.choice(ui,len(ui),replace=True)
    boots.append(np.concatenate([idx_by[u] for u in s]))
res={n:np.array([pq(a[i]) for i in boots]) for n,a in cnt.items()}
for n,r in res.items(): print(n,'IC95',np.percentile(r,[2.5,97.5]).round(4),'sd',r.std().round(4))
for a,b in [('M2','M1'),('M4','M1'),('M4','M2')]:
    d=res[a]-res[b]; print(a,'-',b,'delta',round(pq(cnt[a])-pq(cnt[b]),4),'IC95',np.percentile(d,[2.5,97.5]).round(4),'P(d>0)',(d>0).mean().round(3))
# bootstrap por anotacion (ingenuo) para comparar
res2=[]
for b in range(B):
    i=rng.integers(0,len(va),len(va)); res2.append(pq(cnt['M1'][i]))
print('M1 naive-annotation sd',np.std(res2).round(4))
# optimismo: elegir mejor config de rejilla en mitad A, evaluar en mitad B (por imagen), 200 particiones, sobre M2
z=np.load('runs/ens_ms_learned_cont+ms_learned_cons_cont/val_probs.npz')
grid=[(t,m,c) for t in (0.6,0.7,0.75,0.8,0.85) for m in (120,160) for c in (0,2)]
G={}
for g in grid:
    G[g]=np.array([fil.pq_counts(np.asarray(labs[k]), fil.instances(z[str(meta['ann'][k]['img'])].astype(np.float32),*g)) for k in va])
ins=[];outs=[];orc=[]
for r in range(100):
    perm=rng.permutation(ui); A=set(perm[:len(ui)//2])
    ia=np.array([j for j in range(len(va)) if img[j] in A]); ib=np.array([j for j in range(len(va)) if img[j] not in A])
    sa={g:pq(G[g][ia]) for g in grid}; best=max(sa,key=sa.get)
    ins.append(sa[best]); outs.append(pq(G[best][ib])); orc.append(max(pq(G[g][ib]) for g in grid))
print('optimismo medio (mitad A tuned - mitad B)',round(np.mean(ins)-np.mean(outs),4),'sd partición',round(np.std(np.array(ins)-np.array(outs)),4))
print('rango PQ de la rejilla (completo):',round(min(pq(G[g]) for g in grid),4),round(max(pq(G[g]) for g in grid),4))
