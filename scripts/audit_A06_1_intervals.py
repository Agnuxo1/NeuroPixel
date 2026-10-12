"""Recount archived example bootstrap and per-role Wilson intervals independently."""
import json,math
from pathlib import Path
from statistics import NormalDist
import numpy as np
root=Path(__file__).resolve().parents[1]/'results/research/06_A06_1_recovery';r=root/'archive';core=json.loads((r/'core/ablation_records.json').read_text());issues=[];checks=0
for row in core['records']:
    eid=row['evaluation_case']['evaluation_id']
    with np.load(r/'core/evaluations'/eid/'final_predictions.npz') as a:correct=a['prediction']==a['target'];roles=a['role']
    rng=np.random.default_rng(61003);resampled=[]
    for role,name in enumerate(('AGENTE','ACCION','PACIENTE','LUGAR')):
        x=correct[roles==role];n=len(x);p=float(x.mean());z=NormalDist().inv_cdf(.975);den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        if not np.allclose([max(0,center-half),min(1,center+half)],row['metrics']['per_role'][name]['wilson_95'],rtol=0,atol=1e-12):issues.append(eid+' Wilson '+name)
        idx=rng.integers(n,size=(2000,n));resampled.append(x.astype(float)[idx].mean(1));checks+=1
    for k,v in [('macro_all_roles_interval_95',np.mean(resampled,axis=0)),('binding_interval_95',(resampled[0]+resampled[2])/2)]:
        if not np.allclose(np.quantile(v,[.025,.975]),row['metrics'][k],rtol=0,atol=1e-12):issues.append(eid+' '+k)
        checks+=1
out={'status':'verified' if not issues else 'issues_found','checks':checks,'issues':issues,'scope':'Conditional example uncertainty only, not seed replication uncertainty','numpy':np.__version__,'bootstrap_seed':61003,'repetitions':2000}
(root/'audit_intervals.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if issues:raise SystemExit(1)
