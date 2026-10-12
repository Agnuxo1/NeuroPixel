"""Independent saved-array audit; never trains or mutates original evidence."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math
from pathlib import Path
import numpy as np

ROLES = ('AGENTE', 'ACCION', 'PACIENTE', 'LUGAR')
PREFIX = 'runs/research/item6_actions/'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def metrics(pred, target, role, nll=None):
    assert pred.shape == target.shape == role.shape and pred.ndim == 1
    assert np.isfinite(pred).all() and set(np.unique(role)) == {0,1,2,3}
    correct = pred == target
    pr = {name: {'n': int((role == i).sum()), 'correct': int(correct[role == i].sum()),
                 'accuracy': float(correct[role == i].mean())} for i,name in enumerate(ROLES)}
    out = {'n': len(pred), 'correct': int(correct.sum()), 'accuracy': float(correct.mean()),
           'macro_all_roles': float(np.mean([x['accuracy'] for x in pr.values()])),
           'macro_agent_patient_accuracy': (pr['AGENTE']['accuracy'] + pr['PACIENTE']['accuracy'])/2,
           'per_role': pr}
    if nll is not None:
        assert np.isfinite(nll).all() and (nll >= -1e-7).all()
        out['cross_entropy'] = float(np.asarray(nll,dtype=np.float64).mean())
    return out

def paired(values):
    v = np.asarray(values,dtype=float); assert len(v) == 2
    sd = float(v.std(ddof=1)); mu = float(v.mean()); half = (1/math.tan(math.pi*.025))*sd/math.sqrt(2)
    return {'values': v.tolist(), 'mean': mu, 'sample_sd': sd, 't95_df1': [mu-half,mu+half],
            'unit':'fraction; multiply by 100 for percentage points','n_independent_initializations':2,
            'inference':'exploratory, unadjusted, normality not assessable at n=2'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    root=a.archive; repo=Path(__file__).resolve().parents[1]; issues=[]; checks=0
    def check(ok,desc):
        nonlocal checks
        checks+=1
        if not ok: issues.append(desc)
    def same(actual,reported,context):
        for k,v in actual.items():
            if isinstance(v,dict): same(v,reported[k],context+'.'+k)
            else: check(bool(np.isclose(v,reported[k],rtol=0,atol=1e-12)),context+'.'+k)
    manifest=read(root/'artifact_manifest.json'); listed=set()
    for row in manifest['files']:
        rel=row['path'].split(PREFIX,1)[1]; p=root/rel; listed.add(rel)
        check(p.is_file(),f'missing {rel}')
        if p.is_file():
            check(p.stat().st_size==row['bytes'],f'size {rel}')
            check(digest(p)==row['sha256'],f'hash {rel}')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check(actual == listed | {'artifact_manifest.json'},'archive inventory')
    core=read(root/'core/ablation_records.json'); growth=read(root/'growth/growth_records.json')
    plan=read(repo/'docs/research/06_execution_plan.json')
    for f,h in {**core['source']['sha256'],**core['controller_source']}.items():
        check(digest(repo/f)==h,'source '+f)
    check(core['source']['git_commit']=='a009f2a9b520a3c7333d092cd3d4f0026b17c335','execution commit')
    check(core['complete_training_runs']==26 and core['complete_evaluation_cases']==34,'core complete')
    check([x['evaluation_case'] for x in core['records']]==plan['planned_evaluations'],'exact evaluation inventory')
    final_path=root/'core/datasets/split0_test_n4096_seed61002.npz'
    with np.load(final_path,allow_pickle=False) as arr: final={k:arr[k] for k in arr.files}
    final_count=validation_count=0; clean_rows={}; validations=[]; trainings=[]
    for conf in plan['planned_runs']:
        folder=root/'core'/conf['run_id']; t=read(folder/'training.json'); trainings.append(t)
        check(t['status']=='completed','training complete '+conf['run_id'])
        check(all(t['config'][k]==v for k,v in conf.items()),'exact training '+conf['run_id'])
        check(t['curve'][-1]['update']==conf['updates'],'updates '+conf['run_id'])
        check(not t['final_test_accessed'],'no training test access '+conf['run_id'])
        check(digest(folder/'weights.pt')==t['weights_sha256'],'checkpoint '+conf['run_id'])
        check(all(z['available_ram_gib']>=8 for z in t['resources']),'RAM '+conf['run_id'])
        check(all(math.isfinite(float(v)) for z in t['curve'] for k,v in z.items() if isinstance(v,(int,float))), 'finite curve '+conf['run_id'])
        with np.load(folder/'validation_predictions.npz',allow_pickle=False) as arr:
            m=metrics(arr['prediction'],arr['target'],arr['role'],arr['nll']);same(m,t['validation'],conf['run_id']+'.validation');validation_count+=m['n'];validations.append(m)
    opened=next(x['at_utc'] for x in core['events'] if x['event']=='final_dataset_opened')
    check(all(t['completed_at_utc']<opened for t in trainings),'all training before final access')
    for row in core['records']:
        eid=row['evaluation_case']['evaluation_id']; p=root/'core/evaluations'/eid/'final_predictions.npz'
        with np.load(p,allow_pickle=False) as arr:
            check(np.array_equal(arr['target'],final['target']) and np.array_equal(arr['role'],final['roles']),'final labels '+eid)
            check(np.isfinite(arr['confidence']).all() and np.all((arr['confidence']>=0)&(arr['confidence']<=1)),'confidence '+eid)
            m=metrics(arr['prediction'],arr['target'],arr['role'],arr['nll']);same(m,row['metrics'],eid);final_count+=m['n']
        check(row['weights_sha256']==digest(root/'core'/row['config']['run_id']/'weights.pt'),'eval weights '+eid)
        row=dict(row,metrics=m)
        if row['evaluation_case']['kind']=='clean': clean_rows[row['config']['run_id']]=row
    core_effects={}; cells={}
    for seed in (20,21):
        ys={tuple((int(row['config']['variant']['tied']),int(row['config']['school_weight']>0),int(row['config']['variant']['reinject']))):row['metrics']['macro_agent_patient_accuracy']
            for row in clean_rows.values() if row['config']['panel']=='factorial' and row['config']['init_seed']==seed}
        check(len(ys)==8,'factorial '+str(seed)); cells[str(seed)]={''.join(map(str,k)):v for k,v in ys.items()}
        for order in (1,2,3):
            for axes in itertools.combinations(range(3),order):
                name=' x '.join(('tying','school','reinjection')[k] for k in axes)
                effect=sum(v*math.prod(1 if k[j] else -1 for j in axes) for k,v in ys.items())/(2**(3-order))
                core_effects.setdefault(name,[]).append(effect)
    def bind(seed,panel='factorial',kind='clean',school=0.,steps=16,updates=1024):
        rows=[r for r in core['records'] if r['config']['init_seed']==seed and r['config']['panel']==panel and r['evaluation_case']['kind']==kind and r['config']['school_weight']==school and r['config']['steps']==steps and r['config']['updates']==updates and r['config']['variant']['tied'] and r['config']['variant']['reinject']]
        assert len(rows)==1; return rows[0]['metrics']['macro_agent_patient_accuracy']
    additions={}
    for seed in (20,21):
        base=bind(seed); dclean=bind(seed,'damage'); dlesion=bind(seed,'damage','fixed_lesion'); clesion=bind(seed,kind='fixed_lesion')
        for name,val in [('damage train effect clean',dclean-base),('damage train effect lesion',dlesion-clesion),('damage interaction',(dlesion-dclean)-(clesion-base))]: additions.setdefault(name,[]).append(val)
        for depth in (1,4):
            additions.setdefault(f'trained T{depth} minus T16',[]).append(bind(seed,'recurrence',steps=depth)-base)
            row=next(r for r in core['records'] if r['config']['init_seed']==seed and r['config']['panel']=='factorial' and r['config']['variant']['tied'] and r['config']['variant']['reinject'] and r['config']['school_weight']==0 and r['evaluation_case']['kind']=='deployment_truncation' and r['evaluation_case']['steps_override']==depth)
            additions.setdefault(f'same weights evalT{depth} minus T16',[]).append(row['metrics']['macro_agent_patient_accuracy']-base)
        for school in (0.,.3): additions.setdefault(f'8192 minus 1024 school{school}',[]).append(bind(seed,'optimization',school=school,updates=8192)-bind(seed,school=school))
    growth_rows={}; growth_count=0; stage_count=0
    for seed in ('20','21'):
        for stages in ('fixed_stages','novelty_stages','resonance_stages'):
            st=growth['training'][seed][stages]; stage_count+=len(st)
            check(len(st)==3 and all(s['training']['updates']==512 for s in st),'growth stages '+seed+stages)
        growth_rows[seed]={}
        targets=[];roles=[]
        for topic in range(3):
            with np.load(root/f'growth/datasets/final_topic{topic}.npz') as arr: targets.append(arr['target']);roles.append(arr['roles'])
        target=np.concatenate(targets);role=np.concatenate(roles)
        for bank in ('fixed','duplicate','novelty','resonance'):
            row=growth['results'][seed][bank]; ms={}
            with np.load(root/f'growth/seed{seed}/predictions_{bank}.npz') as arr:
                check(np.array_equal(arr['target'],target) and np.array_equal(arr['roles'],role),'growth labels '+seed+bank)
                for router in ('legacy','random','uniform','learned'):
                    ms[router]=metrics(arr['prediction_'+router],target,role);same(ms[router],row['metrics'][router],seed+bank+router);growth_count+=len(target)
                    for topic in range(3):
                        sl=slice(topic*1024,(topic+1)*1024);same(metrics(arr['prediction_'+router][sl],target[sl],role[sl]),row['per_topic']['ABC'[topic]][router],seed+bank+router+str(topic))
                for router in ('legacy','random'):
                    idx=arr['index_'+router];check(np.all((idx>=0)&(idx<row['expert_count'])),'routing range '+seed+bank+router)
                    check(np.bincount(idx,minlength=row['expert_count']).tolist()==row['routing_frequency'][router],'routing frequency '+seed+bank+router)
            growth_rows[seed][bank]={k:v['macro_agent_patient_accuracy'] for k,v in ms.items()}
    check(stage_count==18 and len(growth['gate_fits'])==8,'growth full inventory')
    check(growth['evaluation_gate']['implementation_sha256']==plan['implementation_sha256'],'growth source')
    contrasts={}
    for bank in ('fixed','duplicate','novelty','resonance'):
        for router in ('random','uniform','learned'):
            contrasts[bank+' legacy minus '+router]=paired([growth_rows[s][bank]['legacy']-growth_rows[s][bank][router] for s in ('20','21')])
    for bank in ('novelty','resonance','duplicate'):
        for router in ('legacy','random','uniform','learned'):
            contrasts[bank+' minus fixed '+router]=paired([growth_rows[s][bank][router]-growth_rows[s]['fixed'][router] for s in ('20','21')])
    out={'status':'verified' if not issues else 'issues_found','checks':checks,'issues':issues,
         'manifest_files':len(listed),'archive_sha256':digest(root.parent/'original.zip'),
         'final_core_predictions':final_count,'validation_predictions':validation_count,
         'growth_router_predictions':growth_count,'core_factorial_cells':cells,
         'core_contrasts':{k:paired(v) for k,v in {**core_effects,**additions}.items()},
         'growth_binding':growth_rows,'growth_contrasts':contrasts,
         'budget_limited_trainings':sum(bool(t['optimization_budget_limited']) for t in trainings),
         'minimum_recorded_training_ram_gib':min(z['available_ram_gib'] for t in trainings for z in t['resources']),
         'limits':['Independent arithmetic and archive checks, not external replication.',
                   'Checkpoint finite tensor/state audit and inference replay are separate.',
                   'Single-final-expert predictions and per-example oracle indicators absent from archive; replay needed for independent recount.',
                   'Example bootstrap intervals remain distinct from two-initialization uncertainty.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:out[k] for k in ('status','checks','issues','manifest_files','final_core_predictions','validation_predictions','growth_router_predictions','budget_limited_trainings')}))
    if issues: raise SystemExit(1)

if __name__=='__main__':main()
