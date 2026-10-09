"""Orthogonal saved-byte/array audit. Never imports Torch, predicts, or trains."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.stats import t
from recover_TASK4_archives import PREFIX,ROOT,DEST,AUDIT

OUT=ROOT/'results/research/TASK4_review'
checks=[]; issues=[]; summaries={}; descriptors=[]
def check(name,ok,details=None):
    checks.append(dict(name=name,passed=bool(ok),details=details))
    if not ok:issues.append(name)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(d):
    d=np.asarray(d,dtype=np.float64);m=float(d.mean());h=float(t.ppf(.975,len(d)-1)*d.std(ddof=1)/math.sqrt(len(d)))
    return dict(mean=m,t95=[m-h,m+h],n=len(d),df=len(d)-1)
def main():
    for sha,prefix in PREFIX.items():
        root=DEST/sha[:8]; mp=root/'archive_manifest.json';manifest=json.loads(mp.read_bytes())
        files=manifest['files'];rows=files if isinstance(files,list) else [dict(path=k,**v) for k,v in files.items()]
        declared={x['path'] for x in rows};actual={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file()}
        check(sha[:8]+' exact inventory',actual==declared|{'archive_manifest.json'},dict(expected=len(declared)+1,actual=len(actual)))
        for row in rows:
            p=root/row['path'];check(sha[:8]+' manifest '+row['path'],p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'])
        run=int(prefix.split('/')[1].split('-')[0]);provider=json.loads((AUDIT/'provider'/f'{run}.json').read_bytes())
        check(sha[:8]+' terminal provider',provider['status']=='completed' and provider['conclusion']=='success' and provider['run_attempt']==1)
        source=manifest.get('source_commit');check(sha[:8]+' source/provider',source==provider['head_sha'],dict(archive_source=source,provider_source=provider['head_sha']))
        summaries[sha[:8]]=dict(prefix=prefix,manifest_sha256=digest(mp),files=len(actual),bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),source=source,run=run)
        for p in root.rglob('*.npz'):
            with np.load(p,allow_pickle=False) as z:
                desc=dict(archive=sha,path=str(p.relative_to(root)),arrays=[])
                for k in z.files:
                    a=z[k];desc['arrays'].append(dict(name=k,shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes))
                    # Explicit diagnostics may retain NaN for censored/unsupported rows.
                    if a.dtype.kind=='f':desc['arrays'][-1]['nonfinite']=int(np.count_nonzero(~np.isfinite(a)))
                descriptors.append(desc)
                pk=next((k for k in ['prediction_id','prediction','pred'] if k in z),None)
                if pk and 'logits' in z and z['logits'].shape[:-1]==z[pk].shape:
                    check(sha[:8]+' argmax '+str(p.relative_to(root)),np.array_equal(z['logits'].argmax(-1),z[pk]))
    # Primary item9: exact same archived populations, five paired seeds, base condition0.
    r=DEST/'b64d0e2b'/'study'; accuracy={}
    with np.load(r/'final_data/examples.npz',allow_pickle=False) as data:
        for family in ['neuropixel','relative_transformer']:
            vals=[]
            for seed in range(40,45):
                with np.load(r/f'final_predictions/{family}_seed{seed}.npz',allow_pickle=False) as z:
                    for k in ['target','role','condition_index','record_id']:
                        check(f'item9 {family} {seed} data {k}',np.array_equal(z[k],data[k]))
                    scores=[]
                    for role in [0,2]:
                        mask=(z['condition_index']==0)&(z['role']==role);check(f'item9 {family} {seed} denominator role{role}',int(mask.sum())==512)
                        scores.append(float(np.mean(z['pred'][mask]==z['target'][mask])))
                    vals.append(sum(scores)/2)
            accuracy[family]=vals
    contrast=interval(np.array(accuracy['neuropixel'])-accuracy['relative_transformer'])
    check('item9 report mean NP',np.isclose(np.mean(accuracy['neuropixel']),.0962890625,atol=1e-14,rtol=0))
    check('item9 report mean TF',np.isclose(np.mean(accuracy['relative_transformer']),.4935546875,atol=1e-14,rtol=0))
    check('item9 primary mean',np.isclose(contrast['mean'],-.397265625,atol=1e-14,rtol=0))
    summaries['item9_primary']=dict(accuracy=accuracy,contrast=contrast,scope='retrospective archived predictions; no new final access or model evaluation')
    # Official bAbI test rows, already evaluated in item10; no fresh scoring.
    r=DEST/'0e135ad1'/'final_predictions';accuracy={}
    for family in ['neuropixel','relative_transformer']:
        vals=[]
        for seed in range(60,65):
            with np.load(r/f'train_{family}_s{seed}.npz',allow_pickle=False) as z:
                check(f'item10 {family} {seed} denominator',len(z['gold_id'])==1000)
                vals.append(float(np.mean(z['prediction_id']==z['gold_id'])))
        accuracy[family]=vals
    contrast=interval(100*(np.array(accuracy['neuropixel'])-accuracy['relative_transformer']))
    check('item10 report mean NP',np.isclose(100*np.mean(accuracy['neuropixel']),50.4,atol=1e-12,rtol=0))
    check('item10 report mean TF',np.isclose(100*np.mean(accuracy['relative_transformer']),88.66,atol=1e-12,rtol=0))
    check('item10 report contrast',np.isclose(contrast['mean'],-38.26,atol=1e-12,rtol=0))
    summaries['item10_primary']=dict(accuracy=accuracy,contrast_pp=contrast)
    # Five learned finite-dynamics panels, recompute from all retained states.
    growth=[];gain=[];changed=0;nll_increases=0
    for seed in range(40,45):
        with np.load(DEST/'205507b7'/f'native/seeds/seed{seed}.npz',allow_pickle=False) as z:
            s=z['states'].astype(np.float64);rms=np.sqrt(np.mean(s*s,axis=(-3,-2,-1)))
            pr=np.sqrt(np.mean((s[:,0]-s[:,1])**2,axis=(-3,-2,-1)));pg=pr/pr[:,0,None]
            check(f'item15 seed{seed} complete finite',s.shape==(4,2,257,48,10,8) and np.isfinite(s).all())
            check(f'item15 seed{seed} RMS',np.allclose(rms,z['state_rms'],atol=1e-12,rtol=1e-12))
            check(f'item15 seed{seed} pair RMS',np.allclose(pr,z['pair_rms'],atol=1e-12,rtol=1e-12))
            check(f'item15 seed{seed} gain',np.allclose(pg,z['pair_gain'],atol=1e-10,rtol=1e-12))
            growth.extend((rms[:,0,-1]/rms[:,0,0]).tolist());gain.extend(pg[:,-1].tolist())
            logits=z['logits'];pred=logits.argmax(-1);changed+=int(np.sum(pred[:,0,0]!=pred[:,0,-1]));nll_increases+=int(np.sum(z['token_nll'][:,0,-1]>z['token_nll'][:,0,0]))
    check('item15 endpoint predictions changed18of20',changed==18);check('item15 endpoint NLL increased19of20',nll_increases==19)
    summaries['item15_learned']=dict(state_growth_range=[min(growth),max(growth)],pair_gain_range=[min(gain),max(gain)],base_predictions_changed=changed,base_nll_increased=nll_increases,independent_scenes=1,horizon=256)
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(status='passed' if not issues else 'issues_preserved',checks=len(checks),issues=issues,details=checks,summaries=summaries,npz_descriptors=descriptors,model_loads=0,new_predictions=0,training_updates=0,external_replication=False)
    (OUT/'saved_evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],checks=len(checks),issues=issues,npz_files=len(descriptors),main9=contrast if False else summaries['item9_primary']['contrast'],main10=summaries['item10_primary']['contrast_pp'],learned15=summaries['item15_learned'])))
    return 0 if not issues else 1
if __name__=='__main__':sys.exit(main())
