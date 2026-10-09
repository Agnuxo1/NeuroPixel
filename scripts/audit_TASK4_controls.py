"""Separate finite-set, memory-gate, isolation and repair recounts, no Torch."""
import json,sys
from pathlib import Path
import numpy as np
from audit_TASK4_saved_evidence import check,checks,issues,OUT,DEST

def main():
    j=json.loads((DEST/'f0aa1f17/split_pools.json').read_bytes());pools=[]
    for row in j['seeds']:
        seed=row['seed']
        if seed in [0,1,2]:pools.append((f'historical{seed}',row['direct_role_task']))
        if seed in [0,101,202]:pools.append((f'research{seed}',row['direct_research_role_task']))
    for row in j['growth_partitions']:pools.append((f'growth{row["topic"]}',row['triples']))
    union=set()
    for name,pool in pools:
        sets={k:{tuple(t) for t in v} for k,v in pool.items()}
        check(name+' unique lists',all(len(sets[k])==len(pool[k]) for k in pool))
        for k,a in sets.items():
            for l,b in sets.items():
                if k<l:check(name+' disjoint '+k+'/'+l,not(a&b))
        union|=sets['train']
    check('nine partition census',len(pools)==9);check('1319 train eligible union',len(union)==1319)
    sel=json.loads((DEST/'1b175977/selection.json').read_bytes());check('memory main not admitted',sel['admission_passed'] is False)
    counts={}
    for family,row in sel['families'].items():
        run=row['run_id'];r=DEST/'1b175977/study/runs'/run
        near=r/'normal_d2.npz'
        with np.load(near,allow_pickle=False) as z:
            correct=int(np.sum(z['prediction']==z['target']));n=len(z['target']);counts[family]=dict(correct=correct,n=n)
            check(f'memory {family} saved selection',correct/n==row['near_accuracy'] and n==48)
            check(f'memory {family} admission threshold',row['near_competence_pass']==(correct>=46))
    check('memory selected NP38 GRU48',counts['neuropixel']['correct']==38 and counts['gru']['correct']==48)
    isolation=DEST/'5381f677/isolation';changes=[]
    for stage in range(1,9):
        with np.load(isolation/f'stage{stage}_before.npz',allow_pickle=False) as a,np.load(isolation/f'stage{stage}_after.npz',allow_pickle=False) as b:
            check(f'isolation stage{stage} array inventory',set(a.files)==set(b.files))
            changed=0
            for k in a.files:
                if k.startswith('expert'):
                    expert=int(k.split('_')[0][6:]);same=np.array_equal(a[k],b[k])
                    if expert<stage-1:check(f'isolation stage{stage} old {k}',same)
                    else:changed+=int(not same)
            check(f'isolation stage{stage} actual update',changed>0);changes.append(changed)
    with np.load(isolation/'alias_before.npz',allow_pickle=False) as a,np.load(isolation/'alias_after.npz',allow_pickle=False) as b:
        changed=sum(not np.array_equal(a[k],b[k]) for k in a.files if k.startswith('expert0_'))
        check('alias negative control detected',changed>0)
    r=DEST/'10c694cd/native';report=json.loads((r/'report.json').read_bytes());recipe=report['recipe']
    with np.load(r/'traces.npz',allow_pickle=False) as z:
        pred=z['recovery_logits'].argmax(-1);pre=z['prefix_logits'].argmax(-1)[:,:,-1];cues=z['cues']
        check('repair raw complete dimensions',pred.shape==(3,4,3,6,9))
        for row in report['rows']:
            m=recipe['models'].index(row['model']);l=recipe['lesions'].index(row['lesion']);s=recipe['sources'].index(row['source']);step=row['recovery_step']
            immediate=pred[m,l,s,:,0]==cues;late=pred[m,l,s,:,step]==cues;initial=pre[m]==cues
            expected=dict(n=6,pre_correct=int(initial.sum()),immediate_correct=int(immediate.sum()),late_correct=int(late.sum()),initially_correct_then_lost=int((initial&~immediate).sum()),lost_then_recovered=int((initial&~immediate&late).sum()),survived_and_still_correct=int((initial&immediate&late).sum()),matched_sham_correct=int(np.sum(pred[m,0,s,:,step]==cues)))
            for key,val in expected.items():check('repair '+row['model']+'/'+row['lesion']+'/'+row['source']+'/'+str(step)+' '+key,row[key]==val)
    OUT.mkdir(parents=True,exist_ok=True);result=dict(status='passed' if not issues else 'issues_preserved',checks=len(checks),issues=issues,details=checks,memory_selected=counts,train_eligible_union=len(union),isolation_updated_tensor_counts=changes,repair_rows=len(report['rows']),model_loads=0,training_updates=0)
    (OUT/'control_recount.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='details'}));return 0 if not issues else 1
if __name__=='__main__':sys.exit(main())
