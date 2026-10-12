"""Read-only CPU inference replay of A06.1, using frozen models and saved data."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from neuropixel.research.ablations import training_model, evaluation_view
from neuropixel.research.experiment import evaluate
from neuropixel.research.growth import make_core, state_sha256, score_routers
from neuropixel.research.data import ResearchRoleTask

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def load_dataset(p):
    with np.load(p,allow_pickle=False) as a:return tuple(torch.from_numpy(a[k].copy()) for k in ('canvas','target','roles'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=True
    r=a.archive;c=read(r/'core/ablation_records.json');g=read(r/'growth/growth_records.json');issues=[];maxerr={'confidence':0.,'nll':0.};start=time.monotonic();tensor_count=0
    for p in sorted(r.rglob('*.pt')):
        st=torch.load(p,map_location='cpu',weights_only=True)
        if not all(torch.isfinite(v).all().item() for v in st.values()):issues.append('nonfinite checkpoint '+str(p))
        tensor_count+=len(st)
    dataset=load_dataset(r/'core/datasets/split0_test_n4096_seed61002.npz');ncore=0
    with torch.inference_mode():
        for row in c['records']:
            conf=row['config'];case=row['evaluation_case'];model=training_model(conf)
            model.load_state_dict(torch.load(r/'core'/conf['run_id']/'weights.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
            keep=None
            if row['lesion_keep_mask'] is not None:
                p=r/row['lesion_keep_mask']['path'].split('runs/research/item6_actions/',1)[1]
                with np.load(p) as arr:keep=torch.from_numpy(arr['keep'].copy())
            view=evaluation_view(model,conf,case,keep);_,pred=evaluate(view,dataset,torch.device('cpu'),intervals=False)
            with np.load(r/'core/evaluations'/case['evaluation_id']/'final_predictions.npz') as original:
                for k in ('prediction','target','role'):
                    if not np.array_equal(pred[k],original[k]):issues.append(case['evaluation_id']+' '+k+' mismatch')
                for k in ('confidence','nll'):
                    err=float(np.max(np.abs(pred[k]-original[k])));maxerr[k]=max(maxerr[k],err)
                    if err>1e-4:issues.append(case['evaluation_id']+' '+k+' maxerr='+str(err))
            ncore+=1;print(json.dumps({'core_replay':ncore,'of':34}),flush=True)
        parts=[load_dataset(r/f'growth/datasets/final_topic{t}.npz') for t in range(3)];full=tuple(torch.cat([x[k] for x in parts]) for k in range(3));task=ResearchRoleTask(8,8,seed=0)
        banks={};expert_index={}
        for p in sorted((r/'growth').rglob('*.pt')):
            model=make_core(20);model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True);model.eval();expert_index[state_sha256(model)]=model
        ngrowth=0;derived={}
        for seed in ('20','21'):
            derived[seed]={}
            for bank in ('fixed','duplicate','novelty','resonance'):
                row=g['results'][seed][bank];models=[expert_index[h] for h in row['state_sha256']]
                with np.load(r/f'growth/seed{seed}/gate_{bank}.npz') as arr:gw=arr['weight']
                score=score_routers(models,*full,task.role_ids,gw,torch.device('cpu'),random_seed=62013)
                with np.load(r/f'growth/seed{seed}/predictions_{bank}.npz') as old:
                    for router in ('legacy','random','uniform','learned'):
                        if not np.array_equal(score['predictions'][router],old['prediction_'+router]):issues.append(seed+bank+router+' prediction mismatch')
                    for router in ('legacy','random'):
                        if not np.array_equal(score['indices'][router],old['index_'+router]):issues.append(seed+bank+router+' index mismatch')
                for k in ('accuracy','binding_accuracy'):
                    if abs(score['hard_selection_oracle'][k]-row['hard_selection_oracle'][k])>1e-12:issues.append(seed+bank+' oracle '+k)
                derived[seed][bank]={'oracle':score['hard_selection_oracle'],'checkpoint_states':row['state_sha256']}
                ngrowth+=1;print(json.dumps({'growth_bank_replay':ngrowth,'of':8}),flush=True)
            last=expert_index[g['results'][seed]['single_final_expert']['state_sha256']];met,arr=evaluate(last,full,torch.device('cpu'),intervals=False)
            original=g['results'][seed]['single_final_expert']['metrics']
            for k in ('accuracy','macro_agent_patient_accuracy'):
                if abs(met[k]-original[k])>1e-12:issues.append(seed+' single_final '+k)
            derived[seed]['single_final_expert']=met
    out={'status':'verified' if not issues else 'issues_found','issues':issues,'core_cases':ncore,'growth_banks':ngrowth,'core_saved_predictions_equal':not any('mismatch' in x for x in issues),'finite_checkpoint_files':len(list(r.rglob('*.pt'))),'finite_checkpoint_tensors':tensor_count,'maximum_absolute_numeric_replay_error':maxerr,'numeric_tolerance':1e-4,'elapsed_seconds':time.monotonic()-start,'derived':derived,'environment':{'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':2},'scope':'CPU inference replay, no training; same frozen implementation, not external architectural replication'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='derived'}),flush=True)
    if issues:raise SystemExit(1)
if __name__=='__main__':main()
