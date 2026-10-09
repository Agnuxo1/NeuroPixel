"""Prospectively frozen development-only study; resumable, no test access."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():
    return [{'run_id':f'OPT03D_pad{pad}_fire{fire}_seed{seed}','seed':seed,'freeze_pad':bool(pad),'fire_rate':fire,'updates':8192}
            for seed in (100,101,102) for pad in (0,1) for fire in (.5,1.)]
def load_plan(path):
    p=read(path)
    if p['status']!='frozen' or p['runs']!=inventory():raise ValueError('unfrozen/mismatched inventory')
    for f,h in p['source_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('changed frozen source '+f)
    return p

def run_one(conf,directory,plan_path,device_name,stop_after=None):
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('minimum available RAM8GiB before runtime admission')
    import torch
    import torch.nn.functional as F
    from neuropixel.research.models import build_model
    from neuropixel.research.data import ResearchRoleTask
    from neuropixel.research.experiment import (configure_runtime,resource_sample,ResourceStop,
        dataset_artifact,evaluate,environment_record,save_json,atomic_binary,file_sha256,utc_now)
    device=configure_runtime(device_name,2);plan=load_plan(plan_path);fingerprint=sha(plan_path)
    directory.mkdir(parents=True,exist_ok=True);done=directory/'result.json'
    if done.exists():
        old=read(done)
        if old['status']=='completed' and old['plan_sha256']==fingerprint and old['config']==conf:
            if sha(directory/'checkpoint.pt')!=old['checkpoint_sha256']:raise ValueError('completed checkpoint changed')
            print(json.dumps({'event':'reuse_complete','run_id':conf['run_id']}),flush=True);return
    env=environment_record(device);resource_sample(device,8)
    task=ResearchRoleTask(8,8,seed=0)
    protocol={'data':{}} # No final data are referenced anywhere in this controller.
    probe,_=dataset_artifact(task,'train',2048,76001,directory.parent/'datasets')
    validation,_=dataset_artifact(task,'validation',4096,76002,directory.parent/'datasets')
    torch.manual_seed(conf['seed']);model=build_model('neuropixel',vocab=35,h=8,w=8,steps=16,
        freeze_pad=conf['freeze_pad'],fire_rate=conf['fire_rate']).to(device)
    torch.manual_seed(9001);sampler=torch.Generator(device='cpu').manual_seed(9002)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    checkpoint=directory/'checkpoint.pt';curve=[];evals=[];resources=[];start_update=0;elapsed=0.
    if checkpoint.exists():
        prior=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if prior['plan_sha256']!=fingerprint or prior['config']!=conf or prior['environment']!=env:raise ValueError('resume identity/environment differs')
        model.load_state_dict(prior['model'],strict=True);opt.load_state_dict(prior['optimizer'])
        torch.set_rng_state(prior['cpu_rng']);sampler.set_state(prior['sampler_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(prior['cuda_rng'])
        start_update=prior['update'];curve=prior['curve'];evals=prior['evaluations'];elapsed=prior['elapsed_training_seconds'];resources=prior['resources']
    def persist(update):
        payload={'config':conf,'plan_sha256':fingerprint,'environment':env,'update':update,
                 'model':model.state_dict(),'optimizer':opt.state_dict(),'cpu_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'sampler_rng':sampler.get_state(),
                 'curve':curve,'evaluations':evals,'elapsed_training_seconds':elapsed,'resources':resources}
        with atomic_binary(checkpoint) as f:torch.save(payload,f)
        save_json(directory/'progress.json',{'config':conf,'plan_sha256':fingerprint,'environment':env,
            'status':'training','completed_updates':update,'curve':curve,'evaluations':evals,'resources':resources,'test_accessed':False})
    window=[];start=time.monotonic();model.train()
    try:
        for update in range(start_update+1,conf['updates']+1):
            if update%128==1:resources.append(resource_sample(device,8))
            x,y=task.sample(64,'train',sampler,device='cpu');x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            out=model(x,lens_every=4);answer=F.cross_entropy(out['logits'],y);count=out['lens'].shape[1]
            occupied=(x!=0).unsqueeze(1).expand(-1,count,-1,-1);labels=x.unsqueeze(1).expand(-1,count,-1,-1)
            school=F.cross_entropy(out['lens'][occupied],labels[occupied]);loss=answer+.3*school
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            window.append((float(answer.detach()),float(school.detach()),float(loss.detach())))
            if update%128==0 or update==conf['updates']:
                if device.type=='cuda':torch.cuda.synchronize(device)
                elapsed+=time.monotonic()-start;curve.append({'update':update,'answer_loss_mean':sum(v[0] for v in window)/len(window),'school_loss_mean':sum(v[1] for v in window)/len(window),'total_loss_mean':sum(v[2] for v in window)/len(window),'last_gradient_norm':float(grad),'elapsed_training_seconds':elapsed});window=[]
                if update in (1024,4096,8192):
                    pm,_=evaluate(model,probe,device,intervals=False);vm,_=evaluate(model,validation,device,intervals=False)
                    evals.append({'update':update,'probe':pm,'validation':vm,'test_accessed':False});model.train()
                persist(update);resources.append(resource_sample(device,8));start=time.monotonic()
                print(json.dumps({'event':'OPT03_development_progress','run_id':conf['run_id'],**curve[-1]}),flush=True)
                if stop_after and update>=stop_after:return
        last=evals[-1];gate=last['probe']['macro_agent_patient_accuracy']>=.95 and last['validation']['macro_agent_patient_accuracy']>=.90
        save_json(done,{'status':'completed','config':conf,'plan_sha256':fingerprint,'environment':env,
            'curve':curve,'evaluations':evals,'competence_gate_passed':gate,'checkpoint_sha256':file_sha256(checkpoint),'test_accessed':False,'completed_utc':utc_now(),'resources':resources})
    except Exception as e:
        save_json(directory/'incident.json',{'error_type':type(e).__name__,'message':str(e),'run_id':conf['run_id'],
            'last_saved_update':curve[-1]['update'] if curve else start_update,'test_accessed':False,'resume_supported':checkpoint.exists()})
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--describe',action='store_true');ap.add_argument('--plan',type=Path,default=ROOT/'docs/research/OPT03_execution_plan.json');ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--output',type=Path,default=ROOT/'results/research/OPT03_development');ap.add_argument('--run-id');ap.add_argument('--stop-after',type=int);a=ap.parse_args()
    if a.describe:print(json.dumps({'runs':inventory(),'trainings':12,'updates':98304,'final_test_access':False}));return
    p=load_plan(a.plan)
    for conf in p['runs']:
        if a.run_id is None or a.run_id==conf['run_id']:run_one(conf,a.output/conf['run_id'],a.plan,a.device,a.stop_after)
if __name__=='__main__':main()
