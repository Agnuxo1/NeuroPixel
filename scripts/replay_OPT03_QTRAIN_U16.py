"""Read-only U16 replay: actual continuation states, optimizer/RNG and endpoint decisions."""
import argparse,hashlib,json,os,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--run-id');a=ap.parse_args()
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM below8GiB before replay admission')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from neuropixel.research.models import build_model
    from neuropixel.research.data import ResearchRoleTask,frozen_dataset
    from neuropixel.research.experiment import configure_runtime,environment_record
    from neuropixel.research.qtrain_budget import validate_state,training_state,state_digest as full_state_digest,restore,verify_environment
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM below8GiB after runtime load; no replay started')
    device=configure_runtime('cpu',2);env=environment_record(device);plan=read(ROOT/'docs/research/OPT03_QTRAIN_U16_plan.json');plan_hash=sha(ROOT/'docs/research/OPT03_QTRAIN_U16_plan.json');issues=[];checks=0;count=0;max_nll_mean_error=0.;start=time.monotonic();cases=[]
    if plan_hash!='a4093bb8d559404b9f68587b753f8cea180a1e75d33363a04c1f2c7ebcb22dc0':raise ValueError('study identity changed')
    for f,h in plan['source_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('frozen source differs '+f)
    def check(ok,label):
        nonlocal checks
        checks+=1
        if not ok:issues.append(label)
    task=ResearchRoleTask(8,8,seed=0);datasets={}
    for panel,split,n,seed in [('probe','train',2048,77001),('validation','validation',4096,77002)]:
        dataset=frozen_dataset(task,split,n,seed);datasets[panel]=dataset
        with np.load(a.root/'datasets'/f'{panel}_n{n}_seed{seed}.npz',allow_pickle=False) as raw:
            for name,tensor in zip(('canvas','target','roles'),dataset):check(np.array_equal(raw[name],tensor.numpy()),panel+' regenerate '+name)
    def state_digest(state):
        h=hashlib.sha256()
        for name,v in sorted(state.items()):h.update(name.encode());h.update(v.contiguous().numpy().tobytes())
        return h.hexdigest()
    def dataset_digest(ds):
        h=hashlib.sha256()
        for tensor in ds:
            arr=np.ascontiguousarray(tensor.numpy());h.update(str(arr.dtype).encode());h.update(json.dumps(list(arr.shape)).encode());h.update(arr.tobytes())
        return h.hexdigest()
    def inference(model,canvas,target,seed):
        pred=[];nll=[]
        with torch.random.fork_rng(devices=[]),torch.inference_mode():
            torch.manual_seed(seed)
            for i in range(0,len(canvas),256):
                logits=model(canvas[i:i+256])['logits'];pred.append(logits.argmax(-1));nll.append(F.cross_entropy(logits,target[i:i+256],reduction='none'))
        return torch.cat(pred).numpy(),torch.cat(nll).numpy()
    for conf in plan['runs']:
        run=conf['run_id'];folder=a.root/run
        if a.run_id and run!=a.run_id:continue
        if not (folder/'result.json').exists():continue
        if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM stop before next checkpoint replay')
        before={str(p):sha(p) for p in folder.glob('checkpoint*.pt')};record=read(folder/'result.json');cases.append(run)
        for endpoint in record['evaluations']:
            update=endpoint['update'];p=folder/f'checkpoint_u{update}.pt';cp=torch.load(p,map_location='cpu',weights_only=True)
            check(cp['update']==update and cp['config']==conf and cp['plan_sha256']==plan_hash,run+' endpoint identity')
            actual_digest=validate_state(cp,update,conf,plan_hash);check(actual_digest==cp['training_state_digest'],run+' actual model/optimizer/threeRNG digest');verify_environment(cp['environment'],env)
            check(all(torch.isfinite(v).all().item() for v in cp['model'].values()),run+' finite model')
            for state in cp['optimizer']['state'].values():
                check(float(state['step'])==update,run+' optimizer update count')
                check(all(torch.isfinite(v).all().item() for v in state.values() if isinstance(v,torch.Tensor)),run+' finite optimizer')
            for group in cp['optimizer']['param_groups']:check(group['lr']==.001 and tuple(group['betas'])==(.9,.999) and group['weight_decay']==.0001 and group['eps']==1e-8,run+' optimizer recipe')
            model=build_model('neuropixel',vocab=35,h=8,w=8,steps=16,freeze_pad=False,fire_rate=.5);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001);sampler=torch.Generator();query_rng=torch.Generator();restore(model,optimizer,sampler,query_rng,cp,actual_digest);check(sum(x.numel() for x in model.parameters())==29824,run+' real capacity')
            for panel,payload in endpoint['panels'].items():
                canvas,target,roles=datasets[panel]
                for mode in payload['metrics']:
                    model.fire_rate=.5 if mode=='matched' else 1.;model.train(mode=='matched')
                    with np.load(folder/'decisions'/f'u{update}_{panel}_{mode}.npz',allow_pickle=False) as stored:
                        check(str(stored['state_sha256'].item())==state_digest(cp['model']),run+panel+mode+' actual state hash')
                        check(str(stored['dataset_content_sha256'].item())==dataset_digest(datasets[panel]),run+panel+mode+' actual dataset hash')
                        cf=canvas.clone();noun=torch.isin(roles,torch.tensor([0,2]));new_roles=torch.where(noun,2-roles,roles);cf[:,7,6]=new_roles+1;cy=torch.from_numpy(stored['query_target'].copy())
                        for k,seed in enumerate(stored['mask_seeds'].tolist()):
                            pred,nll=inference(model,canvas,target,seed);qpred,qnll=inference(model,cf,cy,seed)
                            check(np.array_equal(pred,stored['predictions'][k]),run+panel+mode+str(seed)+' original decisions exact')
                            check(np.array_equal(qpred,stored['query_predictions'][k]),run+panel+mode+str(seed)+' query decisions exact')
                            err=max(abs(float(nll.astype(np.float64).mean())-float(stored['nll'][k].astype(np.float64).mean())),abs(float(qnll.astype(np.float64).mean())-float(stored['query_nll'][k].astype(np.float64).mean())));max_nll_mean_error=max(max_nll_mean_error,err);check(err<=1e-4,run+panel+mode+str(seed)+' mean NLL tolerance');count+=len(pred)*2
            check(full_state_digest(training_state(model,optimizer,sampler,query_rng))==actual_digest,run+' replay preserves actual training-state streams/moments')
            print(json.dumps({'case_replay':run,'endpoint':update}),flush=True)
        check(all(sha(pathlib.Path(p))==h for p,h in before.items()),run+' original endpoint weights unchanged')
    out={'status':'verified_available_endpoint_replay' if not issues else 'issues_found','checks':checks,'issues':issues,'cases':cases,'individual_decisions_replayed':count,'max_mean_nll_error':max_nll_mean_error,'elapsed_seconds':time.monotonic()-start,'environment':env,'policy':'exact decisions and meanNLL1e-4; checkpoint/datahashes from actual tensors','scope':'Within-project continuation-state inference replay, no optimizer step/test; not external replication','complete_scientific_task3':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','checks','issues','cases','individual_decisions_replayed','max_mean_nll_error')}))
    if issues:raise SystemExit(1)
if __name__=='__main__':main()
