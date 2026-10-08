"""Prospectively frozen query-policy x school training, resumable and archived."""
import argparse,hashlib,importlib.util,json,os,pathlib,shutil,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
s=importlib.util.spec_from_file_location('qtrain_operations',ROOT/'neuropixel/research/qtrain.py');operations=importlib.util.module_from_spec(s);s.loader.exec_module(operations)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def load_plan(path,expected_hash):
    if sha(path)!=expected_hash:raise ValueError('exact execution plan hash mismatch')
    p=read(path)
    if p['status']!='frozen' or p['runs']!=operations.inventory():raise ValueError('unfrozen/mismatched study inventory')
    for f,h in p['source_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('changed frozen source '+f)
    return p

def run_one(conf,out,plan_path,expected_hash,stop_after=None):
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM admission below8GiB')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from neuropixel.research.models import build_model
    from neuropixel.research.data import ResearchRoleTask,frozen_dataset
    from neuropixel.research.experiment import configure_runtime,resource_sample,environment_record,save_json,atomic_binary,utc_now
    plan=load_plan(plan_path,expected_hash);device=configure_runtime('cpu',2);env=environment_record(device);out.mkdir(parents=True,exist_ok=True);done=out/'result.json'
    if done.exists():
        old=read(done)
        if old['status']!='completed' or old['config']!=conf or old['plan_sha256']!=expected_hash or sha(out/'checkpoint.pt')!=old['checkpoint_sha256']:raise ValueError('completed result differs')
        return old
    resource_sample(device,8);task=ResearchRoleTask(8,8,seed=0);data={};dataset_records={}
    for panel,split,n,seed in [('probe','train',2048,77001),('validation','validation',4096,77002)]:
        ds=frozen_dataset(task,split,n,seed);data[panel]=ds;p=out.parent/'datasets'/f'{panel}_n{n}_seed{seed}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            with np.load(p,allow_pickle=False) as old:
                if any(not np.array_equal(old[k],v.numpy()) for k,v in zip(('canvas','target','roles'),ds)):raise ValueError('existing immutable dataset differs')
        else:np.savez_compressed(p,canvas=ds[0].numpy(),target=ds[1].numpy(),roles=ds[2].numpy())
        dataset_records[panel]={'file':p.name,'sha256':sha(p),'content_sha256':operations.tensor_digest(*ds),'split':split,'n':n,'sample_seed':seed}
    torch.manual_seed(conf['init_seed']);model=build_model('neuropixel',vocab=35,h=8,w=8,steps=16,freeze_pad=False,fire_rate=.5);torch.manual_seed(92001)
    sampler=torch.Generator(device='cpu').manual_seed(92002);query_rng=torch.Generator(device='cpu').manual_seed(92003);opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    check=out/'checkpoint.pt';curve=[];evaluations=[];resources=[];update0=0;elapsed=0.
    if check.exists():
        old=torch.load(check,map_location='cpu',weights_only=True)
        if old['config']!=conf or old['plan_sha256']!=expected_hash or old['environment']!=env:raise ValueError('resume identity/environment mismatch')
        model.load_state_dict(old['model'],strict=True);opt.load_state_dict(old['optimizer']);sampler.set_state(old['sampler_rng']);query_rng.set_state(old['query_rng']);torch.set_rng_state(old['cpu_rng'])
        update0=old['update'];curve=old['curve'];evaluations=old['evaluations'];resources=old['resources'];elapsed=old['elapsed_training_seconds']
    def persist(update):
        payload={'config':conf,'plan_sha256':expected_hash,'environment':env,'update':update,'model':model.state_dict(),'optimizer':opt.state_dict(),'cpu_rng':torch.get_rng_state(),'sampler_rng':sampler.get_state(),'query_rng':query_rng.get_state(),'curve':curve,'evaluations':evaluations,'resources':resources,'elapsed_training_seconds':elapsed,'datasets':dataset_records}
        with atomic_binary(check) as f:torch.save(payload,f)
        save_json(out/'progress.json',{'status':'training','completed_updates':update,'config':conf,'plan_sha256':expected_hash,'curve':curve,'evaluations':evaluations,'test_accessed':False})
    def endpoint_at(update):
        endpoint={'update':update,'test_accessed':False,'panels':{}}
        for panel,ds in data.items():endpoint['panels'][panel]=operations.evaluate_queries(model,ds,out/'decisions',f'u{update}_{panel}',plan['evaluation_mask_seeds'],include_masked=update==8192)
        evaluations.append(endpoint);persist(update);p=out/f'checkpoint_u{update}.pt'
        if not p.exists():shutil.copyfile(check,p)
        elif torch.load(p,map_location='cpu',weights_only=True)['update']!=update:raise ValueError('existing endpoint checkpoint differs')
    if update0 in (1024,4096,8192) and update0 not in [x['update'] for x in evaluations]:endpoint_at(update0)
    model.train();window=[];contexts_hash=hashlib.sha256();role_counts=torch.zeros(4,dtype=torch.long);start=time.monotonic()
    try:
        for update in range(update0+1,conf['updates']+1):
            if update%128==1:resources.append(resource_sample(device,8))
            x,y,r,witness=operations.sample_batch(task,sampler,query_rng,conf['policy']);contexts_hash.update(witness['base_canvas'].numpy().tobytes());contexts_hash.update(witness['fillers'].numpy().tobytes());role_counts+=torch.bincount(r,minlength=4)
            if update==1:
                p=out/'first_batch.npz'
                arrays={'canvas':x.numpy(),'target':y.numpy(),'roles':r.numpy(),**{k:v.numpy() for k,v in witness.items()}}
                if p.exists():
                    with np.load(p,allow_pickle=False) as old:
                        if any(not np.array_equal(old[k],v) for k,v in arrays.items()):raise ValueError('preserved first witness differs')
                else:np.savez_compressed(p,**arrays)
            opt.zero_grad(set_to_none=True);output=model(x,lens_every=4 if conf['school_weight'] else 0);answer=F.cross_entropy(output['logits'],y);school=answer.new_zeros(())
            if conf['school_weight']:
                n=output['lens'].shape[1];occupied=(x!=0).unsqueeze(1).expand(-1,n,-1,-1);labels=x.unsqueeze(1).expand(-1,n,-1,-1);school=F.cross_entropy(output['lens'][occupied],labels[occupied])
            loss=answer+conf['school_weight']*school
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite total loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();window.append((float(answer.detach()),float(school.detach()),float(loss.detach())))
            if update%128==0:
                elapsed+=time.monotonic()-start;curve.append({'update':update,'answer_loss_mean':sum(v[0] for v in window)/128,'school_loss_mean':sum(v[1] for v in window)/128,'total_loss_mean':sum(v[2] for v in window)/128,'last_unclipped_gradient_norm':float(grad),'role_counts':role_counts.tolist(),'context_window_sha256':contexts_hash.hexdigest(),'elapsed_training_seconds':elapsed});window=[];role_counts.zero_();contexts_hash=hashlib.sha256()
                persist(update)
                if update in (1024,4096,8192):endpoint_at(update)
                resources.append(resource_sample(device,8));start=time.monotonic();print(json.dumps({'event':'qtrain_progress','run':conf['run_id'],'update':update,'answer_loss':curve[-1]['answer_loss_mean'],'school_loss':curve[-1]['school_loss_mean']}),flush=True)
                if stop_after and update>=stop_after:return None
        final=evaluations[-1]['panels'];gate=final['probe']['metrics']['matched']['joint_query_binding']>=.95 and final['validation']['metrics']['matched']['joint_query_binding']>=.90
        result={'status':'completed','config':conf,'plan_sha256':expected_hash,'environment':env,'parameter_count':sum(p.numel() for p in model.parameters()),'curve':curve,'evaluations':evaluations,'resources':resources,'datasets':dataset_records,'checkpoint_sha256':sha(check),'matched_joint_competence_gate_passed':gate,'test_accessed':False,'completed_utc':utc_now()};save_json(done,result);return result
    except Exception as error:
        save_json(out/'incident.json',{'error_type':type(error).__name__,'message':str(error),'last_checkpoint_update':curve[-1]['update'] if curve else update0,'test_accessed':False});raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--describe',action='store_true');ap.add_argument('--plan',type=pathlib.Path,default=ROOT/'docs/research/OPT03_QTRAIN_plan.json');ap.add_argument('--expected-plan-sha256');ap.add_argument('--output',type=pathlib.Path,default=ROOT/'results/research/OPT03_QTRAIN');ap.add_argument('--stop-after',type=int);a=ap.parse_args()
    if a.describe:print(json.dumps({'runs':operations.inventory(),'training_count':12,'updates':98304,'test_access':False}));return
    if not a.expected_plan_sha256:raise ValueError('independently frozen expected plan hash required')
    plan=load_plan(a.plan,a.expected_plan_sha256)
    for conf in plan['runs']:
        result=run_one(conf,a.output/conf['run_id'],a.plan,a.expected_plan_sha256,a.stop_after)
        if result is not None and os.environ.get('QTRAIN_PUBLISH_AFTER_CASE')=='1':
            import subprocess
            subprocess.run([sys.executable,'scripts/archive_OPT03_QTRAIN_case.py','--root',str(a.output),'--run-id',conf['run_id'],'--plan',str(a.plan)],cwd=ROOT,check=True)
if __name__=='__main__':main()
