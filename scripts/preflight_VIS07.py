"""Fixed preflight: genuine visual memorization and contracts, no official test."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 admission refused')
    plan_path=ROOT/'docs/research/VIS07_preflight_execution_plan.json';plan=json.loads(plan_path.read_bytes())
    for name,h in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Preflight frozen source differs '+name)
    import numpy as np,torch
    from neuropixel.research.vis07_data import original,development,pixels,identity
    from neuropixel.research.vis07_models import make
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True,exist_ok=False);raw=original(a.output/'original_uci.zip');all_rows,train,dev=development(raw)
    source_ids=np.concatenate([np.flatnonzero(all_rows[train,-1]==label)[:5] for label in range(10)]);selected=train[source_ids];x,y=pixels(all_rows[selected]);image=torch.from_numpy(x);target=torch.from_numpy(y)
    checks=[];runs=[];start=time.perf_counter();counts=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    check('exact sample/group separation',len(train)==3523 and len(dev)==300 and not set(identity(all_rows[train]))&set(identity(all_rows[dev])))
    check('balanced fifty TRAIN-only images',len(y)==50 and np.bincount(y,minlength=11)[1:].tolist()==[5]*10)
    check('image-only sample dimensions',x.shape==(50,3,8,8) and np.array_equal(x[:,0],x[:,1]) and np.array_equal(x[:,1],x[:,2]))
    for family in ['nca','cnn']:
        if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 refused before new pilot')
        model=make(family,83999);params=sum(p.numel() for p in model.parameters());counts.append(params);opt=torch.optim.Adam(model.parameters(),lr=.001);curve=[];t0=time.perf_counter();c0=time.process_time()
        for update in range(1,513):
            if update%64==1 and psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 during pilot')
            opt.zero_grad(set_to_none=True);logits=model(image);loss=torch.nn.functional.cross_entropy(logits,target);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            if update in [1,128,256,512]:
                curve.append(dict(update=update,loss=float(loss.detach()),gradient=float(gradient)))
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),update=update,torch_rng=torch.get_rng_state(),curve=curve),a.output/f'{family}_checkpoint_u{update}.pt')
                print(json.dumps(dict(family=family,update=update,loss=curve[-1]['loss'])),flush=True)
        model.eval()
        with torch.inference_mode():logits=model(image);pred=logits.argmax(-1).numpy();accuracy=float(np.mean(pred==y))
        check(f'{family} finite learning',all(np.isfinite(row['loss']) and np.isfinite(row['gradient']) for row in curve) and curve[-1]['loss']<curve[0]['loss'])
        check(f'{family} admission45of50',int((pred==y).sum())>=45)
        np.savez_compressed(a.output/f'{family}_memorization.npz',image=x,target=y,pred=pred,logits=logits.numpy(),source_indices=selected)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),update=512,torch_rng=torch.get_rng_state()),a.output/f'{family}_checkpoint.pt')
        runs.append(dict(family=family,parameters=params,accuracy=accuracy,correct=int((pred==y).sum()),updates=512,curve=curve,wall_seconds=time.perf_counter()-t0,cpu_seconds=time.process_time()-c0))
    check('near nominal parameter count within one percent',abs(counts[0]-counts[1])/max(counts)<.01)
    check('RAM8 finish',psutil.virtual_memory().available/2**30>=8)
    check('frozen preflight source remains unchanged',all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in plan['source_sha256'].items()))
    result=dict(status='admitted' if all(x['passed'] for x in checks) else 'not_admitted',checks=checks,runs=runs,source_commit=os.environ.get('GITHUB_SHA'),workflow_run_id=os.environ.get('GITHUB_RUN_ID'),plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),original_test_scored=False,official_UCI_test_scored=False,main_trainings=0,scientific_pilot_trainings=2,software=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),wall_seconds=time.perf_counter()-start,available_ram_finish_gib=psutil.virtual_memory().available/2**30)
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));return 0 if result['status']=='admitted' else 2
if __name__=='__main__':sys.exit(main())
