"""Five paired pixel-only learning realizations, then sealed official test access."""
import argparse,hashlib,json,os,sys,time,subprocess,io,zipfile,urllib.request,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import psutil
    def ram():
        v=psutil.virtual_memory().available/2**30
        if v<8:raise RuntimeError('RAM8 refused')
        return v
    ram();plan_path=ROOT/'docs/research/VIS07_execution_plan.json';plan=json.loads(plan_path.read_bytes());plan_hash=sha(plan_path)
    for k,h in plan['source_sha256'].items():
        if sha(ROOT/k)!=h:raise ValueError('Frozen source differs '+k)
    # Exact archived prerequisite admitted both fixed pilot fits before this recipe.
    prereq=plan['preflight'];raw=urllib.request.urlopen(prereq['zip_url'],timeout=60).read(8*1024**2)
    if hashlib.sha256(raw).hexdigest()!=prereq['zip_sha256']:raise ValueError('Preflight ZIP differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:pre=json.loads(z.read('preflight/receipt.json'))
    if pre['status']!='admitted' or not all(c['passed'] for c in pre['checks']) or pre['plan_sha256']!=prereq['plan_sha256']:raise ValueError('Preflight not admitted')
    import numpy as np,torch
    from neuropixel.research.vis07_data import original,development,pixels,final,identity
    from neuropixel.research.vis07_models import make
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    ram();a.output.mkdir(parents=True,exist_ok=True);official=original(a.output/'original_uci.zip');rows,train_ids,dev_ids=development(official);train_x,train_y=pixels(rows[train_ids]);dev_x,dev_y=pixels(rows[dev_ids]);selected=[];training_records=[]
    np.savez_compressed(a.output/'development_indices.npz',train=train_ids,dev=dev_ids)
    save(a.output/'source_record.json',dict(source_commit=os.environ.get('GITHUB_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'),plan_sha256=plan_hash,software=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),train_examples=3523,dev_examples=300,test_scored=False,original264_test_scored=False))
    def evaluate(model,x,y):
        model.eval();logits=[]
        with torch.inference_mode():
            for at in range(0,len(x),128):logits.append(model(torch.from_numpy(x[at:at+128])).cpu())
        log=torch.cat(logits);gold=torch.from_numpy(y);nll=torch.nn.functional.cross_entropy(log,gold,reduction='none').numpy();pred=log.argmax(-1).numpy()
        return dict(correct=int((pred==y).sum()),n=len(y),accuracy=float(np.mean(pred==y)),mean_nll=float(nll.astype(np.float64).mean())),dict(logits=log.numpy(),prediction=pred,target=y,nll=nll)
    def publish(root,registration):
        if os.environ.get('VIS07_PUBLISH')=='1':subprocess.run([sys.executable,str(ROOT/'scripts/publish_sprint_evidence.py'),'--root',str(root),'--registration',registration,'--branch',plan['evidence_branch']],check=True,capture_output=True)
    for seed in plan['seeds']:
        rng=np.random.default_rng(1000000+seed);orders=np.stack([rng.permutation(len(train_ids)) for _ in range(plan['epochs'])]);np.savez_compressed(a.output/f'orders_seed{seed}.npz',orders=orders)
        for family in ['nca','cnn']:
            ram();folder=a.output/'cases'/f'seed{seed}'/family;folder.mkdir(parents=True,exist_ok=True);done=folder/'result.json'
            if done.exists():
                r=json.loads(done.read_bytes());assert r['plan_sha256']==plan_hash and r['seed']==seed and r['family']==family and sha(folder/'best.pt')==r['selected_sha256'];selected.append(r['selection']);training_records.append(r);continue
            model=make(family,seed);optimizer=torch.optim.Adam(model.parameters(),lr=.001);curve=[];best=None;epoch0=0;resume=folder/'checkpoint.pt';start=time.perf_counter();cpu=time.process_time()
            if resume.exists():
                cp=torch.load(resume,map_location='cpu',weights_only=True);assert cp['plan_sha256']==plan_hash and cp['seed']==seed and cp['family']==family
                model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['torch_rng']);random.setstate(cp['python_rng']);curve=cp['curve'];best=cp['best'];epoch0=cp['epoch']
            for epoch in range(epoch0,plan['epochs']):
                ram();model.train();total=0.;seen=0
                for at in range(0,len(train_ids),64):
                    ix=orders[epoch,at:at+64];image=torch.from_numpy(train_x[ix]);target=torch.from_numpy(train_y[ix]);optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(image),target)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    if not torch.isfinite(gradient):raise ValueError('Nonfinite gradient')
                    optimizer.step();total+=float(loss.detach())*len(ix);seen+=len(ix)
                dev,_=evaluate(model,dev_x,dev_y);row=dict(epoch=epoch+1,train_loss=total/seen,dev=dev);curve.append(row);key=(-dev['correct'],dev['mean_nll'],epoch+1)
                if best is None or tuple(key)<tuple(best['key']):
                    best=dict(epoch=epoch+1,key=key,dev=dev);torch.save(dict(model=model.state_dict(),seed=seed,family=family,epoch=epoch+1,plan_sha256=plan_hash),folder/'best.pt')
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_order_rng=rng.bit_generator.state,epoch=epoch+1,curve=curve,best=best,seed=seed,family=family,plan_sha256=plan_hash),resume)
                save(folder/'curve.json',curve)
                if (epoch+1)%10==0:print(json.dumps(dict(seed=seed,family=family,epoch=epoch+1,dev_accuracy=dev['accuracy'])),flush=True)
            selection=dict(seed=seed,family=family,path=str(folder/'best.pt'),sha256=sha(folder/'best.pt'),epoch=best['epoch'],dev=best['dev'])
            r=dict(status='closed_fixed_forty_epoch_fit',seed=seed,family=family,plan_sha256=plan_hash,epochs=40,updates=40*((len(train_ids)+63)//64),parameters=sum(p.numel() for p in model.parameters()),selected_sha256=selection['sha256'],selection=selection,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,test_scored=False)
            save(done,r);selected.append(selection);training_records.append(r);publish(folder,f'VIS07SEALED{seed}{family.upper()}')
    if len(selected)!=10 or {(r['seed'],r['family']) for r in selected}!={(s,f) for s in plan['seeds'] for f in ['nca','cnn']}:raise ValueError('Incomplete all-ten gate')
    gate=dict(status='all_ten_selected_models_sealed_before_official_test',selected_models=selected,plan_sha256=plan_hash,created_at_unix=time.time(),scope='Public official UCI test, not blind custody or independent external replication')
    gate_root=a.output/'gate';save(gate_root/'selection_gate.json',gate);publish(gate_root,'VIS07FINALGATE')
    # Access is consumed durably before parsing/scoring the official test.
    save(a.output/'final_access.json',dict(status='consumed',selection_gate_sha256=sha(gate_root/'selection_gate.json'),at_unix=time.time(),plan_sha256=plan_hash))
    test_rows=final(official,gate);test_x,test_y=pixels(test_rows);np.savez_compressed(a.output/'official_test_data.npz',image=test_x,target=test_y)
    scores=[]
    for selection in selected:
        ram();model=make(selection['family'],selection['seed']);cp=torch.load(selection['path'],map_location='cpu',weights_only=True);model.load_state_dict(cp['model']);folder=a.output/'final'/f'seed{selection["seed"]}'/selection['family'];folder.mkdir(parents=True,exist_ok=True)
        conditions={'base':test_x,'half_intensity':test_x*.5,'shift_right_crop':np.pad(test_x[:,:,:,:-1],((0,0),(0,0),(0,0),(1,0)))}
        for condition,x in conditions.items():
            metrics,arrays=evaluate(model,x,test_y);np.savez_compressed(folder/f'{condition}.npz',**arrays);scores.append(dict(seed=selection['seed'],family=selection['family'],condition=condition,**metrics))
        if selection['family']=='nca':
            core=model.core;core.eval();depth_rows=[]
            with torch.inference_mode():
                for at in range(0,len(test_x),128):
                    image=torch.from_numpy(test_x[at:at+128]);ids=core.retina(image);state=core.seed(ids)
                    for step in range(33):
                        if step in [0,4,8,16,32]:
                            logits=core.read(state[:,:,4,4])@core.dictionary().T;logits[:,0]=-10000
                            depth_rows.append(dict(step=step,at=at,logits=logits.numpy().copy(),state_rms=np.sqrt(np.mean(state.numpy().astype(np.float64)**2,axis=(1,2,3)))))
                        if step<32:state=state+core.f2(torch.relu(core.f1(torch.cat([state,core.perceive(state),ids],1))))
            arrays={}
            for step in [0,4,8,16,32]:
                part=[r for r in depth_rows if r['step']==step];log=np.concatenate([r['logits'] for r in part]);arrays[f'logits_T{step}']=log;arrays[f'state_rms_T{step}']=np.concatenate([r['state_rms'] for r in part]);scores.append(dict(seed=selection['seed'],family='nca',condition=f'held_depth{step}',correct=int((log.argmax(-1)==test_y).sum()),n=len(test_y),accuracy=float(np.mean(log.argmax(-1)==test_y)),finite_logits=bool(np.isfinite(log).all())))
            np.savez_compressed(folder/'held_depth_controls.npz',target=test_y,**arrays)
    for k,h in plan['source_sha256'].items():
        if sha(ROOT/k)!=h:raise ValueError('Scientific source drift')
    save(a.output/'result.json',dict(status='completed_all_five_pairs_and_registered_official_test_conditions',plan_sha256=plan_hash,source_commit=os.environ.get('GITHUB_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'),training_records=training_records,selection_gate_sha256=sha(gate_root/'selection_gate.json'),scores=scores,original264_test_scored=False,external_replication=False,available_ram_finish_gib=ram()))
    print(json.dumps(dict(status='completed_all_five_pairs',selected_models=len(selected),score_rows=len(scores))))
if __name__=='__main__':main()
