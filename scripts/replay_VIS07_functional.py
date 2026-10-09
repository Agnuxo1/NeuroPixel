"""Read-only functional-convolution replay of ten archived models and export."""
import argparse,hashlib,io,json,os,sys,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
URL='https://raw.githubusercontent.com/Agnuxo1/NeuroPixel/a783bb77d8e9a73a1c6f3fb624562ffde242ebd7/results/research/VIS07MAIN_cloud/37908785698/raw.zip'
ZIP_SHA='6677eb3c5d8e04fad5d6ed2947c707bc5ab0761a12f654b4b5972cf9bd648ca3'
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 admission refused')
    import numpy as np,torch
    import torch.nn.functional as F
    from neuropixel.research.vis07_data import parse,pixels,development,TRAIN_SHA,TEST_SHA
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True,exist_ok=False);raw=urllib.request.urlopen(URL,timeout=60).read(128*1024**2)
    if hashlib.sha256(raw).hexdigest()!=ZIP_SHA:raise ValueError('Original complete main ZIP differs')
    inp=a.output/'inputs';inp.mkdir()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('archive_manifest.json'));assert manifest['source_commit']=='8cbc49a29c3526846712c2b5f415d3b8ec0fdb50' and set(z.namelist())==set(manifest['files'])|{'archive_manifest.json'}
        for name,v in manifest['files'].items():
            p=inp/name;p.resolve().relative_to(inp.resolve());b=z.read(name);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    original=json.loads((inp/'result.json').read_bytes());assert original['status']=='completed_all_five_pairs_and_registered_official_test_conditions'
    test=parse((inp/'original_uci.zip').read_bytes(),'optdigits.tes',TEST_SHA);x,y=pixels(test);checks=[];decisions=0;max_logit_error=0.;max_nll_error=0.
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def nca(w,image,steps=8):
        ids=image
        for ix,dilation in [(0,1),(2,2),(4,4),(6,8)]:ids=F.relu(F.conv2d(ids,w[f'core.retina.net.{ix}.weight'],w[f'core.retina.net.{ix}.bias'],padding=dilation,dilation=dilation))
        ids=F.conv2d(ids,w['core.retina.net.8.weight'],w['core.retina.net.8.bias']);state=F.conv2d(ids,w['core.seed.weight'],w['core.seed.bias'])
        for _ in range(steps):
            perception=F.conv2d(state,w['core.perceive.weight'],padding=1,groups=16);hidden=F.relu(F.conv2d(torch.cat([state,perception,ids],1),w['core.f1.weight'],w['core.f1.bias']));state=state+F.conv2d(hidden,w['core.f2.weight'],w['core.f2.bias'])
        logits=F.linear(state[:,:,4,4],w['core.read.weight'],w['core.read.bias'])@w['core.embed.weight'].T;logits[:,0]=-10000;return logits,state
    def cnn(w,image):
        h=F.relu(F.conv2d(image,w['features.0.weight'],w['features.0.bias'],padding=1));h=F.relu(F.conv2d(h,w['features.2.weight'],w['features.2.bias'],padding=1));h=F.adaptive_avg_pool2d(h,1).flatten(1);log=F.linear(h,w['head.weight'],w['head.bias']);log[:,0]=-10000;return log
    for seed in range(240,245):
        for family in ['nca','cnn']:
            if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 before archived model load')
            folder=inp/'cases'/f'seed{seed}'/family;record=json.loads((folder/'result.json').read_bytes());check(f'{seed}/{family} selected original hash',hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()==record['selected_sha256']);w=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['model']
            for condition,images in [('base',x),('half_intensity',x*.5),('shift_right_crop',np.pad(x[:,:,:,:-1],((0,0),(0,0),(0,0),(1,0))))]:
                with np.load(inp/'final'/f'seed{seed}'/family/f'{condition}.npz',allow_pickle=False) as z:
                    actual=[]
                    with torch.inference_mode():
                        for at in range(0,len(images),128):
                            image=torch.from_numpy(images[at:at+128]);actual.append((nca(w,image)[0] if family=='nca' else cnn(w,image)).cpu())
                    log=torch.cat(actual);pred=log.argmax(-1).numpy();nll=F.cross_entropy(log,torch.from_numpy(y),reduction='none').numpy();error=float(np.max(np.abs(log.numpy()-z['logits'])));ne=float(abs(nll.astype(np.float64).mean()-z['nll'].astype(np.float64).mean()));max_logit_error=max(max_logit_error,error);max_nll_error=max(max_nll_error,ne)
                    check(f'{seed}/{family}/{condition} targets',np.array_equal(y,z['target']));check(f'{seed}/{family}/{condition} full logits',np.all(np.abs(log.numpy()-z['logits'])<=1e-4+1e-5*np.abs(z['logits'])),error);check(f'{seed}/{family}/{condition} all decisions',np.array_equal(pred,z['prediction']));check(f'{seed}/{family}/{condition} mean NLL',ne<=1e-4,ne);decisions+=len(y)
            if family=='nca':
                with np.load(inp/'final'/f'seed{seed}'/family/'held_depth_controls.npz',allow_pickle=False) as z:
                    for depth in [0,4,8,16,32]:
                        out=[]
                        with torch.inference_mode():
                            for at in range(0,len(x),128):out.append(nca(w,torch.from_numpy(x[at:at+128]),depth)[0])
                        log=torch.cat(out).numpy();ref=z[f'logits_T{depth}'];check(f'{seed}/depth{depth} logits',np.all(np.abs(log-ref)<=1e-4+1e-5*np.abs(ref)));check(f'{seed}/depth{depth} decisions',np.array_equal(log.argmax(-1),ref.argmax(-1)));decisions+=len(y)
                export=dict(seed=seed,source_checkpoint_sha256=record['selected_sha256'],config=dict(c=16,cid=8,hidden=32,width=8,height=8,vocab=11,steps=8),weights={k.removeprefix('core.'):dict(shape=list(v.shape),values=v.flatten().tolist()) for k,v in w.items()},scope='Selected complete main model, TRAIN/DEV selected before official test; pixel-only inference')
                ep=a.output/'export'/f'weights_seed{seed}.json';ep.parent.mkdir(exist_ok=True);ep.write_text(json.dumps(export,separators=(',',':'))+'\n',encoding='utf-8')
        print(json.dumps(dict(seed=seed,verified_decisions=decisions)),flush=True)
    train,train_ids,dev_ids=development((inp/'original_uci.zip').read_bytes());gallery=[]
    for label in range(10):
        index=int(next(i for i in train_ids if train[i,-1]==label));gallery.append(dict(label=label,original_train_index=index,pixels=train[index,:64].tolist()))
    (a.output/'export/gallery_train.json').write_text(json.dumps(dict(source='Official TRAIN examples, one fixed original row per class, no target as model input',license='CC BY4.0',citation='Alpaydin & Kaynak1998,UCI DOI10.24432/C50P49',samples=gallery),indent=2)+'\n',encoding='utf-8')
    result=dict(status='verified_functional_no_training_VIS07_replay' if all(x['passed'] for x in checks) else 'failed_preserved',checks=checks,individual_decisions=decisions,max_absolute_logit_error=max_logit_error,max_mean_nll_error=max_nll_error,training_updates=0,source_archive_sha256=ZIP_SHA,source_commit=os.environ.get('GITHUB_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'),external_replication=False)
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],checks=len(checks),decisions=decisions)));return 0 if result['status'].startswith('verified') else 1
if __name__=='__main__':sys.exit(main())
