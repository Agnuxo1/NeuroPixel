"""Balanced query-supervision interventions; no final-test data access."""
from __future__ import annotations
import hashlib,json

def inventory():
    return [{'run_id':f'QTRAIN_{policy}_school{school}_seed{seed}','init_seed':seed,'policy':policy,'school_weight':school,'updates':8192}
            for seed in (200,201,202) for policy in ('paired','repeated') for school in (0.,.3)]

def assemble_batch(canvas,fillers,assignment,role_ids,policy):
    import torch
    if policy not in ('paired','repeated'):raise ValueError('unknown query policy')
    if canvas.shape!=(16,8,8) or fillers.shape!=(16,4) or assignment.shape!=(16,):raise ValueError('sixteen native contexts required')
    if not torch.equal(torch.bincount(assignment,minlength=4),torch.full((4,),4)):raise ValueError('balanced assignment required')
    contexts=torch.arange(16).repeat_interleave(4);roles=torch.arange(4).repeat(16) if policy=='paired' else assignment.repeat_interleave(4)
    x=canvas[contexts].clone();x[:,7,6]=role_ids[roles];target=fillers[contexts,roles]
    if not torch.equal(torch.bincount(roles,minlength=4),torch.full((4,),16)):raise ValueError('unbalanced sixty-four examples')
    return x,target,roles,contexts

def sample_batch(task,sampler,query_rng,policy):
    import torch
    canvas,_,meta=task.sample(16,'train',sampler,device='cpu',meta=True)
    assignment=(torch.arange(16)%4)[torch.randperm(16,generator=query_rng)]
    x,y,roles,contexts=assemble_batch(canvas,meta['fillers'],assignment,task.role_ids,policy)
    return x,y,roles,{'base_canvas':canvas,'fillers':meta['fillers'],'assignment':assignment,'context_indices':contexts}

def tensor_digest(*arrays):
    import numpy as np
    h=hashlib.sha256()
    for array in arrays:
        v=np.ascontiguousarray(array.detach().cpu().numpy() if hasattr(array,'detach') else array)
        h.update(str(v.dtype).encode());h.update(json.dumps(list(v.shape)).encode());h.update(v.tobytes())
    return h.hexdigest()

def evaluate_queries(model,dataset,output,prefix,mask_seeds,include_masked):
    """Return scalar records and immutable decision arrays; preserve firing RNG/mode."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from neuropixel.research.OPT03_role_views import transform
    canvas,target,roles=dataset
    cf,cy,cr=transform(canvas.numpy(),target.numpy(),roles.numpy(),'query_flip')
    transformed=(torch.from_numpy(cf),torch.from_numpy(cy),torch.from_numpy(cr));original_mode=model.training;original_fire=model.fire_rate
    records={};artifacts={};paths=Path(output);paths.mkdir(parents=True,exist_ok=True)
    model_h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):model_h.update(name.encode());model_h.update(value.detach().cpu().contiguous().numpy().tobytes())
    state_identity=model_h.hexdigest();dataset_identity=tensor_digest(*dataset)
    def forward(ds,seed):
        x,y,r=ds;torch.manual_seed(seed);pred=[];loss=[]
        with torch.inference_mode():
            for start in range(0,len(x),256):
                logits=model(x[start:start+256])['logits'];pred.append(logits.argmax(-1));loss.append(F.cross_entropy(logits,y[start:start+256],reduction='none'))
        return torch.cat(pred).numpy(),torch.cat(loss).numpy()
    with torch.random.fork_rng(devices=[]):
        try:
            for name,seeds,fire in [('dense',[mask_seeds[0]],1.)]+([('matched',mask_seeds,.5)] if include_masked else []):
                file=paths/f'{prefix}_{name}.npz';model.fire_rate=fire;model.train(fire<1)
                if file.exists():
                    with np.load(file,allow_pickle=False) as saved:
                        if str(saved['state_sha256'].item())!=state_identity or str(saved['dataset_content_sha256'].item())!=dataset_identity or saved['mask_seeds'].tolist()!=list(seeds):raise ValueError('existing endpoint identity differs')
                        if not np.array_equal(saved['target'],target.numpy()) or not np.array_equal(saved['query_target'],cy):raise ValueError('existing endpoint labels differ')
                        pred=saved['predictions'].copy();nll=saved['nll'].copy();cpred=saved['query_predictions'].copy();cnll=saved['query_nll'].copy()
                else:
                    pp=[];nn=[];cc=[];cn=[]
                    for seed in seeds:
                        p,n=forward(dataset,seed);c,l=forward(transformed,seed);pp.append(p);nn.append(n);cc.append(c);cn.append(l)
                    pred=np.stack(pp);nll=np.stack(nn);cpred=np.stack(cc);cnll=np.stack(cn)
                    np.savez_compressed(file,predictions=pred,nll=nll,query_predictions=cpred,query_nll=cnll,target=target.numpy(),roles=roles.numpy(),query_target=cy,query_roles=cr,mask_seeds=seeds,state_sha256=state_identity,dataset_content_sha256=dataset_identity)
                gold=target.numpy();r=roles.numpy();noun=np.isin(r,[0,2]);good=pred==gold[None,:];cf_good=cpred==cy[None,:];per=[float(good[:,r==i].mean()) for i in range(4)]
                if not np.array_equal(pred[:,~noun],cpred[:,~noun]):raise ValueError('query-control decisions differ under common masks')
                records[name]={'accuracy':float(good.mean()),'binding':(per[0]+per[2])/2,'joint_query_binding':float((good&cf_good)[:,noun].mean()),'per_role':per,'mean_nll':float(nll.astype(np.float64).mean()),'mask_seeds':list(seeds),'training_initializations_not_mask_count':True}
                artifacts[file.name]={'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
        finally:model.fire_rate=original_fire;model.train(original_mode)
    return {'metrics':records,'decision_artifacts':artifacts,'test_accessed':False}
