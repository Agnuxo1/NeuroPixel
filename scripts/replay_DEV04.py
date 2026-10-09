"""Read-only fresh-birth/state and endpoint replay; no optimizer step or effects."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PLAN='2484f3c5917e474ca3755bcdbbaca7d322077615ba47d51653116042d980f20a'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def recount_body(stored):
    import numpy as np
    pred,query_pred=stored['predictions'],stored['query_predictions'];y,z,roles=stored['target'],stored['query_target'],stored['roles']
    if pred.ndim!=2 or query_pred.shape!=pred.shape or pred.shape[1]!=len(y) or set(roles.tolist())!={0,1,2,3}:raise ValueError('Complete native body decision bank required')
    nll=stored['nll']
    if nll.shape!=pred.shape or not np.isfinite(nll).all():raise ValueError('Finite original body NLL bank required')
    good,other=pred==y[None],query_pred==z[None];noun=np.isin(roles,[0,2])
    if not np.array_equal(pred[:,~noun],query_pred[:,~noun]):raise ValueError('Unchanged query controls differ')
    per=[float(good[:,roles==role].mean()) for role in range(4)]
    return dict(accuracy=float(good.mean()),binding=(per[0]+per[2])/2,joint_query_binding=float((good&other)[:,noun].mean()),per_role=per,mean_nll=float(nll.astype(np.float64).mean()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--case-folder',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    args=p.parse_args()
    from neuropixel.research.qtrain_budget import require_ram
    ram=require_ram()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from neuropixel.research import dev04 as ex
    from neuropixel.research.qtrain_budget import state_digest,validate_state,restore,verify_environment,training_state
    from neuropixel.research.experiment import configure_runtime,environment_record
    from neuropixel.research.OPT03_role_views import transform
    from neuropixel.research.frozen_readout import FrozenReadout,body_digest
    from neuropixel.research.readout_experiment import extract_bank,training_state as head_state,restore as restore_head
    from scripts.replay_READ03 import recount
    require_ram();start=time.monotonic();plan_path=ROOT/'docs/research/DEV04_execution_plan.json'
    if sha(plan_path)!=PLAN:raise ValueError('Recipe differs')
    plan=json.loads(plan_path.read_text(encoding='utf-8'));config=next(r for r in plan['runs'] if r['run_id']==args.case)
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed')
    env=environment_record(configure_runtime('cpu',2));folder=args.case_folder
    before={str(f):sha(f) for f in folder.rglob('*') if f.is_file()}
    task=ex.DevelopmentRoleTask(config['partition_seed']);datasets=ex.panels(task)
    issues=[];checks=0;decisions=0;max_error=0.;body_endpoints=[];head_endpoints={}
    def check(ok,label):
        nonlocal checks
        checks+=1
        if not bool(ok):issues.append(label)
    check(task.partition_record()==plan['partitions'][str(config['partition_seed'])],'actual partition membership/hash')
    closed=json.loads((folder/'result.json').read_text(encoding='utf-8')) if (folder/'result.json').exists() else None
    if closed:
        check(closed['config']==config and closed['plan_sha256']==PLAN and closed['historical_checkpoint_loaded'] is False
            and closed['new_body_initialization'] is True and closed['test_scored'] is False,'closed case complete recipe/lineage')
        check(sha(folder/'body/result.json')==closed['body_result_sha256'] and sha(folder/'body/checkpoint.pt')==closed['body_checkpoint_sha256'],'closed body result/checkpoint hashes')
    for name,data in datasets.items():
        with np.load(folder/'datasets'/f'{name}.npz',allow_pickle=False) as stored:
            for key,tensor in zip(('canvas','target','roles'),data):check(np.array_equal(tensor.numpy(),stored[key]),'regenerated dataset '+name+'/'+key)
    from neuropixel.research.readout_experiment import paired_training_dataset
    with np.load(folder/'datasets/head_train.npz',allow_pickle=False) as stored:
        for key,tensor in zip(('canvas','target','roles'),paired_training_dataset(task)):check(np.array_equal(tensor.numpy(),stored[key]),'actual head training '+key)
    def simple_state_digest(values):
        h=hashlib.sha256()
        for name,v in sorted(values.items()):h.update(name.encode());h.update(v.contiguous().numpy().tobytes())
        return h.hexdigest()
    def data_digest(data):
        h=hashlib.sha256()
        for t in data:
            a=np.ascontiguousarray(t.numpy());h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
        return h.hexdigest()
    def body_forward(model,canvas,target,seed):
        pred=[];nll=[]
        with torch.random.fork_rng(devices=[]),torch.inference_mode():
            torch.manual_seed(seed)
            for at in range(0,len(canvas),256):
                logits=model(canvas[at:at+256])['logits'];pred.append(logits.argmax(-1));nll.append(F.cross_entropy(logits,target[at:at+256],reduction='none'))
        return torch.cat(pred),torch.cat(nll)
    checkpoint_files=list(sorted((folder/'body').glob('checkpoint_u*.pt')))
    latest=folder/'body/checkpoint.pt'
    if latest.exists():checkpoint_files.append(latest)
    if not checkpoint_files:raise ValueError('No actual body checkpoints')
    body=None
    for path in checkpoint_files:
        require_ram();payload=torch.load(path,map_location='cpu',weights_only=True)
        model,opt,sampler,query_rng=ex.new_objects(config);birth=state_digest(model.state_dict())
        check(birth==plan['initial_body_model_digests'][args.case],'recreated fresh initialization')
        ex.check_resume(payload,config,PLAN,birth,env);digest=validate_state(payload,payload['update'],config,PLAN)
        restore(model,opt,sampler,query_rng,payload,digest)
        check([w['update'] for w in payload['curve']]==list(range(128,payload['update']+1,128)),'body complete update windows')
        check([w['update'] for w in payload['resources']]==list(range(1,payload['update']+1,128))
            and all(w['available_ram_gib']>=8 for w in payload['resources']),'body original RAM admission windows')
        check(all(len(w['batch_witness_sha256'])==64 and all(np.isfinite(w[k]) for k in ('answer_loss_mean','school_loss_mean','total_loss_mean','last_unclipped_gradient_norm','elapsed_update_seconds'))
            and w['role_counts']==[2048,2048,2048,2048] for w in payload['curve']),'body finite losses/gradient/supervision/witnesses')
        check(sum(v.numel() for v in model.parameters())==29824,'actual body capacity')
        check(payload['partition']==task.partition_record(),'checkpoint dataset membership')
        if path.name!='checkpoint.pt':
            update=payload['update'];body_endpoints.append(update)
            endpoint=next(r for r in payload['evaluations'] if r['update']==update)
            for panel,record in endpoint['panels'].items():
                canvas,target,roles=datasets[panel]
                cf=transform(*(t.numpy() for t in datasets[panel]),'query_flip');cx,cy,cr=(torch.from_numpy(v) for v in cf)
                for mode in ('dense','matched'):
                    model.fire_rate=1. if mode=='dense' else .5;model.train(mode=='matched')
                    artifact=folder/'body/decisions'/f'u{update}_{panel}_{mode}.npz'
                    check(sha(artifact)==record['decision_artifacts'][artifact.name]['sha256'],'actual body array bytes '+artifact.name)
                    with np.load(artifact,allow_pickle=False) as stored:
                        counted=recount_body(stored)
                        for key,value in counted.items():check(np.allclose(value,record['metrics'][mode][key],rtol=0,atol=1e-12),'independent body scalar '+artifact.name+'/'+key)
                        check(stored['mask_seeds'].tolist()==([95000] if mode=='dense' else list(range(95000,95008))),'fixed body rollout masks '+artifact.name)
                        check(str(stored['state_sha256'].item())==simple_state_digest(payload['model']),'actual body state hash '+artifact.name)
                        check(str(stored['dataset_content_sha256'].item())==data_digest(datasets[panel]),'actual input hash '+artifact.name)
                        for k,seed in enumerate(stored['mask_seeds'].tolist()):
                            pred,nll=body_forward(model,canvas,target,seed);qpred,qnll=body_forward(model,cx,cy,seed)
                            check(np.array_equal(pred.numpy(),stored['predictions'][k]),'exact body decisions '+artifact.name+'/'+str(seed))
                            check(np.array_equal(qpred.numpy(),stored['query_predictions'][k]),'exact body queryflip '+artifact.name+'/'+str(seed))
                            check(np.array_equal(stored['target'],target.numpy()) and np.array_equal(stored['query_target'],cy.numpy()) and np.array_equal(stored['roles'],roles.numpy()),'actual body labels '+artifact.name+'/'+str(seed))
                            error=max(abs(float(nll.double().mean())-float(stored['nll'][k].astype(np.float64).mean())),abs(float(qnll.double().mean())-float(stored['query_nll'][k].astype(np.float64).mean())))
                            max_error=max(max_error,error);check(error<=1e-4,'original body NLL tolerance '+artifact.name+'/'+str(seed));decisions+=len(pred)*2
        check(state_digest(training_state(model,opt,sampler,query_rng))==digest,'replay preserves body optimizer/three streams')
        if path.name=='checkpoint.pt':body=model
    # Heads can only be audited against the exact finished body that produced them.
    head_dirs=[mode for mode in ('query_attention','local_query') if (folder/mode/'progress.json').exists()]
    if head_dirs:
        if payload['update']!=16384:raise ValueError('Heads require completed fresh body')
        verify_environment(payload['environment'],env)
        for parameter in body.parameters():parameter.requires_grad_(False);parameter.grad=None
        banks={}
        for panel,data in datasets.items():
            counter=tuple(torch.from_numpy(v) for v in transform(*(t.numpy() for t in data),'query_flip'))
            banks[panel]=extract_bank(body,data,ex.EVAL_MASKS);banks[panel+'_query_flip']=extract_bank(body,counter,ex.EVAL_MASKS)
            with np.load(folder/'body/decisions'/f'u16384_{panel}_matched.npz',allow_pickle=False) as stored:
                check(np.array_equal(banks[panel]['predictions'].numpy(),stored['predictions'])
                    and np.array_equal(banks[panel+'_query_flip']['predictions'].numpy(),stored['query_predictions']),'independent latent bank original-body decisions '+panel)
        for mode in head_dirs:
            head_endpoints[mode]=[]
            record=json.loads((folder/mode/'result.json').read_text(encoding='utf-8')) if (folder/mode/'result.json').exists() else None
            peer='local_query' if mode=='query_attention' else 'query_attention'
            if record:
                peer_record=json.loads((folder/peer/'result.json').read_text(encoding='utf-8'))
                check([w['batch_witness_sha256'] for w in record['curve']]==[w['batch_witness_sha256'] for w in peer_record['curve']],'paired heads same batches and masks '+mode)
                check(record['new_backbone_updates']==0 and record['test_accessed'] is False and record['head_nominal_parameters']==3168,'frozen body/no test/nominal capacity '+mode)
                check(record['checkpoint_sha256']==sha(folder/mode/'checkpoint.pt')==sha(folder/mode/'checkpoint_u8192.pt'),'actual final head checkpoint equals evaluated endpoint '+mode)
                if closed:check(sha(folder/mode/'result.json')==closed['head_results_sha256'][mode],'closed head result hash '+mode)
            for path in sorted((folder/mode).glob('checkpoint_u*.pt')):
                require_ram();hp=torch.load(path,map_location='cpu',weights_only=True);update=hp['update']
                check(hp['config']==dict(parent_case=args.case,mode=mode,head_seed=500+config['init_seed']-200,updates=8192) and hp['plan_sha256']==PLAN,'actual head recipe '+mode)
                check(hp['effective_gradient_parameters']==3168,'actual active head capacity '+mode)
                check([w['update'] for w in hp['curve']]==list(range(128,update+1,128)),'head complete update windows '+mode)
                check([w['update'] for w in hp['resources']]==list(range(1,update+1,128)) and all(w['available_ram_gib']>=8 for w in hp['resources']),'head original RAM windows '+mode)
                check(all(torch.isfinite(v).all() for v in hp['head'].values()),'finite actual head '+mode)
                check(all(hp[k].dtype==torch.uint8 and hp[k].ndim==1 and hp[k].numel()>0 for k in ('cpu_rng','context_rng','mask_rng')),'head three actual RNG streams '+mode)
                for group in hp['optimizer']['param_groups']:
                    check(group['lr']==.001 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.0001
                        and not any(group[k] for k in ('amsgrad','maximize','capturable','differentiable')),'actual head AdamW recipe '+mode)
                reader=FrozenReadout(body,mode);optimizer=torch.optim.AdamW(reader.head_parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
                context_rng,mask_rng=torch.Generator(),torch.Generator();restore_head(reader,optimizer,context_rng,mask_rng,hp)
                original=state_digest(head_state(reader,optimizer,context_rng,mask_rng));head_endpoints[mode].append(update)
                for state in hp['optimizer']['state'].values():
                    check(float(state['step'])==update and all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)),'actual head optimizer finite/step '+mode)
                endpoint=next(r for r in hp['evaluations'] if r['update']==update)
                for panel in ('probe','validation'):
                    artifact=folder/mode/'decisions'/f'u{update}_{panel}.pt';stored=torch.load(artifact,map_location='cpu',weights_only=True)
                    check(sha(artifact)==endpoint['panels'][panel]['artifact_sha256'],'actual head arrays '+mode+panel+str(update))
                    counted=recount(stored)
                    for key,value in counted.items():check(np.allclose(value,endpoint['panels'][panel]['metrics'][key],atol=1e-12,rtol=0),'independent head scalar '+mode+panel+key+str(update))
                    for name,bank in [('original',banks[panel]),('query_flip',banks[panel+'_query_flip'])]:
                        check(torch.equal(stored[name]['target'],bank['target']) and torch.equal(stored[name]['roles'],bank['roles']),'regenerated head labels/roles '+mode+panel+name+str(update))
                        for k in range(8):
                            pp,nn=[],[]
                            with torch.inference_mode():
                                for at in range(0,len(bank['canvas']),256):
                                    x=bank['canvas'][at:at+256];facts=bank['facts'][k,at:at+256];flat=torch.zeros((len(x),64,48),dtype=facts.dtype)
                                    positions=bank['positions'][at:at+256];flat.scatter_(1,positions[:,:,None].expand(-1,-1,48),facts);flat[:,63]=bank['local'][k,at:at+256]
                                    layout=torch.channels_last if bank['channels_last'] else torch.contiguous_format
                                    state=flat.reshape(len(x),8,8,48).permute(0,3,1,2).contiguous(memory_format=layout)
                                    ids=F.embedding(x,body.dictionary()).permute(0,3,1,2);local=state[:,:,7,7]
                                    if mode=='local_query':
                                        head=reader.head(local,ids[:,:,7,6]);identity=body.read(local+head['delta'])*head['identity_gate']
                                    else:
                                        head=reader.head(state,ids,x,(7,6));identity=body.read(local+head['delta'])
                                    logits=identity@body.decoder_dictionary().T;logits[:,0]=-1e4
                                    pp.append(logits.argmax(-1));nn.append(F.cross_entropy(logits,bank['target'][at:at+256],reduction='none'))
                            pred,nll=torch.cat(pp),torch.cat(nn);check(torch.equal(pred,stored[name]['predictions'][k]),'exact head decisions '+mode+panel+str(update)+'/'+str(k))
                            error=abs(float(nll.double().mean())-float(stored[name]['nll'][k].double().mean()));max_error=max(max_error,error)
                            check(error<=1e-4,'original head NLL tolerance '+mode+panel+str(update)+'/'+str(k));decisions+=len(pred)
                check(state_digest(head_state(reader,optimizer,context_rng,mask_rng))==original,'head optimizer/three RNG preserved '+mode+str(update));reader.assert_frozen()
            if record:
                final=record['evaluations'][-1]['panels'];gate=final['probe']['metrics']['joint_query_binding']>=.95 and final['validation']['metrics']['joint_query_binding']>=.90
                check(gate==record['matched_joint_competence_gate_passed'],'registered competence gate recount '+mode)
    if closed:
        check(set(body_endpoints)=={8192,16384} and set(head_endpoints)=={'query_attention','local_query'}
            and all(set(v)=={1024,4096,8192} for v in head_endpoints.values()),'all registered body/head endpoints present')
        check(decisions==811008,'complete individual-decision replay count')
    check(all(sha(pathlib.Path(name))==digest for name,digest in before.items()),'all original case files unchanged')
    receipt=dict(status='verified_available_readonly_DEV04_state_and_endpoints' if not issues else 'issues_found',case=args.case,
        checks=checks,issues=issues,body_endpoints=body_endpoints,head_endpoints=head_endpoints,individual_decisions_replayed=decisions,
        max_mean_nll_error=max_error,plan_sha256=PLAN,input_files_sha256=before,available_ram_gib_at_admission=ram,environment=env,elapsed_seconds=time.monotonic()-start,
        new_training_updates=0,test_scored=False,external_replication=False,full_cohort_verified=False,scientific_task3_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:receipt[k] for k in ('status','case','checks','issues','individual_decisions_replayed','max_mean_nll_error')}))
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
