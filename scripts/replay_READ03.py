"""Independent read-only endpoint/array audit of frozen READ03 heads.

No optimizer step, fitting, selection, final-test access or cohort effect estimate.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN_SHA = 'd273b8284d9a387b49266dc4ec62cfad76aefaa11c5f82c7ab6a97ba0e431ea0'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recount(arrays):
    """Recompute scalar metrics without the experiment's evaluation function."""
    import numpy as np
    a, b = arrays['original'], arrays['query_flip']
    p, q = np.asarray(a['predictions']), np.asarray(b['predictions'])
    y, z, roles = np.asarray(a['target']), np.asarray(b['target']), np.asarray(a['roles'])
    if p.shape != (8, len(y)) or q.shape != p.shape or y.shape != roles.shape:
        raise ValueError('Eight complete individual mask banks required')
    noun = (roles == 0) | (roles == 2)
    expected_roles = np.where(noun,2-roles,roles)
    if not np.array_equal(expected_roles, np.asarray(b['roles'])) or set(roles.tolist()) != {0,1,2,3}:
        raise ValueError('Counterfactual roles differ')
    if np.any(p < 1) or np.any(p >= 35) or np.any(q < 1) or np.any(q >= 35):
        raise ValueError('Prediction outside non-PAD vocabulary')
    good, cf_good = p == y[None], q == z[None]
    per_role = [float(good[:, roles == r].mean(dtype=np.float64)) for r in range(4)]
    nll = np.asarray(a['nll'])
    if nll.shape != p.shape or not np.isfinite(nll).all() or (nll < 0).any():
        raise ValueError('Invalid finite nonnegative NLL')
    if not np.array_equal(p[:,~noun], q[:,~noun]) or not np.array_equal(y[~noun], z[~noun]):
        raise ValueError('Unchanged query-control decisions/targets differ')
    return dict(accuracy=float(good.mean(dtype=np.float64)),binding=sum(per_role[::2])/2,
                joint_query_binding=float((good & cf_good)[:,noun].mean(dtype=np.float64)),
                per_role=per_role,mean_nll=float(nll.mean(dtype=np.float64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit-id', required=True)
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    a = ap.parse_args()
    from neuropixel.research.qtrain_budget import require_ram
    require_ram()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from neuropixel.research.models import ResearchNCA
    from neuropixel.research.data import ResearchRoleTask, frozen_dataset
    from neuropixel.research.experiment import configure_runtime, environment_record
    from neuropixel.research.OPT03_role_views import transform
    from neuropixel.research.qtrain_budget import validate_state, verify_environment, state_digest
    from neuropixel.research.frozen_readout import FrozenReadout, body_digest
    from neuropixel.research.readout_experiment import extract_bank, save_state, restore, training_state
    from scripts.research_READ03 import load_plan
    require_ram()
    plan, inventory = load_plan(ROOT/'docs/research/READ03_execution_plan.json', PLAN_SHA)
    conf = next(row for row in plan['runs'] if row['run_id'] == a.fit_id)
    parent = next(row for row in inventory['parents'] if row['parent_case'] == conf['parent_case'])
    folder = a.root/conf['parent_case']/conf['mode']
    record = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    expected_config = {k:conf[k] for k in ('parent_case','mode','head_seed','updates')}
    issues, checks, decisions, max_error = [], 0, 0, 0.
    start = time.monotonic()
    def check(ok, label):
        nonlocal checks
        checks += 1
        if not bool(ok): issues.append(label)
    before = {str(p):sha(p) for p in folder.rglob('*') if p.is_file()}
    environment = environment_record(configure_runtime('cpu',2))
    verify_environment(parent['environment'], environment)
    cp_path = ROOT/parent['checkpoint_path']
    if sha(cp_path) != parent['checkpoint_sha256']:
        raise ValueError('Frozen parent bytes differ')
    cp = torch.load(cp_path, map_location='cpu', weights_only=True)
    validate_state(cp,16384,expected_plan=inventory['parent_plan_sha256'])
    body = ResearchNCA(35,(7,7),c_id=16,c=48,hidden=128,steps=16,fire_rate=.5,
                       tied=True,reinject=True,freeze_pad=False)
    body.load_state_dict(cp['model'])
    for parameter in body.parameters(): parameter.requires_grad_(False)
    digest = body_digest(body)
    check(record['config'] == expected_config and record['plan_sha256'] == PLAN_SHA,'final recipe identity')
    check(record['status']=='completed' and record['new_backbone_updates']==0 and record['test_accessed'] is False,'final scope')
    check(record['frozen_body_digest']==digest,'actual parent/body/decoder digest')
    check(record['head_nominal_parameters']==3168,'head nominal capacity')
    effective = 1856 if conf['mode']=='uniform_global' else 3168
    check(record['effective_gradient_parameters']==effective,'effective gradient capacity')
    check([e['update'] for e in record['evaluations']]==[1024,4096,8192],'complete fixed endpoints')
    check([w['update'] for w in record['curve']]==list(range(128,8193,128)),'all loss windows')
    check(all(np.isfinite(w['answer_loss_mean']) and w['answer_loss_mean']>=0 for w in record['curve']),'finite loss windows')
    check(len(record['resources'])==64 and all(r['available_ram_gib']>=8 for r in record['resources']),'RAM admission at every window')
    check(sha(folder/'checkpoint.pt')==record['checkpoint_sha256'],'final actual checkpoint hash')
    check((folder/'checkpoint.pt').read_bytes()==(folder/'checkpoint_u8192.pt').read_bytes(),'final equals evaluated checkpoint')
    # Batch/mask witnesses must agree across all head variants, independently of losses.
    for mode in ('query_attention','uniform_global','local_query'):
        peer = a.root/conf['parent_case']/mode/'result.json'
        if peer.exists():
            rr=json.loads(peer.read_text(encoding='utf-8'))
            check([w['batch_witness_sha256'] for w in rr['curve']]==[w['batch_witness_sha256'] for w in record['curve']],'common batch/mask stream '+mode)
    task = ResearchRoleTask(seed=0)
    banks, datasets = {}, {}
    old_record=json.loads((cp_path.parent/'result.json').read_text(encoding='utf-8'))
    if sha(cp_path.parent/'result.json')!=parent['result_sha256']:raise ValueError('Parent record changed')
    for panel, split, n, seed in [('probe','train',2048,77001),('validation','validation',4096,77002)]:
        datasets[panel]=frozen_dataset(task,split,n,seed)
        datasets[panel+'_query_flip']=tuple(torch.from_numpy(v) for v in transform(*(t.numpy() for t in datasets[panel]),'query_flip'))
        for view in (panel,panel+'_query_flip'):
            require_ram()
            data=datasets[view]
            identity=dict(parent_checkpoint_sha256=parent['checkpoint_sha256'],dataset_digest=state_digest(data),mask_seeds=list(range(79000,79008)),plan_sha256=PLAN_SHA)
            cache=a.root.parent/'readonly_cache'/conf['parent_case']/(view+'.pt')
            if cache.exists():
                saved=torch.load(cache,map_location='cpu',weights_only=True)
                if saved['identity']!=identity or saved['bank_digest']!=state_digest(saved['bank']):raise ValueError('Read-only latent cache differs')
                bank=saved['bank']
            else:
                bank=extract_bank(body,data,identity['mask_seeds'])
                save_state(cache,dict(identity=identity,bank=bank,bank_digest=state_digest(bank)))
            check(bank['frozen_body_digest']==digest,'bank actual frozen body '+view)
            ref=cp_path.parent/'decisions'/f'u16384_{panel}_matched.npz'
            artifact=old_record['evaluations'][-1]['panels'][panel]['decision_artifacts'][ref.name]
            check(sha(ref)==artifact['sha256'],'closed parent prediction hash '+view)
            with np.load(ref,allow_pickle=False) as raw:
                field='query_predictions' if view.endswith('_query_flip') else 'predictions'
                check(np.array_equal(bank['predictions'].numpy(),raw[field]),'regenerated parent decisions exact '+view)
            banks[view]=bank
    for endpoint in record['evaluations']:
        update=endpoint['update'];path=folder/f'checkpoint_u{update}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=True)
        check(payload['config']==expected_config and payload['plan_sha256']==PLAN_SHA and payload['update']==update,'endpoint identity '+str(update))
        check(payload['frozen_body_digest']==digest,'endpoint original body digest '+str(update))
        check(payload['effective_gradient_parameters']==effective,'endpoint effective capacity '+str(update))
        check(all(torch.isfinite(v).all() for v in payload['head'].values()),'finite head '+str(update))
        for name in ('cpu_rng','context_rng','mask_rng'):
            check(payload[name].dtype==torch.uint8 and payload[name].numel()>0,'actual RNG '+name+str(update))
        reader=FrozenReadout(body,conf['mode'])
        optimizer=torch.optim.AdamW(reader.head_parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
        context_rng,mask_rng=torch.Generator(),torch.Generator()
        restore(reader,optimizer,context_rng,mask_rng,payload)
        restored=state_digest(training_state(reader,optimizer,context_rng,mask_rng))
        for group in payload['optimizer']['param_groups']:
            check(group['lr']==.001 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.0001,'optimizer recipe '+str(update))
        check(len(payload['optimizer']['state'])==len(reader.head_parameters())-(4 if conf['mode']=='uniform_global' else 0),'active optimizer parameter families '+str(update))
        for value in payload['optimizer']['state'].values():
            check(float(value['step'])==update,'actual optimizer step '+str(update))
            check(all(torch.isfinite(v).all() for v in value.values() if isinstance(v,torch.Tensor)),'finite optimizer '+str(update))
        for panel in ('probe','validation'):
            artifact=folder/'decisions'/f'u{update}_{panel}.pt'
            check(sha(artifact)==endpoint['panels'][panel]['artifact_sha256'],'actual endpoint array hash '+panel+str(update))
            stored=torch.load(artifact,map_location='cpu',weights_only=True)
            counted=recount(stored)
            for key,value in counted.items():
                check(np.max(np.abs(np.asarray(value)-np.asarray(endpoint['panels'][panel]['metrics'][key])))<=1e-12,'independent metric '+key+panel+str(update))
            for view_name, view in [('original',panel),('query_flip',panel+'_query_flip')]:
                bank=banks[view]
                check(torch.equal(stored[view_name]['target'],datasets[view][1]) and torch.equal(stored[view_name]['roles'],datasets[view][2]),'actual dataset labels '+view+str(update))
                for k in range(8):
                    pp,nn=[],[]
                    with torch.inference_mode():
                        for at in range(0,len(bank['canvas']),256):
                            # Independent assembly and head invocation; no fitting/evaluate_bank call.
                            x=bank['canvas'][at:at+256];features=bank['facts'][k,at:at+256];local=bank['local'][k,at:at+256]
                            flat=torch.zeros((len(x),64,48),dtype=features.dtype)
                            pos=bank['positions'][at:at+256]
                            flat.scatter_(1,pos[:,:,None].expand(-1,-1,48),features);flat[:,63]=local
                            layout=torch.channels_last if bank['channels_last'] else torch.contiguous_format
                            state=flat.reshape(len(x),8,8,48).permute(0,3,1,2).contiguous(memory_format=layout)
                            ids=F.embedding(x,body.dictionary()).permute(0,3,1,2);local=state[:,:,7,7]
                            if conf['mode']=='local_query':
                                out=reader.head(local,ids[:,:,7,6]);identity=body.read(local+out['delta'])*out['identity_gate']
                            else:
                                out=reader.head(state,ids,x,(7,6));identity=body.read(local+out['delta'])
                            logits=identity@body.decoder_dictionary().T;logits[:,0]=-1e4
                            pp.append(logits.argmax(-1));nn.append(F.cross_entropy(logits,bank['target'][at:at+256],reduction='none'))
                    pred,nll=torch.cat(pp),torch.cat(nn)
                    check(torch.equal(pred,stored[view_name]['predictions'][k]),'exact head decisions '+view+str(update)+'/'+str(k))
                    error=abs(float(nll.double().mean())-float(stored[view_name]['nll'][k].double().mean()))
                    max_error=max(max_error,error);check(error<=1e-4,'original mean NLL tolerance '+view+str(update)+'/'+str(k))
                    decisions+=len(pred)
        check(state_digest(training_state(reader,optimizer,context_rng,mask_rng))==restored,'replay preserves head/optimizer/threeRNG '+str(update))
        reader.assert_frozen()
        print(json.dumps(dict(fit_replay=a.fit_id,endpoint=update)),flush=True)
    last=record['evaluations'][-1]['panels']
    gate=last['probe']['metrics']['joint_query_binding']>=.95 and last['validation']['metrics']['joint_query_binding']>=.90
    check(gate==record['matched_joint_competence_gate_passed'],'registered final gate recount')
    check(all(sha(pathlib.Path(p))==h for p,h in before.items()),'all scientific fit files unchanged')
    check(sha(cp_path)==parent['checkpoint_sha256'] and body_digest(body)==digest,'original parent unchanged')
    result=dict(status='verified_readonly_fit_replay' if not issues else 'issues_found',cases=[a.fit_id],checks=checks,issues=issues,
        individual_decisions_replayed=decisions,max_mean_nll_error=max_error,plan_sha256=PLAN_SHA,
        environment=environment,elapsed_seconds=time.monotonic()-start,new_training_updates=0,new_backbone_updates=0,
        test_accessed=False,independent_external_replication=False,scientific_task3_complete=False,
        scope='Within-project fixed-head endpoint replay and independent metric recount; no new scientific replication')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:result[k] for k in ('status','checks','issues','individual_decisions_replayed','max_mean_nll_error')}))
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
