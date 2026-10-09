"""GLOB05 head-only fitting on immutable DEV04 bodies; no old fit is replayed."""
import hashlib
import json
import math
import pathlib
import shutil
import time
import torch
from neuropixel.research.query_gated_uniform import FrozenGatedUniformReadout
from neuropixel.research.query_attention import QueryAttentionReadout
from neuropixel.research.qtrain_budget import require_ram,state_digest,verify_environment
from neuropixel.research.readout_experiment import save_json,save_state,sha,training_state,restore,step,evaluate_bank
from neuropixel.research.frozen_readout import body_digest

REG='NP-GLOB05-20261009'
MODE='gated_uniform_global'

def validate_head(payload,config,plan_hash,body_hash,environment):
    if (payload['registration_id']!=REG or payload['config']!=config or payload['plan_sha256']!=plan_hash
        or payload['frozen_body_digest']!=body_hash or payload['new_backbone_updates']!=0):raise ValueError('Foreign head/body/recipe state')
    verify_environment(payload['environment'],environment);update=payload['update']
    if not 0<=update<=config['updates'] or update%128:raise ValueError('Invalid durable head update')
    fields=('head','optimizer','cpu_rng','context_rng','mask_rng','frozen_body_digest')
    if state_digest({k:payload[k] for k in fields})!=payload['training_state_digest']:raise ValueError('Actual head/optimizer/three RNG digest differs')
    def finite(v):
        if isinstance(v,torch.Tensor):return bool(torch.isfinite(v).all())
        if isinstance(v,dict):return all(finite(x) for x in v.values())
        if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
        return not isinstance(v,float) or math.isfinite(v)
    if not finite({k:payload[k] for k in fields}):raise ValueError('Nonfinite training state')
    for key in ('cpu_rng','context_rng','mask_rng'):
        if payload[key].dtype!=torch.uint8 or payload[key].ndim!=1 or payload[key].numel()==0:raise ValueError('Invalid RNG stream')
    for group in payload['optimizer']['param_groups']:
        if (group['lr']!=.001 or tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8 or group['weight_decay']!=.0001
            or any(group[k] for k in ('amsgrad','maximize','capturable','differentiable'))):raise ValueError('Optimizer recipe differs')
    if update and (not payload['optimizer']['state'] or any(float(v['step'])!=update for v in payload['optimizer']['state'].values())):raise ValueError('Actual optimizer steps differ')
    if [r['update'] for r in payload['curve']]!=list(range(128,update+1,128)):raise ValueError('Repeated/missing closed training prefix')
    if [r['update'] for r in payload['resources']]!=list(range(1,update+1,128)) or any(r['available_ram_gib']<8 for r in payload['resources']):raise ValueError('RAM admission windows differ')
    return payload['training_state_digest']

def fit_control(body,banks,output,config,plan_hash,environment,parent,baseline_witnesses,
                endpoints=(1024,4096,8192),stop_after=None,archive_callback=None):
    output=pathlib.Path(output);final=output/'result.json';body_hash=body_digest(body)
    if config['mode']!=MODE:raise ValueError('Only registered new control can be fitted')
    fixture=config['parent_case'].startswith('fixture_')
    if not fixture and (config['updates']!=8192 or tuple(endpoints)!=(1024,4096,8192)):raise ValueError('Fixed scientific budget/endpoints required')
    if final.exists():
        prior=json.loads(final.read_bytes())
        if prior['config']!=config or prior['plan_sha256']!=plan_hash or prior['frozen_body_digest']!=body_hash or sha(output/'checkpoint.pt')!=prior['checkpoint_sha256']:raise ValueError('Preserve different closed control')
        return prior
    require_ram();torch.manual_seed(config['head_seed']);reader=FrozenGatedUniformReadout(body)
    birth=state_digest(reader.head.state_dict())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['head_seed']);original_initial=QueryAttentionReadout()
        if state_digest(original_initial.state_dict())!=birth:raise ValueError('Attention/control initial tensors differ')
    optimizer=torch.optim.AdamW(reader.head_parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    context_rng=torch.Generator().manual_seed(89004);mask_rng=torch.Generator().manual_seed(89005)
    checkpoint=output/'checkpoint.pt';curve=[];evaluations=[];resources=[];elapsed=0.;last=0
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True);validate_head(payload,config,plan_hash,body_hash,environment)
        if payload['initial_head_digest']!=birth or payload['parent']!=parent:raise ValueError('Original initialization/parent identity differs')
        restore(reader,optimizer,context_rng,mask_rng,payload)
        curve,evaluations,resources=payload['curve'],payload['evaluations'],payload['resources'];last=payload['update'];elapsed=payload['elapsed_update_seconds']
    def persist(update):
        state=training_state(reader,optimizer,context_rng,mask_rng)
        payload=dict(**state,registration_id=REG,config=config,plan_sha256=plan_hash,environment=environment,
            parent=parent,initial_head_digest=birth,update=update,curve=curve,evaluations=evaluations,resources=resources,
            elapsed_update_seconds=elapsed,effective_gradient_parameters=3168 if update else 0,new_backbone_updates=0,
            training_state_digest=state_digest(state))
        save_state(checkpoint,payload);save_json(output/'progress.json',dict(registration_id=REG,config=config,plan_sha256=plan_hash,
            update=update,checkpoint_sha256=sha(checkpoint),frozen_body_digest=body_hash,initial_head_digest=birth,parent=parent,
            new_backbone_updates=0,test_accessed=False))
    def endpoint(update):
        panels={}
        for panel in ('probe','validation'):
            result=evaluate_bank(reader,banks[panel],banks[panel+'_query_flip']);path=output/'decisions'/f'u{update}_{panel}.pt'
            save_state(path,result['arrays']);panels[panel]=dict(metrics=result['metrics'],artifact_sha256=sha(path))
        evaluations.append(dict(update=update,panels=panels,test_accessed=False));persist(update)
        snapshot=output/f'checkpoint_u{update}.pt'
        if snapshot.exists() and snapshot.read_bytes()!=checkpoint.read_bytes():raise ValueError('Different closed evaluated control checkpoint')
        if not snapshot.exists():shutil.copyfile(checkpoint,snapshot)
    if not checkpoint.exists():persist(0)
    if last in endpoints and last not in [e['update'] for e in evaluations]:endpoint(last)
    losses=[];witness=hashlib.sha256()
    for update in range(last+1,config['updates']+1):
        if update%128==1:resources.append(dict(update=update,available_ram_gib=require_ram()))
        started=time.monotonic();loss,gradient,value=step(reader,optimizer,banks['train'],context_rng,mask_rng)
        elapsed+=time.monotonic()-started;losses.append(loss);witness.update(value.encode())
        if sum(p.numel() for p in reader.head_parameters() if p.grad is not None)!=3168:raise ValueError('Inactive nominal projection parameters')
        if update%128==0:
            digest=witness.hexdigest();index=update//128-1
            if index>=len(baseline_witnesses) or digest!=baseline_witnesses[index]:raise ValueError('Original attention batch/mask stream differs')
            curve.append(dict(update=update,answer_loss_mean=sum(losses)/len(losses),last_unclipped_gradient_norm=gradient,
                batch_witness_sha256=digest,elapsed_update_seconds=elapsed));losses=[];witness=hashlib.sha256();persist(update)
            if update in endpoints:endpoint(update)
            if archive_callback and update%1024==0 and update<config['updates']:
                before=state_digest(training_state(reader,optimizer,context_rng,mask_rng));archive_callback()
                if state_digest(training_state(reader,optimizer,context_rng,mask_rng))!=before:raise ValueError('Archival changed real training state')
            print(json.dumps(dict(event='glob05_head_progress',case=config['parent_case'],update=update)),flush=True)
            if stop_after and update>=stop_after:return None
    reader.assert_frozen();last_panels=evaluations[-1]['panels']
    result=dict(registration_id=REG,status='completed',config=config,plan_sha256=plan_hash,parent=parent,initial_head_digest=birth,
        frozen_body_digest=body_hash,head_nominal_parameters=3168,effective_gradient_parameters=3168,new_backbone_updates=0,
        checkpoint_sha256=sha(checkpoint),curve=curve,evaluations=evaluations,resources=resources,environment=environment,
        matched_joint_competence_gate_passed=last_panels['probe']['metrics']['joint_query_binding']>=.95 and last_panels['validation']['metrics']['joint_query_binding']>=.90,
        original_DEV04_precision_reinterpreted=False,test_accessed=False)
    save_json(final,result)
    if archive_callback:archive_callback()
    return result
