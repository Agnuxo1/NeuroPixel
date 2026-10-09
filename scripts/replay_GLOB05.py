"""Independent no-update GLOB05 endpoint replay with separate projection arithmetic."""
import argparse,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))

def independent_logits(body,weights,bank,k,at,end):
    import torch
    import torch.nn.functional as F
    import math
    x=bank['canvas'][at:end];facts=bank['facts'][k,at:end];positions=bank['positions'][at:end]
    flat=torch.zeros((len(x),64,48),dtype=facts.dtype);flat.scatter_(1,positions[:,:,None].expand(-1,-1,48),facts);flat[:,63]=bank['local'][k,at:end]
    layout=torch.channels_last if bank['channels_last'] else torch.contiguous_format
    state=flat.reshape(len(x),8,8,48).permute(0,3,1,2).contiguous(memory_format=layout)
    identities=F.embedding(x,body.dictionary()).permute(0,3,1,2);features=torch.cat((state,identities),1).permute(0,2,3,1).reshape(len(x),64,64)
    query=F.linear(identities[:,:,7,6],weights['query.weight'],weights['query.bias']).reshape(len(x),4,4)
    keys=F.linear(features,weights['key.weight'],weights['key.bias']).reshape(len(x),64,4,4).transpose(1,2)
    values=F.linear(features,weights['value.weight'],weights['value.bias']).reshape(len(x),64,4,4).transpose(1,2)
    visible=(x!=0).clone();visible[:,7,6]=False;valid=visible.flatten(1)
    uniform=(valid[:,None,:].to(state.dtype)/valid.sum(1).clamp_min(1)[:,None,None]).expand(len(x),4,64)
    mean_key=(uniform[:,:,:,None]*keys).sum(2);mean_value=(uniform[:,:,:,None]*values).sum(2)
    context=(mean_value*(1+torch.tanh(query*mean_key/math.sqrt(4)))).reshape(len(x),16)
    delta=F.linear(context,weights['output.weight'],weights['output.bias'])*valid.any(1)[:,None].to(context.dtype)
    identity=body.read(state[:,:,7,7]+delta);logits=identity@body.decoder_dictionary().T;logits[:,0]=-1e4
    return logits

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--case-folder',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--plan',type=pathlib.Path,default=ROOT/'docs/research/GLOB05_execution_plan.json');p.add_argument('--expected-plan-sha256',required=True);a=p.parse_args()
    from research_GLOB05 import load_plan,load_inventory,admit_parent,sha
    plan=load_plan(a.plan,a.expected_plan_sha256);inventory=load_inventory(ROOT/plan['parent_inventory_path'],plan['parent_inventory_sha256'])
    descriptor=next(r for r in inventory['parents'] if r['case']==a.case);config=next(r for r in plan['runs'] if r['parent_case']==a.case)
    from neuropixel.research.qtrain_budget import require_ram,state_digest
    ram=require_ram()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from neuropixel.research.experiment import configure_runtime,environment_record
    from neuropixel.research.readout_experiment import extract_bank,training_state,restore
    from neuropixel.research.query_gated_uniform import FrozenGatedUniformReadout
    from neuropixel.research.glob05_experiment import validate_head
    from replay_READ03 import recount
    environment=environment_record(configure_runtime('cpu',2));start=time.monotonic();body,parent_folder,datasets,old_attention=admit_parent(descriptor,inventory,environment,ROOT/'results/research/GLOB05_parent_recovery/cohort')
    folder=a.case_folder;before={f.relative_to(folder).as_posix():sha(f) for f in folder.rglob('*') if f.is_file() and 'cache' not in f.relative_to(folder).parts};record=json.loads((folder/'result.json').read_bytes())
    issues=[];checks=0;decisions=0;maximum=0.
    def check(ok,label):
        nonlocal checks
        checks+=1
        if not bool(ok):issues.append(label)
    check(record['config']==config and record['plan_sha256']==a.expected_plan_sha256 and record['new_backbone_updates']==0 and record['test_accessed'] is False,'exact new-control recipe/scope')
    check([r['update'] for r in record['evaluations']]==[1024,4096,8192],'all original frozen endpoints')
    check([w['batch_witness_sha256'] for w in record['curve']]==[w['batch_witness_sha256'] for w in old_attention['curve']],'actual same attention batches/masks all8192updates')
    for name,expected in descriptor['dataset_sha256'].items():check(sha(folder/'datasets'/f'{name}.npz')==expected,'same actual original dataset '+name)
    banks={}
    for name,data in datasets.items():
        if name=='train':continue
        require_ram();bank=extract_bank(body,data,list(range(95000,95008)));banks[name]=bank
        base=name.replace('_query_flip','');key='query_predictions' if name.endswith('_query_flip') else 'predictions'
        with np.load(parent_folder/'body/decisions'/f'u16384_{base}_matched.npz',allow_pickle=False) as original:
            check(np.array_equal(bank['predictions'].numpy(),original[key]),'original body decisions preserved while regenerating '+name)
    for endpoint in record['evaluations']:
        update=endpoint['update'];path=folder/f'checkpoint_u{update}.pt';require_ram();payload=torch.load(path,map_location='cpu',weights_only=True)
        validate_head(payload,config,a.expected_plan_sha256,descriptor['frozen_body_digest'],environment)
        reader=FrozenGatedUniformReadout(body);optimizer=torch.optim.AdamW(reader.head_parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001);context_rng,mask_rng=torch.Generator(),torch.Generator();restore(reader,optimizer,context_rng,mask_rng,payload)
        saved=state_digest(training_state(reader,optimizer,context_rng,mask_rng))
        check(payload['effective_gradient_parameters']==3168 and sum(p.numel() for p in reader.head_parameters())==3168,'actual active/nominal capacity '+str(update))
        for panel in ('probe','validation'):
            artifact=folder/'decisions'/f'u{update}_{panel}.pt';check(sha(artifact)==endpoint['panels'][panel]['artifact_sha256'],'original endpoint prediction bytes')
            stored=torch.load(artifact,map_location='cpu',weights_only=True);counted=recount(stored)
            for key,value in counted.items():check(np.allclose(value,endpoint['panels'][panel]['metrics'][key],rtol=0,atol=1e-12),'independent scalar '+key+panel+str(update))
            for name,bank in [('original',banks[panel]),('query_flip',banks[panel+'_query_flip'])]:
                check(torch.equal(stored[name]['target'],bank['target']) and torch.equal(stored[name]['roles'],bank['roles']),'actual original labels/roles '+panel+name)
                for k in range(8):
                    pp=[];nn=[]
                    with torch.inference_mode():
                        for at in range(0,len(bank['canvas']),256):
                            logits=independent_logits(body,payload['head'],bank,k,at,min(at+256,len(bank['canvas'])));pp.append(logits.argmax(-1));nn.append(F.cross_entropy(logits,bank['target'][at:at+256],reduction='none'))
                    pred,nll=torch.cat(pp),torch.cat(nn);check(torch.equal(pred,stored[name]['predictions'][k]),'exact control decisions '+panel+name+str(update)+'/'+str(k))
                    error=abs(float(nll.double().mean())-float(stored[name]['nll'][k].double().mean()));maximum=max(maximum,error);check(error<=1e-4,'unchanged meanNLL tolerance '+panel+name+str(update)+'/'+str(k));decisions+=len(pred)
        check(state_digest(training_state(reader,optimizer,context_rng,mask_rng))==saved,'no head/optimizer/threeRNG update '+str(update));reader.assert_frozen()
    check(decisions==294912,'all fixed head endpoints/masks/views replayed')
    check(all(sha(folder/name)==h for name,h in before.items()),'all scientific control files unchanged')
    check(record['checkpoint_sha256']==sha(folder/'checkpoint.pt')==sha(folder/'checkpoint_u8192.pt'),'actual closed endpoint checkpoint')
    final=record['evaluations'][-1]['panels'];gate=final['probe']['metrics']['joint_query_binding']>=.95 and final['validation']['metrics']['joint_query_binding']>=.9
    check(gate==record['matched_joint_competence_gate_passed'],'registered gate recount')
    result=dict(status='verified_readonly_GLOB05_control_replay' if not issues else 'issues_found',case=a.case,checks=checks,issues=issues,individual_decisions_replayed=decisions,
        max_mean_nll_error=maximum,plan_sha256=a.expected_plan_sha256,input_files_sha256=before,parent_body_checkpoint_sha256=descriptor['body_checkpoint_sha256'],
        original_DEV04_precision_reinterpreted=False,new_body_initializations=0,new_training_updates=0,old_head_fits_repeated=0,original_test_scored=False,
        available_ram_gib_at_admission=ram,environment=environment,elapsed_seconds=time.monotonic()-start,external_replication=False,scientific_task3_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:result[k] for k in ('status','case','checks','issues','individual_decisions_replayed','max_mean_nll_error')}))
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
