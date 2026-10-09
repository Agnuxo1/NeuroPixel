"""Frozen-parent admission and new active-global heads; never refits DEV04 heads."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k,v in [('OMP_NUM_THREADS','2'),('MKL_NUM_THREADS','2'),('CUBLAS_WORKSPACE_CONFIG',':4096:8')]:os.environ.setdefault(k,v)
REG='NP-GLOB05-20261009'
PARENT_PLAN='2484f3c5917e474ca3755bcdbbaca7d322077615ba47d51653116042d980f20a'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def expected_cases():return [f'DEV04_partition{s}_seed{400+4*j+k}' for j,s in enumerate((101,102,103)) for k in range(4)]

def load_inventory(path,expected):
    if sha(path)!=expected:raise ValueError('Independently fixed parent inventory hash differs')
    inventory=json.loads(path.read_bytes())
    if (inventory['registration_id']!=REG or inventory['parent_plan_sha256']!=PARENT_PLAN or inventory['parent_count']!=12
        or [r['case'] for r in inventory['parents']]!=expected_cases() or inventory['new_body_initializations']!=0
        or inventory['original_DEV04_precision_target']!='failed_preserved'):raise ValueError('Different/full-cohort parent inventory required')
    for r in inventory['parents']:
        if r['config']['updates']!=16384 or r['new_body_training_updates']!=0 or r['old_head_trainings_repeated']!=0:raise ValueError('No fresh-body or old-head fitting permitted')
    return inventory

def load_plan(path,expected):
    if sha(path)!=expected:raise ValueError('Independently fixed GLOB05 plan hash differs')
    plan=json.loads(path.read_bytes())
    configs=[dict(parent_case=c,mode='gated_uniform_global',head_seed=700+i,updates=8192) for i,c in enumerate(expected_cases())]
    if (plan['registration_id']!=REG or plan['status']!='frozen_before_scientific_training' or plan['runs']!=configs
        or plan['new_body_initializations']!=0 or plan['new_head_fits']!=12 or plan['old_head_fits_repeated']!=0
        or plan['original_test_scored'] is not False):raise ValueError('Frozen full fixed-cohort plan required')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Scientific governing file changed '+name)
    load_inventory(ROOT/plan['parent_inventory_path'],plan['parent_inventory_sha256'])
    return plan

def admit_parent(descriptor,inventory,environment,parent_root):
    from neuropixel.research import glob05_transport as archive,dev04 as development
    from neuropixel.research.qtrain_budget import require_ram,state_digest,validate_state,restore
    from neuropixel.research.frozen_readout import body_digest
    from neuropixel.research.readout_experiment import paired_training_dataset
    from neuropixel.research.OPT03_role_views import transform
    import torch
    import numpy as np
    require_ram();folder=archive.recover_parent(descriptor,parent_root);config=descriptor['config']
    proof_raw=archive.old.fetch('https://raw.githubusercontent.com/Agnuxo1/NeuroPixel/'+inventory['parent_verification_git_commit']+'/'+inventory['parent_replay_prefix']+descriptor['case']+'/replay.json')
    if hashlib.sha256(proof_raw).hexdigest()!=descriptor['neural_replay_proof_sha256']:raise ValueError('Previously completed neural parent proof differs')
    proof=json.loads(proof_raw)
    if (proof['case']!=descriptor['case'] or proof['status']!='verified_available_readonly_DEV04_state_and_endpoints'
        or proof['issues'] or proof['individual_decisions_replayed']!=811008 or proof['verification_returncode']!=0
        or proof['max_mean_nll_error']>1e-4 or proof['plan_sha256']!=PARENT_PLAN):raise ValueError('Verified closed parent required')
    task=development.DevelopmentRoleTask(config['partition_seed']);datasets=dict(train=paired_training_dataset(task),**development.panels(task))
    for panel,arrays in datasets.items():
        native='head_train' if panel=='train' else panel
        with np.load(folder/'datasets'/f'{native}.npz',allow_pickle=False) as stored:
            if any(not np.array_equal(stored[k],v.numpy()) for k,v in zip(('canvas','target','roles'),arrays)):raise ValueError('Parent dataset regeneration differs')
    model,opt,sampler,query_rng=development.new_objects(config);birth=state_digest(model.state_dict())
    parent_plan=json.loads((ROOT/'docs/research/DEV04_execution_plan.json').read_bytes())
    if birth!=parent_plan['initial_body_model_digests'][descriptor['case']]:raise ValueError('Original birth digest differs')
    payload=torch.load(folder/'body/checkpoint.pt',map_location='cpu',weights_only=True)
    development.check_resume(payload,config,PARENT_PLAN,birth,environment);digest=validate_state(payload,16384,config,PARENT_PLAN)
    if payload['partition']!=task.partition_record():raise ValueError('Original fixed partition differs')
    restore(model,opt,sampler,query_rng,payload,digest)
    if body_digest(model)!=descriptor['frozen_body_digest']:raise ValueError('Exact trained body/decoder digest differs')
    for parameter in model.parameters():parameter.requires_grad_(False);parameter.grad=None
    for panel in ('probe','validation'):
        datasets[panel+'_query_flip']=tuple(torch.from_numpy(v) for v in transform(*(a.numpy() for a in datasets[panel]),'query_flip'))
    attention=json.loads((folder/'query_attention/result.json').read_bytes());local=json.loads((folder/'local_query/result.json').read_bytes())
    if ([w['batch_witness_sha256'] for w in attention['curve']]!=[w['batch_witness_sha256'] for w in local['curve']]
        or attention['new_backbone_updates']!=0 or attention['config']['updates']!=8192):raise ValueError('Original closed paired-head provenance differs')
    return model,folder,datasets,attention

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=pathlib.Path,default=ROOT/'docs/research/GLOB05_execution_plan.json');p.add_argument('--expected-plan-sha256')
    p.add_argument('--preflight',action='store_true');p.add_argument('--parent-inventory',type=pathlib.Path,default=ROOT/'docs/research/GLOB05_parent_inventory.json');p.add_argument('--expected-parent-inventory-sha256')
    p.add_argument('--case');p.add_argument('--recover-cloud',action='store_true');p.add_argument('--output',type=pathlib.Path,default=ROOT/'results/research/GLOB05');a=p.parse_args()
    if a.preflight:
        if not a.expected_parent_inventory_sha256:raise ValueError('Preflight requires independently fixed inventory hash')
        inventory=load_inventory(a.parent_inventory,a.expected_parent_inventory_sha256);plan=None
        from scripts.research_DEV04 import load_plan as old_plan
        old_plan(ROOT/'docs/research/DEV04_execution_plan.json',PARENT_PLAN)
    else:
        if not a.expected_plan_sha256:raise ValueError('Frozen plan required before any scientific fitting')
        plan=load_plan(a.plan,a.expected_plan_sha256);inventory=load_inventory(ROOT/plan['parent_inventory_path'],plan['parent_inventory_sha256'])
    from neuropixel.research.qtrain_budget import require_ram
    require_ram()
    import torch
    from neuropixel.research.experiment import configure_runtime,environment_record
    from neuropixel.research.readout_experiment import extract_bank,save_state,save_json
    from neuropixel.research.qtrain_budget import state_digest
    from neuropixel.research.glob05_experiment import fit_control
    from neuropixel.research import glob05_transport as archive
    environment=environment_record(configure_runtime('cpu',2));publish=os.environ.get('GLOB05_PUBLISH')=='1'
    descriptors=inventory['parents']
    if a.case:
        descriptors=[r for r in descriptors if r['case']==a.case]
        if len(descriptors)!=1:raise ValueError('Unknown independent execution shard')
    if publish:
        if a.preflight or len(descriptors)!=1:raise ValueError('Only one fixed case per evidence shard')
        archive.select_case_archive_branch(descriptors[0]['case'])
        if a.recover_cloud:archive.recover(a.output,descriptors[0]['case'],a.expected_plan_sha256)
    parent_root=ROOT/'results/research/GLOB05_parent_recovery/cohort';admitted=[]
    for descriptor in descriptors:
        case=descriptor['case'];folder=a.output/case
        if not a.preflight and (folder/'result.json').exists():
            old=json.loads((folder/'result.json').read_bytes());config=next(r for r in plan['runs'] if r['parent_case']==case)
            if old['config']!=config or old['plan_sha256']!=a.expected_plan_sha256:raise ValueError('Preserve different closed head')
            continue
        body,parent_folder,datasets,attention=admit_parent(descriptor,inventory,environment,parent_root)
        admitted.append(dict(case=case,body_checkpoint_sha256=descriptor['body_checkpoint_sha256'],frozen_body_digest=descriptor['frozen_body_digest'],datasets_regenerated_exact=True,new_body_updates=0,old_head_trainings_repeated=0))
        if a.preflight:
            del body,datasets;continue
        parent=dict(parent_case=case,original_zip_sha256=descriptor['original_zip_sha256'],body_checkpoint_sha256=descriptor['body_checkpoint_sha256'],
            frozen_body_digest=descriptor['frozen_body_digest'],parent_inventory_sha256=plan['parent_inventory_sha256'])
        save_json(folder/'parent_admission.json',dict(parent=parent,source_descriptor=descriptor,environment=environment))
        for panel in ('head_train','probe','validation'):
            raw=(parent_folder/'datasets'/f'{panel}.npz').read_bytes();archive.old.immutable(folder/'datasets'/f'{panel}.npz',raw)
        banks={}
        for panel,data in datasets.items():
            require_ram();seeds=list(range(94000,94008)) if panel=='train' else list(range(95000,95008))
            identity=dict(parent=parent,dataset_digest=state_digest(data),mask_seeds=seeds,plan_sha256=a.expected_plan_sha256)
            cache=folder/'cache'/(panel+'.pt')
            if cache.exists():
                stored=torch.load(cache,map_location='cpu',weights_only=True)
                if stored['identity']!=identity or stored['bank_digest']!=state_digest(stored['bank']):raise ValueError('Closed latent cache differs')
                bank=stored['bank']
            else:
                bank=extract_bank(body,data,seeds)
                if panel!='train':
                    import numpy as np
                    reference=parent_folder/'body/decisions'/f'u16384_{panel.replace("_query_flip","")}_matched.npz'
                    with np.load(reference,allow_pickle=False) as original:
                        key='query_predictions' if panel.endswith('_query_flip') else 'predictions'
                        if not np.array_equal(bank['predictions'].numpy(),original[key]):raise ValueError('Latent extraction changes original body decisions')
                save_state(cache,dict(identity=identity,bank=bank,bank_digest=state_digest(bank)))
            banks[panel]=bank
        config=next(r for r in plan['runs'] if r['parent_case']==case);callback=(lambda:archive.publish_case(a.output,case,a.plan)) if publish else None
        fit_control(body,banks,folder,config,a.expected_plan_sha256,environment,parent,[w['batch_witness_sha256'] for w in attention['curve']],archive_callback=callback)
        del body,banks,datasets
    if a.preflight:
        receipt=dict(status='verified_available_exact_frozen_parents' if len(admitted)==len(descriptors) else 'incomplete',parents=admitted,environment=environment,
            parent_inventory_sha256=a.expected_parent_inventory_sha256,new_scientific_body_fits=0,new_scientific_head_fits=0,original_test_scored=False)
        save_json(ROOT/'results/research/GLOB05_preflight/parent_admission_receipt.json',receipt)
        print(json.dumps(dict(status=receipt['status'],admitted_parents=len(admitted),scientific_fits=0)))
    else:print(json.dumps(dict(completed_shards=len(descriptors),expected_new_heads=12,new_body_fits=0,old_HEAD_trainings_repeated=0,test_accessed=False)))

if __name__=='__main__':main()
