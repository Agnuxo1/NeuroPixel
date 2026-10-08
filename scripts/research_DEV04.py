"""Registered fresh-body development study, guarded against old weights/test."""
import argparse
import hashlib
import json
import os
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for name,value in [('OMP_NUM_THREADS','2'),('MKL_NUM_THREADS','2'),('CUBLAS_WORKSPACE_CONFIG',':4096:8')]:os.environ.setdefault(name,value)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_inventory():
    return [dict(run_id=f'DEV04_partition{split}_seed{400+4*j+k}',partition_seed=split,init_seed=400+4*j+k,
                 policy='paired',school_weight=.3,updates=16384) for j,split in enumerate((101,102,103)) for k in range(4)]


def load_plan(path,expected):
    if sha(path)!=expected:raise ValueError('Independent DEV04 plan hash differs')
    plan=json.loads(path.read_text(encoding='utf-8'))
    if (plan['registration_id']!='NP-DEV04-20261008' or plan['status']!='frozen_before_scientific_training'
        or plan['runs']!=expected_inventory() or plan['body_fits']!=12 or plan['head_fits']!=24
        or plan['modes']!=['query_attention','local_query'] or plan['original_test_fixed_excluded'] is not True):raise ValueError('Unfrozen/different full-cohort DEV04 recipe')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Governing file changed '+name)
    return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=pathlib.Path,default=ROOT/'docs/research/DEV04_execution_plan.json')
    parser.add_argument('--expected-plan-sha256');parser.add_argument('--describe',action='store_true');parser.add_argument('--case')
    parser.add_argument('--recover-cloud',action='store_true');parser.add_argument('--output',type=pathlib.Path,default=ROOT/'results/research/DEV04')
    args=parser.parse_args()
    if args.describe:
        print(json.dumps(dict(body_fits=12,head_fits=24,partition_seeds=[101,102,103],init_seeds=list(range(400,412)),historical_checkpoints_loaded=0,test_scored=False)));return
    if not args.expected_plan_sha256:raise ValueError('Freeze independently hashed plan before scientific training')
    plan=load_plan(args.plan,args.expected_plan_sha256)
    from neuropixel.research.qtrain_budget import require_ram
    require_ram()
    import numpy as np
    import torch
    from neuropixel.research import dev04 as study
    from neuropixel.research import dev04_transport as archive
    from neuropixel.research.qtrain_budget import state_digest,validate_state,restore,verify_environment
    from neuropixel.research.experiment import configure_runtime,environment_record
    from neuropixel.research.OPT03_role_views import transform
    from neuropixel.research.readout_experiment import paired_training_dataset,extract_bank,fit_head,save_state,save_json
    environment=environment_record(configure_runtime('cpu',2));publish=os.environ.get('DEV04_PUBLISH')=='1'
    selected=plan['runs']
    if args.case:
        selected=[r for r in selected if r['run_id']==args.case]
        if len(selected)!=1:raise ValueError('Unknown execution shard; selection is not a score filter')
    if publish and len(selected)!=1:raise ValueError('Cloud execution requires one independent evidence branch per case shard')
    if publish:archive.select_case_archive_branch(selected[0]['run_id'])
    if args.recover_cloud:archive.recover(args.output,{r['run_id'] for r in selected},args.expected_plan_sha256)
    for conf in selected:
        case=conf['run_id'];folder=args.output/case;complete=folder/'result.json'
        if complete.exists():
            old=json.loads(complete.read_text(encoding='utf-8'))
            if old['config']!=conf or old['plan_sha256']!=args.expected_plan_sha256:raise ValueError('Closed case identity differs')
            continue
        require_ram();task=study.DevelopmentRoleTask(conf['partition_seed']);partition=task.partition_record()
        if partition!=plan['partitions'][str(conf['partition_seed'])]:raise ValueError('Prospective partition hashes/counts differ')
        save_json(folder/'partition.json',partition);data=study.panels(task)
        for panel,arrays in data.items():
            path=folder/'datasets'/(panel+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                with np.load(path,allow_pickle=False) as saved:
                    if any(not np.array_equal(saved[name],value.numpy()) for name,value in zip(('canvas','target','roles'),arrays)):raise ValueError('Preserved panel differs')
            else:np.savez_compressed(path,**{name:value.numpy() for name,value in zip(('canvas','target','roles'),arrays)})
        callback=(lambda:archive.publish_case(args.output,case,args.plan)) if publish else None
        body_record=study.fit_body(conf,task,data,folder/'body',args.expected_plan_sha256,environment,archive_callback=callback,
            expected_birth=plan['initial_body_model_digests'][case])
        require_ram();checkpoint=folder/'body/checkpoint.pt'
        if sha(checkpoint)!=body_record['checkpoint_sha256']:raise ValueError('Fresh completed body changed')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model,opt,sampler,query_rng=study.new_objects(conf)
        birth=state_digest(model.state_dict());study.check_resume(payload,conf,args.expected_plan_sha256,birth,environment)
        actual=validate_state(payload,conf['updates'],conf,args.expected_plan_sha256);restore(model,opt,sampler,query_rng,payload,actual)
        for parameter in model.parameters():parameter.requires_grad_(False);parameter.grad=None
        datasets=dict(train=paired_training_dataset(task),**data)
        for panel,arrays in data.items():datasets[panel+'_query_flip']=tuple(torch.from_numpy(v) for v in transform(*(x.numpy() for x in arrays),'query_flip'))
        banks={}
        for panel,arrays in datasets.items():
            seeds=study.TRAIN_MASKS if panel=='train' else study.EVAL_MASKS
            identity=dict(plan_sha256=args.expected_plan_sha256,body_checkpoint_sha256=sha(checkpoint),dataset_digest=state_digest(arrays),mask_seeds=seeds)
            path=folder/'cache'/(panel+'.pt')
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if saved['identity']!=identity or saved['bank_digest']!=state_digest(saved['bank']):raise ValueError('Cached state identity differs')
                bank=saved['bank']
            else:
                bank=extract_bank(model,arrays,seeds)
                if panel!='train':
                    base=panel.replace('_query_flip','');reference=folder/'body/decisions'/f'u{conf["updates"]}_{base}_matched.npz'
                    field='query_predictions' if panel.endswith('_query_flip') else 'predictions'
                    with np.load(reference,allow_pickle=False) as prior:
                        if not np.array_equal(bank['predictions'].numpy(),prior[field]):raise ValueError('State extraction changed original body decisions')
                save_state(path,dict(identity=identity,bank=bank,bank_digest=state_digest(bank)))
            banks[panel]=bank
        heads={}
        for mode in plan['modes']:
            parent=dict(parent_case=case,init_seed=conf['init_seed'])
            heads[mode]=fit_head(model,mode,banks,folder/mode,args.expected_plan_sha256,parent,archive_callback=callback)
        result=dict(registration_id=study.REGISTRATION,status='completed',config=conf,plan_sha256=args.expected_plan_sha256,
            body_result_sha256=sha(folder/'body/result.json'),head_results_sha256={m:sha(folder/m/'result.json') for m in plan['modes']},
            body_checkpoint_sha256=sha(checkpoint),partition=partition,historical_checkpoint_loaded=False,new_body_initialization=True,test_scored=False)
        save_json(complete,result)
        if callback:callback()
        del banks,model,opt,payload
    print(json.dumps(dict(completed_execution_shards=len(selected),expected_bodies=12,expected_head_fits=24,test_scored=False,scientific_task3_complete=False)))


if __name__=='__main__':main()
