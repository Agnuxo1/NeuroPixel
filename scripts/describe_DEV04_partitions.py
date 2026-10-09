"""Pinned development membership/hash inventory; no test sampling or fitting."""
import json
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from neuropixel.research.qtrain_budget import require_ram
    require_ram()
    from neuropixel.research.dev04 import DevelopmentRoleTask,inventory,new_objects
    from neuropixel.research.qtrain_budget import state_digest
    tasks={seed:DevelopmentRoleTask(seed) for seed in (101,102,103)}
    partitions={str(seed):task.partition_record() for seed,task in tasks.items()}
    if len({p['composition_sha256']['test'] for p in partitions.values()})!=1:raise ValueError('Original test changed across policies')
    overlaps={}
    for first,a in tasks.items():
        for second,b in tasks.items():
            overlaps[f'{first}_{second}']={f'{x}_{y}':len(set(getattr(a,x+'_triples'))&set(getattr(b,y+'_triples'))) for x in ('train','validation') for y in ('train','validation')}
    births={}
    for conf in inventory():
        require_ram();model,opt,_,_=new_objects(conf)
        if opt.state_dict()['state'] or sum(p.numel() for p in model.parameters())!=29824:raise ValueError('Fresh object admission differs')
        births[conf['run_id']]=state_digest(model.state_dict())
    if len(set(births.values()))!=12:raise ValueError('Distinct new initialization states required')
    receipt=dict(status='verified_twelve_initialization_and_three_development_partition_inventory',partitions=partitions,
        overlaps=overlaps,initial_body_model_digests=births,body_fits=12,head_fits=24,scientific_training_updates=0,test_sampled=False,
        scope='Fresh constructor and partition metadata inventory before scientific recipe freeze, not twelve scientific fits')
    path=ROOT/'results/research/DEV04_contracts/partition_inventory.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({k:receipt[k] for k in ('status','body_fits','head_fits','scientific_training_updates','test_sampled')}))


if __name__=='__main__':main()
