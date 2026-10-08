"""Registered all-parent frozen-readout experiment; refuses unfrozen execution."""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path, expected):
    if sha(path) != expected:
        raise ValueError('Independent execution-plan hash differs')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan['status'] != 'frozen_before_head_fitting' or plan['scientific_head_fits'] != 36:
        raise ValueError('Exact prospective 36-fit plan required')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Governing file changed: ' + name)
    inventory_path = ROOT/'docs/research/READ03_parent_inventory.json'
    if sha(inventory_path) != plan['parent_inventory_sha256']:
        raise ValueError('Parent inventory differs')
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    if len(inventory['parents']) != 12:
        raise ValueError('All twelve parents required')
    return plan, inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=pathlib.Path, default=ROOT/'docs/research/READ03_execution_plan.json')
    parser.add_argument('--expected-plan-sha256')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--parent-case')
    parser.add_argument('--recover-cloud', action='store_true')
    parser.add_argument('--output', type=pathlib.Path, default=ROOT/'results/research/READ03')
    args = parser.parse_args()
    if args.describe:
        inventory = json.loads((ROOT/'docs/research/READ03_parent_inventory.json').read_text(encoding='utf-8'))
        print(json.dumps(dict(parents=len(inventory['parents']), head_fits=36, new_backbone_updates=0,
                              modes=['query_attention','uniform_global','local_query'])))
        return
    if not args.expected_plan_sha256:
        raise ValueError('Freeze and hash the recipe before fitting')
    plan, inventory = load_plan(args.plan, args.expected_plan_sha256)
    from neuropixel.research.qtrain_budget import require_ram
    require_ram()
    if args.recover_cloud:
        import importlib.util
        spec = importlib.util.spec_from_file_location('transport', ROOT/'neuropixel/research/qtrain_budget_transport.py')
        transport = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transport)
        transport.recover_existing(ROOT/'results/research/OPT03_QTRAIN_U16_recovery/37771588722/cohort',
            {d['parent_case'] for d in inventory['parents']}, 'NP-OPT03-QTRAIN-U16-20261008', inventory['parent_plan_sha256'])
    require_ram()
    import numpy as np
    import torch
    from neuropixel.research.qtrain_budget import validate_state, verify_environment, state_digest
    from neuropixel.research.models import ResearchNCA
    from neuropixel.research.data import ResearchRoleTask, frozen_dataset
    from neuropixel.research.experiment import configure_runtime, environment_record
    from neuropixel.research.OPT03_role_views import transform
    from neuropixel.research.readout_experiment import paired_training_dataset, extract_bank, fit_head, save_state, save_json, MODES
    environment = environment_record(configure_runtime('cpu',2))
    task = ResearchRoleTask(seed=0)
    datasets = {'train':paired_training_dataset(task)}
    for name, split, n, seed in [('probe','train',2048,77001), ('validation','validation',4096,77002)]:
        data = frozen_dataset(task,split,n,seed)
        datasets[name] = data
        cf = transform(*(v.numpy() for v in data),'query_flip')
        datasets[name+'_query_flip'] = tuple(torch.from_numpy(v) for v in cf)
    selected = inventory['parents']
    if args.parent_case:
        selected = [d for d in selected if d['parent_case'] == args.parent_case]
        if len(selected) != 1:
            raise ValueError('Unknown parent; subset is an execution shard, not a selection rule')
    completed = []
    for descriptor in selected:
        case = descriptor['parent_case']
        parent_folder = args.output/case
        # Entire completed shards are omitted before constructing another model.
        if all((parent_folder/mode/'result.json').exists() for mode in MODES):
            for mode in MODES:
                old = json.loads((parent_folder/mode/'result.json').read_text())
                if old['plan_sha256'] != args.expected_plan_sha256 or old['config']['mode'] != mode:
                    raise ValueError('Closed fit identity differs')
                completed.append(dict(parent=case,mode=mode,skipped_closed=True))
            continue
        require_ram()
        cp = ROOT/descriptor['checkpoint_path']
        if sha(cp) != descriptor['checkpoint_sha256']:
            raise ValueError('Parent checkpoint differs')
        payload = torch.load(cp,map_location='cpu',weights_only=True)
        validate_state(payload,16384,expected_plan=inventory['parent_plan_sha256'])
        verify_environment(descriptor['environment'],environment)
        body = ResearchNCA(35,(7,7),c_id=16,c=48,hidden=128,steps=16,fire_rate=.5,
                           tied=True,reinject=True,freeze_pad=False)
        body.load_state_dict(payload['model'])
        for p in body.parameters():
            p.requires_grad_(False)
        parent_result_path = cp.parent/'result.json'
        if sha(parent_result_path) != descriptor['result_sha256']:
            raise ValueError('Closed result bytes differ')
        record = json.loads(parent_result_path.read_text())
        banks = {}
        for panel, dataset in datasets.items():
            cache = parent_folder/'cache'/f'{panel}.pt'
            identity = dict(plan_sha256=args.expected_plan_sha256,parent_checkpoint_sha256=sha(cp),
                            dataset_digest=state_digest(dataset),mask_seeds=list(range(88000,88008)) if panel=='train' else list(range(79000,79008)))
            if cache.exists():
                saved = torch.load(cache,map_location='cpu',weights_only=True)
                if saved['identity'] != identity or saved['tensor_digest'] != state_digest(saved['bank']):
                    raise ValueError('Preserved cache differs')
                bank = saved['bank']
            else:
                bank = extract_bank(body,dataset,identity['mask_seeds'])
                if panel != 'train':
                    original_panel = panel.replace('_query_flip','')
                    field = 'query_predictions' if panel.endswith('_query_flip') else 'predictions'
                    relative = f'u16384_{original_panel}_matched.npz'
                    path = cp.parent/'decisions'/relative
                    archived = record['evaluations'][-1]['panels'][original_panel]['decision_artifacts'][relative]
                    if sha(path) != archived['sha256']:
                        raise ValueError('Closed reference prediction bytes differ')
                    with np.load(path,allow_pickle=False) as prior:
                        if not np.array_equal(bank['predictions'].numpy(),prior[field]):
                            raise ValueError('New latent extraction changed reference decisions')
                save_state(cache,dict(identity=identity,bank=bank,tensor_digest=state_digest(bank)))
            banks[panel] = bank
        for mode in MODES:
            result = fit_head(body,mode,banks,parent_folder/mode,args.expected_plan_sha256,descriptor)
            completed.append(dict(parent=case,mode=mode,checkpoint_sha256=result['checkpoint_sha256']))
        del banks, body, payload
    save_json(args.output/'cohort_progress.json',dict(plan_sha256=args.expected_plan_sha256,
              completed=completed,execution_shard=args.parent_case,scientific_head_fits_expected=36,
              new_backbone_updates=0,test_accessed=False))


if __name__ == '__main__':
    main()
