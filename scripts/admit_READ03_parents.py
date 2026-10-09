"""Load and freeze all pinned U16 states; no inference, fitting or test data."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-inventory-sha256', required=True)
    parser.add_argument('--recover-cloud', action='store_true')
    args = parser.parse_args()
    from neuropixel.research.qtrain_budget import require_ram, state_digest, validate_state, verify_environment
    ram = require_ram()
    path = ROOT/'docs/research/READ03_parent_inventory.json'
    if sha(path) != args.expected_inventory_sha256:
        raise ValueError('Independent pinned parent inventory hash differs')
    inventory = json.loads(path.read_text(encoding='utf-8'))
    expected = {x['parent_case'] for x in inventory['parents']}
    if len(expected) != 12 or len(inventory['parents']) != 12:
        raise ValueError('All twelve original bodies required')
    if args.recover_cloud:
        spec = importlib.util.spec_from_file_location('closed_transport', ROOT/'neuropixel/research/qtrain_budget_transport.py')
        transport = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transport)
        recovered = transport.recover_existing(ROOT/'results/research/OPT03_QTRAIN_U16_recovery/37771588722/cohort',
            expected, 'NP-OPT03-QTRAIN-U16-20261008', inventory['parent_plan_sha256'])
        if {r['case'] for r in recovered if r['status'] == 'completed'} != expected:
            raise ValueError('Recovery did not supply twelve completed bodies')
    require_ram()
    import torch
    from neuropixel.research.models import ResearchNCA
    from neuropixel.research.experiment import configure_runtime, environment_record
    from neuropixel.research.frozen_readout import FrozenReadout, body_digest
    environment = environment_record(configure_runtime('cpu', 2))
    rows = []
    for descriptor in inventory['parents']:
        require_ram()
        checkpoint = ROOT/descriptor['checkpoint_path']
        if sha(checkpoint) != descriptor['checkpoint_sha256']:
            raise ValueError('Actual parent checkpoint bytes differ')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        validate_state(payload, 16384, expected_plan=inventory['parent_plan_sha256'])
        differences = verify_environment(descriptor['environment'], environment)
        body = ResearchNCA(35, (7, 7), c_id=16, c=48, hidden=128, steps=16,
                           fire_rate=.5, tied=True, reinject=True, freeze_pad=False)
        body.load_state_dict(payload['model'])
        if state_digest(body.state_dict()) != state_digest(payload['model']):
            raise ValueError('Restored original tensors/buffers differ')
        original = body_digest(body)
        for mode in ('query_attention', 'uniform_global', 'local_query'):
            reader = FrozenReadout(body, mode)
            reader.assert_frozen()
            if sum(p.numel() for p in reader.head_parameters()) != 3168 or body_digest(body) != original:
                raise ValueError('Frozen body/head capacity admission differs')
            del reader
        rows.append(dict(case=descriptor['parent_case'], actual_body_digest=original,
                         checkpoint_sha256=sha(checkpoint), environment_differences=differences))
        del body, payload
    receipt = dict(status='verified_twelve_actual_frozen_bodies_no_inference_or_fitting',
                   parents=rows, inventory_sha256=sha(path), available_ram_before_torch_gib=ram,
                   environment=environment, new_backbone_updates=0, scientific_head_fits=0,
                   scientific_dataset_evaluations=0, test_accessed=False)
    dest = ROOT/'results/research/READ03_contracts/parent_admission_receipt.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:receipt[k] for k in ('status', 'scientific_head_fits', 'new_backbone_updates')}))


if __name__ == '__main__':
    main()
