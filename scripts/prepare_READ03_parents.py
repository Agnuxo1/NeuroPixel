"""Pin all twelve closed U16 parents for a new readout diagnostic; no Torch."""
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review = ROOT/'results/research/OPT03_QTRAIN_U16_review/37771588722'
    closure_path = review/'closure_receipt.json'
    closure = json.loads(closure_path.read_text(encoding='utf-8'))
    assert closure['status'] == 'closed_exact_U16_development_recipe_no_eligible_condition'
    assert closure['cases'] == 12 and closure['issues'] == []
    for name, digest in closure['inputs_sha256'].items():
        assert sha(ROOT/name) == digest
    plan_path = ROOT/'docs/research/OPT03_QTRAIN_U16_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert sha(plan_path) == 'a4093bb8d559404b9f68587b753f8cea180a1e75d33363a04c1f2c7ebcb22dc0'
    cohort = ROOT/'results/research/OPT03_QTRAIN_U16_recovery/37771588722/cohort'
    parents = []
    for conf in plan['runs']:
        case = conf['run_id']
        result_path = cohort/case/'result.json'
        checkpoint = cohort/case/'checkpoint_u16384.pt'
        record = json.loads(result_path.read_text(encoding='utf-8'))
        assert record['status'] == 'completed' and record['last_update'] == 16384
        assert record['parameter_count'] == 29824 and record['test_accessed'] is False
        assert sha(cohort/case/'checkpoint.pt') == record['checkpoint_sha256']
        assert checkpoint.read_bytes() == (cohort/case/'checkpoint.pt').read_bytes()
        replay_path = review/'pinned_replay_receipts'/case/'replay.json'
        replay = json.loads(replay_path.read_text(encoding='utf-8'))
        assert replay['status'] == 'verified_available_endpoint_replay' and replay['issues'] == []
        assert replay['cases'] == [case] and replay['verification_returncode'] == 0
        original = ROOT/f'results/research/OPT03_QTRAIN_U16_recovery/37771588722/originals/{case}/recovery_receipt.json'
        parents.append(dict(parent_case=case, init_seed=conf['init_seed'], parent_policy=conf['policy'],
                            parent_school_weight=conf['school_weight'], parent_updates=16384,
                            checkpoint_path=str(checkpoint.relative_to(ROOT)).replace('\\','/'),
                            checkpoint_sha256=sha(checkpoint), result_sha256=sha(result_path),
                            endpoint_replay_sha256=sha(replay_path), recovery_receipt_sha256=sha(original),
                            datasets=record['datasets'], environment=record['environment']))
    assert len(parents) == 12 and len({p['parent_case'] for p in parents}) == 12
    inventory = dict(status='all_twelve_closed_parents_pinned_no_head_fitting',
                     scientific_source_commit='adab1f3dc0ded9da3c42a3e7e5adcb96cdaba67e',
                     scientific_run_id=37771588722, parent_plan_sha256=sha(plan_path),
                     closure_receipt_sha256=sha(closure_path), parents=parents,
                     backbone_initializations=3, head_variants=['query_attention', 'uniform_global', 'local_query'],
                     prospective_head_fits=36, selected_parent_subset=False,
                     head_fitting_started=False, new_backbone_updates=0)
    path = ROOT/'docs/research/READ03_parent_inventory.json'
    data = (json.dumps(inventory, indent=2)+'\n').encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Pinned inventory differs; preserve and investigate')
    path.write_bytes(data)
    print(json.dumps({'parents':12, 'prospective_head_fits':36, 'inventory_sha256':sha(path),
                      'new_backbone_updates':0, 'head_fitting_started':False}))


if __name__ == '__main__':
    main()
