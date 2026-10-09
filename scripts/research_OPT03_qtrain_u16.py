"""Registered all-parent continuation; first new update8193, never refit closed cases."""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key, value in [('OMP_NUM_THREADS', '2'), ('MKL_NUM_THREADS', '2'),
                   ('CUBLAS_WORKSPACE_CONFIG', ':4096:8')]:
    os.environ.setdefault(key, value)
REGISTRATION = 'NP-OPT03-QTRAIN-U16-20261008'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


budget = module('budget_primitives', 'neuropixel/research/qtrain_budget.py')
transport = module('budget_transport', 'neuropixel/research/qtrain_budget_transport.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(parents):
    return [{'run_id': p['extension_case'], 'parent_case': p['parent_case'],
             'init_seed': p['parent_config']['init_seed'], 'policy': p['parent_config']['policy'],
             'school_weight': p['parent_config']['school_weight'], 'updates': 16384}
            for p in parents['parents']]


def load_plan(path, expected_hash):
    raw = path.read_bytes()
    if b'\r\n' in raw or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('Canonical frozen plan differs')
    plan = read(path)
    parents = read(ROOT / plan['parent_inventory_file'])
    if (plan['registration_id'] != REGISTRATION or plan['status'] != 'frozen'
            or len(parents['parents']) != 12 or plan['runs'] != inventory(parents)
            or plan['parent_updates'] != 8192 or plan['target_updates'] != 16384
            or plan['endpoints'] != [12288, 16384]
            or plan['evaluation_mask_seeds'] != list(range(79000, 79008))):
        raise ValueError('Unfrozen or changed continuation recipe')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Changed frozen governing source ' + name)
    return plan, parents


def objects(config):
    import torch
    from neuropixel.research.models import build_model
    torch.manual_seed(config['init_seed'])
    model = build_model('neuropixel', vocab=35, h=8, w=8, steps=16,
                        freeze_pad=False, fire_rate=.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999),
                                 eps=1e-8, weight_decay=.0001)
    sampler = torch.Generator().manual_seed(92002)
    query_rng = torch.Generator().manual_seed(92003)
    return model, optimizer, sampler, query_rng


def admit_parent(parents, descriptor, parent_root, environment):
    budget.require_ram()
    import torch
    path = transport.recover_parent(parents, descriptor, parent_root)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    digest = budget.validate_state(payload, 8192, descriptor['parent_config'], parents['parent_plan_sha256'])
    differences = budget.verify_environment(payload['environment'], environment)
    model, optimizer, sampler, query_rng = objects(descriptor['parent_config'])
    if sum(p.numel() for p in model.parameters()) != 29824:
        raise ValueError('Backbone capacity differs')
    restored = budget.restore(model, optimizer, sampler, query_rng, payload, digest)
    budget.require_ram()
    return payload, (model, optimizer, sampler, query_rng), {
        'parent_checkpoint_sha256': descriptor['checkpoint_sha256'], 'parent_state_digest': digest,
        'restored_state_digest': restored, 'restored_update': 8192,
        'environment_differences': differences, 'new_updates': 0, 'test_accessed': False}


def run_one(config, descriptor, output, parent_root, plan, parents, plan_hash, stop_after=None,
            archive_callback=None):
    done = output / 'result.json'
    if done.exists():
        old = read(done)
        if (old['status'] != 'completed' or old['config'] != config or old['plan_sha256'] != plan_hash
                or old['last_update'] != 16384 or sha(output / 'checkpoint.pt') != old['checkpoint_sha256']):
            raise ValueError('Existing completed extension differs')
        return old
    budget.require_ram()
    import numpy as np
    import torch
    from neuropixel.research.data import ResearchRoleTask, frozen_dataset
    from neuropixel.research.experiment import (configure_runtime, environment_record, resource_sample,
                                                save_json, atomic_binary, utc_now)
    from neuropixel.research.qtrain import evaluate_queries, tensor_digest
    device = configure_runtime('cpu', 2)
    environment = environment_record(device)
    parent, actors, admission = admit_parent(parents, descriptor, parent_root, environment)
    model, optimizer, sampler, query_rng = actors
    task = ResearchRoleTask(8, 8, seed=0)
    data, dataset_records = {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for panel, split, count, seed in [('probe', 'train', 2048, 77001), ('validation', 'validation', 4096, 77002)]:
        values = frozen_dataset(task, split, count, seed)
        record = descriptor['datasets'][panel]
        source = parent_root / 'datasets' / record['file']
        if sha(source) != record['sha256'] or tensor_digest(*values) != record['content_sha256']:
            raise ValueError('Parent development dataset differs')
        destination = output.parent / 'datasets' / record['file']
        transport.immutable(destination, source.read_bytes())
        data[panel], dataset_records[panel] = values, record
    # Dataset creation is not allowed to alter the inherited firing RNG.
    if budget.state_digest(budget.training_state(model, optimizer, sampler, query_rng)) != admission['parent_state_digest']:
        raise ValueError('Dataset preparation consumed inherited RNG or changed state')
    check = output / 'checkpoint.pt'
    curve, evaluations, resources, elapsed, update0 = [], [], [], 0., 8192
    if check.exists():
        saved = torch.load(check, map_location='cpu', weights_only=True)
        if (saved['config'] != config or saved['plan_sha256'] != plan_hash
                or saved['parent_checkpoint_sha256'] != descriptor['checkpoint_sha256']):
            raise ValueError('Resume lineage/recipe differs')
        if not 8192 <= saved['update'] <= 16384 or saved['update'] % 128:
            raise ValueError('Invalid durable continuation step')
        budget.validate_state(saved, saved['update'], config, plan_hash)
        budget.verify_environment(saved['environment'], environment)
        budget.restore(model, optimizer, sampler, query_rng, saved, saved['training_state_digest'])
        curve, evaluations, resources = saved['curve'], saved['evaluations'], saved['resources']
        elapsed, update0 = saved['elapsed_training_seconds'], saved['update']
        if [x['update'] for x in curve] != list(range(8320, update0 + 1, 128)):
            raise ValueError('Incomplete or repeated durable prefix')
    else:
        save_json(output / 'parent_admission.json', admission)
    parent_reference = {'parent_case': descriptor['parent_case'],
                        'parent_checkpoint_sha256': descriptor['checkpoint_sha256'],
                        'parent_result_sha256': descriptor['result_sha256'],
                        'parent_source_commit': parents['parent_source_commit'],
                        'parent_archive_git_commit': parents['parent_archive_git_commit'],
                        'parent_updates': 8192, 'baseline_evaluations': parent['evaluations']}
    transport.immutable(output / 'parent_reference.json', (json.dumps(parent_reference, indent=2) + '\n').encode())

    def persist(update):
        state = budget.training_state(model, optimizer, sampler, query_rng)
        payload = {'config': config, 'plan_sha256': plan_hash, 'environment': environment,
                   'update': update, **state, 'training_state_digest': budget.state_digest(state),
                   'parent_checkpoint_sha256': descriptor['checkpoint_sha256'],
                   'curve': curve, 'evaluations': evaluations, 'resources': resources,
                   'elapsed_training_seconds': elapsed, 'datasets': dataset_records}
        with atomic_binary(check) as handle:
            torch.save(payload, handle)
        save_json(output / 'progress.json', {'status': 'training', 'config': config,
                  'parent_case': descriptor['parent_case'], 'plan_sha256': plan_hash,
                  'completed_updates': update, 'new_updates': update - 8192,
                  'checkpoint_sha256': sha(check), 'test_accessed': False})

    def endpoint(update):
        item = {'update': update, 'test_accessed': False, 'panels': {}}
        for panel, values in data.items():
            item['panels'][panel] = evaluate_queries(model, values, output / 'decisions',
                f'u{update}_{panel}', plan['evaluation_mask_seeds'], include_masked=True)
        evaluations.append(item)
        persist(update)
        snapshot = output / f'checkpoint_u{update}.pt'
        if snapshot.exists():
            old = torch.load(snapshot, map_location='cpu', weights_only=True)
            if old['update'] != update or old['training_state_digest'] != budget.state_digest(budget.training_state(model, optimizer, sampler, query_rng)):
                raise ValueError('Preserved endpoint state differs')
        else:
            shutil.copyfile(check, snapshot)
    if not check.exists():
        persist(8192)
    if update0 in plan['endpoints'] and update0 not in [x['update'] for x in evaluations]:
        endpoint(update0)
    model.train()
    window, roles, context_hash = [], torch.zeros(4, dtype=torch.long), hashlib.sha256()
    start = time.monotonic()
    try:
        for update in range(update0 + 1, 16385):
            if update % 128 == 1:
                resources.append(resource_sample(device, 8))
            value = budget.update_once(model, optimizer, task, sampler, query_rng,
                                       config['policy'], config['school_weight'])
            witness = value['witness']
            context_hash.update(witness['base_canvas'].numpy().tobytes())
            context_hash.update(witness['fillers'].numpy().tobytes())
            roles += torch.bincount(value['roles'], minlength=4)
            if update == 8193:
                arrays = {key: value[key].numpy() for key in ('canvas', 'target', 'roles')}
                arrays.update({key: item.numpy() for key, item in witness.items()})
                path = output / 'first_new_batch.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as old:
                        if set(old.files) != set(arrays) or any(not np.array_equal(old[key], item) for key, item in arrays.items()):
                            raise ValueError('Preserved first new batch differs')
                else:
                    np.savez_compressed(path, **arrays)
            window.append((value['answer_loss'], value['school_loss'], value['total_loss']))
            if update % 128 == 0:
                elapsed += time.monotonic() - start
                curve.append({'update': update,
                    'answer_loss_mean': sum(x[0] for x in window) / 128,
                    'school_loss_mean': sum(x[1] for x in window) / 128,
                    'total_loss_mean': sum(x[2] for x in window) / 128,
                    'last_unclipped_gradient_norm': value['gradient_norm'], 'role_counts': roles.tolist(),
                    'context_window_sha256': context_hash.hexdigest(), 'elapsed_training_seconds': elapsed})
                window = []
                roles.zero_()
                context_hash = hashlib.sha256()
                persist(update)
                if update in plan['endpoints']:
                    endpoint(update)
                if archive_callback is not None and update % 1024 == 0 and update < 16384:
                    before_archive = budget.state_digest(budget.training_state(model, optimizer, sampler, query_rng))
                    archive_callback(config['run_id'])
                    if budget.state_digest(budget.training_state(model, optimizer, sampler, query_rng)) != before_archive:
                        raise ValueError('Archival consumed training RNG or changed state')
                resources.append(resource_sample(device, 8))
                start = time.monotonic()
                print(json.dumps({'event': 'u16_progress', 'case': config['run_id'], 'update': update}), flush=True)
                if stop_after and update >= stop_after:
                    return None
        final = evaluations[-1]['panels']
        gate = (final['probe']['metrics']['matched']['joint_query_binding'] >= .95
                and final['validation']['metrics']['matched']['joint_query_binding'] >= .90)
        result = {'status': 'completed', 'config': config, 'parent_case': descriptor['parent_case'],
                  'plan_sha256': plan_hash, 'environment': environment, 'parameter_count': 29824,
                  'last_update': 16384, 'first_new_update': 8193, 'new_updates': 8192,
                  'new_initializations': 0, 'parent_admission': admission, 'curve': curve,
                  'evaluations': evaluations, 'resources': resources, 'datasets': dataset_records,
                  'checkpoint_sha256': sha(check), 'matched_joint_competence_gate_passed': gate,
                  'test_accessed': False, 'completed_utc': utc_now()}
        save_json(done, result)
        return result
    except Exception as error:
        save_json(output / 'incident.json', {'error_type': type(error).__name__,
                  'last_durable_update': curve[-1]['update'] if curve else update0,
                  'test_accessed': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=pathlib.Path, default=ROOT / 'docs/research/OPT03_QTRAIN_U16_plan.json')
    parser.add_argument('--expected-plan-sha256')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--admit-parents-only', action='store_true')
    parser.add_argument('--parents-root', type=pathlib.Path, default=ROOT / 'results/research/OPT03_QTRAIN_U16/parents')
    parser.add_argument('--output', type=pathlib.Path, default=ROOT / 'results/research/OPT03_QTRAIN_U16/cohort')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.describe:
        parents = read(ROOT / 'docs/research/OPT03_QTRAIN_U16_parent_inventory.json')
        print(json.dumps({'extensions': inventory(parents), 'new_updates_total': 98304,
                          'new_initializations': 0, 'closed_prefix_repeated': False}))
        return
    if args.admit_parents_only:
        budget.require_ram()
        from neuropixel.research.experiment import configure_runtime, environment_record, save_json
        environment = environment_record(configure_runtime('cpu', 2))
        original = read(ROOT / 'docs/research/OPT03_QTRAIN_plan.json')
        for name, digest in original['source_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Closed parent scientific source differs')
        parents = read(ROOT / 'docs/research/OPT03_QTRAIN_U16_parent_inventory.json')
        admissions = []
        for descriptor in parents['parents']:
            _, actors, admission = admit_parent(parents, descriptor, args.parents_root, environment)
            admissions.append({'case': descriptor['parent_case'], **admission})
            del actors
        save_json(args.output.parent / 'parent_admission_receipt.json',
                  {'status': 'verified_twelve_parent_states_no_updates', 'parents': admissions,
                   'new_updates': 0, 'new_model_evaluations': 0, 'test_accessed': False})
        print(json.dumps({'status': 'verified_twelve_parent_states_no_updates', 'parents': len(admissions)}))
        return
    if not args.expected_plan_sha256:
        raise ValueError('Independent frozen plan hash required')
    plan, parents = load_plan(args.plan, args.expected_plan_sha256)
    if os.environ.get('QTRAIN_U16_PUBLISH_AFTER_CASE') == '1':
        transport.recover_existing(args.output, {x['run_id'] for x in plan['runs']},
                                   REGISTRATION, args.expected_plan_sha256)
    by_name = {x['parent_case']: x for x in parents['parents']}
    for config in plan['runs']:
        archival = (lambda case: transport.publish_case(args.output, case, args.plan, REGISTRATION, allow_partial=True)) if os.environ.get('QTRAIN_U16_PUBLISH_AFTER_CASE') == '1' else None
        result = run_one(config, by_name[config['parent_case']], args.output / config['run_id'],
                         args.parents_root, plan, parents, args.expected_plan_sha256, args.stop_after, archival)
        if result is not None and os.environ.get('QTRAIN_U16_PUBLISH_AFTER_CASE') == '1':
            transport.publish_case(args.output, config['run_id'], args.plan, REGISTRATION)


if __name__ == '__main__':
    main()
