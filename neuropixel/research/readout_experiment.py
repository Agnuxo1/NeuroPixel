"""Deterministic cache/head-only fitting primitives; no closed body updates."""
import hashlib
import json
import pathlib
import shutil
import time

import torch
from torch.nn import functional as F

from neuropixel.research.frozen_readout import FrozenReadout, body_digest, pack_visible_state
from neuropixel.research.qtrain_budget import require_ram, state_digest

MODES = ('query_attention', 'uniform_global', 'local_query')


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def save_state(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def save_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def paired_training_dataset(task, n=4096):
    from neuropixel.research.qtrain import sample_batch
    if n < 64 or n % 64:
        raise ValueError('Whole paired batches required')
    sampler = torch.Generator().manual_seed(89002)
    query_rng = torch.Generator().manual_seed(89003)
    arrays = [[], [], []]
    for _ in range(n//64):
        x, y, r, _ = sample_batch(task, sampler, query_rng, 'paired')
        for bucket, value in zip(arrays, (x, y, r)):
            bucket.append(value)
    return tuple(torch.cat(bucket) for bucket in arrays)


def extract_bank(body, dataset, mask_seeds, batch_size=256):
    """New latent observable; original decisions/NLL available for audit only."""
    require_ram()
    x, y, roles = dataset
    before = body_digest(body)
    original_mode, original_fire = body.training, body.fire_rate
    facts, local, predictions, nll = [], [], [], []
    layout = None
    positions = None
    start = time.monotonic()
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        try:
            body.train(True)
            body.fire_rate = .5
            for seed in mask_seeds:
                torch.manual_seed(seed)
                ff, ll, pp, nn = [], [], [], []
                for at in range(0, len(x), batch_size):
                    require_ram()
                    out = body(x[at:at+batch_size])
                    packed = pack_visible_state(out['state'], x[at:at+batch_size])
                    if layout is None:
                        layout = packed['channels_last']
                    if layout != packed['channels_last']:
                        raise ValueError('State layout changed within bank')
                    ff.append(packed['facts'])
                    ll.append(packed['local'])
                    pp.append(out['logits'].argmax(-1))
                    nn.append(F.cross_entropy(out['logits'], y[at:at+batch_size], reduction='none'))
                facts.append(torch.cat(ff))
                local.append(torch.cat(ll))
                predictions.append(torch.cat(pp))
                nll.append(torch.cat(nn))
            # Position selection is derived only from visible inputs.
            mask = (x != 0).clone()
            mask[:, 7, 6] = False
            positions = mask.flatten(1).nonzero(as_tuple=False)[:, 1].reshape(len(x), 8)
        finally:
            body.train(original_mode)
            body.fire_rate = original_fire
    if body_digest(body) != before:
        raise ValueError('Extracting state changed frozen body')
    # Clone outside inference_mode: these detached constants must be usable as
    # saved autograd inputs when training the new head.
    result = dict(canvas=x.detach().clone(), target=y.detach().clone(), roles=roles.detach().clone(),
                  positions=positions.clone(), facts=torch.stack(facts).clone(),
                  local=torch.stack(local).clone(), predictions=torch.stack(predictions).clone(),
                  nll=torch.stack(nll).clone(), channels_last=layout,
                  mask_seeds=list(mask_seeds), frozen_body_digest=before,
                  elapsed_extraction_seconds=time.monotonic()-start)
    if not torch.isfinite(result['facts']).all() or not torch.isfinite(result['local']).all():
        raise ValueError('Nonfinite cached representation')
    return result


def select_bank(bank, indices, masks):
    if isinstance(masks, int):
        masks = torch.full_like(indices, masks)
    return dict(canvas=bank['canvas'][indices], positions=bank['positions'][indices],
                facts=bank['facts'][masks, indices], local=bank['local'][masks, indices],
                channels_last=bank['channels_last'])


def batch_indices(bank, context_rng, mask_rng):
    count = len(bank['canvas'])
    if count % 4 or not torch.equal(bank['roles'], torch.arange(4).repeat(count//4)):
        raise ValueError('Paired context groups/order required')
    groups = torch.randint(count//4, (16,), generator=context_rng)
    indices = (groups[:, None]*4 + torch.arange(4)).flatten()
    masks = torch.randint(len(bank['mask_seeds']), (16,), generator=mask_rng).repeat_interleave(4)
    return indices, masks


def step(reader, optimizer, bank, context_rng, mask_rng):
    indices, masks = batch_indices(bank, context_rng, mask_rng)
    optimizer.zero_grad(set_to_none=True)
    logits = reader(select_bank(bank, indices, masks))
    loss = F.cross_entropy(logits, bank['target'][indices])
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite readout loss')
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(reader.head_parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    reader.assert_frozen()
    witness = hashlib.sha256(indices.numpy().tobytes()+masks.numpy().tobytes()).hexdigest()
    return float(loss.detach()), float(gradient), witness


def evaluate_bank(reader, original, counterfactual, batch_size=256):
    """Eight individual realizations; no ensemble, mask or endpoint selection."""
    output = {}
    with torch.inference_mode():
        for name, bank in [('original', original), ('query_flip', counterfactual)]:
            pred, nll = [], []
            for k in range(len(bank['mask_seeds'])):
                pp, nn = [], []
                for at in range(0, len(bank['canvas']), batch_size):
                    indices = torch.arange(at, min(at+batch_size, len(bank['canvas'])))
                    logits = reader(select_bank(bank, indices, k))
                    pp.append(logits.argmax(-1))
                    nn.append(F.cross_entropy(logits, bank['target'][indices], reduction='none'))
                pred.append(torch.cat(pp))
                nll.append(torch.cat(nn))
            output[name] = dict(predictions=torch.stack(pred), nll=torch.stack(nll),
                                target=bank['target'], roles=bank['roles'])
    good = output['original']['predictions'] == original['target'][None]
    cf_good = output['query_flip']['predictions'] == counterfactual['target'][None]
    noun = (original['roles'] == 0) | (original['roles'] == 2)
    if not torch.equal(output['original']['predictions'][:, ~noun], output['query_flip']['predictions'][:, ~noun]):
        raise ValueError('Unchanged query-control decisions differ')
    per_role = [float(good[:, original['roles'] == r].double().mean()) for r in range(4)]
    return dict(metrics=dict(accuracy=float(good.double().mean()), binding=(per_role[0]+per_role[2])/2,
                             joint_query_binding=float((good & cf_good)[:, noun].double().mean()),
                             per_role=per_role,
                             mean_nll=float(output['original']['nll'].double().mean())), arrays=output)


def training_state(reader, optimizer, context_rng, mask_rng):
    return dict(head=reader.head.state_dict(), optimizer=optimizer.state_dict(),
                cpu_rng=torch.get_rng_state(), context_rng=context_rng.get_state(),
                mask_rng=mask_rng.get_state(), frozen_body_digest=body_digest(reader.backbone))


def restore(reader, optimizer, context_rng, mask_rng, payload):
    if payload['frozen_body_digest'] != body_digest(reader.backbone):
        raise ValueError('Resume body identity differs')
    reader.head.load_state_dict(payload['head'])
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['cpu_rng'])
    context_rng.set_state(payload['context_rng'])
    mask_rng.set_state(payload['mask_rng'])
    actual = training_state(reader, optimizer, context_rng, mask_rng)
    if state_digest(actual) != state_digest({k:payload[k] for k in actual}):
        raise ValueError('Resume head/optimizer/streams changed')
    reader.assert_frozen()


def fit_head(body, mode, banks, output, plan_hash, parent, updates=8192, endpoints=(1024,4096,8192), stop_after=None, archive_callback=None):
    """Stateful fitting of new head only, with intact restart/closed-run guards."""
    output = pathlib.Path(output)
    final_path = output/'result.json'
    config = dict(parent_case=parent['parent_case'], mode=mode,
                  head_seed=500+parent['init_seed']-200, updates=updates)
    if final_path.exists():
        done = json.loads(final_path.read_text(encoding='utf-8'))
        if done['config'] != config or done['plan_sha256'] != plan_hash:
            raise ValueError('Preserved closed fit identity differs')
        return done
    require_ram()
    torch.manual_seed(config['head_seed'])
    reader = FrozenReadout(body, mode)
    optimizer = torch.optim.AdamW(reader.head_parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=.0001)
    context_rng = torch.Generator().manual_seed(89004)
    mask_rng = torch.Generator().manual_seed(89005)
    curve, evaluations, resources, elapsed, last, effective_parameters = [], [], [], 0., 0, 0
    checkpoint = output/'checkpoint.pt'
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['config'] != config or payload['plan_sha256'] != plan_hash:
            raise ValueError('Resume recipe differs')
        restore(reader, optimizer, context_rng, mask_rng, payload)
        curve, evaluations, resources = payload['curve'], payload['evaluations'], payload['resources']
        last, elapsed = payload['update'], payload['elapsed_update_seconds']
        effective_parameters = payload['effective_gradient_parameters']
    else:
        save_state(checkpoint, dict(**training_state(reader,optimizer,context_rng,mask_rng),config=config,
                   plan_sha256=plan_hash,update=0,curve=[],evaluations=[],resources=[],elapsed_update_seconds=0.,
                   effective_gradient_parameters=0))
    def persist(update):
        save_state(checkpoint, dict(**training_state(reader,optimizer,context_rng,mask_rng),config=config,
                   plan_sha256=plan_hash,update=update,curve=curve,evaluations=evaluations,
                   resources=resources,elapsed_update_seconds=elapsed,
                   effective_gradient_parameters=effective_parameters))
        save_json(output/'progress.json',dict(config=config,plan_sha256=plan_hash,update=update,
                  frozen_body_digest=body_digest(body),checkpoint_sha256=sha(checkpoint),test_accessed=False))
    def endpoint(update):
        panels = {}
        for panel in ('probe', 'validation'):
            result = evaluate_bank(reader, banks[panel], banks[panel+'_query_flip'])
            path = output/'decisions'/f'u{update}_{panel}.pt'
            save_state(path, result['arrays'])
            panels[panel] = dict(metrics=result['metrics'], artifact_sha256=sha(path))
        evaluations.append(dict(update=update, panels=panels, test_accessed=False))
        persist(update)
        snapshot = output/f'checkpoint_u{update}.pt'
        if snapshot.exists() and snapshot.read_bytes() != checkpoint.read_bytes():
            raise ValueError('Preserved evaluated head checkpoint differs')
        if not snapshot.exists():
            shutil.copyfile(checkpoint,snapshot)
    if last in endpoints and last not in [e['update'] for e in evaluations]:
        endpoint(last)
    losses, witnesses = [], hashlib.sha256()
    for update in range(last+1, updates+1):
        if update % 128 == 1:
            resources.append(dict(update=update, available_ram_gib=require_ram()))
        started = time.monotonic()
        loss, gradient, witness = step(reader, optimizer, banks['train'], context_rng, mask_rng)
        effective_parameters = sum(p.numel() for p in reader.head_parameters() if p.grad is not None)
        elapsed += time.monotonic()-started
        losses.append(loss)
        witnesses.update(witness.encode())
        if update % 128 == 0:
            curve.append(dict(update=update, answer_loss_mean=sum(losses)/len(losses),
                              batch_witness_sha256=witnesses.hexdigest(), last_unclipped_gradient_norm=gradient,
                              elapsed_update_seconds=elapsed))
            losses, witnesses = [], hashlib.sha256()
            persist(update)
            if update in endpoints:
                endpoint(update)
            if archive_callback is not None and update % 1024 == 0 and update < updates:
                before_archive=state_digest(training_state(reader,optimizer,context_rng,mask_rng))
                archive_callback()
                if state_digest(training_state(reader,optimizer,context_rng,mask_rng)) != before_archive:
                    raise ValueError('Archival changed head/optimizer/RNG')
            print(json.dumps({'event':'head_progress','parent':parent['parent_case'],'mode':mode,'update':update}),flush=True)
            if stop_after and update >= stop_after:
                return None
    reader.assert_frozen()
    final = evaluations[-1]['panels']
    gate = final['probe']['metrics']['joint_query_binding'] >= .95 and final['validation']['metrics']['joint_query_binding'] >= .90
    result = dict(status='completed', config=config, plan_sha256=plan_hash, curve=curve,
                  evaluations=evaluations, resources=resources, checkpoint_sha256=sha(checkpoint),
                  frozen_body_digest=body_digest(body), new_backbone_updates=0,
                  head_nominal_parameters=sum(p.numel() for p in reader.head_parameters()),
                  effective_gradient_parameters=effective_parameters,
                  matched_joint_competence_gate_passed=gate,test_accessed=False)
    save_json(final_path,result)
    if archive_callback is not None:
        archive_callback()
    return result
