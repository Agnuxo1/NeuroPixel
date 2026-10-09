"""Exact-state continuation primitives; imports Torch only after caller admission."""
import hashlib
import math

STATE_FIELDS = ('model', 'optimizer', 'cpu_rng', 'sampler_rng', 'query_rng')
SOFTWARE_FIELDS = ('python', 'versions', 'device', 'threads', 'cuda_runtime',
                   'deterministic_algorithms', 'cublas_workspace_config',
                   'cudnn_benchmark', 'cudnn_deterministic', 'matmul_tf32', 'cudnn_tf32')


def require_ram(minimum=8):
    import psutil
    available = psutil.virtual_memory().available / 2**30
    if available < minimum:
        raise RuntimeError('RAM admission below8GiB')
    return available


def state_digest(value):
    """Hash actual values and structure, not noncanonical Torch ZIP serialization."""
    import torch
    digest = hashlib.sha256()

    def frame(body):
        digest.update(len(body).to_bytes(8, 'big'))
        digest.update(body)

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            frame(repr(('tensor', str(tensor.dtype), tuple(tensor.shape))).encode())
            frame(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            frame(repr(('dict', len(item))).encode())
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            frame(repr((type(item).__name__, len(item))).encode())
            for element in item:
                visit(element)
        elif item is None or isinstance(item, (bool, str, int, float)):
            frame(repr((type(item).__name__, item)).encode())
        else:
            raise TypeError('Unsupported training-state value ' + type(item).__name__)
    visit(value)
    return digest.hexdigest()


def training_state(model, optimizer, sampler, query_rng):
    import torch
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'cpu_rng': torch.get_rng_state(), 'sampler_rng': sampler.get_state(),
            'query_rng': query_rng.get_state()}


def validate_state(payload, expected_update, expected_config=None, expected_plan=None):
    import torch
    if payload['update'] != expected_update:
        raise ValueError('Checkpoint update differs')
    if expected_config is not None and payload['config'] != expected_config:
        raise ValueError('Checkpoint configuration differs')
    if expected_plan is not None and payload['plan_sha256'] != expected_plan:
        raise ValueError('Checkpoint plan differs')
    if not set(STATE_FIELDS).issubset(payload):
        raise ValueError('Incomplete model/optimizer/three RNG state')

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(x) for x in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(x) for x in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not finite({key: payload[key] for key in STATE_FIELDS}):
        raise ValueError('Nonfinite checkpoint state')
    state = payload['optimizer']['state']
    if not state or any(float(item['step']) != expected_update for item in state.values()):
        raise ValueError('AdamW counter differs')
    for group in payload['optimizer']['param_groups']:
        if (group['lr'] != .001 or tuple(group['betas']) != (.9, .999)
                or group['eps'] != 1e-8 or group['weight_decay'] != .0001
                or group['amsgrad'] or group['maximize'] or group['capturable']
                or group['differentiable']):
            raise ValueError('AdamW recipe differs')
    for key in ('cpu_rng', 'sampler_rng', 'query_rng'):
        if payload[key].dtype != torch.uint8 or payload[key].ndim != 1:
            raise ValueError('Invalid RNG state')
    return state_digest({key: payload[key] for key in STATE_FIELDS})


def verify_environment(parent, current):
    differences = {key: {'parent': parent.get(key), 'current': current.get(key)}
                   for key in parent.keys() | current.keys() if parent.get(key) != current.get(key)}
    if any(parent.get(key) != current.get(key) for key in SOFTWARE_FIELDS):
        raise ValueError('Unplanned scientific software/device/determinism change')
    # Standard public CPU hosts/kernels can differ; record this rather than asserting
    # physical-host identity or cross-host bitwise equivalence of future updates.
    return differences


def restore(model, optimizer, sampler, query_rng, payload, expected_digest):
    import torch
    if state_digest({key: payload[key] for key in STATE_FIELDS}) != expected_digest:
        raise ValueError('Expected parent/resume state digest differs')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    sampler.set_state(payload['sampler_rng'])
    query_rng.set_state(payload['query_rng'])
    torch.set_rng_state(payload['cpu_rng'])
    actual = state_digest(training_state(model, optimizer, sampler, query_rng))
    if actual != expected_digest:
        raise ValueError('Restoration altered model/AdamW/three RNG streams')
    return actual


def update_once(model, optimizer, task, sampler, query_rng, policy, school_weight):
    """The same mathematical update as closed QTRAIN; no prefix replay."""
    import torch
    import torch.nn.functional as functional
    from neuropixel.research.qtrain import sample_batch
    if not model.training:
        raise ValueError('Training mode not restored')
    canvas, target, roles, witness = sample_batch(task, sampler, query_rng, policy)
    optimizer.zero_grad(set_to_none=True)
    output = model(canvas, lens_every=4 if school_weight else 0)
    answer = functional.cross_entropy(output['logits'], target)
    school = answer.new_zeros(())
    if school_weight:
        count = output['lens'].shape[1]
        occupied = (canvas != 0).unsqueeze(1).expand(-1, count, -1, -1)
        labels = canvas.unsqueeze(1).expand(-1, count, -1, -1)
        school = functional.cross_entropy(output['lens'][occupied], labels[occupied])
    loss = answer + school_weight * school
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite total loss')
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return {'answer_loss': float(answer.detach()), 'school_loss': float(school.detach()),
            'total_loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'canvas': canvas, 'target': target, 'roles': roles, 'witness': witness}
