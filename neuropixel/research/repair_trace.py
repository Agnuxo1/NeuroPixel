"""Token-only, evaluation-only state-repair traces for item14 research.

This helper leaves the native model and streaming APIs unchanged. It follows
NeuroPixel.forward through initial seeding and local updates, applies one state
mask after an explicit warmup boundary, then changes only the identity input
for subsequent updates. It never seeds again after the boundary. Source access
and state survival are therefore separate interventions.

All returned snapshots are detached independent copies. These traces alone
establish no learned competence, reconstruction of lost information or
long-horizon stability.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from neuropixel.model import NeuroPixel


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _canvas(canvas, *, shape, model, name):
    _require(isinstance(canvas, torch.Tensor) and canvas.dtype == torch.long,
             name + " must be a torch.long tensor")
    _require(canvas.ndim == 3 and all(size > 0 for size in canvas.shape),
             name + " must have nonempty [batch,height,width] shape")
    if shape is not None:
        _require(tuple(canvas.shape) == shape, name + " shape differs from the initial canvas")
    _require(canvas.device == model.embed.weight.device, name + " and model devices differ")
    _require(bool(((canvas >= 0) & (canvas < model.embed.num_embeddings)).all()),
             name + " contains a token outside the vocabulary")


@torch.no_grad()
def trace_repair(model, canvas, *, warmup_steps, recovery_steps, keep_mask, recovery_canvas):
    """Return exact-boundary snapshots for a native, deterministic token trace.

    warmup_steps is a positive integer and recovery_steps a nonnegative integer.
    keep_mask is bool[B,1,H,W], applied after the last warmup update. The initial
    canvas is seeded once. recovery_canvas supplies identities only; it adds no
    seed, even when it contains non-PAD tokens or differs from the initial input.

    Returns prefix_states[B,warmup+1,C,H,W], pre_damage[B,C,H,W],
    post_damage[B,C,H,W], recovery_states[B,recovery+1,C,H,W],
    logits[B,recovery+1,V], and prefix_logits[B,warmup+1,V].
    Prefix index0 is the seed; recovery index0 is the immediate damaged state.
    Readout uses model.out_pos and the native tied dictionary/PAD-logit policy.
    """
    _require(type(model) is NeuroPixel, "only the native NeuroPixel class is supported")
    _require(not model.training, "trace_repair requires model.eval()")
    _require(type(warmup_steps) is int and warmup_steps >= 1,
             "warmup_steps must be a positive integer")
    _require(type(recovery_steps) is int and recovery_steps >= 0,
             "recovery_steps must be a nonnegative integer")
    _canvas(canvas, shape=None, model=model, name="canvas")
    batch, height, width = canvas.shape
    _canvas(recovery_canvas, shape=tuple(canvas.shape), model=model, name="recovery_canvas")
    _require(isinstance(keep_mask, torch.Tensor) and keep_mask.dtype == torch.bool,
             "keep_mask must be a boolean tensor")
    _require(tuple(keep_mask.shape) == (batch, 1, height, width)
             and keep_mask.device == canvas.device,
             "keep_mask must have shape [batch,1,height,width] on the input device")
    position = model.out_pos
    _require(isinstance(position, (tuple, list)) and len(position) == 2
             and all(type(value) is int for value in position)
             and 0 <= position[0] < height and 0 <= position[1] < width,
             "model.out_pos is outside the canvas")
    # A token-only trace does not use a camera/retina path or caller hooks.
    dictionary = model.dictionary()
    _require(dictionary.device == model.seed.weight.device
             and dictionary.dtype == model.seed.weight.dtype,
             "dictionary and seed dtype/device differ")
    _require(bool(torch.isfinite(dictionary).all()), "effective dictionary must be finite")
    ids = F.embedding(canvas, dictionary).permute(0, 3, 1, 2)
    future_ids = F.embedding(recovery_canvas, dictionary).permute(0, 3, 1, 2)
    state = model.seed(ids) * (canvas != 0).unsqueeze(1)

    def update(current, identities):
        joined = torch.cat([current, model.perceive(current), identities], dim=1)
        return current + model.f2(F.relu(model.f1(joined)))

    def read(current):
        logits = model.read(current[:, :, position[0], position[1]]) @ dictionary.T
        logits[:, 0] = -1e4
        return logits.detach().clone()

    def snapshot(current):
        _require(bool(torch.isfinite(current).all()), "nonfinite recurrent state")
        return current.detach().clone()

    prefix_states, prefix_logits = [snapshot(state)], [read(state)]
    for _ in range(warmup_steps):
        state = update(state, ids)
        prefix_states.append(snapshot(state))
        prefix_logits.append(read(state))
    pre_damage = snapshot(state)
    state = state * keep_mask
    post_damage = snapshot(state)
    recovery_states, logits = [snapshot(state)], [read(state)]
    for _ in range(recovery_steps):
        state = update(state, future_ids)
        recovery_states.append(snapshot(state))
        logits.append(read(state))
    return {
        "prefix_states": torch.stack(prefix_states, dim=1),
        "pre_damage": pre_damage,
        "post_damage": post_damage,
        "recovery_states": torch.stack(recovery_states, dim=1),
        "logits": torch.stack(logits, dim=1),
        "prefix_logits": torch.stack(prefix_logits, dim=1),
    }
