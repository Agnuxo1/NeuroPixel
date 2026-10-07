"""Strict, explicit state continuation for item-11 research.

The original model and phase3 APIs remain unchanged. These helpers add input
validation and opt-in caller-owned state/interventions; they do not persist
state on a model. NP dynamics match phase3.np_stream on valid baseline inputs.
A swap means new_state[recipient] = old_state[permutation[recipient]].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _frames(frames, embedding):
    _require(isinstance(frames, (list, tuple)) and len(frames) > 0,
             "frames must be a nonempty explicit list or tuple")
    shape = None
    for frame in frames:
        _require(isinstance(frame, torch.Tensor) and frame.dtype == torch.long,
                 "every frame must be a torch.long tensor")
        _require(frame.ndim == 3 and all(n > 0 for n in frame.shape),
                 "frames must have nonempty [batch,height,width] shape")
        if shape is None:
            shape = tuple(frame.shape)
        _require(tuple(frame.shape) == shape, "frame shapes differ")
        _require(frame.device == embedding.weight.device, "frame/model devices differ")
        _require(bool(((frame >= 0) & (frame < embedding.num_embeddings)).all()),
                 "frame token is outside the model vocabulary")
    return shape


def _state(state, shape, parameter):
    if state is not None:
        _require(isinstance(state, torch.Tensor) and tuple(state.shape) == tuple(shape),
                 "initial or intervened state has the wrong shape")
        _require(state.dtype == parameter.dtype and state.device == parameter.device,
                 "state/model dtype or device differs")
        _require(bool(torch.isfinite(state).all()), "state must be finite")
    return state


def _intervention(value, count, batch):
    if value is None:
        return None
    _require(isinstance(value, dict), "intervention must be a mapping")
    kind = value.get("kind")
    expected = {"before_frame", "kind"} | ({"permutation"} if kind == "swap" else set())
    _require(kind in ("zero", "swap") and set(value) == expected,
             "intervention must specify only zero or swap at a frame boundary")
    index = value["before_frame"]
    _require(type(index) is int and 0 <= index < count, "invalid intervention frame index")
    if kind == "swap":
        permutation = value["permutation"]
        _require(isinstance(permutation, (list, tuple))
                 and all(type(v) is int for v in permutation)
                 and sorted(permutation) == list(range(batch)),
                 "swap permutation must be a complete batch bijection")
    return value


def _apply(state, intervention, axis):
    _require(state is not None, "intervention requires an existing carried state")
    if intervention["kind"] == "zero":
        return torch.zeros_like(state)
    order = torch.tensor(intervention["permutation"], dtype=torch.long, device=state.device)
    return state.index_select(axis, order)


def strict_np_stream(model, frames, steps, *, initial_state=None,
                     intervention=None, capture_states=False, out_pos=None):
    """Run existing local NP dynamics with explicit, optional state continuation.

    Interventions occur before the indexed frame's seed is added. Captured
    frame_states have shape [B,F,C,H,W] and are detached snapshots after each
    frame. The returned state retains its graph for training/continuation.
    """
    batch, height, width = _frames(frames, model.embed)
    _require(isinstance(steps, (list, tuple)) and len(steps) == len(frames),
             "frame and step inventories must have the same nonzero length")
    _require(all(type(n) is int and n > 0 for n in steps),
             "each frame requires a positive integer update count")
    position = model.out_pos if out_pos is None else out_pos
    _require(isinstance(position, (list, tuple)) and len(position) == 2
             and all(type(n) is int for n in position)
             and 0 <= position[0] < height and 0 <= position[1] < width,
             "readout position is outside the frame")
    state_shape = (batch, model.seed.out_channels, height, width)
    s = _state(initial_state, state_shape, model.seed.weight)
    intervention = _intervention(intervention, len(frames), batch)
    snapshots = []
    for index, (canvas, count) in enumerate(zip(frames, steps)):
        if intervention is not None and index == intervention["before_frame"]:
            s = _apply(s, intervention, 0)
        ids = F.embedding(canvas, model.dictionary()).permute(0, 3, 1, 2)
        new = model.seed(ids) * (canvas != 0).unsqueeze(1)
        s = new if s is None else s + new
        for _ in range(count):
            h = torch.cat([s, model.perceive(s), ids], 1)
            ds = model.f2(F.relu(model.f1(h)))
            if model.training and model.fire_rate < 1:
                ds = ds * (torch.rand_like(ds[:, :1]) < model.fire_rate)
            s = s + ds
        if capture_states:
            snapshots.append(s.detach().clone())
    row, col = position
    logits = model.read(s[:, :, row, col]) @ model.dictionary().T
    logits[:, 0] = -1e4
    result = {"logits": logits, "state": s}
    if capture_states:
        result["frame_states"] = torch.stack(snapshots, 1)
    return result


def strict_gru_stream(model, frames, *, initial_state=None,
                      intervention=None, capture_states=False):
    """Continue the existing one-layer FrameGRU with caller-owned hidden state.

    State shape is [1,B,Hid]; captured frame_states are [B,F,Hid]. Without an
    intervention, one GRU call processes the full sequence, as forward_frames
    does. Prefix/suffix GRU calls implement a declared boundary intervention.
    """
    batch, height, width = _frames(frames, model.emb)
    _require(height * width * model.emb.embedding_dim == model.proj.in_features,
             "frame area differs from the configured FrameGRU projection")
    _require(model.gru.num_layers == 1 and not model.gru.bidirectional,
             "this helper supports the existing one-layer unidirectional FrameGRU")
    s = _state(initial_state, (1, batch, model.gru.hidden_size), model.gru.weight_hh_l0)
    intervention = _intervention(intervention, len(frames), batch)
    x = torch.stack([F.relu(model.proj(model.emb(f).flatten(1))) for f in frames], 1)
    if intervention is None:
        outputs, s = model.gru(x, s)
    else:
        boundary = intervention["before_frame"]
        parts = []
        if boundary:
            prefix, s = model.gru(x[:, :boundary], s)
            parts.append(prefix)
        s = _apply(s, intervention, 1)
        suffix, s = model.gru(x[:, boundary:], s)
        parts.append(suffix)
        outputs = torch.cat(parts, 1)
    logits = model.head(s[-1])
    logits[:, 0] = -1e4
    result = {"logits": logits, "state": s}
    if capture_states:
        result["frame_states"] = outputs.detach().clone()
    return result
