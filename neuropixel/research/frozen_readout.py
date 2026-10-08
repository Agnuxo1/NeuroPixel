"""Readout-only causal diagnostic helpers, separate from the closed U16 recipe."""
from __future__ import annotations

import torch
from torch import nn

from neuropixel.research.query_attention import QueryAttentionReadout
from neuropixel.research.query_attention_controls import QueryConditionedLocalReadout
from neuropixel.research.qtrain_budget import state_digest

QUERY_POS, OUTPUT_POS = (7, 6), (7, 7)


def body_digest(backbone):
    return state_digest(backbone.state_dict())


def pack_visible_state(state, canvas):
    """Keep exactly the states consumed by current readouts, using input-only keys.

    Empty/query states are not consumed by the attention heads. The output state
    is retained separately for the residual/local control. No labels enter here.
    """
    if canvas.ndim != 3 or canvas.shape[1:] != (8, 8) or canvas.dtype != torch.long:
        raise ValueError('Native visible int64 8x8 canvases required')
    if state.shape != (len(canvas), 48, 8, 8) or not torch.isfinite(state).all():
        raise ValueError('Finite 48-channel native states required')
    mask = (canvas != 0).clone()
    mask[:, QUERY_POS[0], QUERY_POS[1]] = False
    if not (mask.flatten(1).sum(1) == 8).all():
        raise ValueError('Exactly eight visible fact cells outside query required')
    positions = mask.flatten(1).nonzero(as_tuple=False)[:, 1].reshape(len(canvas), 8)
    flat = state.permute(0, 2, 3, 1).reshape(len(canvas), 64, 48)
    facts = flat.gather(1, positions[:, :, None].expand(-1, -1, 48))
    return dict(canvas=canvas.detach().clone(), positions=positions.detach().clone(),
                facts=facts.detach().clone(), local=state[:, :, OUTPUT_POS[0], OUTPUT_POS[1]].detach().clone())


def unpack_visible_state(packed):
    canvas, positions, facts, local = (packed[k] for k in ('canvas', 'positions', 'facts', 'local'))
    if (positions.shape != (len(canvas), 8) or facts.shape != (len(canvas), 8, 48)
            or local.shape != (len(canvas), 48) or canvas.shape[1:] != (8, 8)
            or canvas.dtype != torch.long or positions.dtype != torch.long
            or not torch.isfinite(facts).all() or not torch.isfinite(local).all()):
        raise ValueError('Packed state contract differs')
    mask = (canvas != 0).clone()
    mask[:, QUERY_POS[0], QUERY_POS[1]] = False
    if not (mask.flatten(1).sum(1) == 8).all():
        raise ValueError('Visible fact count differs')
    expected = mask.flatten(1).nonzero(as_tuple=False)[:, 1].reshape(len(canvas), 8)
    if not torch.equal(positions, expected):
        raise ValueError('Packed positions do not match visible input')
    flat = torch.zeros((len(canvas), 64, 48), dtype=facts.dtype, device=facts.device)
    flat.scatter_(1, positions[:, :, None].expand(-1, -1, 48), facts)
    flat[:, OUTPUT_POS[0]*8+OUTPUT_POS[1]] = local
    return flat.reshape(len(canvas), 8, 8, 48).permute(0, 3, 1, 2)


class FrozenReadout(nn.Module):
    """Only new head parameters may receive optimizer updates; decoder stays fixed."""

    def __init__(self, backbone, mode):
        super().__init__()
        if mode not in ('query_attention', 'uniform_global', 'local_query'):
            raise ValueError('Unknown readout diagnostic variant')
        self.backbone, self.mode = backbone, mode
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.head = (QueryConditionedLocalReadout() if mode == 'local_query' else
                     QueryAttentionReadout(mode='query' if mode == 'query_attention' else 'uniform'))
        self.initial_body_digest = body_digest(backbone)

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.backbone.parameters()):
            raise ValueError('Original parameter gradient/freeze contract violated')
        if body_digest(self.backbone) != self.initial_body_digest:
            raise ValueError('Original backbone/decoder parameters or buffers changed')

    def forward(self, packed):
        self.assert_frozen()
        canvas = packed['canvas']
        state = unpack_visible_state(packed)
        identities = torch.nn.functional.embedding(canvas, self.backbone.dictionary()).permute(0, 3, 1, 2)
        local = packed['local']
        if self.mode == 'local_query':
            out = self.head(local, identities[:, :, QUERY_POS[0], QUERY_POS[1]])
            identity = self.backbone.read(local + out['delta']) * out['identity_gate']
        else:
            out = self.head(state, identities, canvas, QUERY_POS)
            identity = self.backbone.read(local + out['delta'])
        logits = identity @ self.backbone.decoder_dictionary().T
        logits[:, 0] = -1e4
        return logits

    def head_parameters(self):
        self.assert_frozen()
        return list(self.head.parameters())
