"""Separate prospective query-conditioned readout; never used by frozen U16.

The extra global communication path is explicit. Uniform pooling computes the
same Q/K projections but does not use them to choose weights, so its effective
capacity differs; a nominal parameter match is not an effective-capacity claim.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from neuropixel.research.models import ResearchNCA


class QueryAttentionReadout(nn.Module):
    """Visible query identity attends to recurrent states plus visible identities.

    Query and empty cells are excluded as keys. No targets, role assignments,
    generator metadata or split membership enter this module. States are learned
    representations, not oracle role/value pairings. No positional bias or causal
    temporal mask is silently added to this static visible-grid task.
    """

    def __init__(self, c=48, c_id=16, d=16, heads=4, mode='query'):
        super().__init__()
        if min(c, c_id, d, heads) < 1 or d % heads:
            raise ValueError('Positive dimensions and d divisible by heads required')
        if mode not in ('query', 'uniform'):
            raise ValueError('Unknown attention control mode')
        self.c, self.c_id, self.d, self.heads, self.mode = c, c_id, d, heads, mode
        self.query = nn.Linear(c_id, d)
        self.key = nn.Linear(c + c_id, d)
        self.value = nn.Linear(c + c_id, d)
        self.output = nn.Linear(d, c)

    def forward(self, state, identities, canvas, query_pos):
        if canvas.ndim != 3 or canvas.dtype != torch.long:
            raise ValueError('Visible canvas must be int64 [batch,height,width]')
        batch, h, w = canvas.shape
        if min(batch, h, w) < 1:
            raise ValueError('Nonempty input required')
        if state.shape != (batch, self.c, h, w) or identities.shape != (batch, self.c_id, h, w):
            raise ValueError('State/identity shapes differ from visible canvas')
        r, col = query_pos
        if not (0 <= r < h and 0 <= col < w):
            raise ValueError('Query position outside canvas')
        if (state.device != identities.device or canvas.device != state.device
                or state.dtype != identities.dtype):
            raise ValueError('Inputs must share device and floating dtype')
        if not state.is_floating_point() or not torch.isfinite(state).all() or not torch.isfinite(identities).all():
            raise ValueError('Finite floating representations required')
        visible = (canvas != 0).clone()
        visible[:, r, col] = False
        valid = visible.flatten(1)
        features = torch.cat([state, identities], 1).permute(0, 2, 3, 1).reshape(batch, h*w, -1)
        width = self.d // self.heads
        query = self.query(identities[:, :, r, col]).reshape(batch, self.heads, 1, width)
        keys = self.key(features).reshape(batch, h*w, self.heads, width).transpose(1, 2)
        values = self.value(features).reshape(batch, h*w, self.heads, width).transpose(1, 2)
        scores = (query @ keys.transpose(-1, -2)) / math.sqrt(width)
        mask = valid[:, None, None, :]
        if self.mode == 'query':
            scores = scores.masked_fill(~mask, -torch.inf)
            scores = torch.where(valid.any(1)[:, None, None, None], scores, torch.zeros_like(scores))
            weights = scores.softmax(-1) * mask.to(scores.dtype)
        else:
            weights = mask.to(scores.dtype) / valid.sum(1).clamp_min(1)[:, None, None, None]
            weights = weights.expand(batch, self.heads, 1, h*w)
        context = (weights @ values).transpose(1, 2).reshape(batch, self.d)
        delta = self.output(context) * valid.any(1)[:, None].to(context.dtype)
        return {'delta': delta, 'weights': weights.squeeze(2), 'valid_keys': valid}


class AttentiveResearchNCA(ResearchNCA):
    """Experimental hybrid; original recurrent state and lens remain explicit.

    New head: 3,168 nominal parameters at C48/Cid16/d16, total32,992 for
    the audited default backbone. This class is not a registered recipe or a
    demonstrated improvement. RGB/retina inputs are deliberately unsupported.
    """

    def __init__(self, vocab, out_pos, *, query_pos=(7, 6), attention_mode='query',
                 attention_d=16, attention_heads=4, **backbone):
        super().__init__(vocab, out_pos, **backbone)
        self.query_pos = query_pos
        self.attention_head = QueryAttentionReadout(
            self.read.in_features, self.read.out_features,
            attention_d, attention_heads, attention_mode)

    def forward(self, canvas, trace=False, lens_every=0, *, out_pos=None, steps=None, hook=None):
        output = super().forward(canvas, trace=trace, lens_every=lens_every,
                                 out_pos=out_pos, steps=steps, hook=hook)
        identities = F.embedding(canvas, self.dictionary()).permute(0, 3, 1, 2)
        attention = self.attention_head(output['state'], identities, canvas, self.query_pos)
        r, col = out_pos or self.out_pos
        readout_state = output['state'][:, :, r, col] + attention['delta']
        logits = self.read(readout_state) @ self.decoder_dictionary().T
        logits[:, 0] = -1e4
        output.update(local_logits=output['logits'], logits=logits,
                      readout_state=readout_state, attention_weights=attention['weights'],
                      attention_valid_keys=attention['valid_keys'])
        return output
