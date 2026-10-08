"""Active local capacity/query-access control for the separate attention study.

Default head nominal count is exactly3,168, as in QueryAttentionReadout. This
does not equate effective function capacity, optimization, or arithmetic. Unlike
uniform global pooling, this control has no aggregation over distant cells.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from neuropixel.research.models import ResearchNCA


class QueryConditionedLocalReadout(nn.Module):
    """Local output residual plus visible-query-conditioned identity-channel gate."""

    def __init__(self, c=48, c_id=16, hidden=32):
        super().__init__()
        if min(c, c_id, hidden) < 1:
            raise ValueError('Positive channel dimensions required')
        self.c, self.c_id = c, c_id
        self.expand = nn.Linear(c, hidden)
        self.project = nn.Linear(hidden, c)
        self.query_scale = nn.Parameter(torch.zeros(c_id))

    def forward(self, local_state, query_identity):
        if (local_state.ndim != 2 or query_identity.ndim != 2
                or local_state.shape[1] != self.c or query_identity.shape != (len(local_state), self.c_id)
                or not len(local_state)):
            raise ValueError('Matching nonempty local-state/query identity batches required')
        if (local_state.device != query_identity.device or local_state.dtype != query_identity.dtype
                or not local_state.is_floating_point() or not torch.isfinite(local_state).all()
                or not torch.isfinite(query_identity).all()):
            raise ValueError('Finite floating inputs on the same dtype/device required')
        delta = self.project(F.relu(self.expand(local_state)))
        gate = 1 + torch.tanh(self.query_scale * query_identity)
        return {'delta': delta, 'identity_gate': gate}


class QueryConditionedLocalNCA(ResearchNCA):
    """Shares the unchanged recurrent backbone/lens and output dictionary.

    Direct visible query access is present in both this control and the attention
    head. All extra parameters participate in the forward function; there is no
    padding with unused parameters to claim matching. The gate starts neutral
    but can receive gradients immediately. At defaults the total count is32,992.
    """

    def __init__(self, vocab, out_pos, *, query_pos=(7, 6), readout_hidden=32, **backbone):
        super().__init__(vocab, out_pos, **backbone)
        self.query_pos = query_pos
        self.local_head = QueryConditionedLocalReadout(
            self.read.in_features, self.read.out_features, readout_hidden)

    def forward(self, canvas, trace=False, lens_every=0, *, out_pos=None, steps=None, hook=None):
        output = super().forward(canvas, trace=trace, lens_every=lens_every,
                                 out_pos=out_pos, steps=steps, hook=hook)
        r, col = self.query_pos
        if not (0 <= r < canvas.shape[1] and 0 <= col < canvas.shape[2]):
            raise ValueError('Query position outside canvas')
        identities = F.embedding(canvas, self.dictionary()).permute(0, 3, 1, 2)
        out_r, out_col = out_pos or self.out_pos
        local_state = output['state'][:, :, out_r, out_col]
        head = self.local_head(local_state, identities[:, :, r, col])
        readout_state = local_state + head['delta']
        readout_identity = self.read(readout_state) * head['identity_gate']
        logits = readout_identity @ self.decoder_dictionary().T
        logits[:, 0] = -1e4
        output.update(local_logits=output['logits'], logits=logits,
                      readout_state=readout_state, identity_gate=head['identity_gate'])
        return output
