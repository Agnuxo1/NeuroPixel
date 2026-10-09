"""GLOB05 active query-conditioned global-pooling control, outside frozen DEV04.

Same Q/K/V/output dimensions and initial tensors as the attention head. Queries
gate pooled global features; they never choose spatially non-uniform weights.
All projection families remain trainable, unlike the old inactive-QK uniform arm.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from neuropixel.research.query_attention import QueryAttentionReadout
from neuropixel.research.frozen_readout import body_digest,unpack_visible_state,QUERY_POS,OUTPUT_POS


class QueryGatedUniformReadout(QueryAttentionReadout):
    def __init__(self,c=48,c_id=16,d=16,heads=4):
        super().__init__(c=c,c_id=c_id,d=d,heads=heads,mode='query')

    def forward(self,state,identities,canvas,query_pos):
        if canvas.ndim!=3 or canvas.dtype!=torch.long:raise ValueError('Visible int64 canvas required')
        batch,h,w=canvas.shape;r,col=query_pos
        if min(batch,h,w)<1 or not (0<=r<h and 0<=col<w):raise ValueError('Nonempty valid query position required')
        if state.shape!=(batch,self.c,h,w) or identities.shape!=(batch,self.c_id,h,w):raise ValueError('State/identity shapes differ')
        if state.device!=identities.device or canvas.device!=state.device or state.dtype!=identities.dtype:raise ValueError('Common device and floating dtype required')
        if not state.is_floating_point() or not torch.isfinite(state).all() or not torch.isfinite(identities).all():raise ValueError('Finite floating representations required')
        visible=(canvas!=0).clone();visible[:,r,col]=False;valid=visible.flatten(1)
        features=torch.cat((state,identities),1).permute(0,2,3,1).reshape(batch,h*w,-1)
        width=self.d//self.heads
        q=self.query(identities[:,:,r,col]).reshape(batch,self.heads,width)
        k=self.key(features).reshape(batch,h*w,self.heads,width).transpose(1,2)
        v=self.value(features).reshape(batch,h*w,self.heads,width).transpose(1,2)
        weights=valid[:,None,:].to(state.dtype)/valid.sum(1).clamp_min(1)[:,None,None]
        weights=weights.expand(batch,self.heads,h*w)
        mean_k=(weights[:,:,:,None]*k).sum(2)
        mean_v=(weights[:,:,:,None]*v).sum(2)
        gate=1+torch.tanh(q*mean_k/math.sqrt(width))
        context=(mean_v*gate).reshape(batch,self.d)
        delta=self.output(context)*valid.any(1)[:,None].to(context.dtype)
        return dict(delta=delta,weights=weights,valid_keys=valid,global_query_gate=gate)


class FrozenGatedUniformReadout(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone;self.mode='gated_uniform_global'
        for parameter in backbone.parameters():parameter.requires_grad_(False);parameter.grad=None
        self.head=QueryGatedUniformReadout();self.initial_body_digest=body_digest(backbone)

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.backbone.parameters()):raise ValueError('Body gradient/freeze violation')
        if body_digest(self.backbone)!=self.initial_body_digest:raise ValueError('Body/decoder changed')

    def head_parameters(self):
        self.assert_frozen();return list(self.head.parameters())

    def forward(self,packed):
        self.assert_frozen();canvas=packed['canvas'];state=unpack_visible_state(packed)
        ids=F.embedding(canvas,self.backbone.dictionary()).permute(0,3,1,2)
        out=self.head(state,ids,canvas,QUERY_POS);r,col=OUTPUT_POS
        identity=self.backbone.read(state[:,:,r,col]+out['delta'])
        logits=identity@self.backbone.decoder_dictionary().T;logits[:,0]=-1e4
        return logits
