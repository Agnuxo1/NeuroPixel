"""Synthetic-only contracts for GLOB05; admission before Torch, no scientific fits."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import pathlib
import sys
import unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neuropixel.research.qtrain_budget import require_ram
require_ram()
import torch
from torch import nn
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
from neuropixel.research.query_attention import QueryAttentionReadout
from neuropixel.research.query_gated_uniform import QueryGatedUniformReadout,FrozenGatedUniformReadout
from neuropixel.research.frozen_readout import pack_visible_state,body_digest
from neuropixel.research.qtrain_budget import state_digest
from neuropixel.research.readout_experiment import training_state,restore

class TinyBody(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Parameter(torch.randn(35,16)*.1);self.read=nn.Linear(48,16,bias=False)
    def dictionary(self):return self.embedding
    def decoder_dictionary(self):return self.embedding

class Contracts(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(9901);state=torch.randn(4,48,8,8);ids=torch.randn(4,16,8,8);x=torch.zeros(4,8,8,dtype=torch.long)
        coords=[(1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(4,1),(4,2)]
        for j,(r,c) in enumerate(coords):x[:,r,c]=j+1
        x[:,7,6]=1
        return state,ids,x,coords

    def test_capacity_and_identical_initial_tensors(self):
        torch.manual_seed(700);attention=QueryAttentionReadout();torch.manual_seed(700);control=QueryGatedUniformReadout()
        self.assertEqual(sum(p.numel() for p in control.parameters()),3168)
        self.assertEqual(state_digest(control.state_dict()),state_digest(attention.state_dict()))

    def test_all_projection_families_receive_response_gradient(self):
        s,ids,x,_=self.fixture();head=QueryGatedUniformReadout();head(s,ids,x,(7,6))['delta'].square().sum().backward()
        self.assertEqual(sum(p.numel() for p in head.parameters() if p.grad is not None),3168)
        for module in (head.query,head.key,head.value,head.output):self.assertGreater(float(module.weight.grad.abs().sum()),0.)

    def test_uniform_spatial_weights_but_active_query_gate(self):
        s,ids,x,_=self.fixture();head=QueryGatedUniformReadout();a=head(s,ids,x,(7,6));changed=ids.clone();changed[:,:,7,6]+=2.;b=head(s,changed,x,(7,6))
        self.assertTrue(torch.equal(a['weights'],b['weights']));self.assertTrue(torch.equal(a['weights'].sum(-1),torch.ones(4,4)))
        self.assertEqual(torch.unique(a['weights'][a['weights']>0]).tolist(),[.125]);self.assertFalse(torch.allclose(a['delta'],b['delta']))

    def test_query_state_and_padding_do_not_enter_global_pool(self):
        s,ids,x,_=self.fixture();head=QueryGatedUniformReadout();a=head(s,ids,x,(7,6))['delta'];ss=s.clone();ii=ids.clone()
        unused=x==0;ss.permute(0,2,3,1)[unused]+=5.;ii.permute(0,2,3,1)[unused]-=3.;ss[:,:,7,6]+=7.
        self.assertTrue(torch.equal(a,head(ss,ii,x,(7,6))['delta']))

    def test_pool_invariant_to_visible_cell_permutation(self):
        s,ids,x,coords=self.fixture();head=QueryGatedUniformReadout().double();s=s.double();ids=ids.double();a=head(s,ids,x,(7,6))['delta'];ss=s.clone();ii=ids.clone();xx=x.clone()
        for dst,src in zip(coords,coords[1:]+coords[:1]):
            r,c=dst;u,v=src;ss[:,:,r,c]=s[:,:,u,v];ii[:,:,r,c]=ids[:,:,u,v];xx[:,r,c]=x[:,u,v]
        self.assertTrue(torch.allclose(a,head(ss,ii,xx,(7,6))['delta'],rtol=0,atol=1e-12))

    def test_empty_global_keys_and_bad_inputs(self):
        s,ids,x,_=self.fixture();head=QueryGatedUniformReadout();empty=torch.zeros_like(x);empty[:,7,6]=1;out=head(s,ids,empty,(7,6))
        self.assertTrue(torch.equal(out['delta'],torch.zeros(4,48)));self.assertEqual(float(out['weights'].sum()),0.)
        bad=s.clone();bad[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):head(bad,ids,x,(7,6))
        with self.assertRaises(ValueError):head(s,ids,x.float(),(7,6))

    def test_body_freeze_and_exact_head_state_restore(self):
        s,_,x,_=self.fixture();body=TinyBody();reader=FrozenGatedUniformReadout(body);before=body_digest(body);packed=pack_visible_state(s,x)
        optimizer=torch.optim.AdamW(reader.head_parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001);context_rng=torch.Generator().manual_seed(89004);mask_rng=torch.Generator().manual_seed(89005)
        loss=torch.nn.functional.cross_entropy(reader(packed),torch.tensor([5,6,7,8]));loss.backward();optimizer.step();reader.assert_frozen();self.assertEqual(body_digest(body),before)
        state=training_state(reader,optimizer,context_rng,mask_rng);expected=state_digest(state);clone=TinyBody();clone.load_state_dict(body.state_dict());resumed=FrozenGatedUniformReadout(clone);opt=torch.optim.AdamW(resumed.head_parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001);a,b=torch.Generator(),torch.Generator();restore(resumed,opt,a,b,state)
        self.assertEqual(state_digest(training_state(resumed,opt,a,b)),expected);self.assertTrue(torch.equal(reader(packed),resumed(packed)))

if __name__=='__main__':unittest.main()
