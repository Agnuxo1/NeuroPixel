"""Independent arithmetic fixtures before any scientific GLOB05 head training."""
import os,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from neuropixel.research.qtrain_budget import require_ram
require_ram()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
from test_GLOB05_control import TinyBody
from test_GLOB05_resume import fixture
from neuropixel.research.query_gated_uniform import FrozenGatedUniformReadout
from neuropixel.research.readout_experiment import select_bank
from replay_GLOB05 import independent_logits

class Contracts(unittest.TestCase):
    def test_projection_formula_matches_wrapper_and_layout(self):
        for layout in (False,True):
            body,banks,_=fixture();reader=FrozenGatedUniformReadout(body);bank=banks['probe'];bank['channels_last']=layout
            for k in (0,7):
                indices=torch.arange(64);actual=reader(select_bank(bank,indices,k));replayed=independent_logits(body,reader.head.state_dict(),bank,k,0,64)
                self.assertTrue(torch.equal(actual,replayed));self.assertTrue(torch.equal(actual.argmax(-1),replayed.argmax(-1)))

    def test_query_flip_controls_and_partial_batches(self):
        body,banks,_=fixture();reader=FrozenGatedUniformReadout(body)
        for view in ('probe','probe_query_flip'):
            bank=banks[view];indices=torch.arange(7,31);actual=reader(select_bank(bank,indices,3));replayed=independent_logits(body,reader.head.state_dict(),bank,3,7,31)
            self.assertTrue(torch.equal(actual,replayed))

    def test_padding_and_roles_not_oracle_inputs(self):
        body,banks,_=fixture();reader=FrozenGatedUniformReadout(body);bank=banks['probe'];before=independent_logits(body,reader.head.state_dict(),bank,0,0,64)
        bank['target']=bank['target'].roll(1);bank['roles']=bank['roles'].roll(1)
        after=independent_logits(body,reader.head.state_dict(),bank,0,0,64)
        self.assertTrue(torch.equal(before,after));self.assertTrue((before[:,0]==-1e4).all())

if __name__=='__main__':unittest.main()
