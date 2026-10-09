"""Tiny independent recipe tests exact durable continuation and closed-prefix guards."""
import os,pathlib,sys,tempfile,json,hashlib,unittest,copy
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from neuropixel.research.qtrain_budget import require_ram
require_ram()
import torch
from neuropixel.research.glob05_experiment import fit_control,validate_head
from neuropixel.research.readout_experiment import batch_indices
from neuropixel.research.qtrain_budget import state_digest
from neuropixel.research.experiment import configure_runtime,environment_record
from test_GLOB05_control import TinyBody
DEST=ROOT/'results/research/GLOB05_fixture_resume';DEST.mkdir(parents=True,exist_ok=True)
ENV=environment_record(configure_runtime('cpu',2))

def fixture():
    torch.manual_seed(9201);body=TinyBody();x=torch.zeros(64,8,8,dtype=torch.long);roles=torch.arange(4).repeat(16)
    for j in range(4):x[:,j+1,1]=j+1;x[:,j+1,2]=[5,17,6,27][j]
    x[:,7,6]=roles+1;positions=(x!=0).clone();positions[:,7,6]=False;positions=positions.flatten(1).nonzero()[:,1].reshape(64,8)
    bank=dict(canvas=x,target=torch.tensor([5,17,6,27])[roles],roles=roles,positions=positions,
        facts=torch.randn(8,64,8,48)*.1,local=torch.randn(8,64,48)*.1,channels_last=False,mask_seeds=list(range(94000,94008)))
    counter={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in bank.items()};counter['roles']=torch.where((roles==0)|(roles==2),2-roles,roles);counter['canvas'][:,7,6]=counter['roles']+1;counter['target']=torch.tensor([5,17,6,27])[counter['roles']]
    banks=dict(train=bank,probe=bank,validation=bank,probe_query_flip=counter,validation_query_flip=counter)
    a,b=torch.Generator().manual_seed(89004),torch.Generator().manual_seed(89005);w=hashlib.sha256();expected=[]
    for update in range(1,257):
        indices,masks=batch_indices(bank,a,b);value=hashlib.sha256(indices.numpy().tobytes()+masks.numpy().tobytes()).hexdigest();w.update(value.encode())
        if update%128==0:expected.append(w.hexdigest());w=hashlib.sha256()
    return body,banks,expected

class Contracts(unittest.TestCase):
    def config(self):return dict(parent_case='fixture_GLOB05',mode='gated_uniform_global',head_seed=700,updates=256)
    def test_exact_interrupted_resume_matches_uninterrupted_real_states(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            b,banks,w=fixture();first=pathlib.Path(d)/'full';fit_control(b,banks,first,self.config(),'fixture-plan',ENV,{'fixture':'separate'},w,endpoints=(128,256))
            b,banks,w=fixture();second=pathlib.Path(d)/'split';self.assertIsNone(fit_control(b,banks,second,self.config(),'fixture-plan',ENV,{'fixture':'separate'},w,endpoints=(128,256),stop_after=128))
            old=(second/'checkpoint_u128.pt').read_bytes();fit_control(b,banks,second,self.config(),'fixture-plan',ENV,{'fixture':'separate'},w,endpoints=(128,256))
            full=torch.load(first/'checkpoint.pt',weights_only=True);split=torch.load(second/'checkpoint.pt',weights_only=True)
            self.assertEqual(full['training_state_digest'],split['training_state_digest']);self.assertEqual((second/'checkpoint_u128.pt').read_bytes(),old)
            self.assertEqual([r['update'] for r in split['curve']],[128,256]);before=(second/'checkpoint.pt').read_bytes();fit_control(b,banks,second,self.config(),'fixture-plan',ENV,{'fixture':'separate'},w,endpoints=(128,256));self.assertEqual((second/'checkpoint.pt').read_bytes(),before)

    def test_wrong_stream_and_altered_resume_rejected(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            b,banks,w=fixture();folder=pathlib.Path(d)/'changed';fit_control(b,banks,folder,self.config(),'fixture-plan',ENV,{},w,endpoints=(128,256),stop_after=128)
            payload=torch.load(folder/'checkpoint.pt',weights_only=True);bad=copy.deepcopy(payload);bad['cpu_rng'][0]^=1
            with self.assertRaises(ValueError):validate_head(bad,self.config(),'fixture-plan',payload['frozen_body_digest'],ENV)
            bad=copy.deepcopy(payload);bad['optimizer']['param_groups'][0]['lr']=.01
            with self.assertRaises(ValueError):validate_head(bad,self.config(),'fixture-plan',payload['frozen_body_digest'],ENV)
            with self.assertRaises(ValueError):fit_control(b,banks,folder,self.config(),'other-plan',ENV,{},w,endpoints=(128,256))
            b,banks,w=fixture();w[0]='0'*64
            with self.assertRaises(ValueError):fit_control(b,banks,pathlib.Path(d)/'wrong-witness',self.config(),'fixture-plan',ENV,{},w,endpoints=(128,256))

    def test_scientific_budget_not_fixture_budget(self):
        b,banks,w=fixture();c=self.config();c['parent_case']='DEV04_partition101_seed400'
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            with self.assertRaises(ValueError):fit_control(b,banks,pathlib.Path(d),c,'fixture-plan',ENV,{},w,endpoints=(128,256))

if __name__=='__main__':unittest.main()
