"""Runner contracts on a tiny synthetic fixture, not scientific study outcomes."""
import importlib.util,json,pathlib,tempfile,unittest
from unittest.mock import patch
import numpy as np
import torch
root=pathlib.Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('opt03',root/'scripts/research_OPT03_development.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class TinyTask:
    def __init__(self,*args,**kwargs):pass
    def sample(self,n,split,generator,device):
        assert split=='train'
        return torch.randint(1,35,(n,8,8),generator=generator),torch.randint(1,35,(n,),generator=generator)
class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed=torch.nn.Embedding(35,8);self.head=torch.nn.Linear(8,35)
    def forward(self,x,lens_every=4):
        h=self.embed(x);h=h*(torch.rand(x.shape)[...,None]>.5);lens=self.head(h)
        return {'logits':lens.mean((1,2)),'lens':lens[:,None]}
def datasets(task,split,n,seed,output):
    assert split in ('train','validation')
    return (torch.ones(4,8,8,dtype=torch.long),torch.ones(4,dtype=torch.long),torch.arange(4)),{}
def trees_equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(trees_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(trees_equal(x,y) for x,y in zip(a,b))
    return a==b

class Contracts(unittest.TestCase):
    def setUp(self):
        folder=root/'results/research/OPT03_contracts';folder.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=folder);self.p=pathlib.Path(self.tmp.name);self.plan=self.p/'plan.json'
        self.plan.write_text(json.dumps({'status':'frozen','runs':m.inventory(),'source_sha256':{}}))
    def tearDown(self):self.tmp.cleanup()
    def test_inventory(self):
        rows=m.inventory();self.assertEqual(len(rows),12);self.assertEqual(len({x['run_id'] for x in rows}),12);self.assertEqual(sum(x['updates'] for x in rows),98304)
    def test_unfrozen_and_changed_source_rejected(self):
        d=json.loads(self.plan.read_text());d['status']='draft';self.plan.write_text(json.dumps(d))
        with self.assertRaises(ValueError):m.load_plan(self.plan)
        d['status']='frozen';d['source_sha256']={'neuropixel/research/models.py':'0'*64};self.plan.write_text(json.dumps(d))
        with self.assertRaises(ValueError):m.load_plan(self.plan)
    def test_low_ram_refused_before_training(self):
        class Ram:available=int(7.9*2**30)
        with patch('psutil.virtual_memory',return_value=Ram()):
            with self.assertRaisesRegex(RuntimeError,'RAM8GiB'):m.run_one(m.inventory()[0],self.p/'blocked',self.plan,'cpu',128)
        self.assertFalse((self.p/'blocked').exists())
    def test_interrupt_resume_matches_uninterrupted_toy(self):
        with patch('neuropixel.research.models.build_model',side_effect=lambda *a,**k:TinyModel()),patch('neuropixel.research.data.ResearchRoleTask',TinyTask),patch('neuropixel.research.experiment.dataset_artifact',datasets):
            conf=m.inventory()[0];m.run_one(conf,self.p/'full',self.plan,'cpu',256)
            m.run_one(conf,self.p/'resume',self.plan,'cpu',128);m.run_one(conf,self.p/'resume',self.plan,'cpu',256)
        a=torch.load(self.p/'full/checkpoint.pt',weights_only=True);b=torch.load(self.p/'resume/checkpoint.pt',weights_only=True)
        for field in ['model','optimizer','cpu_rng','sampler_rng','cuda_rng']:
            self.assertTrue(trees_equal(a[field],b[field]),field)
        self.assertEqual(a['update'],256);self.assertEqual(b['update'],256)
        self.assertFalse(json.loads((self.p/'resume/progress.json').read_text())['test_accessed'])
        self.plan.write_text(json.dumps({'status':'frozen','runs':m.inventory(),'source_sha256':{},'changed':True}))
        with self.assertRaisesRegex(ValueError,'identity/environment'):
            m.run_one(conf,self.p/'resume',self.plan,'cpu',384)

if __name__=='__main__':
    runner=unittest.TextTestRunner(verbosity=2);out=runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    p=root/'results/research/OPT03_contracts/receipt.json';p.write_text(json.dumps({'status':'passed' if out.wasSuccessful() else 'failed','tests':out.testsRun,'failures':len(out.failures),'errors':len(out.errors),'scope':'Synthetic tiny-model operational contracts; no NeuroPixel scientific development training'},indent=2)+'\n')
    raise SystemExit(0 if out.wasSuccessful() else 1)
