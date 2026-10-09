"""Scientific-control and operational contracts; only native sampling and toy fitting."""
import hashlib,importlib.util,json,pathlib,tempfile,unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('runner',ROOT/'scripts/research_OPT03_qtrain.py');runner=importlib.util.module_from_spec(s);s.loader.exec_module(runner);q=runner.operations
from neuropixel.research.data import ResearchRoleTask

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.e=torch.nn.Embedding(35,4);self.h=torch.nn.Linear(4,35);self.fire_rate=.5
    def forward(self,x,lens_every=0):
        h=self.e(x)
        if self.training:h=h*(torch.rand(x.shape)[...,None]>.5)
        logits=self.h(h);out={'logits':logits.mean((1,2))}
        if lens_every:out['lens']=logits[:,None]
        return out
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

class QTrain(unittest.TestCase):
    def setUp(self):
        folder=ROOT/'results/research/OPT03_QTRAIN_contracts';folder.mkdir(parents=True,exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=folder);self.p=pathlib.Path(self.tmp.name);self.plan=self.p/'plan.json'
        self.plan.write_text(json.dumps({'status':'frozen','runs':q.inventory(),'source_sha256':{},'evaluation_mask_seeds':[79000,79001]}),newline='\n');self.hash=hashlib.sha256(self.plan.read_bytes()).hexdigest()
    def tearDown(self):self.tmp.cleanup()
    def test_inventory_and_exact_plan_admission(self):
        self.assertEqual(len(q.inventory()),12);self.assertEqual(sum(x['updates'] for x in q.inventory()),98304)
        runner.load_plan(self.plan,self.hash)
        with self.assertRaises(ValueError):runner.load_plan(self.plan,'0'*64)
    def test_same_contexts_roles_examples_and_distinct_query_structure(self):
        task=ResearchRoleTask(8,8,seed=0);results={}
        for policy in ('paired','repeated'):
            sg=torch.Generator().manual_seed(321);qr=torch.Generator().manual_seed(123);results[policy]=q.sample_batch(task,sg,qr,policy)
        a,b=results['paired'],results['repeated'];self.assertTrue(torch.equal(a[3]['base_canvas'],b[3]['base_canvas']));self.assertTrue(torch.equal(a[3]['fillers'],b[3]['fillers']))
        for policy,(x,y,r,w) in results.items():
            self.assertEqual(x.shape,(64,8,8));self.assertTrue(torch.equal(torch.bincount(r),torch.full((4,),16)))
            self.assertTrue(torch.equal(y,w['fillers'][w['context_indices'],r]));self.assertTrue(torch.equal(x[:,7,6],task.role_ids[r]))
            for c in range(16):self.assertEqual(len(torch.unique(r[w['context_indices']==c])),4 if policy=='paired' else 1)
            stripped=x.clone();stripped[:,7,6]=0;base=w['base_canvas'].clone();base[:,7,6]=0;self.assertTrue(torch.equal(stripped,base[w['context_indices']]))
    def test_bad_balancing_or_policy_refused(self):
        x=torch.zeros(16,8,8,dtype=torch.long);f=torch.ones(16,4,dtype=torch.long)
        with self.assertRaises(ValueError):q.assemble_batch(x,f,torch.zeros(16,dtype=torch.long),torch.arange(1,5),'paired')
        with self.assertRaises(ValueError):q.assemble_batch(x,f,torch.arange(16)%4,torch.arange(1,5),'wrong')
    def test_semantic_sampler_does_not_consume_firing_rng(self):
        torch.manual_seed(111);before=torch.get_rng_state().clone();task=ResearchRoleTask(8,8,seed=0)
        # Construction consumes no global RNG in this frozen task.
        q.sample_batch(task,torch.Generator().manual_seed(11),torch.Generator().manual_seed(12),'paired')
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_evaluation_preserves_rng_mode_and_reuses_immutable_decisions(self):
        task=ResearchRoleTask(8,8,seed=0)
        from neuropixel.research.data import frozen_dataset
        ds=frozen_dataset(task,'train',16,777);model=Tiny();model.train();before=torch.get_rng_state().clone();mode=model.training;fire=model.fire_rate
        first=q.evaluate_queries(model,ds,self.p/'eval','fixture',[1,2],True)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(model.training,mode);self.assertEqual(model.fire_rate,fire)
        second=q.evaluate_queries(model,ds,self.p/'eval','fixture',[1,2],True);self.assertEqual(first,second)
        with torch.no_grad():model.h.bias.add_(.1)
        with self.assertRaisesRegex(ValueError,'identity differs'):q.evaluate_queries(model,ds,self.p/'eval','fixture',[1,2],True)
    def test_low_ram_refused_before_study(self):
        class Ram:available=int(7.9*2**30)
        with patch('psutil.virtual_memory',return_value=Ram()):
            with self.assertRaisesRegex(RuntimeError,'RAM admission'):runner.run_one(q.inventory()[0],self.p/'blocked',self.plan,self.hash,128)
        self.assertFalse((self.p/'blocked').exists())
    def test_interrupt_resume_same_model_optimizer_and_all_rng_streams_toy(self):
        with patch('neuropixel.research.models.build_model',side_effect=lambda *a,**k:Tiny()):
            conf=q.inventory()[0];runner.run_one(conf,self.p/'full',self.plan,self.hash,256);runner.run_one(conf,self.p/'resume',self.plan,self.hash,128);runner.run_one(conf,self.p/'resume',self.plan,self.hash,256)
        a=torch.load(self.p/'full/checkpoint.pt',weights_only=True);b=torch.load(self.p/'resume/checkpoint.pt',weights_only=True)
        for field in ('model','optimizer','cpu_rng','sampler_rng','query_rng'):self.assertTrue(equal(a[field],b[field]),field)
        self.assertEqual(a['update'],256);self.assertEqual(b['update'],256)
        self.assertEqual([r['context_window_sha256'] for r in a['curve']],[r['context_window_sha256'] for r in b['curve']])
        self.assertTrue(all(r['role_counts']==[2048]*4 for r in a['curve']))

if __name__=='__main__':
    out=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QTrain));p=ROOT/'results/research/OPT03_QTRAIN_contracts/receipt.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'status':'passed' if out.wasSuccessful() else 'failed','tests':out.testsRun,'failures':len(out.failures),'errors':len(out.errors),'scope':'Native batch transformations and tiny-model operational checks; no scientific NeuroPixel training or learned outcomes'},indent=2)+'\n',newline='\n');raise SystemExit(0 if out.wasSuccessful() else 1)
