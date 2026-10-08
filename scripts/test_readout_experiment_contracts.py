"""State-cache/head-resume fixtures, no scientific cohort fitting."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import psutil
    ram = psutil.virtual_memory().available/2**30
    dest = ROOT/'results/research/READ03_controller_contracts'
    dest.mkdir(parents=True,exist_ok=True)
    if ram < 8:
        receipt = dict(status='admission_refused_before_torch_import', available_ram_gib=ram,
                       minimum_ram_gib=8, tests=0, scientific_head_fits=0)
        (dest/'local_admission_refused.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(receipt))
        return 75
    import copy
    import hashlib
    import tempfile
    import unittest
    from unittest.mock import patch
    import numpy as np
    import torch
    from neuropixel.research.models import ResearchNCA
    from neuropixel.research.data import ResearchRoleTask
    from neuropixel.research.qtrain import evaluate_queries
    from neuropixel.research.qtrain_budget import state_digest
    from neuropixel.research.OPT03_role_views import transform
    from neuropixel.research.frozen_readout import FrozenReadout,body_digest
    from neuropixel.research import readout_experiment as ex
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    class Contracts(unittest.TestCase):
        def fixture(self):
            torch.manual_seed(71)
            body = ResearchNCA(35,(7,7),steps=1)
            for p in body.parameters():p.requires_grad_(False)
            task = ResearchRoleTask(seed=0)
            data = ex.paired_training_dataset(task,n=64)
            bank = ex.extract_bank(body,data,[88000,88001])
            cf = tuple(torch.from_numpy(v) for v in transform(*(v.numpy() for v in data),'query_flip'))
            counter = ex.extract_bank(body,cf,[88000,88001])
            return body,data,dict(train=bank,probe=bank,probe_query_flip=counter,
                                  validation=bank,validation_query_flip=counter)

        def test_new_latent_extraction_matches_reference_decisions_and_rng(self):
            body,data,banks = self.fixture()
            rng = torch.get_rng_state().clone()
            before = body_digest(body)
            with tempfile.TemporaryDirectory(dir=dest) as folder:
                old = evaluate_queries(body,data,pathlib.Path(folder),'fixture',[88000,88001],True)
                with np.load(pathlib.Path(folder)/'fixture_matched.npz',allow_pickle=False) as saved:
                    self.assertTrue(np.array_equal(saved['predictions'],banks['probe']['predictions'].numpy()))
                    self.assertTrue(np.array_equal(saved['query_predictions'],banks['probe_query_flip']['predictions'].numpy()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(before,body_digest(body))
            # Cached constants participate in head backprop without inference-tensor errors.
            reader = FrozenReadout(body,'query_attention')
            opt = torch.optim.AdamW(reader.head_parameters(),lr=.001)
            ex.step(reader,opt,banks['train'],torch.Generator().manual_seed(89004),torch.Generator().manual_seed(89005))

        def test_interrupted_and_uninterrupted_fits_match_all_training_states(self):
            body,_,banks = self.fixture()
            parent = dict(parent_case='synthetic_fixture',init_seed=200)
            for mode in ex.MODES:
                with tempfile.TemporaryDirectory(dir=dest) as folder:
                    base=pathlib.Path(folder)
                    full=ex.fit_head(copy.deepcopy(body),mode,banks,base/'full','fixture_plan',parent,
                                     updates=256,endpoints=(128,256))
                    stopped=ex.fit_head(copy.deepcopy(body),mode,banks,base/'resumed','fixture_plan',parent,
                                        updates=256,endpoints=(128,256),stop_after=128)
                    self.assertIsNone(stopped)
                    resumed=ex.fit_head(copy.deepcopy(body),mode,banks,base/'resumed','fixture_plan',parent,
                                        updates=256,endpoints=(128,256))
                    a=torch.load(base/'full/checkpoint.pt',weights_only=True)
                    b=torch.load(base/'resumed/checkpoint.pt',weights_only=True)
                    keys=('head','optimizer','cpu_rng','context_rng','mask_rng','frozen_body_digest')
                    self.assertEqual(state_digest({k:a[k] for k in keys}),state_digest({k:b[k] for k in keys}))
                    self.assertEqual([x['batch_witness_sha256'] for x in full['curve']],
                                     [x['batch_witness_sha256'] for x in resumed['curve']])
                    self.assertEqual(full['new_backbone_updates'],0)
                    self.assertEqual(resumed['effective_gradient_parameters'],1856 if mode=='uniform_global' else 3168)

        def test_closed_fits_skip_before_ram_or_model_use_and_reject_new_recipe(self):
            body,_,banks=self.fixture()
            parent=dict(parent_case='synthetic_fixture',init_seed=200)
            with tempfile.TemporaryDirectory(dir=dest) as folder:
                ex.fit_head(body,'local_query',banks,pathlib.Path(folder),'fixture_plan',parent,
                            updates=128,endpoints=(128,))
                with patch.object(ex,'require_ram',side_effect=AssertionError('closed fit must be omitted')):
                    done=ex.fit_head(None,'local_query',None,pathlib.Path(folder),'fixture_plan',parent,
                                     updates=128,endpoints=(128,))
                    self.assertEqual(done['status'],'completed')
                    with self.assertRaises(ValueError):
                        ex.fit_head(None,'local_query',None,pathlib.Path(folder),'different_plan',parent,
                                    updates=128,endpoints=(128,))

        def test_training_groups_roles_and_private_streams_are_identical_across_heads(self):
            _,_,banks=self.fixture()
            a=torch.Generator().manual_seed(89004);b=torch.Generator().manual_seed(89005)
            c=torch.Generator().manual_seed(89004);d=torch.Generator().manual_seed(89005)
            first=ex.batch_indices(banks['train'],a,b);second=ex.batch_indices(banks['train'],c,d)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(first,second)))
            roles=banks['train']['roles'][first[0]]
            self.assertTrue(torch.equal(torch.bincount(roles,minlength=4),torch.full((4,),16)))

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    files=[pathlib.Path(__file__),ROOT/'neuropixel/research/readout_experiment.py',ROOT/'scripts/research_READ03.py']
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
                 failures=len(result.failures),errors=len(result.errors),available_ram_gib=ram,
                 scientific_head_fits=0,new_backbone_updates=0,test_accessed=False,
                 scope='Generated native training fixtures with random untrained bodies; head-only optimizer/resume tests, not scientific cohort fitting',
                 source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
