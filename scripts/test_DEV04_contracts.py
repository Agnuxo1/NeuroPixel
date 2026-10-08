"""Fresh-partition/body fixtures, never the scientific cohort or final test."""
import json
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DEST=ROOT/'results/research/DEV04_contracts'


def main():
    import psutil
    DEST.mkdir(parents=True,exist_ok=True);ram=psutil.virtual_memory().available/2**30
    if ram<8:
        a=dict(status='admission_refused_before_torch_import',available_ram_gib=ram,minimum_ram_gib=8,tests=0,scientific_body_fits=0)
        (DEST/'local_admission_refused.json').write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(a));return 75
    import copy
    import tempfile
    import unittest
    from unittest.mock import patch
    import torch
    from neuropixel.research import dev04 as ex
    from neuropixel.research.data import ResearchRoleTask
    from neuropixel.research import qtrain_budget as budget
    from neuropixel.research.experiment import configure_runtime,environment_record
    env=environment_record(configure_runtime('cpu',2))

    class Contracts(unittest.TestCase):
        def test_exact_reserved_test_and_counts(self):
            original=ResearchRoleTask(seed=0);test=set(original.test_triples)
            pool=set(original.train_triples)|set(original.validation_triples)
            for seed in (101,102,103):
                task=ex.DevelopmentRoleTask(seed);task.check_partitions()
                self.assertEqual(test,set(task.test_triples));self.assertEqual(pool,set(task.train_triples)|set(task.validation_triples))
                self.assertEqual(task.partition_record()['counts'],dict(train=924,validation=132,test=264))

        def test_private_split_determinism_and_global_rng_preserved(self):
            before=torch.get_rng_state().clone();a=ex.DevelopmentRoleTask(101);b=ex.DevelopmentRoleTask(101);c=ex.DevelopmentRoleTask(102)
            self.assertEqual(a.partition_record(),b.partition_record());self.assertNotEqual(a.train_triples,c.train_triples)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))

        def test_original_test_refused_before_sampling(self):
            task=ex.DevelopmentRoleTask(101);g=torch.Generator().manual_seed(8);before=g.get_state().clone()
            with self.assertRaises(ValueError):task.sample(64,'test',g)
            self.assertTrue(torch.equal(before,g.get_state()))

        def test_sampling_only_current_development_partition(self):
            task=ex.DevelopmentRoleTask(103)
            for split in ('train','validation'):
                x,y,r=ex.frozen_dataset(task,split,64,99)
                from neuropixel.research.OPT03_role_views import parse
                meta=parse(x.numpy(),y.numpy(),r.numpy())
                # Convert visible vocabulary IDs to native composition indices.
                noun=task._nouns_cpu.tolist();verb=task._verbs_cpu.tolist()
                triples={(noun.index(a),verb.index(b),noun.index(c)) for a,b,c in meta['semantic_triple']}
                self.assertTrue(triples<=set(getattr(task,split+'_triples')))

        def test_inventory_and_fresh_objects(self):
            rows=ex.inventory();self.assertEqual(len(rows),12);self.assertEqual(len({r['init_seed'] for r in rows}),12)
            self.assertEqual({r['init_seed'] for r in rows},set(range(400,412)))
            a=ex.new_objects(rows[0]);b=ex.new_objects(rows[0]);self.assertEqual(budget.state_digest(a[0].state_dict()),budget.state_digest(b[0].state_dict()))
            self.assertEqual(a[1].state_dict()['state'],{});self.assertEqual(sum(p.numel() for p in a[0].parameters()),29824)

        def fixture(self):
            config=dict(run_id='DEV04_synthetic_fixture',partition_seed=101,init_seed=400,policy='paired',school_weight=.3,updates=256,steps=4)
            task=ex.DevelopmentRoleTask(101);data=ex.panels(task,64,64);return config,task,data

        def test_exact_interrupted_resume_all_states_and_witnesses(self):
            config,task,data=self.fixture()
            with tempfile.TemporaryDirectory(dir=DEST) as d:
                full=pathlib.Path(d)/'full';split=pathlib.Path(d)/'split'
                ex.fit_body(config,task,data,full,'fixture-plan',env,endpoints=(128,256))
                ex.fit_body(config,task,data,split,'fixture-plan',env,endpoints=(128,256),stop_after=128)
                ex.fit_body(config,task,data,split,'fixture-plan',env,endpoints=(128,256))
                a=torch.load(full/'checkpoint.pt',weights_only=True);b=torch.load(split/'checkpoint.pt',weights_only=True)
                self.assertEqual(a['training_state_digest'],b['training_state_digest'])
                self.assertEqual([w['batch_witness_sha256'] for w in a['curve']],[w['batch_witness_sha256'] for w in b['curve']])
                self.assertEqual(a['evaluations'],b['evaluations'])
                self.assertTrue((split/'checkpoint_u128.pt').exists());self.assertTrue((split/'checkpoint_u256.pt').exists())

        def test_foreign_checkpoint_rejected_before_any_update(self):
            config,task,data=self.fixture()
            with tempfile.TemporaryDirectory(dir=DEST) as d:
                folder=pathlib.Path(d)
                ex.fit_body(config,task,data,folder,'fixture-plan',env,endpoints=(128,256),stop_after=128)
                p=torch.load(folder/'checkpoint.pt',weights_only=True);p['registration_id']='historical-U16'
                ex.save_state(folder/'checkpoint.pt',p)
                with self.assertRaises(ValueError):ex.fit_body(config,task,data,folder,'fixture-plan',env,endpoints=(128,256))

        def test_closed_body_skipped_before_initialization_or_ram(self):
            config,task,data=self.fixture()
            with tempfile.TemporaryDirectory(dir=DEST) as d:
                folder=pathlib.Path(d);ex.fit_body(config,task,data,folder,'fixture-plan',env,endpoints=(128,256))
                with patch.object(ex,'new_objects',side_effect=AssertionError('No repeated initialization')),patch.object(ex.budget,'require_ram',side_effect=AssertionError('Skip first')):
                    self.assertEqual(ex.fit_body(config,task,data,folder,'fixture-plan',env)['status'],'completed')
                wrong=dict(config,init_seed=401)
                with self.assertRaises(ValueError):ex.fit_body(wrong,task,data,folder,'fixture-plan',env,endpoints=(128,256))

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        available_ram_gib=ram,environment=env,scientific_body_fits=0,scientific_head_fits=0,test_scored=False,
        scope='Fresh synthetic body fitting fixtures and reserved-test partition metadata only; not the twelve scientific body trainings')
    (DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(receipt))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
