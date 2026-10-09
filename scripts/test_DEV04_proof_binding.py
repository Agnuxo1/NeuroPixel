"""Real original archive + synthetic receipts exercise tamper/incomplete-proof rejection."""
import copy
import json
import pathlib
import unittest
import zipfile
from summarize_DEV04 import validate_proof,sha,PLAN,RUN,SOURCE

ROOT=pathlib.Path(__file__).resolve().parents[1]
CASE='DEV04_partition101_seed401'
ORIGINAL=ROOT/f'results/research/DEV04_recovery/{RUN}/originals'/CASE

class Contracts(unittest.TestCase):
    def fixture(self):
        with zipfile.ZipFile(ORIGINAL/'original.zip') as z:manifest=json.loads(z.read('archive_manifest.json'))
        proof=dict(case=CASE,status='verified_available_readonly_DEV04_state_and_endpoints',issues=[],checks=500,
            scientific_run_id=RUN,scientific_source_commit=SOURCE,verification_returncode=0,individual_decisions_replayed=811008,
            plan_sha256=PLAN,max_mean_nll_error=0.,new_training_updates=0,test_scored=False,external_replication=False,
            verification_run_id='123',verification_source_commit='a'*40,original_zip_sha256=sha(ORIGINAL/'original.zip'),
            frozen_recipe_source_commit=SOURCE,body_endpoints=[8192,16384],head_endpoints={'query_attention':[1024,4096,8192],'local_query':[1024,4096,8192]},
            input_files_sha256={'/fixture/'+name:digest for name,digest in manifest['files'].items()})
        return proof,manifest

    def validate(self,proof,manifest):validate_proof(proof,CASE,ORIGINAL,manifest,ROOT,'123','a'*40)

    def test_exact_original_file_inventory_accepted(self):
        p,m=self.fixture();self.validate(p,m)

    def test_altered_and_omitted_checkpoint_binding_rejected(self):
        p,m=self.fixture();k=next(x for x in p['input_files_sha256'] if x.endswith('checkpoint.pt'));p['input_files_sha256'][k]='0'*64
        with self.assertRaises(ValueError):self.validate(p,m)
        p,m=self.fixture();p['input_files_sha256'].pop(k)
        with self.assertRaises(ValueError):self.validate(p,m)

    def test_nonfinite_looser_nll_and_retraining_rejected(self):
        original,m=self.fixture()
        for key,value in [('max_mean_nll_error',float('nan')),('max_mean_nll_error',float('inf')),('max_mean_nll_error',.000100001),('new_training_updates',1),('external_replication',True),('checks',0),('scientific_source_commit','b'*40)]:
            p=copy.deepcopy(original);p[key]=value
            with self.assertRaises(ValueError):self.validate(p,m)

    def test_missing_endpoint_and_inflated_mask_decisions_rejected(self):
        p,m=self.fixture();p['head_endpoints']['local_query'].pop()
        with self.assertRaises(ValueError):self.validate(p,m)
        p,m=self.fixture();p['individual_decisions_replayed']*=8
        with self.assertRaises(ValueError):self.validate(p,m)

if __name__=='__main__':unittest.main()
