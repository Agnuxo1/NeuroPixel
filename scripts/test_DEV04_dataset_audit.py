"""Real archived data and alternate valid-row layouts exercise grouping controls."""
import importlib.util
import json
import pathlib
import tempfile
import unittest
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/audit_DEV04_archives.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
BASE=ROOT.parent/'.cognition/dev04-partials/37845945011/DEV04_partition101_seed401_body_u5120/DEV04_partition101_seed401/datasets'
DEST=ROOT/'results/research/DEV04_dataset_audit_contracts';DEST.mkdir(parents=True,exist_ok=True)


class Contracts(unittest.TestCase):
    def arrays(self):
        with np.load(BASE/'head_train.npz',allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}

    def reject(self,arrays):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            path=pathlib.Path(d)/'head_train.npz';np.savez_compressed(path,**arrays)
            with self.assertRaises(ValueError):m.dataset_compositions(path,'head_train')

    def test_actual_four_query_groups_accepted(self):
        self.assertTrue(m.dataset_compositions(BASE/'head_train.npz','head_train'))

    def test_valid_individual_samples_sorted_into_role_blocks_rejected(self):
        arrays=self.arrays();order=np.argsort(arrays['roles'],kind='stable')
        self.reject({k:v[order] for k,v in arrays.items()})

    def test_role_order_preserved_but_context_group_broken_rejected(self):
        arrays=self.arrays()
        for key in arrays:arrays[key][3],arrays[key][7]=arrays[key][7].copy(),arrays[key][3].copy()
        self.reject(arrays)

    def test_native_role_block_probe_remains_valid(self):
        self.assertTrue(m.dataset_compositions(BASE/'probe.npz','probe'))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),scientific_training_updates=0,nn_replay_completed=False)
    (DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(receipt));raise SystemExit(0 if result.wasSuccessful() else 1)
