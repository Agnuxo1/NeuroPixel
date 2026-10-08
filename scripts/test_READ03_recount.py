"""Analytical fixtures for independent array recount, no Torch/body fitting."""
import importlib.util
import pathlib
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('recount',pathlib.Path(__file__).with_name('replay_READ03.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class RecountContracts(unittest.TestCase):
    def arrays(self):
        roles=np.arange(4);y=np.array([5,6,7,8]);z=np.array([7,6,5,8])
        return {name:dict(predictions=np.tile(target,(8,1)),nll=np.ones((8,4)),target=target,roles=roles if name=='original' else np.array([2,1,0,3]))
                for name,target in [('original',y),('query_flip',z)]}

    def test_perfect_joint(self):
        r=m.recount(self.arrays());self.assertEqual(r,dict(accuracy=1.,binding=1.,joint_query_binding=1.,per_role=[1.]*4,mean_nll=1.))

    def test_joint_is_not_nominal_or_ensemble(self):
        a=self.arrays();a['query_flip']['predictions'][0,0]=9
        r=m.recount(a);self.assertEqual(r['binding'],1.);self.assertEqual(r['joint_query_binding'],15/16)

    def test_unchanged_control_rejected(self):
        a=self.arrays();a['query_flip']['predictions'][0,1]=9
        with self.assertRaises(ValueError):m.recount(a)

    def test_invalid_bank_and_nll_rejected(self):
        a=self.arrays();a['original']['predictions']=a['original']['predictions'][:1]
        with self.assertRaises(ValueError):m.recount(a)
        a=self.arrays();a['original']['nll'][0,0]=np.nan
        with self.assertRaises(ValueError):m.recount(a)


if __name__=='__main__':unittest.main()
