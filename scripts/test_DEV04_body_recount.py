"""Analytical fixtures distinguish joint query competence from nominal accuracy."""
import importlib.util
import pathlib
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('replay',pathlib.Path(__file__).with_name('replay_DEV04.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Contracts(unittest.TestCase):
    def bank(self,masks=8):
        y=np.array([5,17,6,27]);z=np.array([6,17,5,27]);r=np.arange(4)
        return dict(predictions=np.tile(y,(masks,1)),query_predictions=np.tile(z,(masks,1)),target=y,query_target=z,roles=r,nll=np.full((masks,4),.5))

    def test_dense_and_matched_perfect(self):
        for k in (1,8):self.assertEqual(m.recount_body(self.bank(k)),dict(accuracy=1.,binding=1.,joint_query_binding=1.,per_role=[1.]*4,mean_nll=.5))

    def test_nominal_correct_but_query_joint_failure(self):
        bank=self.bank();bank['query_predictions'][:,0]=5
        r=m.recount_body(bank);self.assertEqual(r['binding'],1.);self.assertEqual(r['joint_query_binding'],.5)

    def test_individual_masks_not_ensemble(self):
        bank=self.bank();bank['predictions'][0,0]=6;bank['query_predictions'][1,2]=6
        r=m.recount_body(bank);self.assertEqual(r['binding'],15/16);self.assertEqual(r['joint_query_binding'],14/16)

    def test_unchanged_control_and_nonfinite_nll_rejected(self):
        bank=self.bank();bank['query_predictions'][0,1]=18
        with self.assertRaises(ValueError):m.recount_body(bank)
        bank=self.bank();bank['nll'][0,0]=np.nan
        with self.assertRaises(ValueError):m.recount_body(bank)


if __name__=='__main__':unittest.main()
