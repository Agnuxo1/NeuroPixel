"""Analytical fixtures expose pseudoreplication, optional stopping and bad proofs."""
import copy
import math
import unittest
from summarize_DEV04 import registered_summary

class Contracts(unittest.TestCase):
    def rows(self):
        return [dict(partition_seed=s,init_seed=400+4*j+k,mode=m,probe_joint=.96 if m=='query_attention' else .5,
            validation_joint=.95 if m=='query_attention' else .5,validation_binding=.95 if m=='query_attention' else .5)
            for j,s in enumerate((101,102,103)) for k in range(4) for m in ('query_attention','local_query')]

    def test_complete_paired_effect_equal_weights(self):
        d=registered_summary(self.rows());self.assertEqual(d['head_fits'],24);self.assertEqual(d['body_fits'],12)
        self.assertAlmostEqual(d['primary']['mean_pp'],45);self.assertAlmostEqual(d['primary']['half_width_pp'],0)
        self.assertEqual(d['primary']['n_policy_means'],3);self.assertEqual(d['primary']['df'],2);self.assertTrue(d['competence_gate_all_twelve'])

    def test_average_four_initializations_before_policy_interval(self):
        r=self.rows();r[0]['validation_joint']-=.4;d=registered_summary(r)
        self.assertEqual(len(d['pairs']),12);self.assertAlmostEqual(d['policies'][0]['mean_difference_pp'],35)
        self.assertAlmostEqual(d['primary']['mean_pp'],(35+45+45)/3);self.assertFalse(d['primary']['precision_target_passed']);self.assertFalse(d['competence_gate_all_twelve'])

    def test_incomplete_or_duplicate_rows_rejected(self):
        r=self.rows()
        with self.assertRaises(ValueError):registered_summary(r[:-1])
        r[-1]=copy.deepcopy(r[0])
        with self.assertRaises(ValueError):registered_summary(r)

    def test_one_failed_candidate_disqualifies_full_gate(self):
        r=self.rows();r[0]['probe_joint']=.949
        d=registered_summary(r);self.assertFalse(d['competence_gate_all_twelve']);self.assertEqual(d['attention_gate_passed_cases'],11)

    def test_negative_and_wide_intervals_preserved_without_truncation(self):
        r=self.rows()
        for row in r:
            if row['mode']=='query_attention':row['validation_joint']={101:0.,102:.5,103:1.}[row['partition_seed']]
        d=registered_summary(r);self.assertAlmostEqual(d['primary']['mean_pp'],0)
        self.assertLess(d['primary']['t95_pp'][0],-100);self.assertGreater(d['primary']['t95_pp'][1],100);self.assertFalse(d['primary']['precision_target_passed'])

    def test_nan_and_outside_accuracy_rejected(self):
        for value in (float('nan'),float('inf'),-1.,1.01):
            r=self.rows();r[0]['validation_joint']=value
            with self.assertRaises(ValueError):registered_summary(r)

if __name__=='__main__':unittest.main()
