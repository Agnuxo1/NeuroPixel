"""Analytical incomplete-cohort and grouped-estimator tests; no neural imports."""
import unittest
from analyze_GLOB05 import contrast_vectors,interval
from research_GLOB05 import expected_cases

class Contracts(unittest.TestCase):
    def cells(self):return {(c,m):v for c in expected_cases() for m,v in [('attention',.99),('gated_uniform_global',.94),('local',.8)]}
    def test_paired_fixed_groups(self):
        vectors=contrast_vectors(self.cells())
        for v in vectors['attention_minus_active_global']:self.assertAlmostEqual(v,5.)
        for v in vectors['active_global_minus_local']:self.assertAlmostEqual(v,14.)
        self.assertEqual(interval(vectors['attention_minus_active_global'])['df'],2)
    def test_incomplete_rejected(self):
        cells=self.cells();cells.pop((expected_cases()[-1],'gated_uniform_global'))
        with self.assertRaises(ValueError):contrast_vectors(cells)
    def test_four_initializations_averaged_before_interval(self):
        cells=self.cells();cells[expected_cases()[0],'gated_uniform_global']-=.2
        v=contrast_vectors(cells)['attention_minus_active_global'];self.assertAlmostEqual(v[0],10);self.assertAlmostEqual(v[1],5);self.assertAlmostEqual(v[2],5)
    def test_negative_wide_intervals_not_truncated_and_no_replica_inflation(self):
        i=interval([-50.,0.,50.]);self.assertLess(i['t95_pp'][0],-100);self.assertFalse(i['lower_bound_positive'])
        with self.assertRaises(ValueError):interval([0.]*12)

if __name__=='__main__':unittest.main()
