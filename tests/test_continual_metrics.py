import unittest
import numpy as np
from neuropixel.research.continual_metrics import matrix_summary, score_matrices


class ContinualMetricContracts(unittest.TestCase):
    def test_distinct_backward_change_and_two_past_maxima(self):
        r = [[.4, .8, .1], [.9, .6, .1], [.7, .5, .8]]
        s = matrix_summary(r)
        self.assertAlmostEqual(s["final"]["BWT"], .1)
        self.assertAlmostEqual(s["final"]["max_past_forgetting"], .25)
        self.assertAlmostEqual(s["final"]["post_acquisition_forgetting"], .15)
        self.assertAlmostEqual(s["final"]["average_seen_accuracy"], 2 / 3)
        self.assertIsNone(s["stages"][0]["BWT"])
        self.assertIsNone(s["FWT"])

    def test_signed_improvement_and_single_task(self):
        s = matrix_summary([[.2, .1], [.6, .8]])
        self.assertAlmostEqual(s["final"]["BWT"], .4)
        self.assertAlmostEqual(s["final"]["max_past_forgetting"], -.4)
        self.assertAlmostEqual(s["final"]["post_acquisition_forgetting"], -.4)
        one = matrix_summary([[.3]])
        self.assertEqual(one["final"]["average_seen_accuracy"], .3)
        self.assertIsNone(one["final"]["BWT"])
        self.assertIsNone(one["final"]["max_past_forgetting"])

    def test_reject_invalid_or_incomplete_statistical_inputs(self):
        for r in ([], [[.1, .2]], [[float("nan")]], [[1.1]], [[-.1]]):
            with self.subTest(matrix=r), self.assertRaises(ValueError):
                matrix_summary(r)
        logp = np.log(np.full((1, 4, 2), .5))
        for role in (np.zeros(4, dtype=np.int64), np.arange(4, dtype=np.float64)):
            with self.subTest(role=role.dtype), self.assertRaises(ValueError):
                score_matrices(logp, np.zeros(4, dtype=np.int64), role, np.zeros(4, dtype=np.int64))
        with self.assertRaises(ValueError):
            score_matrices(np.zeros((1, 4, 2)), np.zeros(4, dtype=np.int64),
                           np.arange(4), np.zeros(4, dtype=np.int64))

    def test_hand_counted_role_macro_and_hard_oracle(self):
        # Two tasks, one example per role. Four correct in different checkpoints;
        # hard selection can recover all eight, with no probability-mixture claim.
        role = np.tile(np.arange(4), 2)
        topic = np.repeat(np.arange(2), 4)
        y = np.zeros(8, dtype=np.int64)
        probabilities = np.array([
            [[.9, .1]] * 4 + [[.1, .9]] * 4,
            [[.1, .9]] * 4 + [[.9, .1]] * 4])
        s = score_matrices(np.log(probabilities), y, role, topic)
        self.assertEqual(s["measures"]["accuracy"]["matrix"], [[1., 0.], [0., 1.]])
        self.assertEqual(s["measures"]["binding"]["matrix"], [[1., 0.], [0., 1.]])
        self.assertEqual(s["measures"]["binding"]["final"]["BWT"], -1.)
        self.assertEqual([v["correct"] for v in s["hard_selection_oracle_by_topic"]], [4, 4])
