"""Deterministic routing-summary and narrow driver-integration regressions."""
import ast
from pathlib import Path
import unittest

from neuropixel.research.routing_metrics import summarize_routing


ROOT = Path(__file__).resolve().parents[1]


class RoutingMetricsTests(unittest.TestCase):
    def test_equal_mean_ids_have_distinct_distributions(self):
        all_one, mixed = [1, 1], [0, 2]
        self.assertEqual(sum(all_one) / 2, sum(mixed) / 2)
        self.assertEqual(summarize_routing(all_one, 3)["counts"], [0, 2, 0])
        self.assertEqual(summarize_routing(mixed, 3)["counts"], [1, 0, 1])
        self.assertNotEqual(summarize_routing(all_one, 3), summarize_routing(mixed, 3))

    def test_empty_is_invalid_but_unused_slots_are_explicit(self):
        with self.assertRaises(ValueError):
            summarize_routing([], 3)
        result = summarize_routing([1, 1], 4)
        self.assertEqual(result["counts"], [0, 2, 0, 0])
        self.assertEqual(result["fractions"], [0, 1, 0, 0])
        self.assertEqual(result["unique_modal_slot_id"], 1)

    def test_invalid_slot_inventory(self):
        for value in (0, -1, True, 3.0, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_routing([0], value)

    def test_invalid_assignment_ids(self):
        for value in (-1, 3, True, 1.0, "1", float("nan"), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_routing([value], 3)

    def test_frequency_ties_do_not_invent_one_expert(self):
        result = summarize_routing([2, 0], 4)
        self.assertEqual(result["modal_slot_ids"], [0, 2])
        self.assertIsNone(result["unique_modal_slot_id"])
        self.assertIsNone(result["argmax_ties"])

    def test_known_topic_contingency(self):
        result = summarize_routing([0, 1, 1, 2, 1], 3, topics=[7, 7, 9, 9, 9])
        self.assertEqual(result["counts"], [1, 3, 1])
        self.assertEqual([(r["topic_id"], r["n"], r["counts"]) for r in result["by_topic"]],
                         [(7, 2, [1, 1, 0]), (9, 3, [0, 2, 1])])
        self.assertIsNone(result["mapped_slot_agreement"])

    def test_topics_are_never_inferred(self):
        result = summarize_routing([0, 1, 2], 3)
        self.assertIsNone(result["by_topic"])
        self.assertIsNone(result["mapped_slot_agreement"])
        with self.assertRaises(ValueError):
            summarize_routing([0], 2, correct_slot_by_topic={0: 0})

    def test_invalid_or_misaligned_topics(self):
        for topics in ([], [0, 0], [True], [-1], [1.0], ["known"]):
            with self.subTest(topics=topics), self.assertRaises(ValueError):
                summarize_routing([0], 2, topics=topics)

    def test_mapping_must_cover_observed_topics(self):
        with self.assertRaises(ValueError):
            summarize_routing([0, 1], 2, topics=[7, 9], correct_slot_by_topic={7: 0})
        for mapping in ({7: 2}, {7: True}, {True: 0}, {7: -1}, [(7, 0)]):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                summarize_routing([0], 2, topics=[7], correct_slot_by_topic=mapping)

    def test_explicit_mapping_reports_agreement_only(self):
        result = summarize_routing([0, 1, 1, 1], 2, topics=[7, 7, 9, 9],
                                   correct_slot_by_topic={7: 0, 9: 1, 12: 0})
        agreement = result["mapped_slot_agreement"]
        self.assertEqual((agreement["matching_count"], agreement["n"], agreement["fraction"]), (3, 4, .75))
        self.assertIn("not prediction accuracy or oracle optimality", agreement["meaning"])
        self.assertEqual([r["topic_id"] for r in result["by_topic"]], [7, 9])

    def test_slot_renaming_permutes_counts_and_preserves_mapped_agreement(self):
        picks, topics = [0, 2, 2, 1, 2], [7, 7, 9, 9, 9]
        permutation = {0: 2, 1: 0, 2: 1}
        before = summarize_routing(picks, 3, topics=topics, correct_slot_by_topic={7: 0, 9: 2})
        after = summarize_routing([permutation[p] for p in picks], 3, topics=topics,
                                  correct_slot_by_topic={7: permutation[0], 9: permutation[2]})
        for old, new in permutation.items():
            self.assertEqual(before["counts"][old], after["counts"][new])
            self.assertEqual(before["fractions"][old], after["fractions"][new])
        self.assertEqual(before["mapped_slot_agreement"]["fraction"], after["mapped_slot_agreement"]["fraction"])

    def test_pooled_denominator_weights_unequal_topic_sizes(self):
        small, large = [0, 0], [1] * 8
        pooled = summarize_routing(small + large, 2, topics=[0] * 2 + [1] * 8,
                                   correct_slot_by_topic={0: 0, 1: 0})
        self.assertEqual((pooled["n"], pooled["counts"], pooled["fractions"]), (10, [2, 8], [.2, .8]))
        unweighted = sum(r["fractions"][0] for r in pooled["by_topic"]) / 2
        self.assertEqual(unweighted, .5)
        self.assertNotEqual(pooled["fractions"][0], unweighted)
        self.assertEqual(pooled["mapped_slot_agreement"]["fraction"], .2)

    def test_exact_score_ties_use_original_first_maximum_rule(self):
        result = summarize_routing([0, 2, 1], 3, topics=[7, 7, 9],
                                   argmax_scores=[[2, 2, 0], [0, 0, 1], [0, 3, 3]])
        self.assertEqual(result["argmax_ties"]["count"], 2)
        self.assertEqual(result["argmax_ties"]["fraction"], 2 / 3)
        self.assertEqual([row["argmax_tie_count"] for row in result["by_topic"]], [1, 1])
        with self.assertRaises(ValueError):
            summarize_routing([1], 2, argmax_scores=[[2, 2]])

    def test_nonfinite_bad_shape_and_inconsistent_scores_rejected(self):
        for scores in ([], [[1]], [[1, 2, 3]], [[0, 1]], [[float("nan"), 0]],
                       [[float("inf"), 0]], [[True, 0]], [["1", 0]]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                summarize_routing([0], 2, argmax_scores=scores)


class DriverSummaryIntegrationTests(unittest.TestCase):
    def test_both_growth_drivers_use_topic_and_pooled_count_summaries(self):
        tree = ast.parse((ROOT / "scripts/phase3.py").read_text())
        for name in ("t_grow", "t_grow2"):
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "summarize_routing"]
            self.assertEqual(len(calls), 2, name)
            for call in calls:
                self.assertEqual({k.arg for k in call.keywords}, {"topics", "argmax_scores"})
            self.assertNotIn("elige_lienzo", ast.unparse(function))
            self.assertNotIn("pick.float().mean()", ast.unparse(function))

    def test_training_and_original_routing_expressions_unchanged(self):
        old = ast.parse((ROOT / "results/research/08_validation/routing_phase3_original.py").read_text())
        new = ast.parse((ROOT / "scripts/phase3.py").read_text())
        def expressions(tree, name):
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            return [ast.dump(n, include_attributes=False) for n in ast.walk(function)
                    if isinstance(n, ast.Call) and ast.unparse(n.func) in
                    ("train", "NeuroPixel", "torch.stack", "sc.argmax", "preds.gather", "tset")]
        for name in ("t_grow", "t_grow2"):
            self.assertEqual(expressions(old, name), expressions(new, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
