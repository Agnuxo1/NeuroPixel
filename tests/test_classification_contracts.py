"""Meaningful scoring regressions, including the preserved original functions.

Local checks execute the exact function ASTs without importing unrelated Torch
training code. A separate integration case imports the real module in the
declared cloud environment. No random bootstrap is needed for these fixtures.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.research.classification_contracts import (  # noqa: E402
    validate_binary_counts, validate_classification_inputs,
)


def source_functions(source):
    tree = ast.parse(Path(source).read_text())
    wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in {"wilson", "classification_metrics"}]
    if {node.name for node in wanted} != {"wilson", "classification_metrics"}:
        raise ValueError("expected the actual two metric functions")
    role_tree = ast.parse((ROOT / "neuropixel/task.py").read_text())
    role_assign = next(node for node in role_tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "ROLES" for t in node.targets))
    namespace = {"np": np, "math": math, "NormalDist": NormalDist,
                 "ROLES": ast.literal_eval(role_assign.value),
                 "validate_binary_counts": validate_binary_counts,
                 "validate_classification_inputs": validate_classification_inputs}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


SOURCE = Path(os.environ.get("NEUROPIXEL_METRIC_SOURCE", str(ROOT / "neuropixel/research/experiment.py")))
FUNCTIONS = source_functions(SOURCE)


class ClassificationContractTests(unittest.TestCase):
    def score(self, prediction=None, target=None, roles=None, nll=None, **kwargs):
        prediction = [1, 2, 8, 4] if prediction is None else prediction
        target = [1, 2, 3, 4] if target is None else target
        roles = [0, 1, 2, 3] if roles is None else roles
        return FUNCTIONS["classification_metrics"](prediction, target, roles, nll, **kwargs)

    def test_valid_counts_and_macro_role_weighting(self):
        # Unequal role populations: micro accuracy differs from macro role mean.
        score = self.score([1, 8, 8, 2, 3, 8, 4, 8], [1, 1, 1, 2, 3, 3, 4, 4],
                           [0, 0, 0, 1, 2, 2, 3, 3], [0., .5, 1., 1.5, 2., 2.5, 3., 3.5])
        self.assertEqual((score["correct"], score["n"]), (4, 8))
        self.assertEqual(score["accuracy"], .5)
        self.assertAlmostEqual(score["macro_all_roles"], (1/3 + 1 + .5 + .5)/4)
        self.assertAlmostEqual(score["macro_agent_patient_accuracy"], (1/3 + .5)/2)
        self.assertEqual(score["cross_entropy"], 1.75)
        self.assertEqual(sum(row["n"] for row in score["per_role"].values()), score["n"])

    def test_unknown_role_cannot_inflate_global_denominator(self):
        with self.assertRaises(ValueError):
            self.score([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 1, 2, 3, 99])

    def test_nll_must_align_with_every_prediction(self):
        for value in ([.1], .1, [[.1], [.2], [.3], [.4]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.score(nll=value)

    def test_nonfinite_or_negative_nll_is_rejected(self):
        for bad in (float("nan"), float("inf"), -.001):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.score(nll=[.1, .2, .3, bad])

    def test_noninteger_or_boolean_ids_are_rejected(self):
        for kwargs in ({"prediction": [1., 2., 3., 4.]},
                       {"target": [1, 2, 3, float("nan")]},
                       {"roles": [0., 1., 2., 3.]},
                       {"prediction": [True, False, True, False]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.score(**kwargs)

    def test_pad_target_and_negative_prediction_are_rejected(self):
        for kwargs in ({"target": [0, 2, 3, 4]}, {"prediction": [-1, 2, 3, 4]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.score(**kwargs)
        self.assertEqual(self.score(prediction=[0, 2, 3, 4])["correct"], 3)

    def test_missing_roles_and_mismatched_shapes_are_rejected(self):
        for kwargs in ({"roles": [0, 1, 2, 2]}, {"prediction": [1, 2]},
                       {"prediction": [], "target": [], "roles": []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.score(**kwargs)

    def test_interval_flag_and_counts_have_explicit_types(self):
        cases = ({"intervals": "no"}, {"intervals": True, "repetitions": True},
                 {"intervals": True, "bootstrap_seed": True},
                 {"intervals": True, "bootstrap_seed": -1},
                 {"intervals": True, "repetitions": 0})
        with patch.object(np.random, "default_rng", side_effect=AssertionError(
                "invalid inputs must be rejected before RNG construction")):
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.score(**kwargs)

    def test_wilson_rejects_fractional_counts_and_invalid_confidence(self):
        for args in ((1.5, 4, .95), (True, 4, .95), (1, 4, float("nan")),
                     (1, 4, 1.), (1, 4, -1.), (1, 0, .95), (5, 4, .95)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                FUNCTIONS["wilson"](*args)
        interval = FUNCTIONS["wilson"](np.int64(2), np.int64(4), .95)
        self.assertLess(interval[0], .5)
        self.assertGreater(interval[1], .5)

    def test_valid_summary_matches_preserved_original_exactly(self):
        before = source_functions(ROOT / "results/research/08_validation/classification_source_original.py")
        values = ([1, 2, 8, 4], [1, 2, 3, 4], [0, 1, 2, 3], [.1, .2, .3, .4])
        self.assertEqual(FUNCTIONS["classification_metrics"](*values),
                         before["classification_metrics"](*values))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "real module integration requires declared cloud Torch")
    def test_real_module_integration(self):
        from neuropixel.research.experiment import classification_metrics
        values = ([1, 2, 8, 4], [1, 2, 3, 4], [0, 1, 2, 3], [.1, .2, .3, .4])
        self.assertEqual(classification_metrics(*values), self.score(*values))
        with self.assertRaises(ValueError):
            classification_metrics(*values[:3], nll=[.1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
