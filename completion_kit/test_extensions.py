"""Contracts for the 2026-10-12b extension: custody, programme inference, laws and claims."""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.stats import norm, t

from .common import sha256_file, write_json
from .commitments import commit, verify_opening, directory_digest
from .inference_plus import (tost_paired, noninferiority_paired, hierarchical_bootstrap, spending_function,
                             conservative_sequential_plan, online_bonferroni, lond, specification_curve)
from .laws import (contraction_horizons, capacity_ceiling_bits, fano_information_lower_bound, max_log2_items,
                   landauer_reference, fit_saturating_power_law, extrapolation_test)
from .claims_ledger import validate
from .stable_field import StableField


class CommitmentContracts(unittest.TestCase):
    def test_opening_round_trip_and_tampering(self):
        public, private = commit(dict(seed=20261012, generator="abc"), purpose="TEST seed", salt=b"s" * 32)
        self.assertTrue(verify_opening(public, private))
        changed = dict(private, value=dict(seed=20261013, generator="abc"))
        self.assertFalse(verify_opening(public, changed))
        self.assertFalse(verify_opening(public, dict(private, salt_hex=(b"t" * 32).hex())))
        self.assertIsNone(public["timestamp_evidence"])

    def test_short_salt_and_empty_purpose_refused(self):
        with self.assertRaises(ValueError):
            commit(1, purpose="x", salt=b"short")
        with self.assertRaises(ValueError):
            commit(1, purpose=" ")

    def test_salt_bytes_cannot_be_reread_as_value_bytes(self):
        public, private = commit(12345, purpose="scalar seed", salt=b"s" * 32)
        forged = dict(private, value=5, salt_hex=(b"s" * 32 + b"1234").hex())
        self.assertFalse(verify_opening(public, forged))
        self.assertFalse(verify_opening(dict(public, commitment="é" * 64), private))
        self.assertFalse(verify_opening([], private))

    @unittest.skipUnless(os.name == "nt", "Windows junctions only")
    def test_directory_digest_refuses_junctions(self):
        import _winapi
        with tempfile.TemporaryDirectory() as outside, tempfile.TemporaryDirectory() as folder:
            (Path(outside) / "secret.txt").write_text("outside", encoding="utf-8")
            root = Path(folder)
            (root / "model.json").write_text("{}", encoding="utf-8")
            _winapi.CreateJunction(outside, str(root / "jn"))
            try:
                with self.assertRaises(ValueError):
                    directory_digest(root)
            finally:
                os.rmdir(root / "jn")

    def test_random_salt_hides_equal_values(self):
        first, _ = commit(7, purpose="same value")
        second, _ = commit(7, purpose="same value")
        self.assertNotEqual(first["commitment"], second["commitment"])

    def test_directory_digest_tracks_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a").mkdir()
            (root / "a" / "model.json").write_bytes(b"{}")
            (root / "b.txt").write_bytes(b"one")
            first = directory_digest(root)
            self.assertEqual([item["path"] for item in first["files"]], ["a/model.json", "b.txt"])
            self.assertEqual(directory_digest(root)["digest"], first["digest"])
            (root / "b.txt").write_bytes(b"two")
            self.assertNotEqual(directory_digest(root)["digest"], first["digest"])


class InferenceContracts(unittest.TestCase):
    def test_tost_matches_manual_statistics(self):
        values = [0.4, -0.2, 0.1, 0.3, 0.0, 0.2]
        result = tost_paired(values, -1.0, 1.0)
        mean, se = np.mean(values), np.std(values, ddof=1) / math.sqrt(len(values))
        self.assertAlmostEqual(result["p_lower"], t.sf((mean + 1) / se, 5), places=12)
        self.assertAlmostEqual(result["p_upper"], t.cdf((mean - 1) / se, 5), places=12)
        self.assertTrue(result["equivalent"])
        self.assertFalse(tost_paired(values, -0.05, 0.05)["equivalent"])

    def test_tost_decision_equals_interval_inclusion(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            values = rng.normal(rng.normal(0, 1), rng.uniform(0.1, 2), size=int(rng.integers(3, 12)))
            low, high = sorted(rng.normal(0, 1.5, size=2))
            if high - low < 1e-6:
                continue
            result = tost_paired(values.tolist(), float(low), float(high))
            inside = low < result["interval_1_minus_2alpha"][0] and result["interval_1_minus_2alpha"][1] < high
            self.assertEqual(result["equivalent"], inside)

    def test_noninferiority_and_degenerate_variance(self):
        self.assertTrue(noninferiority_paired([0.0, 0.1, -0.1, 0.05, 0.0], 1.0)["noninferior"])
        self.assertFalse(noninferiority_paired([-2.0, -2.1, -1.9], 1.0)["noninferior"])
        degenerate = tost_paired([0.2, 0.2, 0.2], -1, 1)
        self.assertTrue(degenerate["equivalent"])
        self.assertIn("warning", degenerate)
        self.assertFalse(tost_paired([2.0, 2.0, 2.0], -1, 1)["equivalent"])
        self.assertNotIn("warning", tost_paired([0.1, 0.3, 0.2], -1, 1))

    def test_hierarchical_bootstrap_weights_clusters(self):
        groups = dict(big=[0.0] * 50, small=[10.0, 10.0])
        result = hierarchical_bootstrap(groups, replicates=2000, seed=1)
        self.assertEqual(result["estimate"], 5.0)
        self.assertIn("warning", result)
        again = hierarchical_bootstrap(groups, replicates=2000, seed=1)
        self.assertEqual(result["interval"], again["interval"])
        with self.assertRaises(ValueError):
            hierarchical_bootstrap(dict(only=[1.0, 2.0, 3.0]))

    def test_spending_functions(self):
        for kind in ("obrien_fleming", "pocock"):
            self.assertAlmostEqual(spending_function(1.0, 0.05, kind), 0.05, places=12)
        self.assertLess(spending_function(0.25, 0.05, "obrien_fleming"), spending_function(0.25, 0.05, "pocock"))
        plan = conservative_sequential_plan([1 / 3, 2 / 3, 1.0], 0.05)
        self.assertAlmostEqual(plan["total_alpha"], 0.05, places=12)
        with self.assertRaises(ValueError):
            conservative_sequential_plan([0.5, 0.4, 1.0])

    def test_sequential_plan_controls_error_under_correlated_looks(self):
        rng = np.random.default_rng(11)
        fractions = [0.25, 0.5, 0.75, 1.0]
        plan = conservative_sequential_plan(fractions, 0.05)
        increments = rng.normal(size=(40000, 4)) * np.sqrt(np.diff([0.0] + fractions))
        statistics = np.cumsum(increments, axis=1) / np.sqrt(fractions)
        rejected = (norm.sf(statistics) <= np.asarray(plan["p_thresholds"])).any(axis=1)
        self.assertLess(rejected.mean(), 0.05 + 3 * math.sqrt(0.05 * 0.95 / 40000))

    def test_online_procedures(self):
        bonferroni = online_bonferroni([0.001, 0.2, 0.004])
        self.assertEqual(bonferroni["rejected"], [True, False, False])
        result = lond([0.001, 0.2, 0.004], 0.05)
        self.assertEqual(result["rejected"], [True, False, True])
        self.assertAlmostEqual(result["thresholds"][2], 0.05 * 6 / (math.pi**2 * 9) * 2, places=15)
        with self.assertRaises(ValueError):
            lond([0.1, 0.2], gamma=[0.9, 0.9])
        with self.assertRaises(ValueError):
            online_bonferroni([0.1], alpha=2.0)
        self.assertEqual(lond([0.0, 0.0], 0.05, gamma=[0.0, 1.0])["rejected"], [False, True])

    def test_lond_fdr_simulation_under_independence(self):
        rng = np.random.default_rng(12)
        rates = []
        for _ in range(1500):
            null = rng.random(40) < 0.8
            p_values = np.where(null, rng.random(40), norm.sf(rng.normal(3, 1, 40)))
            rejected = np.asarray(lond(p_values.tolist(), 0.1)["rejected"])
            rates.append((rejected & null).sum() / max(1, rejected.sum()))
        self.assertLess(float(np.mean(rates)), 0.1 + 3 * float(np.std(rates)) / math.sqrt(len(rates)))

    def test_specification_curve(self):
        curve = specification_curve([dict(specification="a", effect=1.0, low=0.5, high=1.5),
                                     dict(specification="b", effect=-0.2, low=-0.9, high=0.4)])
        self.assertEqual(curve["share_positive"], 0.5)
        self.assertEqual(curve["share_interval_above_zero"], 0.5)
        with self.assertRaises(ValueError):
            specification_curve([dict(specification="a", effect=2.0, low=0.5, high=1.5)] * 2)


def contractive_field(seed: int = 5) -> StableField:
    """Return a projected field near its registered contraction limit."""
    model = StableField(1, 3, 3, steps=4, alpha=0.5, rho=0.9, seed=seed)
    model.parameters["mix"] *= 30
    model.parameters["kernels"] *= 30
    model.parameters["decoder"] *= 40
    model.project()
    return model


class LawContracts(unittest.TestCase):
    def test_forgetting_and_repair_follow_the_certificate(self):
        model = contractive_field()
        q = model.certificate()["q_float"]
        self.assertGreater(q, 0.9)
        rng = np.random.default_rng(3)
        inputs = rng.normal(size=(4, 1, 5, 5))
        first = rng.uniform(-1, 1, size=(4, 3, 5, 5))
        second = rng.uniform(-1, 1, size=(4, 3, 5, 5))
        bound = contraction_horizons(q, decoder=model.parameters["decoder"], margin=0.5,
                                     damage=float(np.abs(first - second).max()), tolerance=1e-3)
        start = float(np.abs(first - second).max())
        separable = bound["max_separable_step"]
        self.assertIsNotNone(separable)
        self.assertLess(separable, bound["repair_steps_bound"])
        state_a, state_b = first, second
        for depth in range(0, bound["repair_steps_bound"] + 5):
            logits_a, _ = model.forward(inputs, steps=0, initial=state_a)
            logits_b, _ = model.forward(inputs, steps=0, initial=state_b)
            self.assertLessEqual(float(np.abs(state_a - state_b).max()), q**depth * start + 1e-12)
            if depth > separable:
                for row_a, row_b in zip(logits_a, logits_b):
                    ordered_a, ordered_b = np.sort(row_a), np.sort(row_b)
                    both_confident = (ordered_a[-1] - ordered_a[-2] >= 0.5) and (ordered_b[-1] - ordered_b[-2] >= 0.5)
                    self.assertFalse(both_confident and np.argmax(row_a) != np.argmax(row_b))
            if depth >= bound["repair_steps_bound"]:
                self.assertLessEqual(float(np.abs(state_a - state_b).max()), 1e-3 + 1e-12)
            state_a, state_b = model.update(state_a, inputs), model.update(state_b, inputs)

    def test_exact_equality_step_is_admitted(self):
        decoder = np.array([[1.0, 0.0], [0.0, 0.0]])
        result = contraction_horizons(0.9, diameter=2.0, decoder=decoder, margin=0.9**2)
        self.assertEqual(result["max_separable_step"], 2)

    def test_capacity_and_fano(self):
        ceiling = capacity_ceiling_bits(10, 2, 5, 1.0)
        self.assertAlmostEqual(ceiling["ceiling_bits"], 5.0, places=12)
        perfect = fano_information_lower_bound(4000, 4000, 16)
        self.assertGreater(perfect["lower_bound_bits"], 3.98)
        self.assertLessEqual(perfect["lower_bound_bits"], 4.0)
        chance = fano_information_lower_bound(250, 4000, 16)
        self.assertEqual(chance["lower_bound_bits"], 0.0)
        self.assertLess(fano_information_lower_bound(3000, 4000, 16)["lower_bound_bits"],
                        fano_information_lower_bound(3600, 4000, 16)["lower_bound_bits"])
        self.assertAlmostEqual(max_log2_items(12.0, 0.0), 12.0, places=9)
        self.assertGreater(max_log2_items(12.0, 0.1), 12.0)
        self.assertEqual(max_log2_items(0.0, 0.0), 0.0)

    def test_simulated_noisy_erased_store_respects_the_ceiling(self):
        rng = np.random.default_rng(8)
        items, values, erased, sigma = 32, 6, 2, 0.6
        codebook = rng.choice([-1.0, 1.0], size=(items, values))
        labels = rng.integers(0, items, size=20000)
        stored = codebook[labels].copy()
        stored[:, :erased] = 0.0
        observed = stored + rng.normal(0, sigma, size=stored.shape)
        observed[:, :erased] = 0.0
        reference = codebook.copy()
        reference[:, :erased] = 0.0
        decoded = np.argmin(((observed[:, None, :] - reference[None]) ** 2).sum(axis=2), axis=1)
        lower = fano_information_lower_bound(int((decoded == labels).sum()), len(labels), items)
        ceiling = capacity_ceiling_bits(values, 1, erased, sigma)
        self.assertGreater(lower["lower_bound_bits"], 0.0)
        self.assertLessEqual(lower["lower_bound_bits"], ceiling["ceiling_bits"])

    def test_landauer_reference(self):
        result = landauer_reference(2.8710e-21 * 1e9, 1.0, 300.0)
        self.assertAlmostEqual(result["landauer_minimum_joules"], 1.380649e-23 * 300 * math.log(2), places=30)
        self.assertAlmostEqual(result["log10_ratio"], 9.0, places=3)

    def test_power_law_extrapolation(self):
        rng = np.random.default_rng(9)
        law = lambda scale: 0.02 + 0.5 * scale ** -0.7
        units = {scale: (law(scale) + rng.normal(0, 0.002, 5)).tolist() for scale in (1, 2, 4, 8, 16)}
        fit = fit_saturating_power_law(list(units), [float(np.mean(v)) for v in units.values()])
        self.assertAlmostEqual(fit["b"], 0.7, delta=0.15)
        good = extrapolation_test(units, 64, (law(64) + rng.normal(0, 0.002, 5)).tolist(), 0.01,
                                  replicates=300)
        self.assertTrue(good["passed"])
        self.assertTrue(good["reliable"])
        self.assertTrue(fit["adequate"])
        bad = extrapolation_test(units, 64, [0.3, 0.31, 0.29], 0.01, replicates=300)
        self.assertFalse(bad["passed"])
        with self.assertRaises(ValueError):
            extrapolation_test(units, 8, [0.1, 0.1], 0.01)
        zigzag = {1: [0.5, 0.5], 2: [0.1, 0.1], 4: [0.4, 0.4], 8: [0.05, 0.05]}
        rejected = extrapolation_test(zigzag, 32, [0.02, 0.02], 1.0, replicates=300)
        self.assertTrue(rejected["within_tolerance"])
        self.assertFalse(rejected["fit"]["adequate"])
        self.assertFalse(rejected["passed"])


class LedgerContracts(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        for name in ("receipt.json", "protocol.json", "replication.json"):
            (self.root / name).write_text(name, encoding="utf-8")
        public, _ = commit(dict(plan="BET01"), purpose="test registration", salt=b"r" * 32)
        write_json(self.root / "public.json", public)
        self.commitment = public["commitment"]
        self.registration = dict(commitment=self.commitment, timestamp_evidence="https://example.org/proof",
                                 public_record=self.record("public.json"))

    def tearDown(self):
        self.folder.cleanup()

    def record(self, name: str) -> dict:
        return dict(path=name, sha256=sha256_file(self.root / name))

    def ledger(self, *claims) -> dict:
        return dict(schema=1, authors=["Francisco Angulo de Lafuente"], claims=list(claims))

    def claim(self, **changes) -> dict:
        base = dict(id="C1", statement="Repair and forgetting share one rate", population="registered fields",
                    status="claim_supported_within_scope", declared_level=3, receipts=[self.record("receipt.json")],
                    protocol=self.record("protocol.json"), limits=["synthetic task"],
                    preregistration=self.registration)
        base.update(changes)
        return base

    def replication(self, **changes) -> dict:
        base = dict(evaluator="External lab", independent=True, blind=True, outcome="replicated",
                    receipt=self.record("replication.json"))
        base.update(changes)
        return base

    def test_supported_level_three_passes(self):
        report = validate(self.ledger(self.claim()), self.root)
        self.assertTrue(report["valid"], report["violations"])
        self.assertEqual(report["claims"][0]["supported_level"], 3)

    def test_overstated_and_tampered_claims_are_refused(self):
        self.assertFalse(validate(self.ledger(self.claim(declared_level=4)), self.root)["valid"])
        tampered = self.claim(receipts=[dict(self.record("receipt.json"), sha256="0" * 64)])
        self.assertFalse(validate(self.ledger(tampered), self.root)["valid"])
        negative = self.claim(status="claim_not_supported", declared_level=3)
        self.assertFalse(validate(self.ledger(negative), self.root)["valid"])
        self.assertFalse(validate(self.ledger(self.claim(), self.claim()), self.root)["valid"])
        self.assertFalse(validate(self.ledger(self.claim(limits=[])), self.root)["valid"])
        self.assertFalse(validate(self.ledger(self.claim(declared_level=True)), self.root)["valid"])
        unpublished = self.claim(preregistration=dict(self.registration, commitment="a" * 64))
        self.assertFalse(validate(self.ledger(unpublished), self.root)["valid"])
        self.assertFalse(validate(dict(schema=1, claims=[self.claim()]), self.root)["valid"])

    def test_malformed_claims_are_reported_not_raised(self):
        for broken in (self.claim(status=["x"]), self.claim(preregistration=[1]), self.claim(receipts=3)):
            report = validate(self.ledger(broken), self.root)
            self.assertFalse(report["valid"])

    def test_external_replication_needs_independence_and_blinding(self):
        good = self.claim(status="claim_replicated_external", declared_level=4,
                          external_replications=[self.replication()])
        self.assertTrue(validate(self.ledger(good), self.root)["valid"])
        for change in (dict(independent=False), dict(blind=False), dict(evaluator="me"),
                       dict(evaluator="francisco angulo de lafuente")):
            claim = self.claim(status="claim_replicated_external", declared_level=4,
                               external_replications=[self.replication(**change)])
            self.assertFalse(validate(self.ledger(claim), self.root)["valid"], change)


if __name__ == "__main__":
    unittest.main(verbosity=2)
