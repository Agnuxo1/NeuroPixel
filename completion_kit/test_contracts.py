"""Meaningful CPU contracts, independent gradient checks and cohort-admission tests."""
from __future__ import annotations

import copy
import hashlib
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from .common import read_json, write_json, sha256_file, verify_files
from .statistics import paired_interval, holm_adjust, factorial_effect, plan_precision
from .energy import registered_keys, admit_rows, merge_receipts, summarize
from .stable_field import StableField, recurrent_bound, project_recurrence
from .dynamics import (linear_diagnostics, trajectory_products, lyapunov_diagnostic,
                       boolean_noise_information, finite_difference_jacobian)
from .datasets import make_dataset, decode_oracle, encode, oracle
from .model_forward_fix import install
from .publication_manifest import git_blob_and_sha256, verify_manifest
from . import train_reference
from .analyze_cohort import analyze


def energy_fixture() -> tuple[dict, dict]:
    """Construct synthetic admitted arithmetic rows, explicitly not physical evidence."""
    plan = dict(cases=[dict(name=f"case{i}") for i in range(7)],
                methods=["cuda_eager", "cuda_graph", "scalar_render", "vector_render", "render"],
                energy_technical_blocks=3, energy_active_seconds=3,
                resources=dict(actual_min_ram_gib=8))
    rows = []
    for case, block, method in sorted(registered_keys(plan)):
        rows.append(dict(case=case, block=block, method=method, inference_calls=10,
                         active_seconds=3.0, gpu_gross_energy_j=30.0, gpu_gross_j_per_call=3.0,
                         idle_before_w=2.0, idle_after_w=1.0,
                         baseline_adjusted_j_per_call_low=2.4, baseline_adjusted_j_per_call_high=2.7,
                         available_ram_gib=10.0, temperature_start_c=50, temperature_end_c=51))
    receipt = dict(plan_sha256="synthetic_unit_test_only", checks=[dict(passed=True) for _ in range(77)], rows=rows)
    return plan, receipt


class EvidenceContracts(unittest.TestCase):
    def test_git_blob_identity_includes_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.bin"
            path.write_bytes(b"abc")
            git_digest, raw_digest, size = git_blob_and_sha256(path)
            self.assertEqual(git_digest, hashlib.sha1(b"blob 3\0abc").hexdigest())
            self.assertEqual(raw_digest, hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(size, 3)

    def test_publication_manifest_rejects_corrupted_blob(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "raw.bin").write_bytes(b"abc")
            manifest = root / "manifest.json"
            write_json(manifest, dict(raw_entries=[dict(path="raw.bin", mode="100644", type="blob",
                       sha="0" * 40, origin_commit="synthetic_contract")]))
            result = verify_manifest(root, manifest, root / "audit.json")
            self.assertEqual(result["status"], "issues_found")
            self.assertEqual(result["verified"], 0)

    def test_atomic_exclusive_write_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            write_json(path, {"version": 1}, exclusive=True)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json(path, {"version": 2}, exclusive=True)
            self.assertEqual(path.read_bytes(), before)

    def test_hash_guard_rejects_modified_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.txt").write_text("old")
            digest = sha256_file(root / "source.txt")
            verify_files(root, {"source.txt": digest})
            (root / "source.txt").write_text("new")
            with self.assertRaises(ValueError):
                verify_files(root, {"source.txt": digest})

    def test_hash_guard_rejects_root_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                verify_files(Path(folder), {"../outside.txt": "invalid"})

    def test_scoped_forward_patch_preserves_other_methods(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.py"
            source = "class NeuroPixel:\n    def forward(self):\n        return 7\n\n    def dictionary(self):\n        return 9\n"
            path.write_text(source)
            install(path, sha256_file(path))
            after = path.read_text()
            self.assertIn("depth = self.steps if steps is None else steps", after)
            self.assertIn("state.new_zeros(())", after)
            self.assertIn("    def dictionary(self):\n        return 9\n", after)

    def test_scoped_forward_patch_rejects_stale_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.py"
            path.write_text("class NeuroPixel:\n    def forward(self):\n        return 7\n")
            with self.assertRaises(ValueError):
                install(path, "0" * 64)


class EnergyContracts(unittest.TestCase):
    def test_partial_103_has_two_missing_keys(self):
        plan, receipt = energy_fixture()
        receipt["rows"] = receipt["rows"][:103]
        result = merge_receipts(receipt, None, plan)
        self.assertEqual(result["unique_rows"], 103)
        self.assertEqual(len(result["missing"]), 2)
        self.assertFalse(result["complete"])

    def test_recovery_preserves_original_and_recounts_complete(self):
        plan, complete = energy_fixture()
        original = copy.deepcopy(complete)
        original["rows"] = original["rows"][:103]
        recovered = copy.deepcopy(complete)
        before = copy.deepcopy(original)
        result = summarize(original, recovered, plan)
        self.assertEqual(result["unique_rows"], 105)
        self.assertEqual(original, before)
        self.assertFalse(result["wall_energy_measured"])

    def test_duplicate_key_is_rejected(self):
        plan, receipt = energy_fixture()
        with self.assertRaises(ValueError):
            admit_rows(receipt["rows"] + receipt["rows"][:1], plan)

    def test_changed_completed_row_is_rejected(self):
        plan, original = energy_fixture()
        recovery = copy.deepcopy(original)
        recovery["rows"][0]["scope"] = "changed metadata"
        with self.assertRaises(ValueError):
            merge_receipts(original, recovery, plan)

    def test_unregistered_key_is_rejected(self):
        plan, receipt = energy_fixture()
        receipt["rows"][0]["method"] = "new_method"
        with self.assertRaises(ValueError):
            admit_rows(receipt["rows"], plan)

    def test_incomplete_summary_is_rejected(self):
        plan, receipt = energy_fixture()
        receipt["rows"].pop()
        with self.assertRaises(ValueError):
            summarize(receipt, None, plan)

    def test_negative_baseline_estimate_is_preserved(self):
        plan, receipt = energy_fixture()
        row = receipt["rows"][0]
        row.update(idle_before_w=100.0, idle_after_w=100.0,
                   baseline_adjusted_j_per_call_low=-27.0, baseline_adjusted_j_per_call_high=-27.0)
        admitted = admit_rows(receipt["rows"], plan)
        self.assertEqual(next(iter(admitted.values()))["baseline_adjusted_j_per_call_low"], -27)


class StatisticsContracts(unittest.TestCase):
    def test_vis07_integer_recount(self):
        differences = [100 * x / 1797 for x in (101, 45, 67, 46, 57)]
        result = paired_interval(differences)
        self.assertAlmostEqual(result["mean"], 3.5169727323316637, places=12)
        self.assertAlmostEqual(result["interval"][0], 1.93045944879548, places=10)
        self.assertEqual(result["n"], 5)

    def test_holm_original_order(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])

    def test_factorial_known_interaction(self):
        cells = {key: 2*key[0] + 3*key[1] + 5*key[0]*key[1] for key in itertools.product((0, 1), repeat=2)}
        self.assertEqual(factorial_effect(cells, 0), 4.5)

    def test_missing_factorial_cell_rejected(self):
        with self.assertRaises(ValueError):
            factorial_effect({(0, 0): 0, (1, 0): 1}, 0)

    def test_precision_is_a_new_fixed_cohort(self):
        result = plan_precision(3.0, 2.0)
        self.assertGreaterEqual(result["n"], 5)
        self.assertLessEqual(result["expected_half_width_pp"], 2)


class DynamicsContracts(unittest.TestCase):
    def test_nonnormal_transient_amplification_is_detected(self):
        matrix = np.array([[0.8, 2.0], [0, 0.8]])
        result = linear_diagnostics(matrix, np.eye(2), horizon=8)
        self.assertLess(result["spectral_radius"], 1)
        self.assertGreater(max(row["operator_norm"] for row in result["powers"]), 2)
        self.assertFalse(result["rigorous_numerical_certificate"])

    def test_current_display_can_hide_future_observable_state(self):
        matrix = np.array([[0.5, 1.0], [0, 0.5]])
        result = linear_diagnostics(matrix, np.array([[1.0, 0.0]]), horizon=2)
        self.assertEqual(result["observability"]["rank"], 2)

    def test_time_varying_product_order(self):
        first = np.array([[1., 1.], [0., 1.]])
        second = np.diag([2., 3.])
        result = trajectory_products([first, second])
        self.assertAlmostEqual(result[-1], np.linalg.norm(second @ first, 2))

    def test_lyapunov_equation_residual(self):
        result = lyapunov_diagnostic(np.array([[0.5, 0.2], [0, 0.4]]))
        self.assertTrue(result["admitted"])
        self.assertLess(result["residual_norm"], 1e-12)

    def test_boolean_coordinate_attains_information_bound(self):
        values = np.array([i & 1 for i in range(64)])
        result = boolean_noise_information(values, 0.1)
        self.assertAlmostEqual(result["information_bits"], result["proposed_bound_bits"], places=12)

    def test_boolean_parity_loses_more_information(self):
        coordinate = boolean_noise_information(np.array([i & 1 for i in range(64)]), 0.1)
        parity = boolean_noise_information(np.array([i.bit_count() & 1 for i in range(64)]), 0.1)
        self.assertLess(parity["information_bits"], coordinate["information_bits"])

    def test_boolean_constant_and_half_noise(self):
        self.assertAlmostEqual(boolean_noise_information(np.zeros(16), 0.2)["information_bits"], 0)
        self.assertAlmostEqual(boolean_noise_information(np.arange(16) % 2, 0.5)["information_bits"], 0)

    def test_finite_difference_matches_known_map(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = finite_difference_jacobian(lambda x: matrix @ x, np.array([0.2, -0.3]))
        np.testing.assert_allclose(result, matrix, atol=1e-10)


class FieldContracts(unittest.TestCase):
    def test_projection_checks_exact_exported_weight_bound(self):
        mix = np.full((3, 3), 100.0)
        kernels = np.full((3, 3, 3), 5.0)
        project_recurrence(mix, kernels, 0.95)
        from fractions import Fraction
        self.assertLessEqual(recurrent_bound(mix, kernels), Fraction.from_float(0.95))

    def test_empirical_sup_norm_contraction(self):
        model = StableField(2, 3, 2, seed=1)
        rng = np.random.default_rng(2)
        inputs = rng.normal(size=(2, 2, 5, 5))
        first, second = rng.normal(size=(2, 3, 5, 5)), rng.normal(size=(2, 3, 5, 5))
        difference = np.max(np.abs(model.update(first, inputs) - model.update(second, inputs)))
        allowed = model.certificate()["q_float"] * np.max(np.abs(first - second))
        self.assertLessEqual(difference, allowed + 1e-12)

    def test_zero_steps_is_zero_initial_state(self):
        model = StableField(1, 2, 2)
        logits, state = model.forward(np.ones((2, 1, 3, 3)), steps=0)
        self.assertTrue(np.array_equal(state, np.zeros_like(state)))
        self.assertEqual(logits.shape, (2, 2))

    def test_bounded_long_rollout_and_repeat_reset(self):
        model = StableField(1, 2, 2)
        inputs = np.ones((1, 1, 3, 3))
        first = model.forward(inputs, steps=512)
        second = model.forward(inputs, steps=512)
        self.assertLessEqual(np.max(np.abs(first[1])), 1 + 1e-14)
        np.testing.assert_array_equal(first[0], second[0])

    def test_export_roundtrip_and_certificate(self):
        model = StableField(2, 3, 2)
        inputs = np.ones((1, 2, 3, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            model.save(path)
            restored = StableField.load(path)
            np.testing.assert_array_equal(model.forward(inputs)[0], restored.forward(inputs)[0])
            self.assertEqual(model.certificate(), restored.certificate())

    def test_all_parameter_gradient_families(self):
        model = StableField(2, 3, 2, steps=4, seed=3)
        inputs = np.random.default_rng(4).normal(size=(2, 2, 3, 3))
        targets = np.array([0, 1])
        _, analytical = model.loss_gradient(inputs, targets)
        epsilon = 1e-6
        for name, parameter in model.parameters.items():
            index = tuple(0 for _ in parameter.shape)
            original = parameter[index]
            parameter[index] = original + epsilon
            above = model.loss_gradient(inputs, targets)[0]
            parameter[index] = original - epsilon
            below = model.loss_gradient(inputs, targets)[0]
            parameter[index] = original
            self.assertAlmostEqual(analytical[name][index], (above-below)/(2*epsilon), delta=2e-8, msg=name)

    def test_learning_improves_a_supervised_signal(self):
        model = StableField(1, 4, 2, steps=4, seed=8)
        values = np.array([-1., 1.] * 8)
        inputs = np.broadcast_to(values[:, None, None, None], (16, 1, 3, 3)).copy()
        target = (values > 0).astype(int)
        initial = model.loss_gradient(inputs, target)[0]
        for _ in range(80):
            _, gradients = model.loss_gradient(inputs, target)
            for name in model.parameters:
                model.parameters[name] -= 0.5 * gradients[name]
            model.project()
        final = model.loss_gradient(inputs, target)[0]
        self.assertLess(final, initial * 0.65)


class DatasetAndCohortContracts(unittest.TestCase):
    def test_epoch_resume_matches_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(train_reference, "require_resources", return_value=99.0):
            root = Path(folder)
            make_dataset(root / "data", graph_counts=(2, 1, 1), examples_per_graph=2)
            train_reference.register(root / "data/manifest.json", root / "plans", [700], epochs=2, state_channels=2)
            plan = root / "plans/seed700_contractive.json"
            actual_write = train_reference.write_json
            fired = False
            def interrupt_after_checkpoint(path, value, **kwargs):
                nonlocal fired
                actual_write(path, value, **kwargs)
                if Path(path).name == "checkpoint.json" and not fired:
                    fired = True
                    raise RuntimeError("Simulated interruption after committed epoch")
            with patch.object(train_reference, "write_json", side_effect=interrupt_after_checkpoint):
                with self.assertRaises(RuntimeError):
                    train_reference.fit(plan, root / "resumed")
            train_reference.fit(plan, root / "resumed")
            train_reference.fit(plan, root / "uninterrupted")
            resumed = read_json(root / "resumed/checkpoint.json")
            continuous = read_json(root / "uninterrupted/checkpoint.json")
            first = StableField.load(root / "resumed" / resumed["current_model_path"])
            second = StableField.load(root / "uninterrupted" / continuous["current_model_path"])
            for name in first.parameters:
                np.testing.assert_array_equal(first.parameters[name], second.parameters[name])
            self.assertEqual(resumed["updates"], continuous["updates"])

    def test_repeated_relation_oracle(self):
        maps = np.array([[1, 2, 0], [2, 0, 1]])
        query = [0, 0, 1, 0]
        field = encode(maps, 2, query, np.random.default_rng(1))
        self.assertEqual(decode_oracle(field, 3, 2), oracle(maps, 2, query))

    def test_graph_groups_and_encoded_oracles(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "data"
            manifest = make_dataset(output, graph_counts=(3, 2, 2), examples_per_graph=2)
            groups = {}
            for name in manifest["panels"]:
                with np.load(output / (name + ".npz"), allow_pickle=False) as archive:
                    groups[name] = set(archive["groups"])
                    for x, y in zip(archive["inputs"], archive["targets"]):
                        self.assertEqual(decode_oracle(x), y)
            self.assertFalse(groups["train"] & groups["dev"])
            self.assertFalse(groups["dev"] & groups["test_base"])
            self.assertEqual(groups["test_base"], groups["test_long"])

    def test_complete_reference_pipeline_and_closed_fit_guard(self):
        # Resource admission is mocked ONLY for this tiny software contract.
        # These executions are neither research realizations nor physical-GPU evidence.
        with tempfile.TemporaryDirectory() as folder, patch.object(train_reference, "require_resources", return_value=99.0):
            root = Path(folder)
            make_dataset(root / "data", graph_counts=(2, 1, 1), examples_per_graph=2)
            train_reference.register(root / "data/manifest.json", root / "plans", [700, 701], epochs=1, state_channels=2)
            plans = sorted((root / "plans").glob("*.json"))
            fits = []
            for plan in plans:
                output = root / "fits" / plan.stem
                train_reference.fit(plan, output)
                with self.assertRaises(FileExistsError):
                    train_reference.fit(plan, output)
                fits.append(output)
            with self.assertRaises(ValueError):
                train_reference.score_cohort(plans[:2], fits[:2], root / "incomplete_score")
            self.assertFalse((root / "incomplete_score").exists())
            actual_write = train_reference.write_json
            fired = False
            def interrupt_after_score(path, value, **kwargs):
                nonlocal fired
                actual_write(path, value, **kwargs)
                if Path(path).name == "partial_scores.json" and not fired:
                    fired = True
                    raise RuntimeError("Simulated interruption after saved endpoint")
            with patch.object(train_reference, "write_json", side_effect=interrupt_after_score):
                with self.assertRaises(RuntimeError):
                    train_reference.score_cohort(plans, fits, root / "score")
            saved = read_json(root / "score/partial_scores.json")[0]
            original_hash = saved["artifact_sha256"]
            train_reference.score_cohort(plans, fits, root / "score")
            result = read_json(root / "score/result.json")
            self.assertEqual(result["status"], "complete_registered_reference_cohort")
            self.assertEqual(len(result["scores"]), 32)
            self.assertFalse(result["external_replication"])
            self.assertEqual(sha256_file(root / "score" / (saved["key"] + ".npz")), original_hash)
            recount = analyze(root / "score")
            self.assertEqual(recount["rows"], 32)
            self.assertEqual(recount["primary_difference_pp"]["n"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
