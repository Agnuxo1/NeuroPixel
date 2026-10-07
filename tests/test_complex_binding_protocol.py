"""Meaningful failure and mathematical fixtures for the prospective item-9 gate.

Gate files here are deliberately inert stand-ins, never trained models or study
datasets. Numerical model integration cases use arbitrary tiny token tensors,
not the eventual performance collection.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neuropixel.research.complex_binding_metrics import point_metrics
from neuropixel.research.complex_binding_protocol import (
    artifact, canonical_bytes, claim_final_access, create_final_inventory,
    expected_primary, read_json, select_pilot, sha256, validate_pilot_evidence, write_new_json,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE = json.loads((ROOT / "docs/research/09_experiment_recipe.json").read_text())
RATES = {"neuropixel": 0.003, "relative_transformer": 0.001}
CONTEXT = {"source_commit": "a" * 40, "plan_sha256": "b" * 64,
           "recipe_sha256": "c" * 64, "development_identities": {"fixture": "d" * 64}}


def pilot_fixture():
    rows = []
    for family in ("neuropixel", "relative_transformer"):
        for rate in (0.001, 0.003):
            rows.append({"family": family, "learning_rate": rate,
                         "status": "completed", "phase": "pilot", "initialization": 39,
                         "updates_completed": 1024, "context": deepcopy(CONTEXT),
                         "validation": {"binding": 0.5, "cross_entropy": 1.0}})
    return rows


def primary_fixture(root):
    """Create ten inert file identities to exercise orchestration, not training."""
    expected = expected_primary(RECIPE, RATES)
    for spec in expected:
        directory = root / "runs" / spec["run_id"]
        directory.mkdir(parents=True)
        artifacts = {}
        for name in ("checkpoint", "training_log", "probe_predictions", "validation_predictions"):
            path = directory / f"{name}.fixture"
            path.write_bytes(f"inert unit fixture: {spec['run_id']} {name}\n".encode())
            artifacts[name] = artifact(path, root)
        row = {key: spec[key] for key in ("run_id", "family", "initialization", "learning_rate")}
        row.update(status="completed", phase="primary", updates_completed=spec["updates"],
                   context=deepcopy(CONTEXT), artifacts=artifacts,
                   probe={"binding": 0.5, "global": 0.5, "cross_entropy": 1.0},
                   validation={"binding": 0.5, "global": 0.5, "cross_entropy": 1.0})
        write_new_json(directory / "run.json", row)
    write_new_json(root / "primary_runs.json", {"status": "completed",
                                              "runs": [row["run_id"] for row in expected]})


def archive_fixture(root, **overrides):
    value = {"schema_version": 1, "item": 9, "source_commit": CONTEXT["source_commit"],
             "archive_commit": "e" * 40,
             "results_branch": "research/scientific-validation-2026-10-07-cloud-results",
             "final_inventory_sha256": sha256(root / "final_inventory.json"),
             "archived_at_utc": "2000-01-01T00:00:00+00:00"}
    value.update(overrides)
    write_new_json(root / "gate_archive_receipt.json", value)


class PilotSelectionTests(unittest.TestCase):
    def test_authenticated_copy_must_recompute_the_same_winners(self):
        rows = pilot_fixture()
        for i, row in enumerate(rows):
            row["run_id"] = f"pilot_fixture_{i}"
        rows[1]["validation"]["binding"] = 0.6
        runs = {"pilots": rows, "memorization": []}
        runs_hash = hashlib.sha256(canonical_bytes(runs)).hexdigest()
        selection = {"context": deepcopy(CONTEXT), "final_performance_data_generated": False,
                     "rule": RECIPE["training"]["lr_selection"], "selected_learning_rates": deepcopy(RATES),
                     "pilot_runs": [row["run_id"] for row in rows],
                     "development_identities": CONTEXT["development_identities"],
                     "pilot_records": {"path": "preflight_runs.json", "sha256": runs_hash,
                                       "bytes": len(canonical_bytes(runs))}}
        evidence = {"schema_version": 1, "item": 9, "recipe_sha256": CONTEXT["recipe_sha256"],
                    "preflight_source_commit": CONTEXT["source_commit"],
                    "selection": selection, "preflight_runs": runs,
                    "selection_sha256": hashlib.sha256(canonical_bytes(selection)).hexdigest(),
                    "preflight_runs_sha256": runs_hash}
        self.assertEqual(validate_pilot_evidence(evidence, RECIPE, CONTEXT["recipe_sha256"])[0], RATES)
        bad = deepcopy(evidence)
        bad["selection"]["selected_learning_rates"]["neuropixel"] = 0.001
        bad["selection_sha256"] = hashlib.sha256(canonical_bytes(bad["selection"])).hexdigest()
        with self.assertRaises(ValueError):
            validate_pilot_evidence(bad, RECIPE, CONTEXT["recipe_sha256"])
        with self.assertRaises(ValueError):
            validate_pilot_evidence(evidence, RECIPE, "0" * 64)

    def test_score_then_ce_then_lower_rate_without_cherry_picking(self):
        rows = pilot_fixture()
        rows[1]["validation"]["binding"] = 0.6
        self.assertEqual(select_pilot(rows, RECIPE), RATES)
        rows[0]["validation"]["binding"] = 0.6
        rows[1]["validation"]["cross_entropy"] = 0.8
        self.assertEqual(select_pilot(rows, RECIPE), RATES)
        rows[0]["validation"]["cross_entropy"] = 0.8
        self.assertEqual(select_pilot(rows, RECIPE), {"neuropixel": 0.001, "relative_transformer": 0.001})

    def test_rejects_incomplete_duplicate_failed_or_mixed_source_pilot(self):
        cases = []
        rows = pilot_fixture(); cases.append(rows[:-1])
        rows = pilot_fixture(); rows[3] = deepcopy(rows[2]); cases.append(rows)
        rows = pilot_fixture(); rows[0]["status"] = "failed"; cases.append(rows)
        rows = pilot_fixture(); rows[0]["updates_completed"] = 1023; cases.append(rows)
        rows = pilot_fixture(); rows[0]["context"]["source_commit"] = "f" * 40; cases.append(rows)
        rows = pilot_fixture(); rows[0]["validation"]["binding"] = float("nan"); cases.append(rows)
        rows = pilot_fixture(); rows[0]["phase"] = "primary"; cases.append(rows)
        for rows in cases:
            with self.subTest(case=cases.index(rows)), self.assertRaises(ValueError):
                select_pilot(rows, RECIPE)


class FinalGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="item9_gate_fixture_")
        self.root = Path(self.temporary.name)
        primary_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_ten_run_inventory_and_consumption_precede_data(self):
        gate = create_final_inventory(self.root, RECIPE, RATES, CONTEXT)
        self.assertEqual(len(gate["entries"]), 10)
        archive_fixture(self.root)
        claim_final_access(self.root, RECIPE, RATES, CONTEXT)
        self.assertFalse((self.root / "final_data").exists())
        access = read_json(self.root / "final_access.json")
        self.assertTrue(access["consumed_before_final_generation"])
        self.assertFalse(access["retry_allowed"])
        with self.assertRaises(FileExistsError):
            claim_final_access(self.root, RECIPE, RATES, CONTEXT)

    def test_incomplete_panel_blocks_gate(self):
        p = self.root / "primary_runs.json"
        index = read_json(p); index["runs"].pop(); p.write_bytes(canonical_bytes(index))
        with self.assertRaises(ValueError):
            create_final_inventory(self.root, RECIPE, RATES, CONTEXT)
        self.assertFalse((self.root / "final_inventory.json").exists())

    def test_changed_training_identity_blocks_gate(self):
        p = self.root / "runs/neuropixel_seed40/run.json"
        row = read_json(p); row["context"]["development_identities"] = {"changed": True}; p.write_bytes(canonical_bytes(row))
        with self.assertRaises(ValueError):
            create_final_inventory(self.root, RECIPE, RATES, CONTEXT)

    def test_all_four_artifact_types_rechecked_after_inventory(self):
        gate = create_final_inventory(self.root, RECIPE, RATES, CONTEXT)
        archive_fixture(self.root)
        row = read_json(self.root / gate["entries"][0]["summary"]["path"])
        for name in ("checkpoint", "training_log", "probe_predictions", "validation_predictions"):
            path = self.root / row["artifacts"][name]["path"]
            original = path.read_bytes()
            path.write_bytes(original + b"corruption")
            with self.subTest(artifact=name), self.assertRaises(ValueError):
                claim_final_access(self.root, RECIPE, RATES, CONTEXT)
            self.assertFalse((self.root / "final_access.json").exists())
            path.write_bytes(original)

    def test_wrong_remote_inventory_receipt_blocks_access(self):
        create_final_inventory(self.root, RECIPE, RATES, CONTEXT)
        archive_fixture(self.root, final_inventory_sha256="0" * 64)
        with self.assertRaises(ValueError):
            claim_final_access(self.root, RECIPE, RATES, CONTEXT)
        self.assertFalse((self.root / "final_access.json").exists())

    def test_existing_final_data_blocks_gate_and_access(self):
        (self.root / "final_data").mkdir()
        with self.assertRaises(ValueError):
            create_final_inventory(self.root, RECIPE, RATES, CONTEXT)

    def test_artifact_outside_output_is_rejected(self):
        p = self.root / "runs/neuropixel_seed40/run.json"
        row = read_json(p); row["artifacts"]["checkpoint"]["path"] = "../outside.fixture"; p.write_bytes(canonical_bytes(row))
        with self.assertRaises(ValueError):
            create_final_inventory(self.root, RECIPE, RATES, CONTEXT)


class ScenarioMetricTests(unittest.TestCase):
    def test_balanced_scenario_metrics_match_hand_calculation(self):
        roles = [0, 1, 2, 3] * 4
        targets = [5, 17, 6, 27, 7, 18, 8, 28] * 2
        pred = targets[:8] + [0, 17, 6, 0, 0, 18, 8, 0]
        score = point_metrics(pred, targets, roles, [1.25] * 16, [0] * 8 + [1] * 8)
        self.assertEqual(score["correct"], 12)
        self.assertEqual(score["global"], 0.75)
        self.assertEqual(score["per_role"], [0.5, 1.0, 1.0, 0.5])
        self.assertEqual(score["binding"], 0.75)
        self.assertEqual(score["all_eight"], 0.5)
        self.assertEqual(score["cross_entropy"], 1.25)

    def test_pseudoreplication_and_missing_role_blocks_are_rejected(self):
        target = [5, 17, 6, 27, 7, 18, 8, 28]
        with self.assertRaises(ValueError):
            point_metrics(target, target, [0, 1, 2, 3] * 2, [0.0] * 8, list(range(8)))
        with self.assertRaises(ValueError):
            point_metrics(target, target, [0, 1, 2, 3, 0, 1, 1, 3], [0.0] * 8, [0] * 8)
        with self.assertRaises(ValueError):
            point_metrics(target, target, [0, 1, 2, 3] * 2, None, [0] * 8)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "pinned CPU Torch runtime required")
class ModelIntegrationTests(unittest.TestCase):
    def test_one_update_driver_and_checkpoint_prediction_roundtrip(self):
        import torch
        from neuropixel.research import complex_binding_data as data
        from scripts.research_complex_binding import model_for, panel_arrays, predict, render_panel, run_training
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        # This is the same bag as the registered manual grammar fixture, which
        # is excluded from every performance population. Two optimizer updates
        # total exercise serialization and driver integration, not competence.
        bag = {"nouns": [5, 6, 7, 8], "verbs": [17, 18], "places": [27, 28]}
        truth = [[5, 17, 6, 27], [7, 18, 8, 28]]
        scenario = {"schema_version": 1, "group_id": data.group_id(bag), "bag": bag,
                    "facts": [{"event": e, "role": r, "filler": truth[e][r]} for e in range(2) for r in range(4)],
                    "layout": [{"row": i, "start": 0} for i in range(8)]}
        self.assertIn(scenario["group_id"], data.fixture_exposure_ledger()["excluded_groups"])
        arrays = panel_arrays(render_panel([scenario], ("base",)))
        panels = {"probe": arrays, "validation": arrays}
        recipe = deepcopy(RECIPE)
        recipe["training"]["batch_size"] = 8
        with tempfile.TemporaryDirectory(prefix="item9_driver_fixture_") as directory:
            output = Path(directory)
            for family in ("neuropixel", "relative_transformer"):
                result = run_training(output, f"fixture_{family}", "contract_fixture", family,
                                      990921, 0.001, 1, recipe, {"fixture": True}, [scenario], panels)
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual(result["updates_completed"], 1)
                checkpoint = torch.load(output / result["artifacts"]["checkpoint"]["path"],
                                        map_location="cpu", weights_only=True)
                model = model_for(family, recipe)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                actual = predict(model, arrays, 64)
                with np.load(output / result["artifacts"]["validation_predictions"]["path"], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(actual["pred"], saved["pred"])
                    np.testing.assert_array_equal(actual["logits"], saved["logits"])
                    np.testing.assert_array_equal(actual["nll"], saved["nll"])

    def test_actual_model_classes_and_corrected_pad_on_new_grid(self):
        import torch
        from scripts.research_complex_binding import model_for
        from neuropixel.model import NeuroPixel, n_params
        from neuropixel.research.models import RelativeTransformer
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        canvas = torch.zeros(2, 10, 8, dtype=torch.long)
        canvas[:, 0, :3] = torch.tensor([35, 1, 5])
        canvas[:, 9, 5:7] = torch.tensor([35, 1])
        expected = {"neuropixel": (NeuroPixel, 29856), "relative_transformer": (RelativeTransformer, 30157)}
        for family, (klass, count) in expected.items():
            with self.subTest(family=family):
                torch.manual_seed(990901)
                model = model_for(family, RECIPE)
                self.assertIs(type(model), klass)
                self.assertEqual(n_params(model), count)
                if family == "neuropixel":
                    with torch.no_grad():
                        model.embed.weight[0].fill_(3.0)
                    self.assertEqual(torch.count_nonzero(model.dictionary()[0]).item(), 0)
                model.eval()
                with torch.inference_mode():
                    result = model(canvas)
                self.assertEqual(tuple(result["logits"].shape), (2, 37))
                self.assertTrue(torch.isfinite(result["logits"]).all())
                self.assertTrue((result["logits"][:, 0] == -10000).all())


if __name__ == "__main__":
    unittest.main()
