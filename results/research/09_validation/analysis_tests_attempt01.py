"""Synthetic/manual artifact regressions; no study populations or model imports."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("independent_binding_audit", ROOT/"scripts/research_complex_binding_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture():
    """Use only the manual bag already excluded by the frozen fixture ledger."""
    truth = ((5, 17, 6, 27), (7, 18, 8, 28))
    bag = {"nouns": [5, 6, 7, 8], "verbs": [17, 18], "places": [27, 28]}
    scenario = {"schema_version": 1, "group_id": identity(bag), "bag": bag,
        "facts": [{"event": event, "role": role, "filler": truth[event][role]} for event in range(2) for role in range(4)],
        "layout": [{"row": row, "start": 0} for row in range(8)]}
    records = []
    for ci, condition in enumerate(AUDIT.CONDITIONS):
        for event in range(2):
            for role in range(4):
                canvas = [[0]*8 for _ in range(10)]
                for e in range(2):
                    for r in range(4):
                        canvas[e*4+r][:3] = [35+e, 1+r, truth[e][r]]
                visible_event, expected = event, truth[event][role]
                if ci in (1, 2):
                    affected = event if ci == 1 else 1-event
                    a, p = affected*4, affected*4+2
                    canvas[a][2], canvas[p][2] = canvas[p][2], canvas[a][2]
                    if ci == 1 and role in (0, 2):
                        expected = truth[event][2-role]
                elif ci == 3:
                    for row in range(8):
                        canvas[row][0] = 71-canvas[row][0]
                    visible_event = 1-event
                elif ci == 4:
                    visible_event = 1-event
                    expected = truth[1-event][role]
                elif ci == 5:
                    canvas[:8] = canvas[1:8]+canvas[:1]
                canvas[9][5:7] = [35+visible_event, 1+role]
                sid = identity(scenario)
                pid = identity({"scenario_id": sid, "base_query_event": event, "query_role": role})
                record = {"scenario_id": sid, "pair_id": pid, "group_id": scenario["group_id"],
                    "canvas": canvas, "target": expected, "base_target": truth[event][role], "condition": condition,
                    "base_query_event": event, "query_event": visible_event, "query_role": role,
                    "changed_gold": expected != truth[event][role], "group_index": 0, "condition_index": ci}
                record["record_id"] = identity({"pair_id": pid, "condition": condition, "canvas": canvas})
                records.append(record)
    arrays = {}
    for name in AUDIT.ARRAY_FIELDS:
        key = "query_role" if name == "role" else name
        dtype = bool if name == "changed_gold" else str if name in ("group_id", "record_id", "pair_id") else np.int64
        arrays[name] = np.asarray([row[key] for row in records], dtype=dtype)
    return arrays, scenario, records


def predictions(arrays, *, constant=None):
    target = arrays["target"]
    pred = target.copy() if constant is None else np.full(len(target), constant, dtype=np.int64)
    logits = np.full((len(target), 37), -2, dtype=np.float32)
    logits[:, 0] = -10000
    logits[np.arange(len(target)), pred] = 2
    exact_correct_nll = math.log1p(35*math.exp(-4)+math.exp(-10002))
    values = {name: arrays[name].copy() for name in ("target", "role", "group_index", "condition_index", "record_id")}
    values.update(logits=logits, pred=pred, nll=np.where(pred == target, exact_correct_nll, exact_correct_nll+4))
    return values


class ComplexBindingAuditTests(unittest.TestCase):
    def test_manual_visible_truth_and_all_transformations(self):
        arrays, scenario, records = fixture()
        parsed = AUDIT.validate_dataset(arrays, AUDIT.CONDITIONS, 1)
        AUDIT.align_scenarios([scenario], arrays, records)
        self.assertEqual([row["gold"] for row in parsed], arrays["target"].tolist())
        self.assertEqual(arrays["changed_gold"].reshape(6, 8).sum(1).tolist(), [0, 4, 0, 0, 8, 0])

    def test_parser_rejects_duplicate_fact_malformed_query_and_category(self):
        arrays, _, _ = fixture()
        for position, value in (((4, 0), 35), ((9, 7), 5), ((0, 2), 17)):
            canvas = arrays["canvas"][0].copy(); canvas[position] = value
            with self.subTest(position=position), self.assertRaises(AUDIT.AuditError):
                AUDIT.parse_visible(canvas)

    def test_self_consistent_wrong_intervention_fails(self):
        arrays, _, _ = fixture()
        # Replace a queried-event swap by an other-event swap and repair its
        # label/flag. Input-only gold is now consistent, intervention is not.
        index = 8
        arrays["canvas"][index] = arrays["canvas"][16]
        arrays["target"][index] = arrays["target"][16]
        arrays["changed_gold"][index] = False
        self.assertEqual(AUDIT.parse_visible(arrays["canvas"][index])["gold"], arrays["target"][index])
        with self.assertRaisesRegex(AUDIT.AuditError, "transformation"):
            AUDIT.validate_dataset(arrays, AUDIT.CONDITIONS, 1)

    def test_query_metadata_pair_and_record_hash_corruption_fail(self):
        for field in ("query_event", "base_target", "changed_gold", "pair_id"):
            arrays, _, _ = fixture()
            arrays[field][8] = "0"*64 if field == "pair_id" else not arrays[field][8] if field == "changed_gold" else 1
            with self.subTest(field=field), self.assertRaises(AUDIT.AuditError):
                AUDIT.validate_dataset(arrays, AUDIT.CONDITIONS, 1)
        arrays, scenario, records = fixture()
        arrays["record_id"][0] = "0"*64
        with self.assertRaisesRegex(AUDIT.AuditError, "identity"):
            AUDIT.align_scenarios([scenario], arrays, records)

    def test_identity_layout_and_query_varying_layout_fail(self):
        arrays, _, _ = fixture()
        arrays["canvas"][40] = arrays["canvas"][0]
        with self.assertRaisesRegex(AUDIT.AuditError, "identity"):
            AUDIT.validate_dataset(arrays, AUDIT.CONDITIONS, 1)
        arrays, _, _ = fixture()
        arrays["canvas"][41, :8] = np.roll(arrays["canvas"][41, :8], 1, axis=0)
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.validate_dataset(arrays, AUDIT.CONDITIONS, 1)

    def test_stable_nll_and_counts_match_manual_values(self):
        arrays, _, _ = fixture()
        values = predictions(arrays)
        pred, nll, error = AUDIT.recount_predictions(values, arrays)
        self.assertLess(error, 1e-12)
        metrics = AUDIT.point_metrics(pred[:8], arrays["target"][:8], arrays["role"][:8], nll[:8], arrays["group_index"][:8])
        self.assertEqual((metrics["correct"], metrics["n"], metrics["groups"], metrics["binding"], metrics["all_eight"]), (8, 8, 1, 1, 1))
        self.assertAlmostEqual(metrics["cross_entropy"], math.log1p(35*math.exp(-4)), places=13)

    def test_prediction_nll_alignment_and_nonfinite_corruption_fail(self):
        for field in ("pred", "target", "nll", "logits", "role"):
            arrays, _, _ = fixture(); values = predictions(arrays)
            if field == "logits": values[field][0, 2] = np.nan
            elif field == "nll": values[field][0] += 0.01
            else: values[field][0] += 1
            with self.subTest(field=field), self.assertRaises(AUDIT.AuditError):
                AUDIT.recount_predictions(values, arrays)

    def test_argmax_tie_uses_first_index(self):
        arrays, _, _ = fixture(); values = predictions(arrays)
        values["logits"][0, 4] = 2
        with self.assertRaisesRegex(AUDIT.AuditError, "first-index"):
            AUDIT.recount_predictions(values, arrays)

    def test_analytic_controls_are_marginals_not_joint_draws(self):
        arrays, _, _ = fixture(); parsed = AUDIT.validate_dataset(arrays, AUDIT.CONDITIONS, 1)
        noun = np.isin(arrays["role"], [0, 2])
        majority = {"0": 5, "1": 17, "2": 6, "3": 27}
        controls = {"symbolic": np.ones(48), "role_only": np.full(48, .5),
            "event_category": np.where(noun, .5, 1), "bag_category": np.where(noun, .25, .5),
            "train_role_majority": np.asarray([float(majority[str(role)] == target) for role, target in zip(arrays["role"], arrays["target"])])}
        result = AUDIT.verify_controls(controls, arrays, parsed, majority)
        self.assertEqual(result["bag_category"]["global"], .375)
        self.assertEqual(result["event_category"]["binding"], .5)
        self.assertFalse(any("all_eight" in row or "joint" in row for row in result.values()))
        controls["role_only"][0] = 1
        with self.assertRaises(AUDIT.AuditError): AUDIT.verify_controls(controls, arrays, parsed, majority)

    def test_counterfactual_eligibility_and_joint_accuracy(self):
        arrays, _, _ = fixture(); values = predictions(arrays)
        indices = np.zeros((17, 1), dtype=np.int64)
        diagnostics, _ = AUDIT.condition_diagnostics(values["pred"], arrays, values["nll"], indices)
        changed = diagnostics[AUDIT.CONDITIONS[1]]["counterfactual_vs_base"]["all"]
        self.assertEqual(changed["invariant_prediction_equal"]["denominator"], 4)
        self.assertEqual(changed["changed_prediction_different"]["denominator"], 4)
        self.assertEqual(changed["joint_correct"]["estimate"], 1)
        switched = diagnostics["query_switch"]["counterfactual_vs_base"]["all"]
        self.assertEqual(switched["invariant_prediction_equal"]["denominator"], 0)
        self.assertIsNone(switched["invariant_prediction_equal"]["estimate"])
        self.assertEqual(switched["changed_prediction_different"]["denominator"], 8)

    def test_constant_prediction_invariance_does_not_imply_correct_binding(self):
        arrays, _, _ = fixture(); values = predictions(arrays, constant=5)
        diagnostics, _ = AUDIT.condition_diagnostics(values["pred"], arrays, values["nll"], np.zeros((5, 1), dtype=np.int64))
        other = diagnostics["swap_other_agent_patient"]["counterfactual_vs_base"]["all"]
        self.assertEqual(other["invariant_prediction_equal"]["estimate"], 1)
        self.assertEqual(other["joint_correct"]["estimate"], 1/8)
        self.assertEqual(diagnostics["query_switch"]["counterfactual_vs_base"]["all"]["changed_prediction_different"]["estimate"], 0)

    def test_five_training_realization_interval_known_values(self):
        left = dict(zip(AUDIT.SEEDS, [.1, .2, .3, .4, .5])); right = dict.fromkeys(AUDIT.SEEDS, 0)
        result = AUDIT.paired_seed_interval(left, right)
        sd = math.sqrt(.025)
        half = 2.7764451051977987*sd/math.sqrt(5)
        self.assertAlmostEqual(result["mean"], .3)
        self.assertAlmostEqual(result["sample_sd"], sd)
        np.testing.assert_allclose(result["t95"], [.3-half, .3+half], rtol=1e-12)
        self.assertEqual((result["n"], result["df"]), (5, 4))

    def test_incomplete_nonfinite_and_unclipped_seed_intervals(self):
        zero = dict.fromkeys(AUDIT.SEEDS, 0)
        with self.assertRaises(AUDIT.AuditError): AUDIT.paired_seed_interval({40: 1}, {40: 0})
        invalid = dict(zero); invalid[40] = float("nan")
        with self.assertRaises(AUDIT.AuditError): AUDIT.paired_seed_interval(invalid, zero)
        left = dict(zip(AUDIT.SEEDS, [0, 0, 0, 0, 1]))
        self.assertLess(AUDIT.paired_seed_interval(left, zero)["t95"][0], 0)
        self.assertEqual(AUDIT.paired_seed_interval(zero, zero)["t95"], [0, 0])

    def test_common_bootstrap_is_whole_bag_and_signed(self):
        indices = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.int64)
        result = AUDIT.ratio_interval(np.array([1, -1]), np.ones(2), indices)
        self.assertEqual(result["estimate"], 0)
        np.testing.assert_allclose(result["bag_bootstrap95"], [-.925, .925], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(AUDIT.bootstrap_indices(3, 20), AUDIT.bootstrap_indices(3, 20))
        self.assertEqual(AUDIT.bootstrap_indices(3, 20).shape, (20, 3))
        with self.assertRaises(AUDIT.AuditError): AUDIT.ratio_interval([1, 2], [0, 1], indices)

    def test_duplicate_canvas_count_and_disagreement_detection(self):
        arrays, _, _ = fixture(); values = predictions(arrays)
        result = AUDIT.duplicate_agreement(arrays, values)
        self.assertEqual((result["unique_canvases"], result["duplicate_rows_compared_to_first"]), (40, 8))
        values["pred"][32] = 5
        with self.assertRaises(AUDIT.AuditError): AUDIT.duplicate_agreement(arrays, values)

    def test_reader_detects_corruption_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"artifact.json"; path.write_text('{"x":1}')
            reader = AUDIT.Reader(directory)
            record = {"path": "artifact.json", "bytes": path.stat().st_size, "sha256": AUDIT.sha(path)}
            self.assertEqual(reader.verify(record), path)
            path.write_text('{"x":2}')
            with self.assertRaises(AUDIT.AuditError): reader.verify(record)
            with self.assertRaises(AUDIT.AuditError): reader.path("../outside.json")
        with self.assertRaises(AUDIT.AuditError): AUDIT.decode_json('{"x":1,"x":2}')
        with self.assertRaises(AUDIT.AuditError): AUDIT.decode_json('{"x":NaN}')

    def test_matched_training_stream_uses_every_update(self):
        lines = [{"step": index+1, "input_sha256": hashlib.sha256(bytes([index])).hexdigest(),
                  "training_group_indices": [0, 1], "loss": .5, "gradient_norm_before_clip": .1,
                  "elapsed_seconds": index+.1} for index in range(3)]
        digest = hashlib.sha256(b"".join(bytes.fromhex(row["input_sha256"]) for row in lines)).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"training.jsonl"
            path.write_text("".join(json.dumps(row)+"\n" for row in lines))
            result = AUDIT.training_stream(path, 3, {"training_input_sequence_sha256": digest}, 2, 2)
            self.assertEqual(len(result), 3)
            with self.assertRaises(AUDIT.AuditError): AUDIT.training_stream(path, 2, {"training_input_sequence_sha256": digest}, 2, 2)
            lines[1]["step"] = 1
            path.write_text("".join(json.dumps(row)+"\n" for row in lines))
            with self.assertRaises(AUDIT.AuditError): AUDIT.training_stream(path, 3, {"training_input_sequence_sha256": digest}, 2, 2)

    def test_saved_point_metric_corruption_fails(self):
        arrays, _, _ = fixture(); values = predictions(arrays)
        expected = AUDIT.point_metrics(values["pred"][:8], arrays["target"][:8], arrays["role"][:8], values["nll"][:8], arrays["group_index"][:8])
        AUDIT.compare_metrics(expected, expected)
        wrong = copy.deepcopy(expected); wrong["correct"] = 7
        with self.assertRaises(AUDIT.AuditError): AUDIT.compare_metrics(wrong, expected)
        wrong = copy.deepcopy(expected); wrong["binding"] = .5
        with self.assertRaises(AUDIT.AuditError): AUDIT.compare_metrics(wrong, expected)


if __name__ == "__main__":
    unittest.main()
