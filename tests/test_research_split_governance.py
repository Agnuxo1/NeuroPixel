"""Synthetic misuse checks for the prospective local governance boundary.

These fixtures are tiny JSON datasets and opaque checkpoint bytes, not model
outputs or scientific results. The tests use only the Python standard library.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neuropixel.research import split_governance as gov


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


class GovernanceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.directory = self.workspace / "study"
        self.study = gov.FinalStudy(self.workspace, self.directory)
        self.specification = {
            "generator_id": "synthetic-unit-fixture-v1",
            "generator_source_sha256": sha_bytes(b"generator-v1"),
            "vocabulary_sha256": sha_bytes(b"vocabulary-v1"),
            "shape": [2, 2],
            "generation_parameters": {"layout": "fixed"},
            "sampling_seed": 17,
            "rng_runtime": "no random draws; literal stdlib fixture",
            "grouping_rule": "literal synthetic scene identity",
            "grouping_source_sha256": sha_bytes(b"grouping-v1"),
            "transformations": [],
            "preprocessing_fit_partitions": [],
        }
        self.partitions = {}
        self.payloads = {}
        for split in gov.PARTITIONS:
            examples = [{"scene": f"{split}-{i}", "label": i} for i in range(4)]
            payload = gov.canonical_bytes(examples)
            self.payloads[split] = payload
            (self.workspace / f"{split}.json").write_bytes(payload)
            self.partitions[split] = {
                "artifact_sha256": sha_bytes(payload),
                "artifact_bytes": len(payload),
                "records": [
                    {"record_id": f"{split}-{i}", "group_id": f"scene-{split}-{i}",
                     "content_sha256": gov.digest(example)}
                    for i, example in enumerate(examples)
                ],
            }
        self.history = {
            "scope": "synthetic fixtures created by this test only",
            "coverage": "complete_within_declared_scope",
            "evidence_sha256": sha_bytes(b"fixture-history"),
            "exposures": [],
        }
        self.protocol = {
            "protocol_id": "unit-fixture-contract-v1",
            "mode": "confirmatory",
            "candidate_ids": ["a", "b"],
            "selection_rule": list(gov.SELECTION_RULE),
            "validation_artifact_sha256": self.partitions["validation"]["artifact_sha256"],
            "validation_denominators": {"agent": 1, "patient": 1, "all": 4},
        }
        self.candidates = []
        for cid, binding in [("a", 0.5), ("b", 0.4)]:
            config = f"configuration_{cid}.json"
            checkpoint = f"checkpoint_{cid}.bin"
            (self.workspace / config).write_bytes(gov.canonical_bytes({"candidate": cid}))
            (self.workspace / checkpoint).write_bytes(f"opaque checkpoint {cid}".encode())
            self.candidates.append({
                "candidate_id": cid, "status": "completed",
                "configuration_path": config,
                "configuration_sha256": gov.file_digest(self.workspace / config),
                "checkpoint_path": checkpoint,
                "checkpoint_sha256": gov.file_digest(self.workspace / checkpoint),
                "validation_artifact_sha256": self.protocol["validation_artifact_sha256"],
                "binding_accuracy": binding, "cross_entropy": 1.2,
                "parameter_count": 10,
                "denominators": dict(self.protocol["validation_denominators"]),
            })
        (self.workspace / "implementation.py").write_text("# synthetic implementation v1\n")

    def manifest(self, **changes):
        arguments = {"dataset_id": "synthetic-dataset", "specification": self.specification,
                     "partitions": self.partitions, "history": self.history}
        arguments.update(changes)
        return gov.make_manifest(**deepcopy(arguments))

    def freeze(self, *, study=None, **changes):
        arguments = {"manifest": self.manifest(), "protocol": self.protocol,
                     "candidates": self.candidates, "source_paths": ["implementation.py"]}
        arguments.update(changes)
        return (study or self.study).freeze(**deepcopy(arguments))

    def pin(self):
        return gov.file_digest(self.directory / "freeze.json")

    def evaluate(self, callback, **changes):
        arguments = {"expected_freeze_sha256": self.pin(), "final_artifact": "final.json",
                     "callback": callback}
        arguments.update(changes)
        return self.study.evaluate(**arguments)


class ManifestTests(GovernanceFixture):
    def test_order_and_generator_are_part_of_identity_and_old_identity_cannot_be_reused(self):
        original = self.manifest()
        reordered = deepcopy(self.partitions)
        reordered["validation"]["records"].reverse()
        changed_generator = deepcopy(self.specification)
        changed_generator["generator_id"] = "synthetic-unit-fixture-v2"
        new_manifests = [self.manifest(partitions=reordered),
                         self.manifest(specification=changed_generator)]
        for changed in new_manifests:
            with self.subTest(identity=changed["identity_sha256"]):
                self.assertNotEqual(original["identity_sha256"], changed["identity_sha256"])
                changed["identity_sha256"] = original["identity_sha256"]
                with self.assertRaises(gov.ContractError):
                    gov.validate_manifest(changed)
        self.assertEqual(original, gov.validate_manifest(original))

    def test_cross_partition_groups_content_and_duplicate_record_ids_are_rejected(self):
        for field in ["group_id", "content_sha256", "record_id"]:
            with self.subTest(field=field):
                partitions = deepcopy(self.partitions)
                partitions["final"]["records"][0][field] = partitions["train"]["records"][0][field]
                with self.assertRaises(gov.ContractError):
                    self.manifest(partitions=partitions)
        partitions = deepcopy(self.partitions)
        partitions["train"]["records"][1]["record_id"] = partitions["train"]["records"][0]["record_id"]
        with self.assertRaises(gov.ContractError):
            self.manifest(partitions=partitions)

    def test_repeated_content_with_distinct_ids_is_retained_within_one_partition(self):
        partitions = deepcopy(self.partitions)
        first = partitions["train"]["records"][0]
        partitions["train"]["records"][1] = {**first, "record_id": "train-repeat"}
        manifest = self.manifest(partitions=partitions)
        self.assertEqual(len(manifest["partitions"]["train"]["records"]), 4)
        self.assertEqual(manifest["partitions"]["train"]["records"][1]["content_sha256"], first["content_sha256"])

    def test_manifest_detaches_inputs_and_forbids_selection_or_final_fitted_preprocessing(self):
        manifest = self.manifest()
        self.partitions["train"]["records"][0]["group_id"] = "changed-after-registration"
        self.specification["shape"][0] = 99
        self.assertEqual(gov.validate_manifest(manifest), manifest)
        for fitted_on in [["validation"], ["final"], ["train", "validation"]]:
            with self.subTest(fitted_on=fitted_on):
                specification = deepcopy(manifest["specification"])
                specification["preprocessing_fit_partitions"] = fitted_on
                with self.assertRaises(gov.ContractError):
                    self.manifest(specification=specification)

    def test_confirmatory_refuses_exposed_final_groups_and_incomplete_history(self):
        histories = []
        for coverage in ["partial", "unknown"]:
            histories.append({**deepcopy(self.history), "coverage": coverage})
        for use in sorted(gov.EXPOSURE_USES):
            histories.append({**deepcopy(self.history), "exposures": [
                {"group_id": "scene-final-0", "uses": [use]}]})
        for history in histories:
            with self.subTest(history=history):
                with self.assertRaises(gov.ContractError):
                    self.freeze(manifest=self.manifest(history=history))
                self.assertFalse((self.directory / "freeze.json").exists())
        # Exposed populations remain usable for an explicitly exploratory study.
        receipt = self.freeze(manifest=self.manifest(history=histories[-1]),
                              protocol={**self.protocol, "mode": "exploratory"})
        self.assertFalse(receipt["manifest"]["no_conflict_in_declared_complete_history"])
        self.assertEqual(receipt["manifest"]["previously_exposed_final_groups"], ["scene-final-0"])


class SelectionTests(GovernanceFixture):
    def test_exact_inventory_population_rule_and_complete_status_are_required(self):
        scenarios = []
        scenarios.append(({**self.protocol, "candidate_ids": ["a", "a"]}, self.candidates))
        scenarios.append((self.protocol, self.candidates[:1]))
        scenarios.append((self.protocol, [self.candidates[0], self.candidates[0]]))
        for change in [{"candidate_id": "foreign"}, {"status": "failed"},
                       {"validation_artifact_sha256": sha_bytes(b"another population")},
                       {"denominators": {"agent": 1, "patient": 1, "all": 3}}]:
            changed = deepcopy(self.candidates)
            changed[0].update(change)
            scenarios.append((self.protocol, changed))
        scenarios.append(({**self.protocol, "selection_rule": list(reversed(gov.SELECTION_RULE))}, self.candidates))
        scenarios.append(({**self.protocol, "validation_artifact_sha256": sha_bytes(b"wrong")}, self.candidates))
        for protocol, candidates in scenarios:
            with self.subTest(protocol=protocol, candidates=candidates):
                with self.assertRaises(gov.ContractError):
                    self.freeze(protocol=protocol, candidates=candidates)
                self.assertFalse((self.directory / "freeze.json").exists())

    def test_agreed_denominators_must_still_match_manifest_record_count(self):
        protocol = deepcopy(self.protocol)
        protocol["validation_denominators"] = {"agent": 100, "patient": 100, "all": 400}
        candidates = deepcopy(self.candidates)
        for row in candidates:
            row["denominators"] = dict(protocol["validation_denominators"])
        with self.assertRaises(gov.ContractError):
            self.freeze(protocol=protocol, candidates=candidates)

    def test_invalid_metric_types_nonfinite_values_and_counts_cannot_enter_selection(self):
        changes = [(key, value) for key in ["binding_accuracy", "cross_entropy"]
                   for value in [float("nan"), float("inf"), -float("inf"), -1, True, "0.5"]]
        changes += [("binding_accuracy", 1.01), ("parameter_count", 0), ("parameter_count", True)]
        changes.append(("denominators", {"agent": True, "patient": 1, "all": 4}))
        for key, value in changes:
            with self.subTest(key=key, value=value):
                candidates = deepcopy(self.candidates)
                candidates[0][key] = value
                with self.assertRaises(gov.ContractError):
                    self.freeze(candidates=candidates)
                self.assertFalse((self.directory / "freeze.json").exists())

    def test_declared_tie_breakers_are_order_independent(self):
        # Each case makes one successive criterion decisive; no final metric is supplied.
        cases = [({"binding_accuracy": .6, "cross_entropy": 9}, "b"),
                 ({"binding_accuracy": .5, "cross_entropy": 1}, "b"),
                 ({"binding_accuracy": .5, "parameter_count": 9}, "b"),
                 ({"binding_accuracy": .5}, "a")]
        for index, (change, expected) in enumerate(cases):
            with self.subTest(change=change):
                candidates = deepcopy(self.candidates)
                candidates[1].update(change)
                study = gov.FinalStudy(self.workspace, self.workspace / f"tie-{index}")
                receipt = self.freeze(study=study, candidates=list(reversed(candidates)))
                self.assertEqual(receipt["selected_candidate_id"], expected)

    def test_freeze_checks_candidate_bytes_and_cannot_be_replaced(self):
        changed = deepcopy(self.candidates)
        changed[0]["checkpoint_sha256"] = sha_bytes(b"different checkpoint")
        with self.assertRaises(gov.ContractError):
            self.freeze(candidates=changed)
        self.freeze()
        original = (self.directory / "freeze.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.freeze()
        self.assertEqual(original, (self.directory / "freeze.json").read_bytes())


class FinalAccessTests(GovernanceFixture):
    def test_source_configuration_checkpoint_and_external_freeze_pin_prevent_drift(self):
        self.freeze()
        pin = self.pin()
        called = []
        callback = lambda data, frozen: called.append(True) or {}
        for relative in ["implementation.py", "configuration_a.json", "checkpoint_b.bin"]:
            with self.subTest(relative=relative):
                path = self.workspace / relative
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                try:
                    with self.assertRaises(gov.ContractError):
                        self.evaluate(callback, expected_freeze_sha256=pin)
                    self.assertFalse((self.directory / "final_access.json").exists())
                finally:
                    path.write_bytes(original)
        freeze_path = self.directory / "freeze.json"
        altered = gov.read_json(freeze_path)
        altered["selected_candidate_id"] = "b"
        freeze_path.write_bytes(gov.canonical_bytes(altered))
        with self.assertRaises(gov.ContractError):
            self.evaluate(callback, expected_freeze_sha256=pin)
        self.assertFalse(called)
        self.assertFalse((self.directory / "final_access.json").exists())

    def test_access_receipt_exists_before_final_path_is_resolved_and_callback_runs(self):
        self.freeze()
        original_resolver = gov._relative_file
        seen = []

        def resolver(workspace, relative):
            if relative == "final.json":
                access = gov.read_json(self.directory / "final_access.json")
                self.assertTrue(access["access_is_consumed_on_failure"])
                seen.append("resolve")
            return original_resolver(workspace, relative)

        def callback(payload, frozen):
            self.assertEqual(payload, self.payloads["final"])
            self.assertEqual(frozen["selected_candidate_id"], "a")
            self.assertTrue((self.directory / "final_access.json").exists())
            seen.append("callback")
            frozen["selected_candidate_id"] = "mutated callback copy"
            return {"synthetic_score": 0.5}

        with patch.object(gov, "_relative_file", resolver):
            result = self.evaluate(callback)
        self.assertEqual(seen, ["resolve", "callback"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(gov.read_json(self.directory / "freeze.json")["selected_candidate_id"], "a")
        with self.assertRaises(FileExistsError):
            self.evaluate(lambda *_: self.fail("a completed final evaluation was reopened"))

    def test_loader_error_consumes_access_even_after_missing_file_is_restored(self):
        self.freeze()
        final = self.workspace / "final.json"
        final.unlink()
        called = []
        with self.assertRaises(gov.ContractError):
            self.evaluate(lambda *_: called.append(True) or {})
        self.assertFalse(called)
        self.assertEqual(gov.read_json(self.directory / "final_failure.json")["status"], "failed_after_access")
        final.write_bytes(self.payloads["final"])
        with self.assertRaises(FileExistsError):
            self.evaluate(lambda *_: self.fail("a failed access was silently retried"))

    def test_callback_interrupt_and_nonfinite_result_are_retained_as_consumed_failures(self):
        cases = [("interrupt", KeyboardInterrupt), ("nonfinite", ValueError)]
        for name, error_type in cases:
            with self.subTest(case=name):
                study = gov.FinalStudy(self.workspace, self.workspace / name)
                self.freeze(study=study)
                pin = gov.file_digest(study.directory / "freeze.json")

                def callback(*_):
                    if name == "interrupt":
                        raise KeyboardInterrupt("synthetic interruption")
                    return {"invalid": float("nan")}

                with self.assertRaises(error_type):
                    study.evaluate(expected_freeze_sha256=pin, final_artifact="final.json", callback=callback)
                self.assertEqual(gov.read_json(study.directory / "final_failure.json")["status"], "failed_after_access")
                with self.assertRaises(FileExistsError):
                    study.evaluate(expected_freeze_sha256=pin, final_artifact="final.json", callback=lambda *_: {})

    def test_final_byte_drift_and_size_limit_fail_before_callback_but_after_access(self):
        for name in ["bytes", "limit"]:
            with self.subTest(case=name):
                study = gov.FinalStudy(self.workspace, self.workspace / name)
                self.freeze(study=study)
                pin = gov.file_digest(study.directory / "freeze.json")
                path = self.workspace / "final.json"
                original = path.read_bytes()
                if name == "bytes":
                    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                try:
                    with self.assertRaises(gov.ContractError):
                        study.evaluate(expected_freeze_sha256=pin, final_artifact="final.json",
                                       callback=lambda *_: self.fail("invalid final bytes reached evaluator"),
                                       max_artifact_bytes=1 if name == "limit" else 4096)
                    self.assertTrue((study.directory / "final_access.json").exists())
                    self.assertTrue((study.directory / "final_failure.json").exists())
                finally:
                    path.write_bytes(original)

    def test_new_directory_does_not_clear_declared_exposure(self):
        receipt = self.freeze()
        self.assertIn("no blinding or independent custody", receipt["guarantee"])
        self.evaluate(lambda *_: {"synthetic_score": .5})
        history = deepcopy(self.history)
        history["exposures"] = [
            {"group_id": row["group_id"], "uses": ["final_feedback"]}
            for row in self.partitions["final"]["records"]
        ]
        elsewhere = gov.FinalStudy(self.workspace, self.workspace / "new-study-directory")
        with self.assertRaises(gov.ContractError):
            self.freeze(study=elsewhere, manifest=self.manifest(history=history))
        self.assertFalse((elsewhere.directory / "freeze.json").exists())
        # Actual access was recorded in the supplied history. Discovering omitted
        # external access is outside this local contract, not a tested guarantee.


class JSONReceiptTests(unittest.TestCase):
    def test_duplicate_keys_and_nonfinite_json_receipts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            for payload in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
                with self.subTest(payload=payload):
                    path.write_text(payload)
                    with self.assertRaises(gov.ContractError):
                        gov.read_json(path)


if __name__ == "__main__":
    unittest.main()
