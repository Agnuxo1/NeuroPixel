"""Eight bounded controller contracts using only the existing invented fixture ledger.

No official payload is read by this module. All raw-QA contexts are drawn from
tests/test_babi_qa.py DEVELOPMENT_FIXTURES; the same complete ledger is exported
here for prospective exclusion. Synthetic archive/checkpoint bytes used for
gate tests are identity fixtures, never represented as trained model evidence.
Actual model serialization is exercised separately in test_01.
This source was authored without running it; the frozen CPU job runs the tests.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fx = _load("_item10_existing_invented_fixtures", ROOT / "tests/test_babi_qa.py")
study = _load("_item10_study_contract_target", ROOT / "scripts/research_babi_study.py")
qa = study.qa
DEVELOPMENT_FIXTURES = fx.DEVELOPMENT_FIXTURES
DEVELOPMENT_FIXTURE_SCOPE = {
    "source": "reuses the complete test_babi_qa.py invented input-only ledger",
    "additional_raw_contexts": 0, "official_payloads": "none",
    "identity_fixture_scope": "Temporary synthetic file graphs and TARs; no real checkpoint or final archive",
}
try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None


def _recipe():
    return {
        "batch_size": 32, "weight_decay": 1e-4, "gradient_clip": 1.0,
        "preflight": {
            "memorization_seed": 58, "tiny_n": 32, "lr": 0.003,
            "max_updates": 4096, "check_every": 128, "required_streak": 2,
            "target_accuracy": 1.0, "target_ce": 0.05, "pilot_seed": 59,
            "pilot_updates": 1024, "learning_rates": [0.001, 0.003],
        },
        "train": {"seeds": [60, 61, 62, 63, 64], "updates": 4096},
        "test_member": "invented/qa4_two-arg-relations_test.txt",
        "member_byte_cap": 1024 * 1024,
        "development_fixtures": [deepcopy(x) for x in DEVELOPMENT_FIXTURES],
    }


def _encoded(records, encoder):
    rows = []
    for record in records:
        visible = qa.encode_input([f["text"] for f in record["facts"]], record["question"], encoder)
        token = encoder["token_to_id"].get(record["answer"], -1)
        rows.append({
            "record_id": record["record_id"], "input_sha256": record["input_sha256"],
            "encoded_input_sha256": qa.canonical_sha256(visible["canvas"]),
            "canvas": visible["canvas"], "gold": record["answer"],
            "gold_id": token if token >= 2 else None, "supported_gold": token >= 2,
        })
    return rows


def _data(large=False):
    base = fx._records()
    train_raw = (fx._background_records()[:32] if large else []) + [base[0]]
    validation_raw = base[1:]
    encoder = qa.fit_encoder(train_raw)
    train = _encoded(train_raw, encoder)
    validation = _encoded(validation_raw, encoder)
    identity = {"fixture_only": True,
                "train_ids": [r["record_id"] for r in train],
                "validation_ids": [r["record_id"] for r in validation],
                "encoder_sha256": qa.canonical_sha256(encoder)}
    return {
        "encoder": encoder, "train": train, "validation": validation,
        "parsed": {"records": train_raw + validation_raw},
        "identity": identity, "identity_sha256": qa.canonical_sha256(identity),
        "controls": qa.fit_shortcuts(train_raw),
    }


def _source():
    return {"source_commit": "1" * 40, "plan_sha256": "2" * 64,
            "implementation_sha256": {
                name: "3" * 64 for name in (
                    "neuropixel/model.py", "neuropixel/research/models.py",
                    "neuropixel/research/babi_qa.py", "scripts/research_babi_study.py")
            }}


def _selection():
    return {"choices": {family: {"learning_rate": 0.001} for family in study.FAMILIES}}


def _put(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        study.save_json(path, payload)
    else:
        path.write_bytes(payload)
    return study.descriptor(root, path)


def _gate_fixture(root, data=None):
    """Create a complete hash graph; opaque weights are never deserialized."""
    data = _data() if data is None else data
    recipe, source, selection = _recipe(), _source(), _selection()
    selection_identity = {"fixture_selection_sha256": "4" * 64}
    entries = []
    for config in study.primary_configs(data, recipe, selection):
        prefix = "study/runs/" + config["run_id"] + "/"
        checkpoint = _put(root, prefix + "checkpoint.pt", b"opaque identity fixture, not a tensor")
        run = {
            "status": "completed", "run_id": config["run_id"], "config": config,
            "config_sha256": qa.canonical_sha256(config), "source": source,
            "completed_updates": 4096, "checkpoint": checkpoint,
            "config_artifact": _put(root, prefix + "config.json", config),
            "minibatch_indices": _put(root, prefix + "train_indices.npy", b"identity fixture"),
            "training_log": _put(root, prefix + "train_log.jsonl", b"{}\n"),
        }
        for kind in ("validation", "train_probe"):
            run[kind] = {"predictions": _put(root, prefix + kind + ".npz", b"identity fixture")}
        reference = _put(root, prefix + "run.json", run)
        entries.append({"run_id": config["run_id"], "config": config,
                        "config_sha256": qa.canonical_sha256(config),
                        "checkpoint": checkpoint, "summary": reference})
    manifest = {
        "schema_version": 1, "item": 10, "status": "completed",
        "created_at_utc": "2026-10-07T10:00:00+00:00",
        "source_commit": source["source_commit"], "run_key": root.name,
        "ordered_run_ids": [x["run_id"] for x in entries], "runs": entries,
        "recipe_sha256": qa.canonical_sha256(recipe),
        "data_identity_sha256": data["identity_sha256"],
        "selection": selection, "selection_identity": selection_identity,
    }
    study.save_json(root / "training_manifest.json", manifest)
    gate = {
        "schema_version": 1, "item": 10,
        "at_utc": "2026-10-07T10:00:01+00:00",
        "source_commit": source["source_commit"], "run_key": root.name,
        "training_manifest_sha256": study.sha(root / "training_manifest.json"),
        "training_archive_commit": "5" * 40,
    }
    study.save_json(root / "pre_final_archive_receipt.json", gate)
    return data, recipe, source, selection, selection_identity, manifest, gate


def _synthetic_arrays(rows, encoder):
    n, vocabulary = len(rows), len(encoder["vocabulary"])
    gold = np.asarray([r["gold_id"] if r["gold_id"] is not None else -1 for r in rows], dtype=np.int64)
    supported = np.asarray([r["supported_gold"] for r in rows], dtype=np.bool_)
    logits = np.zeros((n, vocabulary), dtype=np.float32)
    logits[:, 0] = -1e4
    for i in np.flatnonzero(supported):
        logits[i, gold[i]] = 12.0
    prediction = logits.argmax(1).astype(np.int64)
    nll = np.zeros(n, dtype=np.float64)
    values = logits[supported].astype(np.float64)
    if len(values):
        nll[supported] = np.logaddexp.reduce(values, axis=1) - values[np.arange(len(values)), gold[supported]]
    return {
        "logits": logits, "prediction_id": prediction, "gold_id": gold,
        "can_encode": np.ones(n, dtype=np.bool_), "gold_supported": supported,
        "nll_supported": supported.copy(), "nll": nll,
        "record_id": np.asarray([r["record_id"] for r in rows], dtype="<U64"),
    }


class BabiStudyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is not None:
            torch.set_num_threads(2)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)

    @unittest.skipIf(torch is None or np is None, "pinned CPU Torch and NumPy are required")
    def test_01_actual_factories_backward_parameter_counts_and_weight_roundtrip(self):
        data = _data()
        encoder = deepcopy(data["encoder"])
        # Add unused lexical identities from the already exposed fixture ledger,
        # solely to exercise the prospectively declared H4/W8/V18 parameter count.
        known = set(encoder["vocabulary"])
        candidates = sorted({t for f in DEVELOPMENT_FIXTURES
                             for line in qa.normalized_input(f["facts"], f["question"])["facts"]
                             for t in line} - known)
        encoder["vocabulary"] += candidates[:18 - len(encoder["vocabulary"])]
        encoder["token_to_id"] = {t: i for i, t in enumerate(encoder["vocabulary"])}
        self.assertEqual((encoder["height"], encoder["width"], len(encoder["vocabulary"])), (4, 8, 18))
        canvas = torch.tensor([data["train"][0]["canvas"]], dtype=torch.long)
        target = torch.tensor([data["train"][0]["gold_id"]], dtype=torch.long)
        expected = {"neuropixel": 29552, "relative_transformer": 29562}
        with tempfile.TemporaryDirectory() as temporary, torch.random.fork_rng(devices=[]):
            for family in study.FAMILIES:
                with self.subTest(family=family):
                    torch.manual_seed(81)
                    config = study.model_configuration(family, encoder)
                    model = study.model_for(config, torch)
                    self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), expected[family])
                    self.assertTrue(model.training)
                    logits = model(canvas)["logits"]
                    self.assertEqual(tuple(logits.shape), (1, 18))
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
                    self.assertTrue(any(torch.count_nonzero(g).item() > 0 for g in grads))
                    torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4).step()
                    path = Path(temporary) / (family + ".pt")
                    study.atomic_tensor(path, model.state_dict(), torch)
                    clone = study.model_for(config, torch)
                    clone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, clone.state_dict()[key]))
                    model.eval()
                    clone.eval()
                    before = torch.get_rng_state().clone()
                    with torch.inference_mode():
                        first = model(canvas)["logits"]
                        second = clone(canvas)["logits"]
                    self.assertTrue(torch.equal(first, second))
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    if family == "neuropixel":
                        self.assertEqual(torch.count_nonzero(model.dictionary()[0]).item(), 0)
                    with self.assertRaises(ValueError):
                        study.atomic_tensor(path, model.state_dict(), torch)

    @unittest.skipIf(torch is None or np is None, "pinned CPU Torch and NumPy are required")
    def test_02_actual_short_training_uses_private_paired_minibatches_and_separate_firing_seed(self):
        data, recipe, source = _data(), _recipe(), _source()
        rows = data["train"] + data["validation"]
        with tempfile.TemporaryDirectory() as temporary, torch.random.fork_rng(devices=[]):
            root = Path(temporary)
            results = []
            for family in study.FAMILIES:
                config = study.run_configuration("contract_" + family, family, 83, 0.001, 2,
                                                 data, recipe, rows, "contract_fixture")
                self.assertEqual((config["sample_seed"], config["firing_seed"]), (100083, 110083))
                self.assertNotEqual(config["init_seed"], config["sample_seed"])
                np.random.seed(70 if family == "neuropixel" else 71)
                global_before = deepcopy(np.random.get_state())
                result, reference = study.train_one(config, rows, data, recipe, root, source, torch, np)
                self.assertEqual(result["status"], "completed", result.get("error"))
                self.assertEqual(result["completed_updates"], 2)
                self.assertEqual(study.read_json(study.verify_reference(root, reference))["status"], "completed")
                actual = np.load(study.verify_reference(root, result["minibatch_indices"]), allow_pickle=False)
                expected = np.random.Generator(np.random.PCG64(100083)).integers(
                    len(rows), size=(2, 32), dtype=np.int64)
                np.testing.assert_array_equal(actual, expected)
                after = np.random.get_state()
                self.assertEqual(global_before[0], after[0])
                np.testing.assert_array_equal(global_before[1], after[1])
                self.assertEqual(global_before[2:], after[2:])
                expected_firing = torch.Generator(device="cpu").manual_seed(110083).get_state()
                self.assertEqual(result["firing_rng_initial_sha256"],
                                 hashlib.sha256(expected_firing.numpy().tobytes()).hexdigest())
                results.append(result)
            self.assertEqual(results[0]["minibatch_indices_content_sha256"],
                             results[1]["minibatch_indices_content_sha256"])
            self.assertEqual(results[0]["firing_rng_initial_sha256"], results[1]["firing_rng_initial_sha256"])
            self.assertNotEqual(results[0]["firing_rng_initial_sha256"], results[0]["firing_rng_final_sha256"])
            self.assertEqual(results[1]["firing_rng_initial_sha256"], results[1]["firing_rng_final_sha256"])

    @unittest.skipIf(torch is None or np is None, "pinned CPU Torch and NumPy are required")
    def test_03_predict_preserves_mode_and_counts_unsupported_overflow_and_empty_subsets(self):
        data = _data()
        row = deepcopy(data["train"][0])
        rows = [row, {**deepcopy(row), "record_id": "a" * 64, "gold": "quorin",
                      "gold_id": None, "supported_gold": False},
                {**deepcopy(row), "record_id": "b" * 64, "can_encode": False, "canvas": None}]
        vocabulary = len(data["encoder"]["vocabulary"])
        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                values = torch.zeros(vocabulary, dtype=torch.float32)
                values[0] = -1e4
                values[row["gold_id"]] = 3.0
                self.register_buffer("values", values)
            def forward(self, canvas):
                return {"logits": self.values.unsqueeze(0).expand(len(canvas), -1)}
        model = Fixed().train()
        arrays, metrics, tokens = study.predict(model, rows, data["encoder"], torch, np)
        self.assertTrue(model.training)
        self.assertEqual((metrics["n"], metrics["correct"], metrics["cross_entropy_n"]), (3, 1, 1))
        self.assertEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual((metrics["can_encode_n"], metrics["cannot_encode_n"]), (2, 1))
        self.assertEqual(arrays["prediction_id"][-1], -1)
        self.assertIsNone(tokens[-1])
        self.assertEqual(arrays["nll_supported"].tolist(), [True, False, False])
        self.assertEqual(arrays["nll"][1:].tolist(), [0.0, 0.0])
        subsets = study.subset_metrics(arrays, rows, {"all": [True] * 3, "empty": [False] * 3},
                                       data["encoder"])
        self.assertTrue(study._same_metrics(metrics, subsets["all"]))
        self.assertEqual(subsets["empty"]["n"], 0)
        self.assertIsNone(subsets["empty"]["accuracy"])
        self.assertIsNone(subsets["empty"]["cross_entropy"])
        _, empty, _ = study.predict(model.eval(), [], data["encoder"], torch, np)
        self.assertFalse(model.training)
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["accuracy"])

    @unittest.skipIf(np is None, "pinned NumPy is required")
    def test_04_validation_selection_ties_missing_runs_and_resealed_config_drift(self):
        data, recipe = _data(), _recipe()
        pilots = []
        for family in study.FAMILIES:
            for lr in (0.001, 0.003):
                config = study.run_configuration("pilot_" + family + "_lr" + study.LR_NAMES[lr] + "_s59",
                                                 family, 59, lr, 1024, data, recipe, data["train"], "pilot")
                pilots.append({
                    "status": "completed", "run_id": config["run_id"], "config": config,
                    "config_sha256": qa.canonical_sha256(config), "checkpoint": {"sha256": "6" * 64},
                    "validation": {"metrics": {"n": len(data["validation"]), "accuracy": 0.5,
                                               "cross_entropy": 1.0}},
                })
        self.assertEqual(study.selection_from(pilots, data, recipe)["choices"]["neuropixel"]["learning_rate"], 0.001)
        pilots[1]["validation"]["metrics"]["cross_entropy"] = 0.9
        self.assertEqual(study.selection_from(pilots, data, recipe)["choices"]["neuropixel"]["learning_rate"], 0.003)
        pilots[0]["validation"]["metrics"].update(accuracy=1.0, cross_entropy=2.0)
        self.assertEqual(study.selection_from(pilots, data, recipe)["choices"]["neuropixel"]["learning_rate"], 0.001)
        with self.assertRaises(ValueError):
            study.selection_from(pilots[:3], data, recipe)
        malformed = deepcopy(pilots)
        malformed[1]["config"]["learning_rate"] = 0.001
        with self.assertRaises(ValueError):
            study.selection_from(malformed, data, recipe)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _preflight_fixture(root)
            plan, output, data, recipe, source = (fixture[k] for k in ("plan", "output", "data", "recipe", "source"))
            with patch.object(study, "recover_input", return_value=fixture["directory"]):
                loaded, identity = study.load_selection(plan, output, data, recipe, source, np)
                self.assertEqual(loaded, fixture["selection"])
                self.assertEqual(identity["selection_sha256"], study.sha(fixture["directory"] / "selection.json"))
                # All file/reference hashes are deliberately recomputed. The
                # failure must arise from the expected scientific configuration,
                # not merely from an obsolete checksum after mutation.
                run = fixture["runs"][2]
                run["config"]["init_seed"] = 60
                run["config_sha256"] = qa.canonical_sha256(run["config"])
                config_path = fixture["directory"] / run["config_artifact"]["path"]
                study.save_json(config_path, run["config"], exclusive=False)
                run["config_artifact"] = study.descriptor(fixture["directory"], config_path)
                run_path = fixture["directory"] / fixture["summary"]["runs"][2]["path"]
                study.save_json(run_path, run, exclusive=False)
                fixture["summary"]["runs"][2] = study.descriptor(fixture["directory"], run_path)
                revised_selection = study.selection_from(fixture["runs"][2:], data, recipe)
                study.save_json(fixture["directory"] / "selection.json", revised_selection, exclusive=False)
                fixture["summary"]["selection"] = study.descriptor(fixture["directory"], fixture["directory"] / "selection.json")
                study.save_json(fixture["directory"] / "preflight_summary.json", fixture["summary"], exclusive=False)
                for name in ("selection.json", "preflight_summary.json"):
                    plan["inputs"]["preflight"]["files_sha256"][name] = study.sha(fixture["directory"] / name)
                with self.assertRaises(ValueError):
                    study.load_selection(plan, output, data, recipe, source, np)

    @unittest.skipIf(np is None, "pinned NumPy is required")
    def test_05_saved_prediction_recount_rejects_nll_argmax_mask_and_population_drift(self):
        data = _data()
        rows, encoder = data["train"] + data["validation"], data["encoder"]
        original = _synthetic_arrays(rows, encoder)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.npz"
            study.atomic_arrays(good, original, np)
            metrics = study.recount_development_predictions(good, rows, encoder, np)
            self.assertEqual((metrics["n"], metrics["correct"]), (3, 3))
            self.assertAlmostEqual(metrics["cross_entropy"], float(original["nll"].mean()), places=12)
            for mutation in ("nll", "argmax", "mask", "population"):
                changed = {k: v.copy() for k, v in original.items()}
                if mutation == "nll":
                    changed["nll"][0] += 1e-8
                elif mutation == "argmax":
                    changed["prediction_id"][0] = 0
                elif mutation == "mask":
                    changed["nll_supported"][0] = False
                else:
                    changed["record_id"] = changed["record_id"][::-1].copy()
                path = root / (mutation + ".npz")
                study.atomic_arrays(path, changed, np)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    study.recount_development_predictions(path, rows, encoder, np)
            with self.assertRaises(ValueError):
                study.atomic_arrays(good, original, np)

    def test_06_ten_checkpoint_gate_authenticates_every_artifact_and_semantic_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, recipe, source, selection, identity, manifest, gate = _gate_fixture(root)
            args = (root, data, recipe, source, selection, identity)
            actual, _ = study.authenticate_gate(*args)
            self.assertEqual(len(actual["runs"]), 10)
            for index, item in enumerate(manifest["runs"]):
                path = root / item["checkpoint"]["path"]
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.subTest(checkpoint=index), self.assertRaises(ValueError):
                    study.authenticate_gate(*args)
                path.write_bytes(original)
            # Test non-weight artifact checks on an otherwise valid graph.
            first_run = study.read_json(root / manifest["runs"][0]["summary"]["path"])
            for key in ("config_artifact", "minibatch_indices", "training_log"):
                path = root / first_run[key]["path"]
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.subTest(artifact=key), self.assertRaises(ValueError):
                    study.authenticate_gate(*args)
                path.write_bytes(original)
            for key in ("validation", "train_probe"):
                path = root / first_run[key]["predictions"]["path"]
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.subTest(artifact=key), self.assertRaises(ValueError):
                    study.authenticate_gate(*args)
                path.write_bytes(original)
            for mutation in ("missing_run", "wrong_lr", "wrong_selection", "early_gate"):
                altered, gate_altered = deepcopy(manifest), deepcopy(gate)
                if mutation == "missing_run":
                    altered["runs"].pop()
                elif mutation == "wrong_lr":
                    altered["runs"][0]["config"]["learning_rate"] = 0.003
                    altered["runs"][0]["config_sha256"] = qa.canonical_sha256(altered["runs"][0]["config"])
                elif mutation == "wrong_selection":
                    altered["selection"]["choices"]["neuropixel"]["learning_rate"] = 0.003
                else:
                    gate_altered["at_utc"] = "2026-10-07T09:59:59+00:00"
                study.save_json(root / "training_manifest.json", altered, exclusive=False)
                gate_altered["training_manifest_sha256"] = study.sha(root / "training_manifest.json")
                study.save_json(root / "pre_final_archive_receipt.json", gate_altered, exclusive=False)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    study.authenticate_gate(*args)
            study.save_json(root / "training_manifest.json", manifest, exclusive=False)
            study.save_json(root / "pre_final_archive_receipt.json", gate, exclusive=False)
            # Even after resealing the full outer graph, an incomplete training
            # budget is not authenticated as a 4,096-update run.
            run_path = root / manifest["runs"][0]["summary"]["path"]
            first_run["completed_updates"] = 4095
            study.save_json(run_path, first_run, exclusive=False)
            manifest["runs"][0]["summary"] = study.descriptor(root, run_path)
            study.save_json(root / "training_manifest.json", manifest, exclusive=False)
            gate["training_manifest_sha256"] = study.sha(root / "training_manifest.json")
            study.save_json(root / "pre_final_archive_receipt.json", gate, exclusive=False)
            with self.assertRaises(ValueError):
                study.authenticate_gate(*args)

    def test_07_final_consumption_is_persisted_before_read_failure_and_prevents_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, recipe, source, selection, identity, _, _ = _gate_fixture(root)
            seen = []
            def failed_read(output, received_data, received_recipe, numpy):
                receipt = study.read_json(output / "final_access.json")
                self.assertEqual(receipt["status"], "consumed")
                self.assertEqual(receipt["training_manifest_sha256"], study.sha(output / "training_manifest.json"))
                self.assertEqual(receipt["pre_final_archive_receipt_sha256"],
                                 study.sha(output / "pre_final_archive_receipt.json"))
                seen.append(True)
                raise RuntimeError("injected failure before any TAR payload is read")
            with patch.object(study, "final_rows", side_effect=failed_read):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    study.final_evaluation(root, data, recipe, source, selection, identity, None, None)
                self.assertEqual(seen, [True])
                with self.assertRaises(FileExistsError):
                    study.final_evaluation(root, data, recipe, source, selection, identity, None, None)
                self.assertEqual(seen, [True])
            self.assertFalse((root / "final_data").exists())
            self.assertFalse((root / "final_predictions").exists())

        # Exercise actual CLI orchestration with inert phase work. A worker
        # owns the unprefixed files before main starts; trainer must preserve
        # them on both successful and failed phase completion.
        for fail in (False, True):
            with self.subTest(main_phase_failure=fail), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = root / "phase"
                output.mkdir()
                data, recipe, source = _data(), _recipe(), _source()
                plan = {"schema_version": 1, "item": 10, "status": "frozen",
                        "phase": "preflight", "freeze_utc": "2026-10-07T10:00:00+00:00",
                        "recipe": recipe}
                plan_path = root / "plan.json"
                study.save_json(plan_path, plan)
                worker_paths = [output / "preflight_environment.json", output / "preflight_status.json"]
                for path in worker_paths:
                    path.write_bytes(b'{"owner":"worker"}\n')
                def phase_work(*args):
                    if fail:
                        raise RuntimeError("injected inert phase failure")
                with patch.object(sys, "argv", [
                        "research_babi_study.py", "--phase", "preflight",
                        "--plan", str(plan_path), "--output", str(output)]), \
                     patch.object(study, "source_record", return_value=source), \
                     patch.object(study, "configure_runtime", return_value=(None, None, {"fixture": True})), \
                     patch.object(study, "load_development", return_value=data), \
                     patch.object(study, "preflight", side_effect=phase_work), \
                     patch.object(study, "admit", return_value=16.0):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "injected inert phase failure"):
                            study.main()
                    else:
                        self.assertEqual(study.main(), 0)
                for path in worker_paths:
                    self.assertEqual(path.read_bytes(), b'{"owner":"worker"}\n')
                self.assertTrue((output / "study_preflight_environment.json").is_file())
                status = study.read_json(output / "study_preflight_status.json")
                self.assertEqual(status["status"], "failed" if fail else "completed")

    def test_08_invented_tar_exact_member_novelty_exposure_and_overflow_denominators(self):
        declared_plan = ROOT / "docs/research/10_preflight_plan.json"
        if declared_plan.is_file():
            frozen = study.read_json(declared_plan)
            self.assertEqual(qa.canonical_bytes(frozen["recipe"]["development_fixtures"]),
                             qa.canonical_bytes(list(DEVELOPMENT_FIXTURES)))
        data, recipe = _data(), _recipe()
        # Every valid context below already appears in DEVELOPMENT_FIXTURES.
        text = (fx._episode(fx.BASE_FACTS, fx.BASE_Q, "neral")
                + fx._episode(fx.BASE_FACTS, fx.INVERSE_Q, "vexa")
                + fx._episode(fx.OOV_FACTS, fx.BASE_Q, "quorin")
                + fx._episode(fx.BASE_FACTS, fx.LONG_Q, "neral")
                + fx._episode([fx.FACT_A] * 7, fx.BASE_Q, "neral"))
        ledger = {qa.canonical_sha256(qa.normalized_input(x["facts"], x["question"]))
                  for x in DEVELOPMENT_FIXTURES}
        parsed = qa.parse_babi(text, {"fixture": "item10-invented-TAR-contract"})
        self.assertTrue(all(r["input_sha256"] in ledger for r in parsed["records"]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "pinned"
            directory.mkdir()
            data["directory"] = directory
            archive_path = directory / "invented.tar.gz"
            def write_archive(members):
                with tarfile.open(archive_path, "w:gz") as archive:
                    for name, value in members:
                        payload = value.encode("utf-8")
                        member = tarfile.TarInfo(name)
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                study.save_json(directory / "selected_source.json",
                                {"archive_filename": archive_path.name,
                                 "archive_sha256": study.sha(archive_path)}, exclusive=False)
            write_archive([("unselected.txt", "not a parser input"),
                           (recipe["test_member"], text)])
            output = root / "valid"
            output.mkdir()
            rows, records, masks, population = study.final_rows(output, data, recipe, None)
            self.assertEqual(len(rows), 5)
            self.assertEqual([r["can_encode"] for r in rows], [True, True, True, False, False])
            self.assertEqual([r["supported_gold"] for r in rows], [True, True, False, True, True])
            self.assertEqual(masks["all_official"], [True] * 5)
            self.assertEqual(masks["novel_vs_optimization_train"], [False, True, True, True, True])
            self.assertEqual(masks["strict_novel_vs_full_exposure"], [False] * 5)
            self.assertEqual(population["unencodable_count"], 2)
            self.assertEqual(population["unsupported_gold_count"], 1)
            self.assertEqual(population["subset_unencodable_counts"]["all_official"], 2)
            self.assertEqual([r["input_sha256"] for r in rows], [r["input_sha256"] for r in records])
            # A deliberately reduced synthetic exposure policy exercises the
            # strict mask branch; the real study must use the exported full ledger.
            reduced = deepcopy(recipe)
            reduced["development_fixtures"] = []
            another = root / "reduced_policy"
            another.mkdir()
            _, _, reduced_masks, _ = study.final_rows(another, data, reduced, None)
            self.assertEqual(reduced_masks["strict_novel_vs_full_exposure"], [False, False, True, True, True])
            for kind, members in (
                ("missing", [("different_member.txt", text)]),
                ("duplicate", [(recipe["test_member"], text), (recipe["test_member"], text)]),
            ):
                write_archive(members)
                target = root / kind
                target.mkdir()
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    study.final_rows(target, data, recipe, None)
                self.assertFalse((target / "final_data").exists())


def _preflight_fixture(root):
    """Build saved prediction/config fixtures, with no model execution."""
    directory, output = root / "preflight", root / "consumer"
    directory.mkdir()
    output.mkdir()
    data, recipe, source = _data(large=True), _recipe(), _source()
    seen, tiny = set(), []
    for row in sorted(data["train"], key=lambda r: r["record_id"]):
        if row["input_sha256"] not in seen:
            seen.add(row["input_sha256"])
            tiny.append(row)
        if len(tiny) == 32:
            break
    specs = [(f"mem_{family}_s58", family, 58, 0.003, 4096, tiny, "memorization")
             for family in study.FAMILIES]
    specs += [(f"pilot_{family}_lr{study.LR_NAMES[lr]}_s59", family, 59, lr, 1024, data["train"], "pilot")
              for family in study.FAMILIES for lr in (0.001, 0.003)]
    runs, references = [], []
    for rid, family, seed, lr, updates, rows, kind in specs:
        config = study.run_configuration(rid, family, seed, lr, updates, data, recipe, rows, kind)
        prefix = "study/runs/" + rid + "/"
        run_dir = directory / prefix
        run_dir.mkdir(parents=True)
        run = {
            "status": "completed", "run_id": rid, "config": config,
            "config_sha256": qa.canonical_sha256(config), "source": source,
            "completed_updates": 256 if kind == "memorization" else 1024,
            "checkpoint": _put(directory, prefix + "checkpoint.pt", b"opaque identity fixture"),
            "config_artifact": _put(directory, prefix + "config.json", config),
            "minibatch_indices": _put(directory, prefix + "train_indices.npy", b"opaque identity fixture"),
            "training_log": _put(directory, prefix + "train_log.jsonl", b"{}\n"),
            "memorization_checks": [], "memorization_pass": True if kind == "memorization" else None,
        }
        for label, population in (("validation", data["validation"]), ("train_probe", data["train"])):
            arrays = _synthetic_arrays(population, data["encoder"])
            path = run_dir / (label + ".npz")
            study.atomic_arrays(path, arrays, np)
            run[label] = {
                "predictions": study.descriptor(directory, path),
                "metrics": study.subset_metrics(arrays, population, {"all": [True] * len(population)},
                                                data["encoder"])["all"],
            }
        if kind == "memorization":
            for check_index, update in enumerate((128, 256), 1):
                arrays = _synthetic_arrays(tiny, data["encoder"])
                path = run_dir / f"memorization_check_{update:05d}.npz"
                study.atomic_arrays(path, arrays, np)
                metric = study.subset_metrics(arrays, tiny, {"all": [True] * len(tiny)}, data["encoder"])["all"]
                run["memorization_checks"].append({
                    "update": update, "accuracy": metric["accuracy"], "cross_entropy": metric["cross_entropy"],
                    "criterion_pass": True, "consecutive_passes": check_index, "n": 32,
                    "record_ids_sha256": config["training_record_ids_sha256"],
                    "predictions": study.descriptor(directory, path),
                })
        references.append(_put(directory, prefix + "run.json", run))
        runs.append(run)
    selection = study.selection_from(runs[2:], data, recipe)
    study.save_json(directory / "selection.json", selection)
    summary = {
        "schema_version": 1, "item": 10, "status": "completed", "admission_passed": True,
        "all_runs_completed": True, "memorization_pass": True,
        "ordered_run_ids": [r["run_id"] for r in runs], "runs": references,
        "recipe_sha256": qa.canonical_sha256(recipe), "data_identity_sha256": data["identity_sha256"],
        "source": source, "selection": study.descriptor(directory, directory / "selection.json"),
    }
    study.save_json(directory / "preflight_summary.json", summary)
    plan = {"inputs": {"preflight": {
        "archive_commit": "7" * 40, "path": "results/research/10_cloud_runs/invented-preflight",
        "files_sha256": {name: study.sha(directory / name)
                        for name in ("preflight_summary.json", "selection.json")},
    }}}
    return {"directory": directory, "output": output, "data": data, "recipe": recipe,
            "source": source, "runs": runs, "summary": summary, "selection": selection, "plan": plan}


if __name__ == "__main__":
    unittest.main()
