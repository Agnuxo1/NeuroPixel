"""Focused item-11 controller contracts on invented, known task fixtures.

These tests do not train study models, estimate memory performance, generate
study RNG streams, or open any final checkpoint inventory. The only optimizer
fixture uses two explicit rows and one update per architecture.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("item11_memory_controller", ROOT / "scripts/research_memory_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def metric(correct=48, ce=0.1):
    return {"n": 48, "correct": correct, "accuracy": correct / 48,
            "cross_entropy": ce, "cross_entropy_n": 48}


def pilots(near_correct=48):
    result = []
    for config in study.preflight_configs():
        result.append({"run_id": config["run_id"], "status": "completed", "config": config,
                       "config_sha256": study.digest(config), "completed_updates": 1024,
                       "development": {f"normal_d{d}": {"metrics": metric(near_correct if d == 2 else 24)}
                                       for d in (2, 3, 4)}})
    return result


def source_fixture():
    paths = ("neuropixel/model.py", "neuropixel/phase3.py",
             "neuropixel/research/stream_memory.py", "scripts/research_memory_study.py")
    return {"source_commit": "b" * 40, "plan_sha256": "c" * 64,
            "implementation_sha256": {p: "d" * 64 for p in paths},
            "tracked_source_clean": True, "recorded_at_utc": "2026-01-01T00:00:00+00:00"}


def add_artifacts(output, run):
    directory = output / "study" / "runs" / run["run_id"]
    directory.mkdir(parents=True)
    for key, name in (("checkpoint", "checkpoint.pt"), ("minibatch_indices", "train_indices.npy"),
                      ("training_delays", "training_delays.npy"), ("training_log", "training_log.jsonl"),
                      ("nonpersistent_buffers", "nonpersistent_buffers.json")):
        path = directory / name
        path.write_bytes(b"declared non-model fixture\n")
        run[key] = study.descriptor(output, path)
    path = directory / "config.json"
    write_json(path, run["config"])
    run["config_artifact"] = study.descriptor(output, path)
    path = directory / "run.json"
    write_json(path, run)
    return study.descriptor(output, path)


def gate_fixture(output):
    source = source_fixture()
    selection = study.selection_from(pilots())
    identity = {"archive_commit": "a" * 40, "selection_sha256": "e" * 64}
    items = []
    for config in study.primary_configs(selection):
        run = {"run_id": config["run_id"], "status": "completed", "config": config,
               "config_sha256": study.digest(config), "completed_updates": 2048,
               "source": source, "development": {}}
        reference = add_artifacts(output, run)
        items.append({"run_id": config["run_id"], "config": config,
                      "config_sha256": study.digest(config), "checkpoint": run["checkpoint"],
                      "summary": reference})
    manifest = {"schema_version": 1, "item": 11, "status": "completed", "run_key": output.name,
                "source_commit": source["source_commit"], "source": source,
                "recipe_sha256": study.digest(study.RECIPE), "selection": selection,
                "selection_identity": identity, "ordered_run_ids": [r["run_id"] for r in items],
                "runs": items, "created_at_utc": "2026-01-01T00:00:01+00:00"}
    write_json(output / "training_manifest.json", manifest)
    gate = {"schema_version": 1, "item": 11, "run_key": output.name,
            "source_commit": source["source_commit"], "training_archive_commit": "f" * 40,
            "training_manifest_sha256": study.sha(output / "training_manifest.json"),
            "at_utc": "2026-01-01T00:00:02+00:00"}
    write_json(output / "pre_final_archive_receipt.json", gate)
    return source, selection, identity, manifest, gate


def development_fixture(output):
    run = {"development": {}}
    # The analytic softmax denominator is exp(2)+7, independent of the scorer.
    loss = math.log(math.exp(2.0) + 7.0) - 2.0
    for delay in (2, 3, 4):
        rows = study.census(delay)
        target = np.asarray([r["target"] for r in rows], dtype=np.int64)
        logits = np.zeros((48, 8), dtype=np.float32)
        logits[np.arange(48), target] = 2.0
        arrays = {"logits": logits, "target": target, "prediction": target.copy(),
                  "nll": np.full(48, loss, dtype=np.float64)}
        path = output / f"normal_d{delay}.npz"
        np.savez(path, **arrays)
        scored = metric(48, loss)
        metadata = {"rows": rows, "rows_sha256": study.digest(rows),
                    "metrics": scored, "predictions": study.descriptor(output, path)}
        meta_path = output / f"normal_d{delay}.json"
        write_json(meta_path, metadata)
        run["development"][f"normal_d{delay}"] = {"metrics": scored,
             "predictions": study.descriptor(output, path), "metadata": study.descriptor(output, meta_path)}
    return run


class MemoryStudyTests(unittest.TestCase):
    def test_inventory_census_and_independent_distractors(self):
        cases = study.evaluation_cases()
        self.assertEqual(len(cases), 16)
        self.assertEqual(sum(c["n"] for c in cases), 1488)
        self.assertEqual(len({c["case_id"] for c in cases}), 16)
        base = study.census(8)
        self.assertEqual(len(base), 48)
        self.assertEqual({(r["target"], tuple(r["cue_position"])) for r in base},
                         {(t, p) for t in range(2, 8) for p in study.POSITIONS})
        for condition in ("blank_repeated", "same_position", "next_position"):
            rows = study.census(8, condition)
            self.assertEqual(len(rows), 288)
            self.assertEqual(sum(r["same_label"] for r in rows), 48)
            self.assertEqual(sum(not r["same_label"] for r in rows), 240)
            for target in range(2, 8):
                for position in range(8):
                    self.assertEqual(sorted(r["distractor_target"] for r in rows
                                            if r["target"] == target and r["cue_position_index"] == position),
                                     list(range(2, 8)))
        with self.assertRaises(ValueError):
            study.census(2, "same_position")

    @unittest.skipIf(torch is None, "Torch unavailable")
    def test_actual_frames_do_not_replay_cue_and_swap_uses_same_position(self):
        for delay in (2, 8):
            for condition in ("normal", "cue_erased", "zero", "swap"):
                rows = study.census(delay, condition)
                frames, steps, intervention = study.frames_for_rows(rows, torch)
                self.assertEqual(len(frames), delay + 2)
                self.assertEqual(sum(steps), 13 + 3 * delay)
                self.assertTrue(all(torch.count_nonzero(f).item() == 0 for f in frames[1:-1]))
                self.assertEqual(torch.count_nonzero(frames[-1]).item(), 48)
                self.assertTrue(torch.equal(frames[-1][:, 2, 1], torch.ones(48, dtype=torch.long)))
                self.assertEqual(torch.count_nonzero(frames[0]).item(), 0 if condition == "cue_erased" else 48)
                if intervention:
                    self.assertEqual(intervention["before_frame"], len(frames) - 1)
                if condition == "swap":
                    self.assertEqual(sorted(intervention["permutation"]), list(range(48)))
                    for i, donor in enumerate(intervention["permutation"]):
                        self.assertEqual(rows[i]["cue_position"], rows[donor]["cue_position"])
                        self.assertEqual(rows[i]["donor_target"], rows[donor]["target"])
                        self.assertNotEqual(rows[i]["target"], rows[donor]["target"])
        for condition in ("same_position", "next_position"):
            rows = study.census(8, condition)
            frames, steps, intervention = study.frames_for_rows(rows, torch)
            self.assertEqual((len(frames), sum(steps), intervention), (10, 37, None))
            self.assertEqual(torch.count_nonzero(frames[1]).item(), 288)
            self.assertTrue(all(torch.count_nonzero(f).item() == 0 for f in frames[2:-1]))
            for i, row in enumerate(rows):
                p = row["distractor_position"]
                self.assertEqual(frames[1][i, p[0], p[1]].item(), row["distractor_target"])

    @unittest.skipIf(np is None, "NumPy unavailable")
    def test_private_pair_stream_ignores_global_rng_and_family(self):
        # Fixture seeds are outside every study/pilot seed.
        configs = [study.run_configuration(f, 110913, lr, 9, "pilot")
                   for f in study.FAMILIES for lr in (0.001, 0.003)]
        saved = np.random.get_state()
        try:
            arrays = []
            for i, config in enumerate(configs):
                np.random.seed(110920 + i)
                np.random.random(11 + i)
                arrays.append(study.training_stream(config, np))
            for pairs, delays in arrays:
                self.assertEqual(pairs.shape, (9, 32))
                self.assertEqual(delays.shape, (9,))
                self.assertTrue(np.array_equal(pairs, arrays[0][0]))
                self.assertTrue(np.array_equal(delays, arrays[0][1]))
                self.assertTrue(((pairs >= 0) & (pairs < 48)).all())
                self.assertTrue(((delays >= 0) & (delays <= 2)).all())
            other = study.training_stream(study.run_configuration("gru", 110914, 0.001, 9, "pilot"), np)
            self.assertFalse(np.array_equal(other[0], arrays[0][0]))
        finally:
            np.random.set_state(saved)

    def test_selection_rule_and_completed_negative_admission(self):
        rows = pilots()
        # NP's greater accuracy wins despite its worse CE.
        for d in (3, 4):
            rows[0]["development"][f"normal_d{d}"]["metrics"] = metric(24, 0.1)
            rows[1]["development"][f"normal_d{d}"]["metrics"] = metric(25, 0.2)
        selected = study.selection_from(rows)
        self.assertEqual(selected["families"]["neuropixel"]["learning_rate"], 0.003)
        self.assertEqual(selected["families"]["gru"]["learning_rate"], 0.001)
        # Equal accuracy: lower CE wins before LR.
        rows[3]["development"]["normal_d3"]["metrics"]["cross_entropy"] = 0.01
        self.assertEqual(study.selection_from(rows)["families"]["gru"]["learning_rate"], 0.003)
        with self.assertRaisesRegex(ValueError, "four ordered pilots"):
            study.selection_from(rows[:-1])
        rows[0]["completed_updates"] = 1023
        with self.assertRaisesRegex(ValueError, "incomplete"):
            study.selection_from(rows)
        negative = pilots(near_correct=45)
        self.assertFalse(study.selection_from(negative)["admission_passed"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            source = source_fixture()
            queue = iter(negative)

            def fake_train(output, config, source, runtime, torch, np):
                run = next(queue)
                run["source"] = source
                reference = add_artifacts(output, run)
                return run, reference

            with mock.patch.object(study, "train_one", side_effect=fake_train) as train:
                summary = study.preflight(output, source, {}, None, None)
            self.assertEqual(train.call_count, 4)
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(summary["all_runs_completed"])
            self.assertFalse(summary["admission_passed"])
            self.assertFalse((output / "training_manifest.json").exists())
            spec = {"archive_commit": "e" * 40, "files_sha256": {
                name: study.sha(output / name) for name in ("preflight_summary.json", "selection.json")}}
            helper = types.SimpleNamespace(recover_input=lambda *args: str(output))
            with mock.patch.dict(sys.modules, {"research_item11_inputs": helper}), \
                    mock.patch.object(study, "recount_development", return_value=None):
                with self.assertRaisesRegex(ValueError, "admission differs or failed"):
                    study.load_selection({"inputs": {"preflight": spec}}, output, source, None)

    @unittest.skipIf(np is None, "NumPy unavailable")
    def test_development_recount_rejects_semantic_corruption_after_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = development_fixture(root)
            study.recount_development(root, run, np)
            key, path = "normal_d2", root / "normal_d2.npz"
            with np.load(path, allow_pickle=False) as opened:
                original = {k: opened[k].copy() for k in opened.files}
            for mutation, message in (("nll", "NLL drift"), ("prediction", "target/prediction drift"),
                                      ("target", "target/prediction drift")):
                arrays = {k: v.copy() for k, v in original.items()}
                arrays[mutation][0] += 1e-8 if mutation == "nll" else 1
                np.savez(path, **arrays)
                # Update hashes deliberately: the scientific contract must also reject it.
                run["development"][key]["predictions"] = study.descriptor(root, path)
                meta_path = root / "normal_d2.json"
                meta = study.read_json(meta_path)
                meta["predictions"] = run["development"][key]["predictions"]
                write_json(meta_path, meta)
                run["development"][key]["metadata"] = study.descriptor(root, meta_path)
                with self.assertRaisesRegex(ValueError, message):
                    study.recount_development(root, run, np)

    def test_gate_requires_all_ten_unchanged_completed_training_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, selection, identity, manifest, gate = gate_fixture(root)
            # Timestamp is phase-specific; scientific identity remains identical.
            source = {**source, "recorded_at_utc": "2026-01-01T00:00:03+00:00"}
            study.authenticate_gate(root, source, selection, identity)
            short = {**manifest, "runs": manifest["runs"][:-1]}
            write_json(root / "training_manifest.json", short)
            gate["training_manifest_sha256"] = study.sha(root / "training_manifest.json")
            write_json(root / "pre_final_archive_receipt.json", gate)
            with self.assertRaisesRegex(ValueError, "checkpoint inventory"):
                study.authenticate_gate(root, source, selection, identity)
            write_json(root / "training_manifest.json", manifest)
            gate["training_manifest_sha256"] = study.sha(root / "training_manifest.json")
            write_json(root / "pre_final_archive_receipt.json", gate)
            checkpoint = root / manifest["runs"][0]["checkpoint"]["path"]
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact identity"):
                study.authenticate_gate(root, source, selection, identity)
            drift = {**source, "plan_sha256": "9" * 64}
            with self.assertRaisesRegex(ValueError, "gate identity"):
                study.authenticate_gate(root, drift, selection, identity)

    def test_consumed_access_precedes_model_and_cannot_silently_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, selection, identity, _, _ = gate_fixture(root)
            with mock.patch.object(study, "make_model", side_effect=RuntimeError("injected before model construction")) as build:
                with self.assertRaisesRegex(RuntimeError, "injected before"):
                    study.final_evaluation(root, source, selection, identity, None, None)
                receipt = study.read_json(root / "final_access.json")
                self.assertEqual(receipt["status"], "consumed")
                self.assertFalse((root / "final_metrics.json").exists())
                with self.assertRaises(FileExistsError):
                    study.final_evaluation(root, source, selection, identity, None, None)
                self.assertEqual(build.call_count, 1)

    @unittest.skipIf(torch is None or np is None, "Torch/NumPy unavailable")
    def test_actual_factories_one_optimizer_fixture_and_weights_only_roundtrip(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        with torch.random.fork_rng(devices=[]):
            rows = [study.census(0)[0], study.census(0)[9]]
            frames, steps, _ = study.frames_for_rows(rows, torch)
            target = torch.tensor([r["target"] for r in rows], dtype=torch.long)
            for family in study.FAMILIES:
                torch.manual_seed(110931)
                model = study.make_model(family, torch).train()
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 29392 if family == "neuropixel" else 29336)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                out = study.stream(model, family, frames, steps)
                loss = torch.nn.functional.cross_entropy(out["logits"], target)
                self.assertTrue(torch.isfinite(loss).item())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all().item()
                                    for p in model.parameters()))
                optimizer.step()
                model.eval()
                if family == "neuropixel":
                    self.assertEqual(torch.count_nonzero(model.dictionary()[0]).item(), 0)
                with torch.inference_mode():
                    expected = study.stream(model, family, frames, steps)["logits"]
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "fixture.pt"
                    torch.save(model.state_dict(), path)
                    restored = study.make_model(family, torch).eval()
                    restored.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
                    with torch.inference_mode():
                        actual = study.stream(restored, family, frames, steps)["logits"]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
