"""Four finite CPU contracts for the item13 battery-helper correction.

Both legacy functions are extracted from a pinned full-source preimage; the
driver itself is never imported and no learning benchmark is executed.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "results/research/13_validation/battery_before.py"
LEGACY_SHA256 = "15f0cd9e6a0986f950c8bb6a142dc4c198b902ec256d2b844bd4ce7f8f5c0dad"
if torch is not None:
    from neuropixel.model import NeuroPixel, TinyTransformer
    from neuropixel.phase3 import expand_vocab


def extract(path, expected_sha256=None):
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AssertionError("battery preimage identity differs")
    selected = [node for node in ast.parse(raw.decode("utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name in ("expand", "with_new_word")]
    if sorted(node.name for node in selected) != ["expand", "with_new_word"]:
        raise AssertionError("two battery functions required")
    scope = {"torch": torch, "copy": copy, "NeuroPixel": NeuroPixel,
             "TinyTransformer": TinyTransformer, "expand_vocab": expand_vocab,
             "DEV": torch.device("cpu")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


class ManualTask:
    """Eight explicit role/replacement combinations, with no sampled task pool."""
    query_pos = (4, 2)

    def __init__(self):
        self.role_ids = torch.tensor([1, 2, 3, 4])

    def sample(self, n, split, generator, meta=False):
        if n != 8 or split != "fixture" or not meta:
            raise ValueError("this manual contract fixture has exactly eight rows")
        canvas = torch.zeros(8, 5, 4, dtype=torch.long)
        fillers = torch.tensor([5, 17, 6, 27])
        for role in range(4):
            canvas[:, role, 0] = role + 1
            canvas[:, role, 1] = fillers[role]
        query = torch.arange(8) % 4
        canvas[:, 4, 2] = self.role_ids[query]
        return canvas, fillers[query].clone(), {
            "rows": torch.arange(4).repeat(8, 1),
            "cols": torch.zeros(8, 4, dtype=torch.long)}


def visible_answer(canvas):
    """Read the unique visible fact with the queried role, without gold/meta."""
    answers = []
    for example in canvas:
        role = int(example[4, 2])
        matching = torch.nonzero(example[:4, 0] == role).flatten()
        if matching.numel() != 1:
            raise AssertionError("ambiguous manual visible role")
        answers.append(int(example[int(matching[0]), 1]))
    return torch.tensor(answers, dtype=torch.long)


@unittest.skipIf(torch is None, "Torch is required for the frozen CPU contracts")
class BatteryExpansionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        cls.old = extract(LEGACY, LEGACY_SHA256)
        cls.current = extract(ROOT / "scripts/battery.py")

    def setUp(self):
        torch.manual_seed(130101)

    def transformed(self, namespace, ask_new):
        # A full 4-query x 2-insertion-role census; no seed-dependent coverage.
        draws = torch.tensor([.25] * 4 + [.75] * 4)
        with patch.object(torch, "rand", return_value=draws):
            return namespace["with_new_word"](ManualTask(), 8, "fixture", 35,
                                               seed=130111, ask_new=ask_new)

    def test_01_old_unforced_query_has_two_stale_labels_fixed_census_has_none(self):
        legacy_canvas, legacy_y = self.transformed(self.old, False)
        canvas, y = self.transformed(self.current, False)
        original, original_y, _ = ManualTask().sample(8, "fixture", None, meta=True)
        truth = visible_answer(canvas)
        wrong = torch.nonzero(legacy_y != truth).flatten().tolist()
        self.assertEqual(wrong, [0, 6])
        torch.testing.assert_close(canvas, legacy_canvas, rtol=0, atol=0)
        torch.testing.assert_close(canvas[:, 4, 2], original[:, 4, 2], rtol=0, atol=0)
        torch.testing.assert_close(y, truth, rtol=0, atol=0)
        self.assertEqual(int((y == 35).sum()), 2)
        untouched = torch.tensor([1, 2, 3, 4, 5, 7])
        torch.testing.assert_close(y[untouched], original_y[untouched], rtol=0, atol=0)
        self.assertEqual(int((canvas != original).sum()), 8)
        print("LEGACY_BATTERY_GOLD_WITNESS " + json.dumps({
            "legacy_preimage_sha256": LEGACY_SHA256, "n": 8,
            "legacy_wrong_indices": wrong, "legacy_targets": legacy_y.tolist(),
            "visible_targets": truth.tolist(), "corrected_wrong": int((y != truth).sum()),
            "scope": "Exhaustive manual query-role by inserted-role fixture, not a dataset score."},
            sort_keys=True))

    def test_02_forced_new_query_preserves_legacy_inputs_and_targets(self):
        legacy_canvas, legacy_y = self.transformed(self.old, True)
        canvas, y = self.transformed(self.current, True)
        torch.testing.assert_close(canvas, legacy_canvas, rtol=0, atol=0)
        torch.testing.assert_close(y, legacy_y, rtol=0, atol=0)
        torch.testing.assert_close(y, visible_answer(canvas), rtol=0, atol=0)
        self.assertEqual(y.tolist(), [35] * 8)

    def test_03_wrapper_preserves_rectangular_transformer_dtype_mode_and_readout(self):
        for mode in (False, True):
            with self.subTest(training=mode):
                base = TinyTransformer(9, 2, 3, (0, 1), d=8, layers=1, heads=2, ff=12).double()
                base.train(mode)
                old_values = {k: v.clone() for k, v in base.state_dict().items()}
                if not mode:
                    with self.assertRaises(RuntimeError) as failure:
                        self.old["expand"](base, 2)
                    print("LEGACY_BATTERY_RECONSTRUCTION_WITNESS " + json.dumps({
                        "family": "TinyTransformer", "type": type(failure.exception).__name__,
                        "message": str(failure.exception)}, sort_keys=True))
                expanded, params, first = self.current["expand"](base, 2)
                self.assertEqual((first, expanded.tok.num_embeddings, expanded.head.out_features), (9, 11, 11))
                self.assertEqual((expanded.training, expanded.out_idx), (mode, 1))
                self.assertEqual(tuple(expanded.pos.shape), (6, 8))
                self.assertEqual(len(expanded.enc.layers), 1)
                self.assertEqual(expanded.enc.layers[0].self_attn.num_heads, 2)
                self.assertEqual(expanded.enc.layers[0].linear1.out_features, 12)
                self.assertEqual({id(p) for p in params}, {
                    id(expanded.tok.weight), id(expanded.head.weight), id(expanded.head.bias)})
                for name, value in expanded.state_dict().items():
                    expected = old_values[name]
                    torch.testing.assert_close(value[:9] if name in ("tok.weight", "head.weight", "head.bias")
                                               else value, expected, rtol=0, atol=0)
                self.assertTrue(all(p.dtype == torch.float64 and p.device.type == "cpu"
                                    for p in expanded.parameters()))
                for name, value in base.state_dict().items():
                    torch.testing.assert_close(value, old_values[name], rtol=0, atol=0)
                canvas = torch.tensor([[[1, 5, 0], [3, 6, 0]], [[1, 6, 0], [3, 5, 0]]])
                base.eval(); expanded.eval()
                with torch.no_grad():
                    torch.testing.assert_close(expanded(canvas)["logits"][:, :9], base(canvas)["logits"],
                                               rtol=1e-12, atol=1e-12)

    def test_04_wrapper_preserves_nondefault_neuropixel_grounding_and_configuration(self):
        rgb = torch.arange(27, dtype=torch.float32).reshape(9, 3) / 100
        mask = torch.zeros(9, dtype=torch.bool)
        mask[[0, 5]] = True
        base = NeuroPixel(9, (0, 2), c_id=4, c=8, hidden=12, steps=3,
                          fire_rate=.25, grounded=(rgb, mask), retina=True).double().eval()
        old_values = {k: v.clone() for k, v in base.state_dict().items()}
        with self.assertRaises(RuntimeError) as failure:
            self.old["expand"](base, 2)
        print("LEGACY_BATTERY_RECONSTRUCTION_WITNESS " + json.dumps({
            "family": "NeuroPixel", "type": type(failure.exception).__name__,
            "message": str(failure.exception)}, sort_keys=True))
        expanded, params, first = self.current["expand"](base, 2)
        self.assertEqual((first, expanded.embed.num_embeddings, len(params)), (9, 11, 1))
        self.assertIs(params[0], expanded.embed.weight)
        self.assertEqual((expanded.out_pos, expanded.steps, expanded.fire_rate, expanded.training),
                         ((0, 2), 3, .25, False))
        self.assertEqual((expanded.seed.in_channels, expanded.seed.out_channels,
                          expanded.f1.out_channels), (4, 8, 12))
        self.assertIsNotNone(expanded.retina)
        self.assertIsNot(expanded.retina, base.retina)
        for name, value in expanded.state_dict().items():
            torch.testing.assert_close(value[:9] if name == "embed.weight" else value,
                                       old_values[name], rtol=0, atol=0)
        torch.testing.assert_close(expanded.g_rgb[:9], base.g_rgb, rtol=0, atol=0)
        torch.testing.assert_close(expanded.g_mask[:9], base.g_mask, rtol=0, atol=0)
        torch.testing.assert_close(expanded.dictionary()[:9], base.dictionary(), rtol=0, atol=0)
        self.assertFalse(expanded.g_mask[9:].any())
        self.assertEqual(int(expanded.g_rgb[9:].count_nonzero()), 0)
        self.assertEqual(expanded.g_rgb.dtype, torch.float64)
        self.assertNotIn("g_rgb", expanded.state_dict())
        self.assertNotIn("g_mask", expanded.state_dict())
        self.assertTrue(all(p.dtype == torch.float64 and p.device.type == "cpu"
                            for p in expanded.parameters()))
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, old_values[name], rtol=0, atol=0)
        canvas = torch.tensor([[[1, 5, 0], [3, 6, 0], [0, 1, 0]],
                               [[1, 6, 0], [3, 5, 0], [0, 1, 0]]])
        with torch.no_grad():
            torch.testing.assert_close(expanded(canvas)["logits"][:, :9], base(canvas)["logits"],
                                       rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
