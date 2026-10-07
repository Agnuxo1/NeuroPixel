"""Three bounded scanner contracts using invented deterministic fixtures.

No historical checkpoint, task sampler, performance dataset or saved image is
opened. The GIF integration test runs the real old/current function bodies,
replacing only rendering sinks and fonts; scanner/model arithmetic remains real.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neuropixel.model import NeuroPixel
from neuropixel.scanner import lens

LEGACY = ROOT / "results/research/16_validation/readme_figures_before.py"
LEGACY_SHA256 = "7b8e64d1599b13d904b17c9a29f651cb630ad709a33559545c8c361140213e4b"


def tiny_model(out_pos=(0, 0)):
    with torch.random.fork_rng(devices=[]):
        model = NeuroPixel(4, out_pos, c_id=4, c=4, hidden=8, steps=4).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.embed.weight.copy_(torch.eye(4, dtype=torch.float64))
        model.read.weight.copy_(torch.eye(4, dtype=torch.float64))
    return model


class InventedTask:
    """Only an invented rendering context, not RoleTask or a held-out panel."""
    v = SimpleNamespace(tokens=["<vacío>", "perro", "gato", "robot"])
    query_pos, out_pos = (7, 6), (7, 7)

    def sample(self, batch, split, generator):
        if batch != 1 or split != "test" or generator.initial_seed() != 11:
            raise AssertionError("unexpected renderer request")
        canvas = torch.zeros((1, 8, 8), dtype=torch.long)
        canvas[0, 7, 6] = 1
        return canvas, torch.tensor([1], dtype=torch.long)


class ScannerContracts(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.get_num_threads(), 1, "run under the frozen one-thread worker")

    def test_negative_logits_and_finite_pad_convention(self):
        model = tiny_model()
        canvas = torch.zeros((1, 1, 1), dtype=torch.long)
        for scores, expected in (([-1.0, -2.0, -3.0], 1),
                                 ([-20000.0, -20001.0, -20002.0], 0)):
            with self.subTest(nonpad_scores=scores):
                with torch.no_grad():
                    model.read.bias.copy_(torch.tensor([0.0] + scores, dtype=torch.float64))
                    out = model(canvas, trace=True, steps=1)
                    raw = model.lens_logits(out["frames"][0])
                self.assertTrue(torch.all(raw.argmax(-1) == 0).item())
                words, confidence = lens(model, out["frames"][0])
                self.assertTrue(torch.all(words == expected).item())
                self.assertEqual(int(out["logits"][0].argmax()), expected)
                self.assertTrue(torch.all(out["logits"][:, 0] == -10000).item())
                expected_confidence = (1.0 / (1.0 + math.exp(-1.0) + math.exp(-2.0))
                                       if expected == 1 else 1.0)
                self.assertAlmostEqual(float(confidence[-1, 0, 0]), expected_confidence, places=14)

    def test_preupdate_lens_and_posthook_trace_have_distinct_times(self):
        model = tiny_model()
        with torch.no_grad():
            model.f2.bias[1] = 1.0
        canvas = torch.zeros((1, 1, 1), dtype=torch.long)

        def hook(step, state):
            if step == 2:
                addition = torch.zeros_like(state)
                addition[:, 2] = 3.0
                return state + addition
            return state

        with torch.no_grad():
            out = model(canvas, trace=True, lens_every=2, steps=4, hook=hook)
        expected = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 3, 0],
                                 [0, 3, 3, 0], [0, 4, 3, 0]], dtype=torch.float64)
        torch.testing.assert_close(out["frames"][0, :, :, 0, 0], expected, rtol=0, atol=0)
        # Numbered updates 2 and 4 sample before the update/hook: states 1 and 3.
        torch.testing.assert_close(out["lens"][0, :, 0, 0],
                                   expected[[1, 3]], rtol=0, atol=0)
        self.assertFalse(torch.equal(out["lens"][0, 0, 0, 0], expected[2]))
        final_expected = expected[4].clone()
        final_expected[0] = -10000
        torch.testing.assert_close(out["logits"][0], final_expected, rtol=0, atol=0)
        words, confidence = lens(model, out["frames"][0])
        self.assertEqual(int(words[-1, 0, 0]), int(out["logits"][0].argmax()))
        self.assertAlmostEqual(float(confidence[-1, 0, 0]),
                               float(out["logits"][0].softmax(-1).max()), places=14)

    def test_real_gif_generator_corrects_legacy_visible_pad_labels(self):
        spec = importlib.util.spec_from_file_location(
            "item16_actual_readme_figures", ROOT / "scripts/make_readme_figures.py")
        figures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(figures)
        original = LEGACY.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), LEGACY_SHA256)
        tree = ast.parse(original.decode("utf-8"), filename=str(LEGACY))
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "thinking_gif"]
        self.assertEqual(len(functions), 1)
        namespace = dict(vars(figures))
        isolated = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
        exec(compile(isolated, str(LEGACY), "exec"), namespace)
        legacy = namespace["thinking_gif"]
        model = tiny_model((7, 7))
        with torch.no_grad():
            model.read.bias.copy_(torch.tensor([0.0, -1.0, -2.0, -3.0], dtype=torch.float64))

        def capture(function, folder):
            tokens, saves = [], []

            class SinkImage:
                def save(self, path, **kwargs):
                    saves.append((Path(path), kwargs))

            class SinkDraw:
                def noop(self, *args, **kwargs):
                    return None
                text = rectangle = rounded_rectangle = noop

            def record_color(token):
                tokens.append(token)
                return np.zeros(3, dtype=float)

            replacements = {
                "Image": SimpleNamespace(new=lambda *args, **kwargs: SinkImage()),
                "ImageDraw": SimpleNamespace(Draw=lambda image: SinkDraw()),
                "font": lambda *args, **kwargs: None,
                "cat_color": record_color, "IMG": Path(folder),
            }
            with mock.patch.multiple(figures, **replacements), mock.patch.dict(namespace, replacements):
                function(model, InventedTask())
            self.assertEqual(len(saves), 1)
            self.assertEqual(saves[0][0], Path(folder) / "thinking.gif")
            self.assertEqual(len(saves[0][1]["append_images"]), 32)
            self.assertEqual(len(saves[0][1]["duration"]), 33)
            self.assertEqual(saves[0][1]["loop"], 0)
            self.assertEqual(len(tokens), 33 * 8 * 8)
            self.assertFalse((Path(folder) / "thinking.gif").exists())
            return tokens

        with tempfile.TemporaryDirectory() as folder:
            old_tokens = capture(legacy, folder)
            current_tokens = capture(figures.thinking_gif, folder)
        self.assertEqual(set(old_tokens), {"<vacío>"})
        self.assertEqual(set(current_tokens), {"perro"})
        self.assertNotEqual(old_tokens, current_tokens)


if __name__ == "__main__":
    unittest.main()
