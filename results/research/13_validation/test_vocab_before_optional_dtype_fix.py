"""Bounded contracts for native vocabulary expansion; no task-learning claim.

The preserved legacy function is executed only from its SHA-256-pinned preimage.
Its observed failures are expected diagnostic witnesses, not silently repaired.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "results/research/13_validation/phase3_before.py"
LEGACY_SHA256 = "57c177757b333ee5ba9fdc953cbbd7ef284892edee612e234cca1df83b6814d0"
if torch is not None:
    from neuropixel.model import NeuroPixel, TinyTransformer
    from neuropixel.phase3 import expand_vocab


@unittest.skipIf(torch is None, "Torch is required for the frozen CPU contracts")
class VocabularyExpansionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        raw = LEGACY.read_bytes()
        if hashlib.sha256(raw).hexdigest() != LEGACY_SHA256:
            raise AssertionError("legacy expansion preimage identity differs")
        tree = ast.parse(raw.decode("utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "expand_vocab"]
        if len(functions) != 1:
            raise AssertionError("exactly one legacy expand_vocab required")
        scope = {"copy": copy, "torch": torch, "nn": nn,
                 "NeuroPixel": NeuroPixel, "TinyTransformer": TinyTransformer}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(LEGACY), "exec"), scope)
        cls.legacy = staticmethod(scope["expand_vocab"])

    def setUp(self):
        torch.manual_seed(130001)

    def model(self, family, dtype=torch.float32):
        if family == "np":
            rgb = torch.arange(27, dtype=torch.float32).reshape(9, 3) / 100
            mask = torch.zeros(9, dtype=torch.bool)
            mask[[0, 5]] = True
            model = NeuroPixel(9, (2, 2), c_id=4, c=8, hidden=12, steps=2,
                               fire_rate=.5, grounded=(rgb, mask))
            with torch.no_grad():
                model.embed.weight[0].fill_(2)
                model.read.bias.copy_(torch.tensor([.05, .1, .15, .2]))
                model.f2.weight.fill_(.005)
                model.f2.bias.fill_(.02)
        else:
            model = TinyTransformer(9, 3, 3, (2, 2), d=8, layers=1, heads=2, ff=12)
        return model.to(dtype=dtype).eval()

    def canvas(self):
        return torch.tensor([[[1, 5, 0], [3, 6, 0], [0, 1, 0]],
                             [[1, 6, 0], [3, 5, 0], [0, 1, 0]]], dtype=torch.long)

    def full_values(self, model):
        return {kind + name: tensor.detach().clone()
                for kind, pairs in (("parameter:", model.named_parameters()),
                                    ("buffer:", model.named_buffers()))
                for name, tensor in pairs}

    def assert_values(self, model, before):
        current = self.full_values(model)
        self.assertEqual(set(current), set(before))
        for name in before:
            torch.testing.assert_close(current[name], before[name], rtol=0, atol=0)

    def assert_old_logits(self, base, expanded):
        with torch.no_grad():
            original = base(self.canvas())
            changed = expanded(self.canvas())
        torch.testing.assert_close(changed["logits"][:, :9], original["logits"],
                                   rtol=1e-5, atol=1e-6)
        if "state" in original:
            torch.testing.assert_close(changed["state"], original["state"], rtol=0, atol=0)
            torch.testing.assert_close(expanded.lens_logits(changed["state"])[..., :9],
                                       base.lens_logits(original["state"]), rtol=1e-5, atol=1e-6)

    def test_01_preserved_legacy_exhibits_grounding_loss_and_dtype_failure(self):
        base = self.model("np")
        before = base.dictionary().detach().clone()
        expanded, _, _ = self.legacy(base, 1)
        self.assertEqual(int(base.g_mask.sum()), 2)
        self.assertEqual(int(expanded.g_mask.sum()), 0)
        self.assertFalse(torch.equal(before, expanded.dictionary()[:9]))
        witness = {"legacy_preimage_sha256": LEGACY_SHA256,
                   "grounded_mask_before": int(base.g_mask.sum()),
                   "grounded_mask_after": int(expanded.g_mask.sum()),
                   "old_dictionary_max_absolute_change":
                       float((before - expanded.dictionary()[:9]).abs().max()),
                   "dtype_failures": {}}
        for family in ("np", "tf"):
            with self.subTest(family=family):
                expanded, _, _ = self.legacy(self.model(family, torch.float64), 1)
                weight = expanded.embed.weight if family == "np" else expanded.tok.weight
                self.assertEqual(weight.dtype, torch.float32)
                with self.assertRaises(RuntimeError) as failure:
                    expanded(self.canvas())
                witness["dtype_failures"][family] = {
                    "type": type(failure.exception).__name__, "message": str(failure.exception)}
        print("LEGACY_EXPANSION_WITNESS " + json.dumps(witness, sort_keys=True))

    def test_02_neuropixel_preserves_grounding_prefixes_configuration_and_outputs(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                base = self.model("np", dtype)
                original = self.full_values(base)
                expanded, params, first = expand_vocab(base, 3)
                self.assertEqual(first, 9)
                self.assertEqual((expanded.embed.num_embeddings, len(params)), (12, 1))
                self.assertIs(params[0], expanded.embed.weight)
                self.assertEqual((expanded.out_pos, expanded.steps, expanded.fire_rate,
                                  expanded.training, expanded.retina), ((2, 2), 2, .5, False, None))
                torch.testing.assert_close(expanded.embed.weight[:9], base.embed.weight, rtol=0, atol=0)
                torch.testing.assert_close(expanded.g_rgb[:9], base.g_rgb, rtol=0, atol=0)
                torch.testing.assert_close(expanded.g_mask[:9], base.g_mask, rtol=0, atol=0)
                self.assertFalse(expanded.g_mask[9:].any())
                self.assertEqual(int(expanded.g_rgb[9:].count_nonzero()), 0)
                torch.testing.assert_close(expanded.dictionary()[:9], base.dictionary(), rtol=0, atol=0)
                self.assertEqual(int(expanded.dictionary()[0].count_nonzero()), 0)
                self.assertEqual(expanded.g_rgb.dtype, dtype)
                self.assertEqual(expanded.embed.weight.dtype, dtype)
                self.assertNotIn("g_rgb", expanded.state_dict())
                self.assertNotIn("g_mask", expanded.state_dict())
                for name, p in expanded.named_parameters():
                    if name != "embed.weight":
                        torch.testing.assert_close(p, dict(base.named_parameters())[name], rtol=0, atol=0)
                        self.assertFalse(p.requires_grad)
                self.assertNotEqual(expanded.embed.weight.data_ptr(), base.embed.weight.data_ptr())
                self.assert_old_logits(base, expanded)
                self.assert_values(base, original)

    def test_03_transformer_preserves_old_tables_configuration_and_outputs(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                base = self.model("tf", dtype)
                before = self.full_values(base)
                expanded, params, first = expand_vocab(base, 2)
                self.assertEqual(first, 9)
                self.assertEqual((expanded.tok.num_embeddings, expanded.head.out_features), (11, 11))
                self.assertEqual({id(p) for p in params},
                                 {id(expanded.tok.weight), id(expanded.head.weight), id(expanded.head.bias)})
                self.assertEqual((expanded.out_idx, expanded.training), (8, False))
                for name, p in expanded.named_parameters():
                    old = dict(base.named_parameters())[name]
                    if name in ("tok.weight", "head.weight", "head.bias"):
                        torch.testing.assert_close(p[:9], old, rtol=0, atol=0)
                        self.assertEqual(p.dtype, dtype)
                        self.assertEqual(p.device, old.device)
                    else:
                        torch.testing.assert_close(p, old, rtol=0, atol=0)
                        self.assertFalse(p.requires_grad)
                self.assert_old_logits(base, expanded)
                self.assert_values(base, before)

    def test_04_fresh_adam_without_decay_changes_only_appended_rows(self):
        for family in ("np", "tf"):
            with self.subTest(family=family):
                base = self.model(family)
                original = self.full_values(base)
                rng = torch.get_rng_state().clone()
                legacy, _, _ = self.legacy(base, 2)
                after_legacy_rng = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                model, params, first = expand_vocab(base, 2)
                torch.testing.assert_close(torch.get_rng_state(), after_legacy_rng, rtol=0, atol=0)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, dict(legacy.named_parameters())[name],
                                               rtol=0, atol=0)
                before = self.full_values(model)
                canvas = self.canvas()
                canvas[:, 0, 1] = first
                optimizer = torch.optim.Adam(params, lr=.01, weight_decay=0)
                loss = F.cross_entropy(model(canvas)["logits"], torch.full((2,), first))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for p in params:
                    self.assertIsNotNone(p.grad)
                    self.assertEqual(int(p.grad[:first].count_nonzero()), 0)
                    self.assertTrue(torch.isfinite(p.grad).all())
                self.assertTrue(any(int(p.grad[first:].count_nonzero()) > 0 for p in params))
                optimizer.step()
                names = {name for name, p in model.named_parameters() if id(p) in {id(q) for q in params}}
                after = self.full_values(model)
                for key, old in before.items():
                    if key.startswith("parameter:") and key[len("parameter:"):] in names:
                        torch.testing.assert_close(after[key][:first], old[:first], rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(after[key], old, rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(after["parameter:" + name][first:],
                                                    before["parameter:" + name][first:]) for name in names))
                self.assert_values(base, original)

    def test_05_hooks_follow_dtype_conversion_and_repeated_expansion(self):
        for family in ("np", "tf"):
            with self.subTest(family=family):
                base = self.model(family)
                once, _, _ = expand_vocab(base, 2)
                original = self.full_values(once)
                twice, params, first = expand_vocab(once, 1)
                self.assertEqual(first, 11)
                for name, p in twice.named_parameters():
                    old = dict(once.named_parameters())[name]
                    torch.testing.assert_close(p[:11] if id(p) in {id(q) for q in params} else p,
                                               old, rtol=0, atol=0)
                self.assert_values(once, original)
                twice.half()
                trainable = [p for p in twice.parameters() if p.requires_grad]
                sum(p.sum() for p in trainable).backward()
                for p in trainable:
                    self.assertEqual(p.grad.dtype, torch.float16)
                    self.assertEqual(int(p.grad[:first].count_nonzero()), 0)
                    torch.testing.assert_close(p.grad[first:], torch.ones_like(p.grad[first:]), rtol=0, atol=0)

    def test_06_invalid_requests_fail_before_copy_or_rng_consumption(self):
        base = self.model("np")
        before = self.full_values(base)
        for new in (0, -1, True, 1.5, "1", None):
            with self.subTest(new=repr(new)):
                rng = torch.get_rng_state().clone()
                with self.assertRaises(ValueError):
                    expand_vocab(base, new)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                self.assert_values(base, before)
        class ExtraTableNeuroPixel(NeuroPixel):
            pass
        for unsupported in (nn.Linear(3, 3), ExtraTableNeuroPixel(9, (2, 2))):
            with self.subTest(model=type(unsupported).__name__):
                rng = torch.get_rng_state().clone()
                with self.assertRaises(TypeError):
                    expand_vocab(unsupported, 1)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_07_old_logits_do_not_imply_old_softmax_or_global_argmax(self):
        base = self.model("tf", torch.float64)
        expanded, _, first = expand_vocab(base, 1)
        with torch.no_grad():
            expanded.head.weight[first].zero_()
            expanded.head.bias[first].fill_(50)
            old = base(self.canvas())["logits"]
            new = expanded(self.canvas())["logits"]
        torch.testing.assert_close(new[:, :first], old, rtol=1e-12, atol=1e-12)
        old_probability, new_probability = old.softmax(-1), new.softmax(-1)
        old_mass = new_probability[:, :first].sum(-1, keepdim=True)
        torch.testing.assert_close(new_probability[:, :first] / old_mass, old_probability,
                                   rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.all(new_probability[:, 1:first] < old_probability[:, 1:]))
        self.assertTrue(torch.all(new.argmax(-1) == first))
        self.assertTrue(torch.all(new[:, :first].argmax(-1) == old.argmax(-1)))

    def test_08_gradient_mask_is_not_an_optimizer_weight_decay_freeze(self):
        base = self.model("np")
        expanded, params, first = expand_vocab(base, 1)
        before = expanded.embed.weight.detach().clone()
        # This deliberately unsupported optimizer setting is a negative control.
        optimizer = torch.optim.AdamW(params, lr=.01, weight_decay=.1)
        optimizer.zero_grad(set_to_none=True)
        expanded.embed.weight.sum().backward()
        self.assertEqual(int(expanded.embed.weight.grad[:first].count_nonzero()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before[1:first], expanded.embed.weight[1:first]))
        print("EXPANSION_DECAY_NEGATIVE_CONTROL old_gradient_zero=true old_rows_changed=true")


if __name__ == "__main__":
    unittest.main()
