"""Tiny deterministic PAD regression fixtures; no dataset sampling or training run.

The preserved pre-fix source is loaded by hash for a positive demonstration of
the defect. Execute that demonstration alone with --legacy-demonstration PATH;
the resulting JSON is exclusive and is evidence of a fixture, not model quality.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropixel.model import NeuroPixel  # noqa: E402
from neuropixel.phase3 import np_stream  # noqa: E402
from neuropixel.research.models import ResearchNCA  # noqa: E402

BEFORE = ROOT / "results/research/08_validation/padding_model_before.py"
BEFORE_SHA = "564045fff799185e15b921873225d344014a4d4bcae6df6deacaf42f6e4bd701"


def legacy_class():
    if hashlib.sha256(BEFORE.read_bytes()).hexdigest() != BEFORE_SHA:
        raise AssertionError("The pre-fix source snapshot differs from its recorded SHA-256")
    spec = importlib.util.spec_from_file_location("padding_model_before", BEFORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NeuroPixel


def fixed_model(cls=NeuroPixel, **kwargs):
    # Constructors consume RNG, so isolate it; every parameter is then explicit.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(0)
        model = cls(4, (1, 1), c_id=4, c=2, hidden=2, steps=2,
                    fire_rate=1.0, **kwargs)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.embed.weight[1:, 0] = torch.tensor([1.0, -1.0, 0.5])
        # At an empty output cell, ds[0] = ReLU(1 + identity[0]).
        # Thus answer CE reaches PAD through reinjection without using the seed.
        model.f1.bias[0] = 1.0
        model.f1.weight[0, -4, 0, 0] = 1.0
        model.f2.weight[0, 0, 0, 0] = 1.0
        model.read.weight[0, 0] = 1.0
    return model


def canvas():
    return torch.tensor([[[1, 0], [0, 0]]], dtype=torch.long)


def objective(model, route):
    if route == "answer_reinjection":
        return F.cross_entropy(model(canvas())["logits"], torch.tensor([2]))
    if route == "lens_readout":
        state = torch.ones(1, 2, 2, 2)
        return F.cross_entropy(model.lens_logits(state).reshape(-1, 4),
                               torch.ones(4, dtype=torch.long))
    raise ValueError("Unknown fixture route")


def legacy_demonstration():
    rows = []
    for route in ("answer_reinjection", "lens_readout"):
        model = fixed_model(legacy_class())
        optimizer = torch.optim.SGD([model.embed.weight], lr=0.1)
        initial = model.dictionary()[0].detach().clone()
        loss = objective(model, route)
        loss.backward()
        gradient = model.embed.weight.grad[0].detach().clone()
        optimizer.step()
        after = model.dictionary()[0].detach().clone()
        if torch.count_nonzero(initial) or not torch.count_nonzero(gradient) or not torch.count_nonzero(after):
            raise AssertionError("Legacy PAD violation was not reproduced")
        rows.append({"route": route, "loss": loss.item(), "pad_before": initial.tolist(),
                     "pad_gradient": gradient.tolist(), "pad_after_sgd": after.tolist(),
                     "zero_pad_contract_violated": True})
    return {"status": "legacy_defect_reproduced", "source_sha256": BEFORE_SHA,
            "torch": torch.__version__, "device": "cpu", "optimizer_steps_per_route": 1,
            "fixture_only": True, "routes": rows}


class PaddingInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)

    def assert_zero(self, value):
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertEqual(torch.count_nonzero(value).item(), 0)

    def test_preserved_source_reproduces_both_legacy_gradient_paths(self):
        evidence = legacy_demonstration()
        self.assertEqual(len(evidence["routes"]), 2)
        self.assertTrue(all(row["zero_pad_contract_violated"] for row in evidence["routes"]))

    def test_actual_answer_and_lens_optimizer_steps_preserve_effective_pad(self):
        for route in ("answer_reinjection", "lens_readout"):
            with self.subTest(route=route):
                model = fixed_model()
                optimizer = torch.optim.SGD([model.embed.weight], lr=0.1)
                other_before = model.embed.weight[1:].detach().clone()
                loss = objective(model, route)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assert_zero(model.embed.weight.grad[0])
                self.assertGreater(model.embed.weight.grad[1:].abs().sum().item(), 0)
                optimizer.step()
                self.assert_zero(model.dictionary()[0])
                self.assert_zero(model.embed.weight[0])
                self.assertFalse(torch.equal(model.embed.weight[1:], other_before))

    def test_checkpoint_nonzero_pad_is_projected_without_rewriting_stored_weights(self):
        old = fixed_model(legacy_class())
        with torch.no_grad():
            old.embed.weight[0] = torch.tensor([0.25, -0.5, 1.0, 2.0])
        checkpoint = deepcopy(old.state_dict())
        corrected = fixed_model().eval()
        corrected.load_state_dict(checkpoint, strict=True)
        clean = deepcopy(corrected)
        with torch.no_grad():
            clean.embed.weight[0].zero_()
        before = corrected.embed.weight.detach().clone()
        self.assert_zero(corrected.dictionary()[0])
        self.assertTrue(torch.equal(corrected.dictionary()[1:], old.dictionary()[1:]))
        with torch.no_grad():
            actual = corrected(canvas(), trace=True, lens_every=1)
            expected = clean(canvas(), trace=True, lens_every=1)
        for key in ("state", "logits", "lens", "frames", "activity"):
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
        self.assertTrue(torch.equal(corrected.embed.weight, before))
        self.assertTrue(torch.equal(corrected.state_dict()["embed.weight"], checkpoint["embed.weight"]))

    def test_resumed_adamw_momentum_cannot_reintroduce_effective_pad(self):
        legacy = fixed_model(legacy_class())
        old_optimizer = torch.optim.AdamW([legacy.embed.weight], lr=0.01, weight_decay=0.1)
        objective(legacy, "answer_reinjection").backward()
        old_optimizer.step()
        corrected = fixed_model()
        corrected.load_state_dict(deepcopy(legacy.state_dict()))
        optimizer = torch.optim.AdamW([corrected.embed.weight], lr=0.01, weight_decay=0.1)
        optimizer.load_state_dict(deepcopy(old_optimizer.state_dict()))
        raw_before = corrected.embed.weight[0].detach().clone()
        self.assertGreater(raw_before.abs().sum().item(), 0)
        objective(corrected, "answer_reinjection").backward()
        self.assert_zero(corrected.embed.weight.grad[0])
        optimizer.step()
        # Old momentum/decay may still move the stored parameter; projection wins.
        self.assertFalse(torch.equal(corrected.embed.weight[0], raw_before))
        self.assert_zero(corrected.dictionary()[0])

    def test_grounded_pad_is_zero_and_other_values_and_gradients_are_unchanged(self):
        rgb = torch.tensor([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                            [0.7, 0.8, 0.9]], requires_grad=True)
        mask = torch.tensor([True, True, False, False])
        model = fixed_model(grounded=(rgb, mask))
        with torch.no_grad():
            model.embed.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10)
        raw = model.embed.weight.detach().clone()
        dictionary = model.dictionary()
        expected_other = torch.cat([torch.where(mask[1:, None], rgb.detach()[1:], raw[1:, :3]), raw[1:, 3:]], 1)
        self.assert_zero(dictionary[0])
        self.assertTrue(torch.equal(dictionary[1:], expected_other))
        dictionary.sum().backward()
        self.assert_zero(model.embed.weight.grad[0])
        self.assert_zero(rgb.grad[0])
        self.assert_zero(model.embed.weight.grad[1, :3])
        self.assertEqual(model.embed.weight.grad[1, 3].item(), 1.0)
        self.assertTrue(torch.equal(model.embed.weight.grad[2:], torch.ones(2, 4)))
        self.assertTrue(torch.equal(rgb.grad[1], torch.ones(3)))
        self.assertTrue(torch.equal(model.embed.weight.detach(), raw))

    def test_nonpad_function_and_gradients_match_legacy_when_pad_has_no_objective(self):
        legacy = fixed_model(legacy_class()).eval()
        corrected = fixed_model().eval()
        corrected.load_state_dict(legacy.state_dict())
        occupied = torch.ones(1, 2, 2, dtype=torch.long)
        expected = legacy(occupied)
        actual = corrected(occupied)
        self.assertTrue(torch.equal(actual["state"], expected["state"]))
        self.assertTrue(torch.equal(actual["logits"], expected["logits"]))
        F.cross_entropy(expected["logits"], torch.tensor([2])).backward()
        F.cross_entropy(actual["logits"], torch.tensor([2])).backward()
        for (left_name, left), (right_name, right) in zip(legacy.named_parameters(), corrected.named_parameters()):
            self.assertEqual(left_name, right_name)
            self.assertTrue(torch.equal(left.grad, right.grad), left_name)

    def test_forward_lens_and_stream_use_the_same_zero_pad_boundary(self):
        model = fixed_model().eval()
        with torch.no_grad():
            model.embed.weight[0].fill_(3.0)
        update_inputs = []
        handle = model.f1.register_forward_pre_hook(
            lambda module, args: update_inputs.append(args[0][:, -4:].detach().clone()))
        try:
            with torch.no_grad():
                forward = model(canvas(), lens_every=1)
                stream = np_stream(model, [canvas()], [2])
                np_stream(model, [canvas(), torch.zeros_like(canvas())], [1, 1])
        finally:
            handle.remove()
        self.assertEqual(len(update_inputs), 6)
        for identities in update_inputs[:5]:
            self.assert_zero(identities[:, :, 1, 1])
        self.assert_zero(update_inputs[-1])
        self.assert_zero(forward["lens"][..., 0])
        self.assertTrue(torch.equal(forward["state"], stream["state"]))
        self.assertTrue(torch.equal(forward["logits"], stream["logits"]))

    def test_zero_identity_and_initial_seed_do_not_imply_zero_recurrent_activity(self):
        model = fixed_model().eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 2, 2, dtype=torch.long), trace=True)
        self.assert_zero(model.dictionary()[0])
        self.assert_zero(out["frames"][:, 0])
        self.assertGreater(out["state"].abs().sum().item(), 0)
        self.assertGreater(out["activity"].item(), 0)

    def test_camera_observation_is_not_discarded_as_padding(self):
        model = fixed_model().eval()
        with torch.no_grad():
            model.seed.weight[0, 0, 0, 0] = 1.0
            model.embed.weight[0].fill_(4.0)
        empty = torch.zeros(1, 2, 2, dtype=torch.long)
        cam = torch.zeros(1, 2, 2, dtype=torch.bool)
        cam[0, 1, 1] = True
        rgb = torch.ones(1, 3, 2, 2)
        with torch.no_grad():
            out = model(empty, rgb=rgb, cam=cam, trace=True)
        self.assertEqual(out["frames"][0, 0, 0, 1, 1].item(), 1.0)
        self.assert_zero(out["frames"][0, 0, :, 0, :])
        self.assert_zero(model.dictionary()[0])

    def test_research_nca_explicit_legacy_policy_survives_core_correction(self):
        legacy = fixed_model(legacy_class()).eval()
        with torch.no_grad():
            legacy.embed.weight[0, 0] = 0.25
        research = fixed_model(ResearchNCA, freeze_pad=False).eval()
        research.load_state_dict(legacy.state_dict(), strict=True)
        projected = fixed_model(ResearchNCA, freeze_pad=True).eval()
        projected.load_state_dict(legacy.state_dict(), strict=True)
        corrected = fixed_model().eval()
        corrected.load_state_dict(legacy.state_dict(), strict=True)
        self.assertTrue(torch.equal(research.dictionary(), legacy.dictionary()))
        self.assertGreater(research.dictionary()[0].abs().sum().item(), 0)
        self.assert_zero(projected.dictionary()[0])
        with torch.no_grad():
            expected = legacy(canvas(), lens_every=1)
            historical = research(canvas(), lens_every=1)
            fixed = corrected(canvas(), lens_every=1)
            opt_in = projected(canvas(), lens_every=1)
        for key in ("state", "logits", "lens", "activity"):
            self.assertTrue(torch.equal(historical[key], expected[key]), key)
            self.assertTrue(torch.equal(opt_in[key], fixed[key]), key)
        self.assertFalse(torch.equal(historical["state"], fixed["state"]))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--legacy-demonstration":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result = legacy_demonstration()
        with Path(sys.argv[2]).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, sort_keys=True))
    else:
        unittest.main()
