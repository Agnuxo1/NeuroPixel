"""Five finite contracts for the item14 evaluation-only repair trace.

These are implementation tests on constructed fixtures, not learned competence
tests. The cloud plan fixes CPU execution, one numerical thread and one interop
thread. No test trains a model or changes the production model implementation.
"""
from __future__ import annotations
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from neuropixel.model import NeuroPixel
    from neuropixel.research.repair_trace import trace_repair


@unittest.skipIf(torch is None, "Torch is required for the declared CPU contracts")
class RepairTraceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)

    def setUp(self):
        torch.manual_seed(140101)

    def active_model(self):
        model = NeuroPixel(9, (2, 3), c_id=4, c=6, hidden=12,
                           steps=5, fire_rate=0.5).double().eval()
        with torch.no_grad():
            model.f2.weight.normal_(0, 0.03)
            model.f2.bias.normal_(0, 0.01)
        return model

    def canvas(self):
        return torch.tensor([[[2, 0, 3, 0], [0, 4, 0, 0], [5, 0, 1, 0]],
                             [[0, 6, 0, 7], [2, 0, 0, 0], [0, 8, 1, 0]]],
                            dtype=torch.long)

    def assert_exact(self, actual, expected):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def constructed(self, rule):
        model = NeuroPixel(4, (1, 1), c_id=4, c=4, hidden=8,
                           steps=3, fire_rate=0.5).double().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.embed.weight.copy_(torch.eye(4, dtype=torch.float64))
            model.seed.weight[:, :, 0, 0].copy_(torch.eye(4, dtype=torch.float64))
            model.read.weight.copy_(torch.eye(4, dtype=torch.float64))
            if rule == "copier":
                for channel in range(4):
                    model.f1.weight[channel, channel, 0, 0] = 1
                    model.f1.weight[4 + channel, 12 + channel, 0, 0] = 1
                    model.f2.weight[channel, channel, 0, 0] = -1
                    model.f2.weight[channel, 4 + channel, 0, 0] = 1
            elif rule != "holder":
                raise ValueError("unknown fixture rule")
        return model

    def test_native_held_source_equivalence(self):
        model, canvas = self.active_model(), self.canvas()
        before = {name: value.detach().clone() for name, value in
                  list(model.named_parameters()) + list(model.named_buffers())}
        input_before, rng_before = canvas.clone(), torch.get_rng_state().clone()
        keep = torch.ones((2, 1, 3, 4), dtype=torch.bool)
        with torch.no_grad():
            native = model(canvas, trace=True, steps=5)
        traced = trace_repair(model, canvas, warmup_steps=2, recovery_steps=3,
                              keep_mask=keep, recovery_canvas=canvas)
        joined = torch.cat((traced["prefix_states"],
                            traced["recovery_states"][:, 1:]), dim=1)
        self.assert_exact(joined, native["frames"])
        self.assert_exact(traced["logits"][:, -1], native["logits"])
        self.assert_exact(traced["pre_damage"], native["frames"][:, 2])
        for time_index in range(3):
            state = native["frames"][:, time_index]
            with torch.no_grad():
                expected = model.read(state[:, :, 2, 3]) @ model.dictionary().T
                expected[:, 0] = -1e4
            self.assert_exact(traced["prefix_logits"][:, time_index], expected)
        self.assertTrue(bool((native["frames"][:, 1] != native["frames"][:, 0]).any()))
        for value in traced.values():
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
            self.assertTrue(bool(torch.isfinite(value).all()))
        after = dict(list(model.named_parameters()) + list(model.named_buffers()))
        for name, value in before.items():
            self.assert_exact(value, after[name])
        self.assert_exact(canvas, input_before)
        self.assert_exact(torch.get_rng_state(), rng_before)

    def test_post_update_boundary_and_snapshot_ownership(self):
        model, canvas = self.active_model(), self.canvas()
        mask = torch.ones((2, 1, 3, 4), dtype=torch.bool)
        mask[:, :, 1, 1] = False
        mask[1, :, 2, 3] = False
        captured = {}
        def lesion(time_index, state):
            if time_index == 2:
                captured["before"] = state.detach().clone()
                return state * mask
            return state
        with torch.no_grad():
            native = model(canvas, trace=True, steps=5, hook=lesion)
        traced = trace_repair(model, canvas, warmup_steps=2, recovery_steps=3,
                              keep_mask=mask, recovery_canvas=canvas)
        self.assert_exact(traced["pre_damage"], captured["before"])
        self.assert_exact(traced["post_damage"], captured["before"] * mask)
        self.assert_exact(traced["prefix_states"][:, :2], native["frames"][:, :2])
        self.assert_exact(traced["recovery_states"], native["frames"][:, 2:])
        self.assert_exact(traced["logits"][:, -1], native["logits"])
        saved = {name: value.clone() for name, value in traced.items()}
        traced["pre_damage"].fill_(123)
        self.assert_exact(traced["prefix_states"], saved["prefix_states"])
        self.assert_exact(traced["post_damage"], saved["post_damage"])
        traced["post_damage"].fill_(-123)
        self.assert_exact(traced["recovery_states"], saved["recovery_states"])
        traced["recovery_states"][:, 0].fill_(456)
        self.assert_exact(traced["recovery_states"][:, 1:], saved["recovery_states"][:, 1:])
        self.assert_exact(traced["logits"], saved["logits"])
        immediate = trace_repair(model, canvas, warmup_steps=2, recovery_steps=0,
                                 keep_mask=mask, recovery_canvas=canvas)
        self.assertEqual(tuple(immediate["recovery_states"].shape), (2, 1, 6, 3, 4))
        self.assert_exact(immediate["recovery_states"][:, 0], saved["post_damage"])
        self.assert_exact(immediate["logits"][:, 0], saved["logits"][:, 0])

    def test_recovery_does_not_reseed(self):
        model = self.constructed("holder")
        cue = torch.full((1, 3, 3), 2, dtype=torch.long)
        replacement = torch.full_like(cue, 3)
        calls = []
        handle = model.seed.register_forward_hook(lambda *args: calls.append(1))
        try:
            for keep_value in (True, False):
                with self.subTest(keep=keep_value):
                    start = len(calls)
                    result = trace_repair(model, cue, warmup_steps=2, recovery_steps=3,
                                          keep_mask=torch.full((1, 1, 3, 3), keep_value,
                                                               dtype=torch.bool),
                                          recovery_canvas=replacement)
                    self.assertEqual(len(calls) - start, 1)
                    expected = result["pre_damage"] if keep_value else torch.zeros_like(result["pre_damage"])
                    for time_index in range(4):
                        self.assert_exact(result["recovery_states"][:, time_index], expected)
        finally:
            handle.remove()

    def test_source_switch_and_complete_erasure_counterfactual(self):
        model = self.constructed("copier")
        cue = torch.full((1, 3, 3), 2, dtype=torch.long)
        replacement = torch.full_like(cue, 3)
        blank = torch.zeros_like(cue)
        mask = torch.zeros((1, 1, 3, 3), dtype=torch.bool)
        traces = {}
        for name, future in (("held", cue), ("removed", blank), ("changed", replacement)):
            traces[name] = trace_repair(model, cue, warmup_steps=1, recovery_steps=2,
                                        keep_mask=mask, recovery_canvas=future)
            self.assert_exact(traces[name]["post_damage"], torch.zeros((1, 4, 3, 3), dtype=torch.float64))
        self.assert_exact(traces["held"]["recovery_states"][:, 1], traces["held"]["pre_damage"])
        self.assertTrue(bool((traces["held"]["logits"][:, 1:].argmax(-1) == 2).all()))
        self.assertTrue(bool((traces["changed"]["logits"][:, 1:].argmax(-1) == 3).all()))
        self.assert_exact(traces["removed"]["recovery_states"],
                          torch.zeros_like(traces["removed"]["recovery_states"]))
        other_history = trace_repair(model, replacement, warmup_steps=1, recovery_steps=2,
                                     keep_mask=mask, recovery_canvas=blank)
        self.assertFalse(torch.equal(other_history["pre_damage"], traces["removed"]["pre_damage"]))
        self.assert_exact(other_history["recovery_states"], traces["removed"]["recovery_states"])
        self.assert_exact(other_history["logits"], traces["removed"]["logits"])

    def test_invalid_contracts(self):
        model, canvas = self.active_model(), self.canvas()
        valid = {"warmup_steps": 2, "recovery_steps": 3,
                 "keep_mask": torch.ones((2, 1, 3, 4), dtype=torch.bool),
                 "recovery_canvas": canvas}
        cases = [
            {"warmup_steps": 0}, {"warmup_steps": True}, {"warmup_steps": 1.5},
            {"recovery_steps": -1}, {"recovery_steps": False},
            {"keep_mask": torch.ones((2, 1, 3, 4), dtype=torch.float64)},
            {"keep_mask": torch.ones((2, 3, 4), dtype=torch.bool)},
            {"recovery_canvas": canvas[:, :, :3]},
            {"recovery_canvas": canvas.to(torch.int32)},
            {"recovery_canvas": torch.full_like(canvas, 9)},
        ]
        for index, changes in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                trace_repair(model, canvas, **{**valid, **changes})
        for invalid_canvas in (canvas.to(torch.int32), torch.full_like(canvas, -1),
                               canvas[:0]):
            with self.subTest(input_shape=tuple(invalid_canvas.shape),
                              dtype=str(invalid_canvas.dtype)), self.assertRaises(ValueError):
                trace_repair(model, invalid_canvas, **valid)
        model.train()
        with self.assertRaises(ValueError):
            trace_repair(model, canvas, **valid)
        model.eval()
        model.out_pos = (3, 0)
        with self.assertRaises(ValueError):
            trace_repair(model, canvas, **valid)
        with self.assertRaises(ValueError):
            trace_repair(object(), canvas, **valid)


if __name__ == "__main__":
    unittest.main()
