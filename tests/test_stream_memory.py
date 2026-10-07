"""Focused item-11 stream contracts; no learned-competence experiment.

All examples are the declared six-label synthetic cue/query fixture. These
tests exercise actual native/helper paths, state ownership and serialization.
They are authored for the frozen CPU job; local collection is not execution.
"""
from __future__ import annotations
import io
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from neuropixel.model import NeuroPixel
    from neuropixel.phase3 import FrameGRU, np_stream
    from neuropixel.research.stream_memory import strict_np_stream, strict_gru_stream


@unittest.skipIf(torch is None, "Torch is required for the declared cloud contracts")
class StreamMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)

    def setUp(self):
        torch.manual_seed(113001)
        self.np_model = NeuroPixel(8, (2, 2), fire_rate=0.5).cpu().eval()
        generator = torch.Generator(device="cpu").manual_seed(113002)
        with torch.no_grad():
            self.np_model.f2.weight.normal_(0.0, 0.01, generator=generator)
            self.np_model.f2.bias.normal_(0.0, 0.01, generator=generator)
        self.gru_model = FrameGRU(8, 3, 3, e=8, d=48, hid=70).cpu().eval()
        cue = torch.zeros(6, 3, 3, dtype=torch.long)
        cue[:, 0, 0] = torch.arange(2, 8)
        blank = torch.zeros_like(cue)
        query = torch.zeros_like(cue)
        query[:, 2, 1] = 1
        self.frames = [cue, blank, query]
        self.steps = [3, 3, 10]
        self.permutation = [1, 2, 3, 4, 5, 0]
        self.order = torch.tensor(self.permutation, dtype=torch.long)

    def close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_np_native_baseline_and_firing_stream_match(self):
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            native = np_stream(self.np_model, self.frames, self.steps)
            strict = strict_np_stream(self.np_model, self.frames, self.steps, capture_states=True)
        self.assertTrue(torch.equal(native["state"], strict["state"]))
        self.assertTrue(torch.equal(native["logits"], strict["logits"]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(tuple(strict["frame_states"].shape), (6, 3, 48, 3, 3))
        self.assertFalse(strict["frame_states"].requires_grad)
        self.assertTrue(torch.equal(strict["frame_states"][:, -1], strict["state"]))
        self.assertGreater(float(strict["state"].abs().sum()), 0)
        # Same global stream produces exactly the same stochastic update masks.
        self.np_model.train()
        torch.manual_seed(113003)
        initial_rng = torch.get_rng_state().clone()
        native = np_stream(self.np_model, self.frames, self.steps)
        end_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        strict = strict_np_stream(self.np_model, self.frames, self.steps)
        self.assertTrue(torch.equal(end_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(native["state"], strict["state"]))
        self.assertTrue(torch.equal(native["logits"], strict["logits"]))

    def test_np_chunk_continuation_retains_graph_and_caller_state(self):
        whole = strict_np_stream(self.np_model, self.frames, self.steps)
        prefix = strict_np_stream(self.np_model, self.frames[:-1], self.steps[:-1])
        saved = prefix["state"].detach().clone()
        prefix["state"].retain_grad()
        continued = strict_np_stream(self.np_model, self.frames[-1:], self.steps[-1:],
                                     initial_state=prefix["state"])
        self.assertTrue(torch.equal(whole["state"], continued["state"]))
        self.assertTrue(torch.equal(whole["logits"], continued["logits"]))
        self.assertTrue(torch.equal(saved, prefix["state"]))
        continued["logits"][:, 2:].square().mean().backward()
        self.assertIsNotNone(prefix["state"].grad)
        self.assertTrue(bool(torch.isfinite(prefix["state"].grad).all()))
        self.assertGreater(float(prefix["state"].grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(self.np_model.f2.weight.grad).all()))
        self.assertGreater(float(self.np_model.f2.weight.grad.abs().sum()), 0)

    def test_gru_native_and_chunk_paths_match_with_backward(self):
        native = self.gru_model.forward_frames(self.frames)
        whole = strict_gru_stream(self.gru_model, self.frames, capture_states=True)
        self.assertTrue(torch.equal(native["logits"], whole["logits"]))
        self.assertEqual(tuple(whole["state"].shape), (1, 6, 70))
        self.assertEqual(tuple(whole["frame_states"].shape), (6, 3, 70))
        self.assertFalse(whole["frame_states"].requires_grad)
        prefix = strict_gru_stream(self.gru_model, self.frames[:-1])
        saved = prefix["state"].detach().clone()
        prefix["state"].retain_grad()
        continued = strict_gru_stream(self.gru_model, self.frames[-1:],
                                      initial_state=prefix["state"])
        self.close(continued["state"], whole["state"])
        self.close(continued["logits"], whole["logits"])
        self.assertTrue(torch.equal(saved, prefix["state"]))
        continued["logits"][:, 2:].square().mean().backward()
        self.assertIsNotNone(prefix["state"].grad)
        self.assertTrue(bool(torch.isfinite(prefix["state"].grad).all()))
        self.assertGreater(float(prefix["state"].grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(self.gru_model.gru.weight_hh_l0.grad).all()))

    def test_zero_boundary_snapshots_and_query_only_reset(self):
        for kind, model, fn in (("np", self.np_model, strict_np_stream),
                                ("gru", self.gru_model, strict_gru_stream)):
            with self.subTest(family=kind), torch.no_grad():
                extra = {"steps": self.steps} if kind == "np" else {}
                prefix_extra = {"steps": self.steps[:-1]} if kind == "np" else {}
                query_extra = {"steps": self.steps[-1:]} if kind == "np" else {}
                prefix = fn(model, self.frames[:-1], **prefix_extra)
                zero = fn(model, self.frames, **extra, capture_states=True,
                          intervention={"before_frame": 2, "kind": "zero"})
                query = fn(model, self.frames[-1:], **query_extra)
                self.close(zero["intervention_before"], prefix["state"])
                self.assertEqual(torch.count_nonzero(zero["intervention_after"]).item(), 0)
                self.close(zero["state"], query["state"])
                self.close(zero["logits"], query["logits"])
                for key in ("intervention_before", "intervention_after"):
                    self.assertFalse(zero[key].requires_grad)
                before = zero["intervention_before"].clone()
                final = zero["state"].clone()
                zero["intervention_after"].fill_(123)
                self.assertTrue(torch.equal(before, zero["intervention_before"]))
                self.assertTrue(torch.equal(final, zero["state"]))

    def test_swap_donor_direction_and_snapshot_no_alias(self):
        for kind, model, fn, axis in (("np", self.np_model, strict_np_stream, 0),
                                     ("gru", self.gru_model, strict_gru_stream, 1)):
            with self.subTest(family=kind), torch.no_grad():
                prefix_extra = {"steps": self.steps[:-1]} if kind == "np" else {}
                query_extra = {"steps": self.steps[-1:]} if kind == "np" else {}
                full_extra = {"steps": self.steps} if kind == "np" else {}
                prefix = fn(model, self.frames[:-1], **prefix_extra)
                original = prefix["state"].clone()
                whole = fn(model, self.frames, **full_extra)
                result = fn(model, self.frames[-1:], **query_extra,
                            initial_state=prefix["state"], capture_states=True,
                            intervention={"before_frame": 0, "kind": "swap",
                                          "permutation": self.permutation})
                self.assertTrue(torch.equal(result["intervention_before"], original))
                self.assertTrue(torch.equal(result["intervention_after"], original.index_select(axis, self.order)))
                self.close(result["state"], whole["state"].index_select(axis, self.order))
                self.close(result["logits"], whole["logits"].index_select(0, self.order))
                final, frame = result["state"].clone(), result["frame_states"].clone()
                after = result["intervention_after"].clone()
                result["intervention_before"].fill_(321)
                self.assertTrue(torch.equal(result["intervention_after"], after))
                self.assertTrue(torch.equal(prefix["state"], original))
                result["intervention_after"].fill_(-321)
                self.assertTrue(torch.equal(prefix["state"], original))
                self.assertTrue(torch.equal(result["state"], final))
                self.assertTrue(torch.equal(result["frame_states"], frame))

    def test_invalid_frame_and_update_inventories_rejected(self):
        bad_frames = ([], [self.frames[0].float()], [self.frames[0], self.frames[0][:5]],
                      [torch.zeros(6, 0, 3, dtype=torch.long)],
                      [torch.full((6, 3, 3), 8, dtype=torch.long)],
                      [torch.full((6, 3, 3), -1, dtype=torch.long)])
        for index, frames in enumerate(bad_frames):
            with self.subTest(invalid_frames=index):
                with self.assertRaises(ValueError):
                    strict_np_stream(self.np_model, frames, [3] * len(frames))
                with self.assertRaises(ValueError):
                    strict_gru_stream(self.gru_model, frames)
        for steps in ([], [3], [3, 3], [3, 3, 0], [3, 3, -1], [3, 3, True],
                      [3, 3, 1.0], [3, 3, 10, 1]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                strict_np_stream(self.np_model, self.frames, steps)
        with self.assertRaises(ValueError):
            strict_gru_stream(self.gru_model, [torch.zeros(6, 2, 3, dtype=torch.long)])
        with self.assertRaises(ValueError):
            strict_np_stream(self.np_model, self.frames, self.steps, out_pos=(3, 2))

    def test_invalid_state_and_interventions_rejected(self):
        for state in (torch.zeros(6, 47, 3, 3), torch.zeros(6, 48, 3, 3, dtype=torch.float64),
                      torch.full((6, 48, 3, 3), float("nan"))):
            with self.subTest(np_state=tuple(state.shape), dtype=str(state.dtype)), self.assertRaises(ValueError):
                strict_np_stream(self.np_model, self.frames, self.steps, initial_state=state)
        for state in (torch.zeros(6, 70), torch.zeros(1, 6, 70, dtype=torch.float64),
                      torch.full((1, 6, 70), float("inf"))):
            with self.subTest(gru_state=tuple(state.shape), dtype=str(state.dtype)), self.assertRaises(ValueError):
                strict_gru_stream(self.gru_model, self.frames, initial_state=state)
        invalid = [
            {"before_frame": 0, "kind": "zero"},
            {"before_frame": True, "kind": "zero"},
            {"before_frame": 3, "kind": "zero"},
            {"before_frame": 2, "kind": "zero", "permutation": self.permutation},
            {"before_frame": 2, "kind": "other"},
            {"before_frame": 2, "kind": "swap", "permutation": [0, 0, 2, 3, 4, 5]},
            {"before_frame": 2, "kind": "swap", "permutation": [False, 1, 2, 3, 4, 5]},
        ]
        for value in invalid:
            with self.subTest(intervention=value):
                with self.assertRaises(ValueError):
                    strict_np_stream(self.np_model, self.frames, self.steps, intervention=value)
                with self.assertRaises(ValueError):
                    strict_gru_stream(self.gru_model, self.frames, intervention=value)

    def test_forward_reset_and_grounding_checkpoint_contract(self):
        with torch.no_grad():
            query = self.frames[-1]
            reference = self.np_model(query, steps=10)
            self.np_model(self.frames[0], steps=3)
            self.np_model(self.frames[1], steps=3)
            after = self.np_model(query, steps=10)
            self.assertTrue(torch.equal(reference["state"], after["state"]))
            self.assertTrue(torch.equal(reference["logits"], after["logits"]))
            rgb = torch.zeros(8, 3)
            mask = torch.zeros(8, dtype=torch.bool)
            rgb[2] = torch.tensor([0.2, 0.4, 0.6])
            mask[2] = True
            grounded = NeuroPixel(8, (2, 2), grounded=(rgb, mask)).eval()
            grounded.load_state_dict(self.np_model.state_dict())
            buffer = io.BytesIO()
            torch.save(grounded.state_dict(), buffer)
            buffer.seek(0)
            saved = torch.load(buffer, map_location="cpu", weights_only=True)
            self.assertNotIn("g_rgb", saved)
            self.assertNotIn("g_mask", saved)
            missing = NeuroPixel(8, (2, 2)).eval()
            missing.load_state_dict(saved, strict=True)
            complete = NeuroPixel(8, (2, 2), grounded=(rgb.clone(), mask.clone())).eval()
            complete.load_state_dict(saved, strict=True)
            self.assertFalse(torch.equal(grounded.dictionary(), missing.dictionary()))
            self.assertTrue(torch.equal(grounded.dictionary(), complete.dictionary()))
            expected, actual = grounded(query, steps=10), complete(query, steps=10)
            self.assertTrue(torch.equal(expected["state"], actual["state"]))
            self.assertTrue(torch.equal(expected["logits"], actual["logits"]))


if __name__ == "__main__":
    unittest.main()
