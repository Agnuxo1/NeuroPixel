"""Run in the real Torch environment AFTER the scoped forward correction."""
import unittest
import torch
from neuropixel.model import NeuroPixel


class OriginalForwardContracts(unittest.TestCase):
    def test_zero_depth_returns_seed_and_one_frame(self):
        model = NeuroPixel(5, (1, 1), c_id=4, c=4, hidden=8, steps=4, fire_rate=1)
        model.eval()
        canvas = torch.ones((2, 3, 3), dtype=torch.long)
        with torch.inference_mode():
            result = model(canvas, steps=0, trace=True)
            ids = torch.nn.functional.embedding(canvas, model.dictionary()).permute(0, 3, 1, 2)
            expected = model.seed(ids)
        torch.testing.assert_close(result["state"], expected, rtol=0, atol=0)
        self.assertEqual(result["frames"].shape[1], 1)
        self.assertEqual(float(result["activity"]), 0.0)

    def test_explicit_depth_and_default_have_equal_outputs(self):
        model = NeuroPixel(5, (1, 1), c_id=4, c=4, hidden=8, steps=4, fire_rate=1)
        model.eval()
        canvas = torch.ones((2, 3, 3), dtype=torch.long)
        with torch.inference_mode():
            first = model(canvas)
            second = model(canvas, steps=4)
        torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)

    def test_invalid_depths_are_rejected(self):
        model = NeuroPixel(5, (1, 1), c_id=4, c=4, hidden=8)
        for depth in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                model(torch.ones((1, 3, 3), dtype=torch.long), steps=depth)


if __name__ == "__main__":
    unittest.main(verbosity=2)
