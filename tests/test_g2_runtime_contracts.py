"""G2.2 runtime contracts of the patched NeuroPixel forward against the frozen original (CPU, Torch).

Run only after the 8 GiB admission (scripts/completion_g2_runtime.py). The frozen checkout at
D:/PROJECTS/NeuroPixel-Frozen-d8721a59 provides the original model source for parity.
"""
import importlib.util
import io
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path("D:/PROJECTS/NeuroPixel-Frozen-d8721a59/neuropixel/model.py")
sys.path.insert(0, str(ROOT))
from neuropixel.model import NeuroPixel  # noqa: E402
from neuropixel.task import NOUNS, PLACES, ROLES, VERBS, Vocab  # noqa: E402


def frozen_class():
    spec = importlib.util.spec_from_file_location("frozen_neuropixel_model", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NeuroPixel


def pair(**kwargs):
    torch.manual_seed(7)
    new = NeuroPixel(35, (7, 7), **kwargs)
    old = frozen_class()(35, (7, 7), **kwargs)
    old.load_state_dict(new.state_dict())
    with torch.no_grad():  # f2 starts at zero; give the update real weights so parity is not trivial
        for module in (new, old):
            torch.manual_seed(11)
            module.f2.weight.normal_(0, 0.05)
            module.f2.bias.normal_(0, 0.01)
    return new, old


def canvas(batch=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, 35, (batch, 8, 8), generator=g)
    c[:, 7, 7] = 1
    return c


class G2RuntimeContracts(unittest.TestCase):
    def test_positive_depth_parity_eval(self):
        new, old = pair()
        new.eval(), old.eval()
        x = canvas()
        with torch.inference_mode():
            for depth in (1, 2, 4, 8, 16, 32):
                a, b = new(x, steps=depth), old(x, steps=depth)
                torch.testing.assert_close(a["logits"], b["logits"], rtol=0, atol=0)
                torch.testing.assert_close(a["state"], b["state"], rtol=0, atol=0)
            torch.testing.assert_close(new(x)["logits"], old(x)["logits"], rtol=0, atol=0)

    def test_positive_depth_parity_training_masks_same_rng(self):
        new, old = pair()
        new.train(), old.train()
        x = canvas()
        for depth in (1, 5, 16):
            torch.manual_seed(1234)
            a = new(x, steps=depth)
            torch.manual_seed(1234)
            b = old(x, steps=depth)
            torch.testing.assert_close(a["logits"], b["logits"], rtol=0, atol=0)

    def test_zero_depth_contract_and_original_discrepancy(self):
        new, old = pair()
        new.eval(), old.eval()
        x = canvas()
        with torch.inference_mode():
            zero = new(x, steps=0)
            ids = torch.nn.functional.embedding(x, new.dictionary()).permute(0, 3, 1, 2)
            torch.testing.assert_close(zero["state"], new.seed(ids) * (x != 0).unsqueeze(1), rtol=0, atol=0)
            self.assertEqual(float(zero["activity"]), 0.0)
            # Documented original behaviour: steps=0 silently ran the default depth.
            torch.testing.assert_close(old(x, steps=0)["logits"], old(x)["logits"], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            new(x, steps=-1)

    def test_pad_class_and_full_logits(self):
        new, old = pair()
        new.eval()
        with torch.inference_mode():
            out = new(canvas())
        self.assertEqual(tuple(out["logits"].shape), (4, 35))
        self.assertTrue(torch.all(out["logits"][:, 0] == -1e4))
        self.assertTrue(torch.all(new.embed.weight[0] == 0))

    def test_class_indices(self):
        vocab = Vocab()
        self.assertEqual(len(vocab), 35)
        self.assertEqual(vocab.tokens[:5], ["<vacío>"] + ROLES)
        self.assertEqual(vocab.ids(NOUNS), list(range(5, 17)))
        self.assertEqual(vocab.ids(VERBS), list(range(17, 27)))
        self.assertEqual(vocab.ids(PLACES), list(range(27, 35)))

    def test_finiteness_long_depth(self):
        new, _ = pair()
        new.eval()
        with torch.inference_mode():
            for depth in (64, 128):
                out = new(canvas(), steps=depth)
                self.assertTrue(torch.isfinite(out["logits"]).all())

    def test_persistence_roundtrip(self):
        new, _ = pair()
        new.eval()
        buffer = io.BytesIO()
        torch.save(new.state_dict(), buffer)
        buffer.seek(0)
        torch.manual_seed(99)
        other = NeuroPixel(35, (7, 7))
        other.load_state_dict(torch.load(buffer, weights_only=True))
        other.eval()
        with torch.inference_mode():
            torch.testing.assert_close(new(canvas())["logits"], other(canvas())["logits"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
