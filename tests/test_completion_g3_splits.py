"""Contract tests for the G3 family-grouped splits and DEV selector."""
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import completion_g3_splits as g3  # noqa: E402
from completion_kit.datasets import encode  # noqa: E402


def brute_isomorphic(a: np.ndarray, b: np.ndarray) -> bool:
    relations, entities = a.shape
    for rel in itertools.permutations(range(relations)):
        for phi in itertools.permutations(range(entities)):
            phi = np.array(phi)
            if np.array_equal(g3.rename(a, phi, np.array(rel)), b):
                return True
    return False


class G3Contracts(unittest.TestCase):
    def test_family_invariant_under_renaming(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            maps = np.stack([rng.permutation(12) for _ in range(3)])
            variant = g3.rename(maps, rng.permutation(12), rng.permutation(3))
            self.assertEqual(g3.family_id(maps), g3.family_id(variant))

    def test_family_is_complete_invariant_on_small_graphs(self):
        rng = np.random.default_rng(2)
        for _ in range(120):
            a = np.stack([rng.permutation(4) for _ in range(2)])
            b = np.stack([rng.permutation(4) for _ in range(2)])
            self.assertEqual(g3.family_id(a) == g3.family_id(b), brute_isomorphic(a, b))

    def test_distinct_structures_differ(self):
        identity = np.stack([np.arange(12)] * 3)
        cycle = np.stack([np.roll(np.arange(12), 1), np.arange(12), np.arange(12)])
        self.assertNotEqual(g3.family_id(identity), g3.family_id(cycle))

    def test_rejects_non_permutation(self):
        with self.assertRaises(ValueError):
            g3.entity_canonical(np.array([[0, 0, 1], [0, 1, 2]]))

    def test_selector_order(self):
        records = [dict(epoch=5, dev_correct=290, dev_nll=0.2), dict(epoch=3, dev_correct=291, dev_nll=0.3),
                   dict(epoch=9, dev_correct=291, dev_nll=0.25), dict(epoch=2, dev_correct=291, dev_nll=0.25)]
        self.assertEqual(g3.select_checkpoint(records)["epoch"], 2)
        with self.assertRaises(ValueError):
            g3.select_checkpoint([])

    def test_maps_recovered_from_input_alone(self):
        rng = np.random.default_rng(3)
        maps = np.stack([rng.permutation(12) for _ in range(3)])
        field = encode(maps, 4, [0, 2, 1], rng)
        self.assertTrue(np.array_equal(g3._maps_from_field(field), maps))

    def test_small_dataset_has_no_family_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ds"
            doc = g3.make_family_dataset(out, seed=7, families={"train": 6, "dev": 3}, variants=2, examples_per_variant=2)
            self.assertEqual(doc["excluded_families"]["count"], 9)
            audit = g3.audit_family_overlap({n: out / (n + ".npz") for n in ("train", "dev", "dev_long")})
            self.assertEqual(audit["overlaps"]["dev|train"], 0)
            self.assertEqual(audit["overlaps"]["dev_long|train"], 0)
            with self.assertRaises(FileExistsError):
                g3.make_family_dataset(out, seed=7, families={"train": 1, "dev": 1}, variants=1, examples_per_variant=1)


if __name__ == "__main__":
    unittest.main()
