"""CPU sampler compatibility and closed final routes; requires Torch in CI."""
import unittest

import torch

from neuropixel.research.data import ResearchRoleTask, frozen_dataset
from neuropixel.research.development_data import DevelopmentRoleTask


class DevelopmentSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_train_and_validation_match_legacy_research_bytes_and_generator_consumption(self):
        underlying = ResearchRoleTask(8, 8, seed=101)
        governed = DevelopmentRoleTask(8, 8, seed=101)
        for split in ["train", "validation"]:
            with self.subTest(split=split):
                left = torch.Generator(device="cpu").manual_seed(62077)
                right = torch.Generator(device="cpu").manual_seed(62077)
                expected = underlying.sample(12, split, left, "cpu", query_role=2, meta=True)
                actual = governed.sample(12, split, right, "cpu", query_role=2, meta=True)
                self.assertTrue(torch.equal(expected[0], actual[0]))
                self.assertTrue(torch.equal(expected[1], actual[1]))
                self.assertEqual(set(expected[2]), set(actual[2]))
                for key in expected[2]:
                    self.assertTrue(torch.equal(expected[2][key], actual[2][key]))
                self.assertTrue(torch.equal(left.get_state(), right.get_state()))
                for x, y in zip(frozen_dataset(underlying, split, 16, 62078),
                                frozen_dataset(governed, split, 16, 62078)):
                    self.assertTrue(torch.equal(x, y))

    def test_final_names_and_typos_are_rejected_without_consuming_sampling_rng(self):
        task = DevelopmentRoleTask()
        for split in ["test", "final", "val", "Validation", "training", "", None]:
            with self.subTest(split=split):
                generator = torch.Generator(device="cpu").manual_seed(62079)
                before = generator.get_state().clone()
                with self.assertRaises(ValueError):
                    task.sample(4, split, generator)
                self.assertTrue(torch.equal(before, generator.get_state()))
                with self.assertRaises(ValueError):
                    frozen_dataset(task, split, 8, 62080)

    def test_inherited_loop_and_camera_routes_are_closed_for_every_partition(self):
        task = DevelopmentRoleTask()
        for name in ["sample_loop", "sample_camera"]:
            for split in ["train", "validation", "test", "final"]:
                with self.subTest(method=name, split=split):
                    generator = torch.Generator(device="cpu").manual_seed(62081)
                    before = generator.get_state().clone()
                    with self.assertRaises(ValueError):
                        getattr(task, name)(4, split, generator)
                    self.assertTrue(torch.equal(before, generator.get_state()))

    def test_adapter_preserves_explicit_cpu_generator_requirement(self):
        task = DevelopmentRoleTask()
        for split in ["train", "validation"]:
            with self.subTest(split=split):
                with self.assertRaises(ValueError):
                    task.sample(4, split, None)


if __name__ == "__main__":
    unittest.main()
