"""Four focused storage-isolation contracts; no continual-learning benchmark."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("item12_isolation", ROOT / "scripts/research_continual_isolation.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU Torch is required by the frozen contract job")
class ContinualIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)

    def setUp(self):
        torch.manual_seed(120101)

    def test_fingerprint_covers_parameters_and_nonpersistent_buffers(self):
        # Generic buffer witness; it does not alter the model's grounding policy.
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.tensor([1.0, 2.0])))
        module.register_buffer("audit_buffer", torch.tensor([3.0]), persistent=False)
        original = probe.fingerprint(module)
        self.assertNotIn("audit_buffer", module.state_dict())
        self.assertIn("buffer:audit_buffer", probe.named_tensors(module))
        with torch.no_grad():
            module.weight.add_(1)
        self.assertNotEqual(probe.fingerprint(module), original)
        with torch.no_grad():
            module.weight.sub_(1)
        self.assertEqual(probe.fingerprint(module), original)
        module.audit_buffer.add_(1)
        self.assertNotEqual(probe.fingerprint(module), original)

    def test_deepcopy_preserves_values_but_has_disjoint_storage(self):
        parent = probe.make_model()
        child = copy.deepcopy(parent)
        self.assertEqual(probe.fingerprint(parent), probe.fingerprint(child))
        inventory = probe.bank_inventory([parent, child])
        first = {r["storage_id"] for r in inventory["storage_aliases"] if r["slot"] == 0}
        second = {r["storage_id"] for r in inventory["storage_aliases"] if r["slot"] == 1}
        self.assertTrue(first.isdisjoint(second))
        self.assertEqual(inventory["parameters_per_expert"], [29824, 29824])
        self.assertEqual(inventory["logical_tensor_payload_bytes"], 2 * probe.bank_inventory([parent])["logical_tensor_payload_bytes"])
        self.assertEqual(inventory["unique_storage_bytes"], inventory["logical_tensor_payload_bytes"])

    def test_one_child_update_preserves_all_parent_tensors_and_logits(self):
        parent = probe.make_model()
        child = copy.deepcopy(parent)
        canvas, target = probe.fixture(torch)
        self.assertTrue(torch.equal(target, torch.arange(5, 17)))
        self.assertTrue(torch.equal(canvas[:, 0, 0], target))
        self.assertTrue(torch.equal(canvas[:, 2, 1], torch.ones(12, dtype=torch.long)))
        before = [probe.fingerprint(m) for m in (parent, child)]
        logits = probe.bank_logits([parent, child], canvas, torch)
        losses = probe.update_fixture(child, canvas, target, torch, updates=1)
        after = probe.bank_logits([parent, child], canvas, torch)
        self.assertEqual(len(losses), 1)
        self.assertEqual(probe.fingerprint(parent), before[0])
        self.assertNotEqual(probe.fingerprint(child), before[1])
        self.assertTrue(torch.equal(logits[0], after[0]))
        self.assertFalse(torch.equal(logits[1], after[1]))
        self.assertFalse(parent.training)
        self.assertFalse(child.training)

    def test_shared_reference_negative_control_detects_old_slot_mutation(self):
        model = probe.make_model()
        bank = [model] * 2
        canvas, target = probe.fixture(torch)
        before = probe.bank_inventory(bank)
        logits = probe.bank_logits(bank, canvas, torch)
        probe.update_fixture(bank[1], canvas, target, torch, updates=1)
        after = probe.bank_inventory(bank)
        current = probe.bank_logits(bank, canvas, torch)
        self.assertIs(bank[0], bank[1])
        self.assertNotEqual(before["expert_fingerprints"][0], after["expert_fingerprints"][0])
        self.assertEqual(after["expert_fingerprints"][0], after["expert_fingerprints"][1])
        self.assertFalse(torch.equal(logits[0], current[0]))
        self.assertTrue(torch.equal(current[0], current[1]))
        self.assertEqual(after["unique_storages"] * 2, after["named_tensor_references"])
        self.assertEqual(after["unique_storage_bytes"] * 2, after["logical_tensor_payload_bytes"])


if __name__ == "__main__":
    unittest.main()
