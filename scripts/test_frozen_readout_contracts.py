"""Synthetic state/cache/freeze contracts; no scientific dataset or parent fitting."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import psutil
    ram = psutil.virtual_memory().available / 2**30
    dest = ROOT/'results/research/READ03_contracts'
    dest.mkdir(parents=True, exist_ok=True)
    if ram < 8:
        row = dict(status='admission_refused_before_torch_import', available_ram_gib=ram,
                   required_ram_gib=8, tests=0, scientific_head_fits=0, new_backbone_updates=0)
        (dest/'local_admission_refused.json').write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps(row))
        return 75
    import hashlib
    import unittest
    import torch
    from neuropixel.research.models import ResearchNCA
    from neuropixel.research.frozen_readout import FrozenReadout, body_digest, pack_visible_state, unpack_visible_state
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    class Contracts(unittest.TestCase):
        def fixture(self):
            canvas = torch.zeros((2, 8, 8), dtype=torch.long)
            for r, c, role, value in [(0, 0, 1, 5), (2, 2, 2, 17), (4, 4, 3, 6), (6, 0, 4, 27)]:
                canvas[:, r, c], canvas[:, r, c+1] = role, value
            canvas[:, 7, 6] = 1
            torch.manual_seed(31)
            state = torch.randn((2, 48, 8, 8), dtype=torch.float64)
            return canvas, state

        def test_compact_state_preserves_all_head_logits_exactly(self):
            canvas, state = self.fixture()
            packed = pack_visible_state(state, canvas)
            for mode in ('query_attention', 'uniform_global', 'local_query'):
                body = ResearchNCA(35, (7, 7)).double()
                reader = FrozenReadout(body, mode).double()
                identities = torch.nn.functional.embedding(canvas, body.dictionary()).permute(0, 3, 1, 2)
                if mode == 'local_query':
                    value = reader.head(state[:, :, 7, 7], identities[:, :, 7, 6])
                    identity = body.read(state[:, :, 7, 7]+value['delta'])*value['identity_gate']
                else:
                    value = reader.head(state, identities, canvas, (7, 6))
                    identity = body.read(state[:, :, 7, 7]+value['delta'])
                expected = identity @ body.decoder_dictionary().T
                expected[:, 0] = -1e4
                torch.testing.assert_close(reader(packed), expected, atol=0, rtol=0)
                self.assertEqual(sum(p.numel() for p in reader.head_parameters()), 3168)

        def test_optimizer_changes_only_new_head_for_each_variant(self):
            canvas, state = self.fixture()
            packed = pack_visible_state(state, canvas)
            for mode in ('query_attention', 'uniform_global', 'local_query'):
                body = ResearchNCA(35, (7, 7)).double()
                reader = FrozenReadout(body, mode).double()
                before = body_digest(body)
                head_before = [p.detach().clone() for p in reader.head_parameters()]
                optimizer = torch.optim.AdamW(reader.head_parameters(), lr=.001)
                loss = torch.nn.functional.cross_entropy(reader(packed), torch.tensor([5, 6]))
                loss.backward()
                optimizer.step()
                reader.assert_frozen()
                self.assertEqual(body_digest(body), before)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in body.parameters()))
                self.assertTrue(any(not torch.equal(old, new) for old,new in zip(head_before, reader.head_parameters())))

        def test_corrupt_positions_or_missing_fact_rejected(self):
            canvas, state = self.fixture()
            packed = pack_visible_state(state, canvas)
            packed['positions'][0, 0] = 63
            with self.assertRaises(ValueError):
                unpack_visible_state(packed)
            canvas[0, 0, 1] = 0
            with self.assertRaises(ValueError):
                pack_visible_state(state, canvas)

        def test_parameter_or_gradient_contract_violation_rejected(self):
            canvas, state = self.fixture()
            body = ResearchNCA(35, (7, 7)).double()
            reader = FrozenReadout(body, 'query_attention').double()
            with torch.no_grad():
                body.read.bias.add_(1.)
            with self.assertRaises(ValueError):
                reader(pack_visible_state(state, canvas))
            body = ResearchNCA(35, (7, 7)).double()
            reader = FrozenReadout(body, 'query_attention').double()
            body.read.bias.requires_grad_(True)
            with self.assertRaises(ValueError):
                reader.assert_frozen()

        def test_cache_detaches_and_owns_storage(self):
            canvas, state = self.fixture()
            state.requires_grad_()
            packed = pack_visible_state(state, canvas)
            copy = packed['facts'].clone()
            with torch.no_grad():
                state.zero_()
            self.assertTrue(torch.equal(copy, packed['facts']))
            self.assertFalse(packed['facts'].requires_grad)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    inputs = [pathlib.Path(__file__), ROOT/'neuropixel/research/frozen_readout.py']
    row = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
               failures=len(result.failures), errors=len(result.errors), available_ram_gib=ram,
               required_ram_gib=8, scientific_head_fits=0, new_backbone_updates=0,
               scope='Synthetic compact-state equivalence and three fixture optimizer steps; not scientific head fitting',
               source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (dest/'receipt.json').write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(row))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
