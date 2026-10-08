"""Numerical/mechanistic contracts for an untrained head; no optimizer/test data.

Resource admission occurs before importing Torch, including for this small
preflight. This never evaluates a scientific dataset or alters U16 checkpoints.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import psutil
    available = psutil.virtual_memory().available / 2**30
    dest = ROOT / 'results/research/query_attention_preflight'
    dest.mkdir(parents=True, exist_ok=True)
    if available < 8:
        receipt = dict(status='admission_refused_before_torch_import', available_ram_gib=available,
                       minimum_ram_gib=8, tests_run=0, new_trainings=0, model_evaluations=0)
        (dest / 'local_admission_refused.json').write_text(json.dumps(receipt, indent=2)+'\n',
                                                        encoding='utf-8', newline='\n')
        print(json.dumps(receipt))
        return 75
    import hashlib
    import math
    import unittest
    import torch
    from neuropixel.research.query_attention import QueryAttentionReadout, AttentiveResearchNCA
    from neuropixel.research.models import ResearchNCA
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    class Contracts(unittest.TestCase):
        def fixture(self, mode='query'):
            head = QueryAttentionReadout(c=2, c_id=2, d=2, heads=1, mode=mode).double()
            with torch.no_grad():
                for parameter in head.parameters():
                    parameter.zero_()
                head.query.weight.copy_(torch.eye(2))
                head.key.weight[:, 2:].copy_(torch.eye(2))
                head.value.weight[:, :2].copy_(torch.eye(2))
                head.output.weight.copy_(torch.eye(2))
            canvas = torch.zeros((1, 2, 3), dtype=torch.long)
            canvas[0, 0, 0], canvas[0, 0, 1], canvas[0, 1, 2] = 5, 6, 1
            state = torch.zeros((1, 2, 2, 3), dtype=torch.float64)
            state[0, :, 0, 0], state[0, :, 0, 1] = torch.tensor([3., 1.]), torch.tensor([1., 5.])
            ids = torch.zeros_like(state)
            ids[0, :, 0, 0], ids[0, :, 0, 1], ids[0, :, 1, 2] = (
                torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([1., 0.]))
            return head, state, ids, canvas

        def test_scaled_attention_matches_hand_calculation_and_query_flip(self):
            head, state, ids, canvas = self.fixture()
            a = math.exp(1/math.sqrt(2)) / (math.exp(1/math.sqrt(2))+1)
            output = head(state, ids, canvas, (1, 2))
            expected = torch.tensor([[3*a+1*(1-a), 1*a+5*(1-a)]], dtype=torch.float64)
            torch.testing.assert_close(output['delta'], expected, atol=1e-12, rtol=0)
            ids[0, :, 1, 2] = torch.tensor([0., 1.])
            flipped = head(state, ids, canvas, (1, 2))
            expected_flip = torch.tensor([[3*(1-a)+a, 1*(1-a)+5*a]], dtype=torch.float64)
            torch.testing.assert_close(flipped['delta'], expected_flip, atol=1e-12, rtol=0)
            self.assertFalse(torch.equal(output['weights'], flipped['weights']))

        def test_padding_and_query_are_not_keys_or_values(self):
            head, state, ids, canvas = self.fixture()
            before = head(state, ids, canvas, (1, 2))
            mask = ~before['valid_keys'].reshape(1, 2, 3)
            state = torch.where(mask[:, None], torch.full_like(state, 1e6), state)
            after = head(state, ids, canvas, (1, 2))
            torch.testing.assert_close(after['delta'], before['delta'], rtol=0, atol=0)
            self.assertEqual(float(after['weights'][..., ~before['valid_keys'][0]].abs().sum()), 0.)
            torch.testing.assert_close(after['weights'].sum(-1), torch.ones((1, 1), dtype=torch.float64))

        def test_empty_facts_have_zero_attention_and_zero_contribution(self):
            head, state, ids, canvas = self.fixture()
            canvas.zero_()
            canvas[0, 1, 2] = 1
            with torch.no_grad():
                head.output.bias.fill_(7.)
            output = head(state, ids, canvas, (1, 2))
            self.assertEqual(float(output['weights'].abs().sum()), 0.)
            self.assertEqual(float(output['delta'].abs().sum()), 0.)
            self.assertTrue(torch.isfinite(output['delta']).all())

        def test_uniform_control_has_same_nominal_parameters_and_ignores_query(self):
            head, state, ids, canvas = self.fixture('uniform')
            first = head(state, ids, canvas, (1, 2))
            torch.testing.assert_close(first['delta'], torch.tensor([[2., 3.]], dtype=torch.float64))
            ids[0, :, 1, 2] = torch.tensor([0., 1.])
            second = head(state, ids, canvas, (1, 2))
            torch.testing.assert_close(first['delta'], second['delta'], atol=0, rtol=0)
            other, *_ = self.fixture('query')
            self.assertEqual(sum(p.numel() for p in head.parameters()), sum(p.numel() for p in other.parameters()))

        def test_gradients_finite_and_head_inputs_are_not_mutated(self):
            head, state, ids, canvas = self.fixture()
            state.requires_grad_()
            ids.requires_grad_()
            copies = [x.detach().clone() for x in (state, ids, canvas)]
            head(state, ids, canvas, (1, 2))['delta'].square().sum().backward()
            for parameter in head.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            for original, copy in zip((state, ids, canvas), copies):
                self.assertTrue(torch.equal(original, copy))

        def test_hybrid_preserves_original_dynamics_and_shared_decoder(self):
            torch.manual_seed(15)
            hybrid = AttentiveResearchNCA(35, (7, 7), steps=2)
            original = ResearchNCA(35, (7, 7), steps=2)
            original.load_state_dict({k:v for k,v in hybrid.state_dict().items() if not k.startswith('attention_head.')})
            hybrid.eval()
            original.eval()
            canvas = torch.zeros((2, 8, 8), dtype=torch.long)
            canvas[:, 0, 0], canvas[:, 0, 1], canvas[:, 7, 6] = 1, 5, 1
            old, new = original(canvas, trace=True, lens_every=1), hybrid(canvas, trace=True, lens_every=1)
            for key in ('state', 'frames', 'lens'):
                torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
            torch.testing.assert_close(old['logits'], new['local_logits'], atol=0, rtol=0)
            expected = hybrid.read(new['readout_state']) @ hybrid.decoder_dictionary().T
            expected[:, 0] = -1e4
            torch.testing.assert_close(expected, new['logits'], atol=0, rtol=0)
            self.assertEqual(sum(p.numel() for p in hybrid.attention_head.parameters()), 3168)
            self.assertEqual(sum(p.numel() for p in hybrid.parameters()), 32992)

        def test_invalid_shapes_positions_and_nonfinite_representations_refused(self):
            head, state, ids, canvas = self.fixture()
            with self.assertRaises(ValueError):
                head(state, ids, canvas, (2, 2))
            with self.assertRaises(ValueError):
                head(state[:, :1], ids, canvas, (1, 2))
            state[0, 0, 0, 0] = math.nan
            with self.assertRaises(ValueError):
                head(state, ids, canvas, (1, 2))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors), available_ram_gib=available,
                   torch=torch.__version__, minimum_ram_gib=8, new_trainings=0, dataset_evaluations=0,
                   scope='Untrained analytic attention and hybrid-preservation contracts; not scientific competence',
                   source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (pathlib.Path(__file__), ROOT/'neuropixel/research/query_attention.py')})
    (dest/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
