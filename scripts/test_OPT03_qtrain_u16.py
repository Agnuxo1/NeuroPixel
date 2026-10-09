"""Contract fixtures only: compare new updater to immutable QTRAIN, no NP fitting."""
import copy
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = load('u16_contract_runner', 'scripts/research_OPT03_qtrain_u16.py')
folder = ROOT / 'results/research/OPT03_QTRAIN_U16_contracts'
folder.mkdir(parents=True, exist_ok=True)
try:
    runner.budget.require_ram()
except RuntimeError:
    (folder / 'receipt_local_admission_refused.json').write_text(json.dumps(
        {'status': 'RAM_admission_refused_before_Torch_and_fixtures', 'tests': 0,
         'scientific_updates': 0, 'limit_gib_unchanged': 8}, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(3)

import numpy as np
import torch
old = load('closed_qtrain_fixture_reference', 'scripts/research_OPT03_qtrain.py')
from neuropixel.research.data import ResearchRoleTask, frozen_dataset
from neuropixel.research.experiment import configure_runtime, environment_record
configure_runtime('cpu', 2)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e = torch.nn.Embedding(35, 4)
        self.h = torch.nn.Linear(4, 35)
        self.fire_rate = .5

    def forward(self, canvas, lens_every=0):
        hidden = self.e(canvas)
        if self.training:
            hidden = hidden * (torch.rand(canvas.shape)[..., None] > .5)
        logits = self.h(hidden)
        result = {'logits': logits.mean((1, 2))}
        if lens_every:
            result['lens'] = logits[:, None]
        return result


def actors():
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999),
                                 eps=1e-8, weight_decay=.0001)
    return model, optimizer, torch.Generator().manual_seed(17), torch.Generator().manual_seed(19)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=folder)
        self.root = pathlib.Path(self.temp.name)
        self.plan = self.root / 'reference_plan.json'
        self.plan.write_text(json.dumps({'status': 'frozen', 'runs': old.operations.inventory(),
            'source_sha256': {}, 'evaluation_mask_seeds': [79000, 79001]}), encoding='utf-8', newline='\n')
        self.hash = runner.sha(self.plan)

    def tearDown(self):
        self.temp.cleanup()

    def parent_fixture(self, conf=None):
        conf = conf or old.operations.inventory()[0]
        with patch('neuropixel.research.models.build_model', side_effect=lambda *a, **k: Tiny()):
            old.run_one(conf, self.root / 'parent', self.plan, self.hash, 128)
        return torch.load(self.root / 'parent/checkpoint.pt', weights_only=True)

    def test_update_matches_closed_reference_all_four_arms(self):
        for index, conf in enumerate(old.operations.inventory()[:4]):
            with patch('neuropixel.research.models.build_model', side_effect=lambda *a, **k: Tiny()):
                old.run_one(conf, self.root / f'full{index}', self.plan, self.hash, 256)
                old.run_one(conf, self.root / f'prefix{index}', self.plan, self.hash, 128)
            parent = torch.load(self.root / f'prefix{index}/checkpoint.pt', weights_only=True)
            expected = torch.load(self.root / f'full{index}/checkpoint.pt', weights_only=True)
            digest = runner.budget.validate_state(parent, 128, conf, self.hash)
            model, optimizer, sampler, query_rng = actors()
            runner.budget.restore(model, optimizer, sampler, query_rng, parent, digest)
            task = ResearchRoleTask(8, 8, seed=0)
            model.train()
            context = hashlib.sha256()
            counts = torch.zeros(4, dtype=torch.long)
            for _ in range(128):
                value = runner.budget.update_once(model, optimizer, task, sampler, query_rng,
                                                conf['policy'], conf['school_weight'])
                context.update(value['witness']['base_canvas'].numpy().tobytes())
                context.update(value['witness']['fillers'].numpy().tobytes())
                counts += torch.bincount(value['roles'], minlength=4)
            actual = runner.budget.training_state(model, optimizer, sampler, query_rng)
            self.assertEqual(runner.budget.state_digest(actual), runner.budget.state_digest(
                {key: expected[key] for key in runner.budget.STATE_FIELDS}))
            self.assertEqual(context.hexdigest(), expected['curve'][-1]['context_window_sha256'])
            self.assertEqual(counts.tolist(), [2048] * 4)

    def test_corrupt_counter_config_plan_or_values_refused(self):
        parent = self.parent_fixture()
        with self.assertRaisesRegex(ValueError, 'update'):
            runner.budget.validate_state(parent, 256)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            runner.budget.validate_state(parent, 128, {})
        with self.assertRaisesRegex(ValueError, 'plan'):
            runner.budget.validate_state(parent, 128, expected_plan='0' * 64)
        changed = copy.deepcopy(parent)
        next(iter(changed['optimizer']['state'].values()))['step'].fill_(127)
        with self.assertRaisesRegex(ValueError, 'counter'):
            runner.budget.validate_state(changed, 128)
        changed = copy.deepcopy(parent)
        next(iter(changed['model'].values())).view(-1)[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            runner.budget.validate_state(changed, 128)

    def test_each_rng_and_parameter_mutation_refused(self):
        parent = self.parent_fixture()
        digest = runner.budget.validate_state(parent, 128)
        for key in ('cpu_rng', 'sampler_rng', 'query_rng', 'model'):
            changed = copy.deepcopy(parent)
            if key == 'model':
                next(iter(changed[key].values())).view(-1)[0] += .01
            else:
                changed[key][0] ^= 1
            with self.assertRaisesRegex(ValueError, 'digest'):
                runner.budget.restore(*actors(), changed, digest)

    def test_state_digest_frames_nested_structure(self):
        self.assertNotEqual(runner.budget.state_digest([['a'], 'b']),
                            runner.budget.state_digest([['a', 'b']]))

    def test_low_ram_refused_and_environment_changes_limited(self):
        class Ram:
            available = int(7.9 * 2**30)
        with patch('psutil.virtual_memory', return_value=Ram()):
            with self.assertRaisesRegex(RuntimeError, 'RAM admission'):
                runner.budget.require_ram()
        environment = environment_record(torch.device('cpu'))
        changed = copy.deepcopy(environment)
        changed['threads'] = 4
        with self.assertRaisesRegex(ValueError, 'software'):
            runner.budget.verify_environment(environment, changed)
        changed = copy.deepcopy(environment)
        changed['platform'] = 'another standard CPU host/kernel'
        self.assertEqual(set(runner.budget.verify_environment(environment, changed)), {'platform'})

    def test_completed_extension_skipped_before_ram_or_model_load(self):
        target = self.root / 'closed'
        target.mkdir()
        (target / 'checkpoint.pt').write_bytes(b'closed fixture checkpoint')
        conf = {'run_id': 'fixture'}
        record = {'status': 'completed', 'config': conf, 'plan_sha256': self.hash,
                  'last_update': 16384, 'checkpoint_sha256': runner.sha(target / 'checkpoint.pt')}
        (target / 'result.json').write_text(json.dumps(record), encoding='utf-8')
        with patch.object(runner.budget, 'require_ram', side_effect=AssertionError('must skip closed')):
            self.assertEqual(runner.run_one(conf, {}, target, self.root, {}, {}, self.hash), record)

    def test_pending_endpoint_recovers_without_repeating_durable_prefix(self):
        parent = self.parent_fixture()
        model, optimizer, sampler, query_rng = actors()
        digest = runner.budget.validate_state(parent, 128)
        runner.budget.restore(model, optimizer, sampler, query_rng, parent, digest)
        for state in optimizer.state.values():
            state['step'].fill_(12288)
        state = runner.budget.training_state(model, optimizer, sampler, query_rng)
        environment = environment_record(torch.device('cpu'))
        conf = {'run_id': 'pending_fixture', 'parent_case': 'parent_fixture', 'init_seed': 200,
                'policy': 'paired', 'school_weight': 0., 'updates': 16384}
        descriptor = {'parent_case': 'parent_fixture', 'checkpoint_sha256': 'a' * 64,
                      'result_sha256': 'c' * 64, 'datasets': {}}
        parent_root = self.root / 'parents'
        (parent_root / 'datasets').mkdir(parents=True)
        task = ResearchRoleTask(8, 8, seed=0)
        for panel, split, count, seed in [('probe', 'train', 2048, 77001), ('validation', 'validation', 4096, 77002)]:
            values = frozen_dataset(task, split, count, seed)
            path = parent_root / 'datasets' / (panel + '.npz')
            np.savez_compressed(path, canvas=values[0].numpy(), target=values[1].numpy(), roles=values[2].numpy())
            descriptor['datasets'][panel] = {'file': path.name, 'sha256': runner.sha(path),
                                            'content_sha256': old.operations.tensor_digest(*values)}
        target = self.root / 'pending'
        target.mkdir()
        payload = {'config': conf, 'plan_sha256': self.hash, 'environment': environment,
                   'parent_checkpoint_sha256': descriptor['checkpoint_sha256'], 'update': 12288,
                   **state, 'training_state_digest': runner.budget.state_digest(state),
                   'curve': [{'update': u} for u in range(8320, 12289, 128)],
                   'evaluations': [], 'resources': [], 'elapsed_training_seconds': 0.}
        torch.save(payload, target / 'checkpoint.pt')
        admission = {'parent_state_digest': runner.budget.state_digest(state)}
        fake_parent = {'evaluations': []}
        calls = []

        def evaluate(*args, **kwargs):
            calls.append(args[3])
            return {'metrics': {'matched': {'joint_query_binding': 0.}}}
        with patch.object(runner, 'admit_parent', return_value=(fake_parent, (model, optimizer, sampler, query_rng), admission)), \
                patch('neuropixel.research.qtrain.evaluate_queries', side_effect=evaluate):
            result = runner.run_one(conf, descriptor, target, parent_root,
                {'endpoints': [12288, 16384], 'evaluation_mask_seeds': [1, 2]},
                {'parent_source_commit': 'fixture', 'parent_archive_git_commit': 'fixture'},
                self.hash, stop_after=12416)
        self.assertIsNone(result)
        restored = torch.load(target / 'checkpoint.pt', weights_only=True)
        self.assertEqual(restored['update'], 12416)
        self.assertEqual([x['update'] for x in restored['evaluations']], [12288])
        self.assertEqual(calls, ['u12288_probe', 'u12288_validation'])
        self.assertEqual(restored['curve'][-1]['update'], 12416)
        self.assertEqual(len(restored['curve']), 33)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    receipt = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors),
               'scope': 'Tiny operational fixtures, old/new updater equivalence; no scientific NP training',
               'scientific_updates': 0}
    (folder / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
