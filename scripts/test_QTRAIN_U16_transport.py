"""Stdlib archive/recovery fixtures; no Torch, network or scientific training."""
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('transport_contracts', ROOT / 'neuropixel/research/qtrain_budget_transport.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def archive(case='fixture', update=10000, status='partial_not_completed', bad_path=False):
    body = b'fixture checkpoint bytes'
    files = {case + '/checkpoint.pt': body}
    if bad_path:
        files['../escape.txt'] = b'escape must be refused'
    manifest = {'registration_id': 'fixture-study', 'plan_sha256': 'p' * 64,
                'case_run_id': case, 'case_status': status, 'last_update': update,
                'source_commit': 'fixture-source', 'workflow_run_id': 'fixture-run',
                'files': {name: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
                          for name, value in files.items()}}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as output:
        for name, value in files.items():
            output.writestr(name, value)
        output.writestr('archive_manifest.json', json.dumps(manifest))
    return buffer.getvalue()


class Transport(unittest.TestCase):
    def setUp(self):
        folder = ROOT / 'results/research/OPT03_QTRAIN_U16_transport_contracts'
        folder.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=folder)
        self.root = pathlib.Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_extract_and_changed_existing_refused(self):
        path = self.root / 'source.zip'
        path.write_bytes(archive())
        transport.unpack(path, self.root / 'cohort', 'fixture')
        transport.unpack(path, self.root / 'cohort', 'fixture')
        self.assertEqual((self.root / 'cohort/fixture/checkpoint.pt').read_bytes(), b'fixture checkpoint bytes')
        with self.assertRaisesRegex(ValueError, 'Preserve'):
            transport.immutable(self.root / 'cohort/fixture/checkpoint.pt', b'changed')

    def test_traversal_and_zip_symlink_refused(self):
        path = self.root / 'bad.zip'
        path.write_bytes(archive(bad_path=True))
        with self.assertRaises(ValueError):
            transport.unpack(path, self.root / 'cohort', 'fixture')
        self.assertFalse((self.root / 'escape.txt').exists())
        with zipfile.ZipFile(path, 'w') as output:
            item = zipfile.ZipInfo('link')
            item.external_attr = (0o120777 << 16)
            output.writestr(item, 'outside')
            output.writestr('archive_manifest.json', json.dumps({'files': {}}))
        with self.assertRaisesRegex(ValueError, 'symlink'):
            transport.unpack(path, self.root / 'cohort', 'fixture')

    def test_wrong_parent_zip_hash_refused(self):
        descriptor = {'parent_case': 'fixture', 'remote_zip_path': 'fixture.zip',
                      'zip_bytes': 10, 'zip_sha256': '0' * 64}
        with patch.object(transport, 'fetch', return_value=b'bad archive'):
            with self.assertRaisesRegex(ValueError, 'ZIP differs'):
                transport.recover_parent({'parent_archive_git_commit': 'fixture'}, descriptor, self.root / 'parents')

    def test_closed_and_latest_partial_recovery_priority(self):
        prefix = 'results/research/OPT03_QTRAIN_U16_cloud/fixture-study/'
        records = {}
        for directory, update, status in [('fixture_partial_u9000', 9000, 'partial_not_completed'),
                                         ('fixture_partial_u10000', 10000, 'partial_not_completed'),
                                         ('fixture', 16384, 'completed')]:
            blob = archive(update=update, status=status)
            receipt = {'registration_id': 'fixture-study', 'plan_sha256': 'p' * 64,
                       'case_run_id': 'fixture', 'case_status': status, 'last_update': update,
                       'zip_bytes': len(blob), 'zip_sha256': hashlib.sha256(blob).hexdigest()}
            records[prefix + directory + '/receipt.json'] = (json.dumps(receipt).encode(), blob)

        def exercise(allow_complete):
            selected = {name: pair for name, pair in records.items()
                        if allow_complete or '/fixture/' not in name}
            tree = {'tree': [{'path': name, 'type': 'blob',
                             'sha': hashlib.sha1(b'blob ' + str(len(pair[0])).encode() + b'\0' + pair[0]).hexdigest()}
                            for name, pair in selected.items()], 'truncated': False}

            def fetch(url):
                if '/git/ref/' in url:
                    return json.dumps({'object': {'sha': 'fixture-commit'}}).encode()
                if '/git/trees/' in url:
                    return json.dumps(tree).encode()
                for name, pair in selected.items():
                    if url.endswith(name):
                        return pair[0]
                    if url.endswith(name.rsplit('/', 1)[0] + '/raw.zip'):
                        return pair[1]
                raise AssertionError('Unexpected network request')
            with patch.dict(os.environ, {'GITHUB_REPOSITORY': 'fixture/repo', 'GITHUB_REF_NAME': 'fixture'}), \
                    patch.object(transport, 'fetch', side_effect=fetch):
                return transport.recover_existing(self.root / ('complete' if allow_complete else 'partial'),
                                                  {'fixture'}, 'fixture-study', 'p' * 64)
        self.assertEqual(exercise(False)[0]['update'], 10000)
        result = exercise(True)[0]
        self.assertEqual(result['update'], 16384)
        self.assertEqual(result['status'], 'completed')

    def test_changed_published_evidence_never_overwritten(self):
        with patch.dict(os.environ, {'GITHUB_REPOSITORY': 'fixture/repo', 'GITHUB_REF_NAME': 'fixture',
                                     'QTRAIN_U16_ARCHIVE_TOKEN': 'fixture-not-a-secret'}), \
                patch.object(transport, 'fetch', return_value=json.dumps({'sha': 'wrong'}).encode()):
            with self.assertRaisesRegex(ValueError, 'Preserve'):
                transport.publish_immutable('fixture.json', b'expected')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Transport))
    path = ROOT / 'results/research/OPT03_QTRAIN_U16_transport_contracts/receipt.json'
    path.write_text(json.dumps({'status': 'passed' if result.wasSuccessful() else 'failed',
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'scope': 'Constructed immutable Git/archive fixtures; no network or model training'}, indent=2)
        + '\n', encoding='utf-8', newline='\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
