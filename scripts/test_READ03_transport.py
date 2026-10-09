"""Bounded immutable readout archives; standard-library fixtures, no Torch."""
import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('readout_archive',ROOT/'neuropixel/research/readout_transport.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
DEST=ROOT/'results/research/READ03_transport_contracts';DEST.mkdir(parents=True,exist_ok=True)


class Contracts(unittest.TestCase):
    def fixture(self,root,complete=True):
        folder=root/'fixture_parent'/'query_attention';folder.mkdir(parents=True)
        (folder/'checkpoint.pt').write_bytes(b'synthetic-checkpoint-bytes')
        plan=root.parent/'plan.json';plan.write_bytes(b'{"scope":"synthetic"}\n')
        record={'config':{'updates':8192},'update':1024,'plan_sha256':m.sha(plan.read_bytes())}
        (folder/('result.json' if complete else 'progress.json')).write_text(json.dumps(record))
        captured={}
        with patch.dict(os.environ,{'GITHUB_SHA':'fixture_source','GITHUB_RUN_ID':'0'}),patch.object(m.old,'publish_immutable',side_effect=lambda path,raw:captured.update({path:raw})):
            m.publish_fit(root,'fixture_parent','query_attention',plan)
        raw=next(v for k,v in captured.items() if k.endswith('/raw.zip'))
        return raw,plan,folder

    def test_deterministic_archive_and_exact_manifest_unpack(self):
        with tempfile.TemporaryDirectory(dir=DEST) as tmp:
            root=pathlib.Path(tmp)/'source';raw,plan,_=self.fixture(root)
            other=pathlib.Path(tmp)/'recovery';manifest=m.unpack(raw,other,'fixture_parent__query_attention',m.sha(plan.read_bytes()))
            self.assertEqual(manifest['last_update'],8192)
            self.assertEqual((other/'fixture_parent/query_attention/checkpoint.pt').read_bytes(),b'synthetic-checkpoint-bytes')
            m.unpack(raw,other,'fixture_parent__query_attention',m.sha(plan.read_bytes()))

    def test_different_closed_local_fit_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=DEST) as tmp:
            root=pathlib.Path(tmp)/'source';raw,plan,_=self.fixture(root)
            other=pathlib.Path(tmp)/'recovery';m.unpack(raw,other,'fixture_parent__query_attention',m.sha(plan.read_bytes()))
            cp=other/'fixture_parent/query_attention/checkpoint.pt';cp.write_bytes(b'preserve-local-changes')
            with self.assertRaises(ValueError):m.unpack(raw,other,'fixture_parent__query_attention',m.sha(plan.read_bytes()))
            self.assertEqual(cp.read_bytes(),b'preserve-local-changes')

    def test_older_partial_does_not_replace_more_advanced_local_state(self):
        with tempfile.TemporaryDirectory(dir=DEST) as tmp:
            root=pathlib.Path(tmp)/'source';raw,plan,_=self.fixture(root,False)
            other=pathlib.Path(tmp)/'recovery';m.unpack(raw,other,'fixture_parent__query_attention',m.sha(plan.read_bytes()))
            folder=other/'fixture_parent/query_attention';p=folder/'progress.json';d=json.loads(p.read_text());d['update']=2048;p.write_text(json.dumps(d));(folder/'checkpoint.pt').write_bytes(b'later-local-state')
            m.unpack(raw,other,'fixture_parent__query_attention',m.sha(plan.read_bytes()))
            self.assertEqual((folder/'checkpoint.pt').read_bytes(),b'later-local-state')

    def test_case_and_plan_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir=DEST) as tmp:
            root=pathlib.Path(tmp)/'source';raw,plan,_=self.fixture(root)
            with self.assertRaises(ValueError):m.unpack(raw,pathlib.Path(tmp)/'bad','another_case',m.sha(plan.read_bytes()))
            with self.assertRaises(ValueError):m.unpack(raw,pathlib.Path(tmp)/'bad','fixture_parent__query_attention','wrong_plan')


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    sources=[pathlib.Path(__file__),ROOT/'neuropixel/research/readout_transport.py']
    d=dict(status='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),
           scope='Only synthetic bytes and local immutable/archive recovery fixtures, no neural fitting',scientific_head_fits=0,
           source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (DEST/'receipt.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(d))
    raise SystemExit(0 if r.wasSuccessful() else 1)
