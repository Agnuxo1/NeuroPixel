"""Execution admission guards without Torch or scientific fitting."""
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cli',ROOT/'scripts/research_DEV04.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
DEST=ROOT/'results/research/DEV04_cli_contracts';DEST.mkdir(parents=True,exist_ok=True)


class Contracts(unittest.TestCase):
    def test_missing_independent_hash_refuses_execution(self):
        with patch('sys.argv',['research_DEV04.py']):
            with self.assertRaisesRegex(ValueError,'Freeze independently'):m.main()

    def fixture(self,path):
        plan=dict(registration_id='NP-DEV04-20261008',status='frozen_before_scientific_training',runs=m.expected_inventory(),body_fits=12,head_fits=24,modes=['query_attention','local_query'],original_test_fixed_excluded=True,source_sha256={})
        path.write_text(json.dumps(plan),encoding='utf-8');return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_draft_or_partial_inventory_rejected(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            path=pathlib.Path(d)/'plan.json';self.fixture(path);plan=json.loads(path.read_text());plan['status']='draft';path.write_text(json.dumps(plan));h=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):m.load_plan(path,h)
            self.fixture(path);plan=json.loads(path.read_text());plan['runs'].pop();path.write_text(json.dumps(plan));h=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):m.load_plan(path,h)

    def test_wrong_hash_or_governing_source_rejected(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            path=pathlib.Path(d)/'plan.json';self.fixture(path)
            with self.assertRaises(ValueError):m.load_plan(path,'wrong')
            plan=json.loads(path.read_text());plan['source_sha256']={'scripts/research_DEV04.py':'changed'};path.write_text(json.dumps(plan));h=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):m.load_plan(path,h)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),scientific_body_fits=0,scientific_head_fits=0)
    (DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(receipt));raise SystemExit(0 if result.wasSuccessful() else 1)
