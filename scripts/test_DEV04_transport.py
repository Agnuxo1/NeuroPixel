"""Archive identity, cache exclusion and stale/closed preservation fixtures."""
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive',ROOT/'neuropixel/research/dev04_transport.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
DEST=ROOT/'results/research/DEV04_transport_contracts';DEST.mkdir(parents=True,exist_ok=True)


class Contracts(unittest.TestCase):
    def fixture(self,d):
        root=pathlib.Path(d)/'source';folder=root/'fixture';(folder/'body').mkdir(parents=True)
        plan=pathlib.Path(d)/'plan.json';plan.write_text(json.dumps(dict(registration_id=m.REG,status='frozen_before_scientific_training',runs=[dict(run_id='fixture')]))+'\n')
        digest=hashlib.sha256(plan.read_bytes()).hexdigest()
        (folder/'body/progress.json').write_text(json.dumps(dict(update=128,plan_sha256=digest)))
        (folder/'body/checkpoint.pt').write_bytes(b'synthetic-not-neural-checkpoint')
        (folder/'cache').mkdir();(folder/'cache/ignored.pt').write_bytes(b'large-regenerable-latent-cache')
        return root,plan,digest

    def test_roundtrip_excludes_cache_and_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            root,plan,h=self.fixture(d);raw,a=m.make_archive(root,'fixture',plan);raw2,b=m.make_archive(root,'fixture',plan)
            self.assertEqual(raw,raw2);self.assertEqual(a,b)
            target=pathlib.Path(d)/'recovered';manifest=m.unpack(raw,target,'fixture',h)
            self.assertFalse((target/'fixture/cache').exists());self.assertEqual((target/'fixture/body/checkpoint.pt').read_bytes(),b'synthetic-not-neural-checkpoint')

    def test_different_closed_case_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            root,plan,h=self.fixture(d);(root/'fixture/result.json').write_text('closed fixture')
            raw,a=m.make_archive(root,'fixture',plan);target=pathlib.Path(d)/'recovered';m.unpack(raw,target,'fixture',h)
            (target/'fixture/body/checkpoint.pt').write_bytes(b'preserve-different')
            with self.assertRaises(ValueError):m.unpack(raw,target,'fixture',h)
            self.assertEqual((target/'fixture/body/checkpoint.pt').read_bytes(),b'preserve-different')

    def test_older_partial_never_overwrites_more_advanced_local_state(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            root,plan,h=self.fixture(d);raw,a=m.make_archive(root,'fixture',plan);target=pathlib.Path(d)/'recovered';m.unpack(raw,target,'fixture',h)
            (target/'fixture/body/progress.json').write_text(json.dumps(dict(update=256,plan_sha256=h)))
            (target/'fixture/body/checkpoint.pt').write_bytes(b'more-advanced')
            m.unpack(raw,target,'fixture',h);self.assertEqual((target/'fixture/body/checkpoint.pt').read_bytes(),b'more-advanced')

    def test_wrong_case_or_plan_rejected(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            root,plan,h=self.fixture(d);raw,a=m.make_archive(root,'fixture',plan)
            with self.assertRaises(ValueError):m.unpack(raw,pathlib.Path(d)/'recovered','other',h)
            with self.assertRaises(ValueError):m.unpack(raw,pathlib.Path(d)/'recovered','fixture','wrong-plan')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),scientific_body_fits=0,scientific_head_fits=0)
    (DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)
