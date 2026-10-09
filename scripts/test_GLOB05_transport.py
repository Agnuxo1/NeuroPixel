"""Bounded archive mutation tests; no Torch, neural loading or scientific fits."""
import hashlib,io,json,pathlib,tempfile,unittest,zipfile,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neuropixel.research import glob05_transport as m
DEST=ROOT/'results/research/GLOB05_transport_fixtures';DEST.mkdir(parents=True,exist_ok=True)

class Contracts(unittest.TestCase):
    def fixture(self,d,closed=False):
        root=pathlib.Path(d)/'source';folder=root/'fixture';folder.mkdir(parents=True)
        plan=pathlib.Path(d)/'plan.json';plan.write_text(json.dumps(dict(registration_id=m.REG,status='frozen_before_scientific_training'))+'\n',encoding='utf-8',newline='\n');h=hashlib.sha256(plan.read_bytes()).hexdigest()
        record=dict(registration_id=m.REG,config=dict(parent_case='fixture',mode='gated_uniform_global',head_seed=700,updates=256),plan_sha256=h,parent=dict(fixture=True),update=128)
        (folder/('result.json' if closed else 'progress.json')).write_text(json.dumps(record)+'\n',encoding='utf-8',newline='\n');(folder/'checkpoint.pt').write_bytes(b'not-neural-fixture');(folder/'cache').mkdir();(folder/'cache/ignored.pt').write_bytes(b'cache')
        return root,plan,h

    def test_roundtrip_deterministic_and_excludes_cache(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            root,p,h=self.fixture(d);raw,r=m.make_archive(root,'fixture',p);self.assertEqual(m.make_archive(root,'fixture',p)[0],raw)
            target=pathlib.Path(d)/'out';m.unpack(raw,target,'fixture',h);self.assertFalse((target/'fixture/cache').exists());self.assertEqual((target/'fixture/checkpoint.pt').read_bytes(),b'not-neural-fixture')

    def test_closed_and_newer_partial_preserved(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            root,p,h=self.fixture(d,True);raw,r=m.make_archive(root,'fixture',p);target=pathlib.Path(d)/'out';m.unpack(raw,target,'fixture',h);(target/'fixture/checkpoint.pt').write_bytes(b'preserve')
            with self.assertRaises(ValueError):m.unpack(raw,target,'fixture',h)
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            root,p,h=self.fixture(d);raw,r=m.make_archive(root,'fixture',p);target=pathlib.Path(d)/'out';m.unpack(raw,target,'fixture',h);(target/'fixture/progress.json').write_text(json.dumps(dict(plan_sha256=h,update=256)));(target/'fixture/checkpoint.pt').write_bytes(b'newer');m.unpack(raw,target,'fixture',h);self.assertEqual((target/'fixture/checkpoint.pt').read_bytes(),b'newer')

    def test_wrong_identity_and_tampered_blob_rejected(self):
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            root,p,h=self.fixture(d);raw,r=m.make_archive(root,'fixture',p)
            with self.assertRaises(ValueError):m.unpack(raw,pathlib.Path(d)/'out','foreign',h)
            with self.assertRaises(ValueError):m.unpack(raw,pathlib.Path(d)/'out','fixture','bad')
            with zipfile.ZipFile(io.BytesIO(raw)) as z:entries={n:z.read(n) for n in z.namelist()}
            entries['fixture/checkpoint.pt']=b'tampered';buf=io.BytesIO()
            with zipfile.ZipFile(buf,'w') as z:
                for n,b in entries.items():z.writestr(n,b)
            with self.assertRaises(ValueError):m.unpack(buf.getvalue(),pathlib.Path(d)/'out','fixture',h)

    def test_path_escape_and_symlink_rejected_before_writes(self):
        for unsafe in ('fixture/../escape','fixture/file:stream','fixture/\\escape'):
            with tempfile.TemporaryDirectory(dir=DEST) as d:
                root,p,h=self.fixture(d);raw,r=m.make_archive(root,'fixture',p)
                with zipfile.ZipFile(io.BytesIO(raw)) as z:entries={n:z.read(n) for n in z.namelist()}
                manifest=json.loads(entries['archive_manifest.json']);body=entries.pop('fixture/checkpoint.pt');manifest['files'].pop('fixture/checkpoint.pt');manifest['files'][unsafe]=hashlib.sha256(body).hexdigest();entries[unsafe]=body;entries['archive_manifest.json']=(json.dumps(manifest)+'\n').encode();buf=io.BytesIO()
                with zipfile.ZipFile(buf,'w') as z:
                    for n,b in entries.items():z.writestr(n,b)
                with self.assertRaises(ValueError):m.unpack(buf.getvalue(),pathlib.Path(d)/'out','fixture',h)
        with tempfile.TemporaryDirectory(dir=DEST) as d:
            root,p,h=self.fixture(d);raw,r=m.make_archive(root,'fixture',p)
            with zipfile.ZipFile(io.BytesIO(raw)) as z:entries={n:z.read(n) for n in z.namelist()}
            buf=io.BytesIO()
            with zipfile.ZipFile(buf,'w') as z:
                for n,b in entries.items():
                    info=zipfile.ZipInfo(n)
                    if n=='fixture/checkpoint.pt':info.external_attr=0o120777<<16
                    z.writestr(info,b)
            with self.assertRaises(ValueError):m.unpack(buf.getvalue(),pathlib.Path(d)/'out','fixture',h)

if __name__=='__main__':unittest.main()
