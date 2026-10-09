"""Read-only generic immutable sprint archive recovery with exact source binding."""
import argparse,hashlib,io,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT.parent/'.cognition/neuropixel-public-evidence.git'
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--registration',required=True);p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sha=json.loads(a.reference.read_bytes())['commit_sha']
    subprocess.run(['python',str(ROOT/'scripts/snapshot_DEV04_git.py'),'--commit',sha,'--reference-proof',str(a.reference),'--output',str(a.snapshot)],check=True,capture_output=True)
    s=json.loads(a.snapshot.read_bytes());entries={x['path']:x for x in s['tree']['tree'] if x['type']=='blob'};g=['git','-c','safe.directory='+CACHE.as_posix(),'--git-dir='+str(CACHE)]
    def get(path):
        e=entries[path];body=subprocess.check_output(g+['cat-file','blob',e['sha']]);assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==e['sha'];return body
    prefix=f'results/research/{a.registration}_cloud/{a.run}/';receipt=json.loads(get(prefix+'receipt.json'));raw=get(prefix+'raw.zip');assert receipt['zip_sha256']==hashlib.sha256(raw).hexdigest() and receipt['zip_bytes']==len(raw) and receipt['source_commit']==a.source
    a.output.mkdir(parents=True,exist_ok=True);original=a.output/'original.zip'
    if original.exists():assert original.read_bytes()==raw
    else:original.write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if sum(i.file_size for i in z.infolist())>256*1024**2:raise ValueError('Unexpected expansion')
        m=json.loads(z.read('archive_manifest.json'));assert m['source_commit']==a.source and m['run_id']==a.run and m['registration']==a.registration
        assert set(z.namelist())==set(m['files'])|{'archive_manifest.json'}
        for name,v in m['files'].items():
            p=a.output/name;p.resolve().relative_to(a.output.resolve());b=z.read(name);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():assert p.read_bytes()==b
            else:p.write_bytes(b)
    r=dict(status='verified_commit_merkle_blob_zip_and_all_original_files',run=a.run,source=a.source,archive_commit=sha,archive_sha256=receipt['zip_sha256'],files=len(m['files']),registration=a.registration,new_training_updates=0)
    (a.output/'recovery_receipt.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
if __name__=='__main__':main()
