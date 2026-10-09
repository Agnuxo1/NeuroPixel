"""Recover two operational SCN06 archives; count one constructed experiment."""
import hashlib,io,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT.parent/'.cognition/neuropixel-public-evidence.git';GIT=['git','-c','safe.directory='+CACHE.as_posix(),'--git-dir='+str(CACHE)]
def main():
    snapshot=json.loads((ROOT/'coord/recovery/SCN06_evidence_snapshot.json').read_bytes());entries={x['path']:x for x in snapshot['tree']['tree'] if x['type']=='blob'};out=ROOT/'results/research/SCN06_review';out.mkdir(parents=True,exist_ok=True);rows=[]
    def get(name):
        e=entries[name];b=subprocess.check_output(GIT+['cat-file','blob',e['sha']]);assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==e['sha'];return b
    for run,source in [(37896280187,'d30b5bd2eab29e11dcfd7d63629646acf5e3c9cc'),(37896335193,'1e949f749d8718efaad57317577e2d9ff6228c60')]:
        prefix=f'results/research/SCN06_cloud/{run}/';raw=get(prefix+'raw.zip');receipt=json.loads(get(prefix+'receipt.json'));assert receipt['zip_sha256']==hashlib.sha256(raw).hexdigest() and receipt['zip_bytes']==len(raw) and receipt['source_commit']==source
        target=out/str(run);target.mkdir(exist_ok=True);(target/'original.zip').write_bytes(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            manifest=json.loads(z.read('archive_manifest.json'));assert manifest['source_commit']==source and manifest['run_id']==str(run)
            assert set(z.namelist())==set(manifest['files'])|{'archive_manifest.json'}
            for name,v in manifest['files'].items():
                p=target/name;p.resolve().relative_to(target.resolve());b=z.read(name);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        native=json.loads((target/'native/result.json').read_bytes());audit=json.loads((target/'native/independent_audit.json').read_bytes());assert native['training_updates']==0 and native['learned_forward_calls']==0 and all(x['passed'] for x in native['checks']) and not audit['issues'] and all(x['passed'] for x in audit['checks'])
        rows.append(dict(run=run,source=source,native_checks=len(native['checks']),arithmetic_checks=len(audit['checks']),npz_sha256=native['native_npz_sha256'],wall_seconds=native['wall_seconds'],cpu_seconds=native['process_cpu_seconds'],archive_sha256=receipt['zip_sha256']))
    assert rows[0]['npz_sha256']==rows[1]['npz_sha256']
    result=dict(status='verified_original_SCN06_native_archives',canonical_run=rows[0]['run'],constructed_experiments=1,redundant_operational_repeat_not_replication=True,rows=rows,archive_commit=snapshot['commit']['sha'],new_training_updates=0,learned_forward_calls=0)
    (out/'recovery_receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
