"""Bind the prepared pre-T0 publication packet to its immutable Git commit."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research/COMPLETION20261012'
PACKAGE='results/research/COMPLETION20261012/G1_owner_publication_package.json'
commit=subprocess.check_output(['git','log','-1','--format=%H','--',PACKAGE],cwd=ROOT,text=True).strip()
package=json.loads((ROOT/PACKAGE).read_text(encoding='utf-8'))
records=[]
with subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as proc:
    for item in package['files']:
        proc.stdin.write((commit+':'+item['path']+'\n').encode('utf-8'))
        proc.stdin.flush()
        header=proc.stdout.readline().decode('ascii').strip().split()
        assert len(header)==3 and header[1]=='blob',item['path']
        remaining=int(header[2])
        digest=hashlib.sha256()
        while remaining:
            data=proc.stdout.read(min(1<<20,remaining))
            assert data
            digest.update(data)
            remaining-=len(data)
        assert proc.stdout.read(1)==b'\n'
        assert digest.hexdigest()==item['sha256'],item['path']
        records.append({'path':item['path'],'git_blob':header[0],'sha256':digest.hexdigest()})
    proc.stdin.close()
    assert proc.wait()==0
locator={'status':'verified_immutable_pre_T0_publication_packet','package_path':PACKAGE,'publication_commit':commit,
         'package_sha256':hashlib.sha256((ROOT/PACKAGE).read_bytes()).hexdigest(),'verified_files':len(records),
         'external_operation_performed':False,'owner_action':'Review or export the packet at this immutable commit. Later branch HEAD contains a separate unvalidated T0 patch and is a different version.',
         'records':records,'source_version_policy':'Historical energy source identity stays at the frozen baseline. T0 runtime and GPU validation remain G2 gates.'}
path=OUT/'G1_owner_publication_locator.json'
assert not path.exists(),path
path.write_text(json.dumps(locator,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v for k,v in locator.items() if k!='records'},indent=2))
