"""Read-only recovery of authenticated immutable archives; no NN load or training."""
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'coord/recovery/TASK4_cloud_audit'
CACHE=ROOT.parent/'.cognition/neuropixel-public-evidence.git'
GIT=['git','-c','safe.directory='+CACHE.as_posix(),'--git-dir='+str(CACHE)]
DEST=ROOT.parent/'NP-T4-20261009'
PREFIX={
'f0aa1f17e08a495e92e7e8084003f357abfe9a8d':'07_cloud_runs/37578991388-1',
'505a7098a077f90421e4a5ed0cd4c06882e98211':'08_cloud_runs/37584119610-1',
'8311c052796aeef5a52484b597cf5d9675ca315e':'09_cloud_runs/37589908205-1-preflight',
'b64d0e2ba222df76caf72bc9870c7602873e7236':'09_cloud_runs/37593731891-1-study',
'2b8203f15ec5f6fe190876c80bf29232034603ce':'09_offline_runs/37603398040-1-offline',
'0e135ad10518e1ccd9b7adc562265aa4e2d94d2b':'10_cloud_runs/37615277149-1-study',
'bb2323ac0e0ba028975a4a4c89d2a97a20c7ff6d':'10_cloud_runs/37618268225-1-audit',
'1b1759775adba1ab0d289ba34b6b306f9ad547c2':'11_cloud_runs/37626660573-1-preflight',
'0a8403b1001881acc8626369a1e4f9dde83cc8e2':'11_cloud_runs/37627489876-1-audit',
'ae82b5154c8d3d7da132c4b95ff05009be253437':'11_cloud_runs/37624600542-1-probe',
'5381f6775ae4be64210b99b14360fe06faea28ec':'12_cloud_runs/37633696345-1-probe',
'e16d0ab07a5aff93c66c4057a6509cc31b744160':'13_cloud_runs/37640756510-1-probe',
'10c694cd299a4114bd447528d22b204d89b00b9e':'14_cloud_runs/37644986366-1-probe',
'ff34cc4f44c535e5f4d00276e1e3be194ec70f21':'15_cloud_runs/37649654847-1-probe',
'205507b71e2f7abb00ce97aa1dbec8321fc0af26':'15_cloud_runs/37654405152-1-preflight'}
def main():
    inventory=[]
    for ref in sorted((AUDIT/'references').glob('*.json')):
        sha=ref.stem; snap=AUDIT/'snapshots'/(sha+'.json')
        subprocess.run([sys.executable,str(ROOT/'scripts/snapshot_DEV04_git.py'),'--commit',sha,'--reference-proof',str(ref),'--output',str(snap)],check=True,capture_output=True)
        tree=json.loads(snap.read_bytes())['tree']['tree']
        prefix='results/research/'+PREFIX[sha]+'/'
        entries=[x for x in tree if x['type']=='blob' and x['path'].startswith(prefix)]
        for e in entries:
            path=DEST/sha[:8]/e['path'][len(prefix):];path.parent.mkdir(parents=True,exist_ok=True)
            data=subprocess.check_output(GIT+['cat-file','blob',e['sha']])
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if blob!=e['sha']:raise ValueError('Git blob mismatch '+e['path'])
            if path.exists() and path.read_bytes()!=data:raise ValueError('Immutable recovery mismatch '+str(path))
            if not path.exists():path.write_bytes(data)
            inventory.append(dict(commit=sha,path=e['path'],local_path=str(path),git_blob=blob,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        print(json.dumps(dict(commit=sha,recovered_blobs=len(entries))),flush=True)
    out=ROOT/'results/research/TASK4_review';out.mkdir(parents=True,exist_ok=True)
    (out/'recovery_inventory.json').write_text(json.dumps(dict(status='git_commit_merkle_and_all_recovered_blob_bytes_verified',files=inventory,neural_loads=0,training_updates=0),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(inventory),bytes=sum(x['bytes'] for x in inventory))))
if __name__=='__main__':main()
