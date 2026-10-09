"""Fetch a connector-pinned commit into a bounded bare cache and verify Git Merkle bytes.

This reads public Git objects without anonymous REST quota; no checkout/reset or NN load.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time
from readout_snapshot import load_snapshot

ROOT=pathlib.Path(__file__).resolve().parents[1]
CACHE=ROOT.parent/'.cognition/neuropixel-public-evidence.git'
REPO='Agnuxo1/NeuroPixel'

def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--reference-proof',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    if len(a.commit)!=40 or any(x not in '0123456789abcdef' for x in a.commit):raise ValueError('Exact SHA1 commit required')
    reference=json.loads(a.reference_proof.read_bytes())
    if reference.get('repository')!=REPO or reference.get('commit_sha')!=a.commit or reference.get('source')!='authenticated_github_connector':raise ValueError('Commit must be pinned by exact authenticated connector reference')
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    common=['git','-c','safe.directory='+CACHE.as_posix(),'-c','pack.threads=2','-c','core.packedGitWindowSize=8m','-c','core.packedGitLimit=64m','-c','pack.deltaCacheSize=32m']
    if not (CACHE/'HEAD').exists():subprocess.run(common+['init','--bare',str(CACHE)],check=True,capture_output=True)
    git=common+['--git-dir='+str(CACHE)]
    start=time.monotonic();exists=subprocess.run(git+['cat-file','-e',a.commit+'^{commit}'],capture_output=True).returncode==0
    if not exists:
        log=a.output.with_suffix('.git-fetch.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('wb') as out:
            subprocess.run(git+['fetch','--no-tags','--depth=1','--no-write-fetch-head','https://github.com/'+REPO+'.git',a.commit],stdout=out,stderr=subprocess.STDOUT,check=True,timeout=180)
    raw=subprocess.check_output(git+['cat-file','commit',a.commit])
    actual=hashlib.sha1(b'commit '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if actual!=a.commit:raise ValueError('Raw commit SHA1 differs')
    tree_sha=raw.splitlines()[0].decode().split()[1]
    entries=[]
    for record in subprocess.check_output(git+['ls-tree','--full-tree','-r','-t','-z',a.commit]).split(b'\0'):
        if not record:continue
        header,name=record.split(b'\t',1);mode,kind,digest=header.decode('ascii').split()
        entries.append(dict(path=name.decode('utf-8'),mode=mode,type=kind,sha=digest))
    snapshot=dict(repository=REPO,commit=dict(sha=a.commit,tree=dict(sha=tree_sha)),tree=dict(sha=tree_sha,tree=entries,truncated=False),
        retrieval='Git transport into isolated bare object cache; authenticated connector reference pinned; raw commit SHA1 and all recursive tree Merkle hashes verified',
        reference_proof_sha256=hashlib.sha256(a.reference_proof.read_bytes()).hexdigest(),source_reference=reference['reference'])
    body=(json.dumps(snapshot,indent=2)+'\n').encode();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists() and a.output.read_bytes()!=body:raise ValueError('Preserve different pinned snapshot')
    if not a.output.exists():a.output.write_bytes(body)
    load_snapshot(a.output,REPO)
    print(json.dumps(dict(status='verified_connector_pinned_git_commit_and_recursive_tree',commit=a.commit,tree=tree_sha,entries=len(entries),cache_hit=exists,
        elapsed_seconds=time.monotonic()-start,output=str(a.output),checkout_modified=False,scientific_training_updates=0)))

if __name__=='__main__':main()
