"""Pin read-only receipts to immutable Git blobs without loading NN weights."""
import datetime
import argparse
import hashlib
import json
import pathlib
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO='Agnuxo1/NeuroPixel'
BRANCH='research/frozen-readout-replay-2026-10-08'
SCIENCE_RUN=37831051538
SCIENCE_SOURCE='a19c0000b4175984c14cd4eaf08392ceffd9e917'
VERIFICATION_RUN=37834470656
VERIFICATION_SOURCE='ce0bdc7ab3080fe851cfe0134b7a68ffc4ec7b98'
PLAN='d273b8284d9a387b49266dc4ec62cfad76aefaa11c5f82c7ab6a97ba0e431ea0'


def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'NeuroPixel-readonly-evidence-collector','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=30) as r:return r.read()


def immutable(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=raw:raise ValueError('Preserve different receipt: '+str(path))
    if not path.exists():path.write_bytes(raw)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tree-snapshot',type=pathlib.Path);args=parser.parse_args()
    api=f'https://api.github.com/repos/{REPO}/'
    if args.tree_snapshot:
        from readout_snapshot import load_snapshot
        snapshot=load_snapshot(args.tree_snapshot,REPO)
        commit=snapshot['commit']['sha'];tree=snapshot['tree']
        if snapshot['repository']!=REPO or snapshot['commit']['tree']['sha']!=tree['sha']:raise ValueError('Snapshot commit/tree binding differs')
    else:
        commit=json.loads(get(api+'git/ref/heads/'+BRANCH))['object']['sha']
        tree=json.loads(get(api+f'git/trees/{commit}?recursive=1'))
    if tree.get('truncated'):raise ValueError('Incomplete receipt tree')
    prefix=f'results/research/READ03_replay_cloud/{SCIENCE_RUN}/{VERIFICATION_RUN}/'
    dest=ROOT/f'results/research/READ03_review/{SCIENCE_RUN}/pinned_replay_receipts'
    expected={row['run_id'] for row in json.loads((ROOT/'docs/research/READ03_execution_plan.json').read_text(encoding='utf-8'))['runs']}
    rows=[];issues=[]
    for entry in tree['tree']:
        path=entry['path']
        if entry['type']!='blob' or not path.startswith(prefix) or not path.endswith('/replay.json'):continue
        case=path.rsplit('/',2)[1]
        if case not in expected:raise ValueError('Unknown fit')
        raw=get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}')
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=entry['sha']:raise ValueError('Receipt Git blob differs')
        receipt=json.loads(raw)
        if (receipt.get('scientific_run_id')!=SCIENCE_RUN or receipt.get('scientific_source_commit')!=SCIENCE_SOURCE
            or str(receipt.get('verification_run_id'))!=str(VERIFICATION_RUN) or receipt.get('verification_source_commit')!=VERIFICATION_SOURCE
            or receipt.get('cases')!=[case] or receipt.get('independent_external_replication') is not False
            or receipt.get('new_training_updates')!=0 or receipt.get('test_accessed') is not False):raise ValueError('Receipt provenance/scope differs')
        verified=(receipt.get('status')=='verified_readonly_fit_replay' and receipt.get('verification_returncode')==0
            and receipt.get('issues')==[] and receipt.get('plan_sha256')==PLAN
            and receipt.get('individual_decisions_replayed')==294912 and receipt.get('checks',0)>0
            and receipt.get('max_mean_nll_error',1)<=1e-4)
        if not verified:issues.append(dict(case=case,status=receipt.get('status'),issues=receipt.get('issues'),returncode=receipt.get('verification_returncode')))
        immutable(dest/case/'replay.json',raw)
        anchor=dict(archive_git_commit=commit,git_blob=blob,remote_path=path,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        if not (dest/case/'git_anchor.json').exists():immutable(dest/case/'git_anchor.json',(json.dumps(anchor,indent=2)+'\n').encode())
        rows.append(dict(case=case,verified=verified,checks=receipt.get('checks',0),decisions=receipt.get('individual_decisions_replayed',0),max_mean_nll_error=receipt.get('max_mean_nll_error'),**anchor))
    now=datetime.datetime.now(datetime.timezone.utc)
    out=dict(observed_utc=now.isoformat(),archive_git_commit=commit,scientific_run_id=SCIENCE_RUN,verification_run_id=VERIFICATION_RUN,
        receipts=rows,missing=sorted(expected-{r['case'] for r in rows}),issues=issues,
        checks=sum(r['checks'] for r in rows),individual_decisions_replayed=sum(r['decisions'] for r in rows),
        verified_fits=sum(r['verified'] for r in rows),expected_fits=36,
        status='replay_discrepancies_require_investigation' if issues else 'verified_complete_internal_endpoint_replay' if len(rows)==36 else 'verified_partial_internal_endpoint_replay',
        independent_external_replication=False,scientific_task3_complete=False)
    if args.tree_snapshot:out['tree_snapshot_sha256']=hashlib.sha256(args.tree_snapshot.read_bytes()).hexdigest()
    immutable(dest/('collection_'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json'),(json.dumps(out,indent=2)+'\n').encode())
    print(json.dumps({k:out[k] for k in ('status','verified_fits','checks','individual_decisions_replayed','issues')}))
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
