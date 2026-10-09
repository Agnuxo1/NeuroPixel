"""Recover a case's latest durable evidence from a pinned connector Git tree."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive',ROOT/'neuropixel/research/dev04_transport.py')
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)
from readout_snapshot import load_snapshot
from collect_READ03_replay_receipts import get
from dev04_execution_registry import validate_receipt

PLAN='2484f3c5917e474ca3755bcdbbaca7d322077615ba47d51653116042d980f20a'
SOURCE='b9431e9b927ba5b6539a172d4b27fe6199776462'
RUN=37845945011
REPO='Agnuxo1/NeuroPixel'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tree-snapshot',type=pathlib.Path,required=True);parser.add_argument('--case',required=True)
    parser.add_argument('--git-object-cache',type=pathlib.Path);args=parser.parse_args()
    plan_path=ROOT/'docs/research/DEV04_execution_plan.json'
    if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=PLAN:raise ValueError('Exact DEV04 plan differs')
    plan=json.loads(plan_path.read_text(encoding='utf-8'));expected={r['run_id']:r for r in plan['runs']}
    if args.case not in expected:raise ValueError('Unknown case')
    snapshot=load_snapshot(args.tree_snapshot,REPO);commit=snapshot['commit']['sha'];tree=snapshot['tree']
    entries_by_path={entry['path']:entry for entry in tree['tree'] if entry['type']=='blob'}
    def pinned_bytes(path):
        entry=entries_by_path.get(path)
        if entry is None:raise ValueError('Required archive path absent from pinned Git tree')
        if args.git_object_cache:
            raw=subprocess.check_output(['git','-c','safe.directory='+args.git_object_cache.resolve().as_posix(),
                '--git-dir='+str(args.git_object_cache.resolve()),'cat-file','blob',entry['sha']])
        else:raw=get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}')
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=entry['sha']:raise ValueError('Pinned Git blob differs')
        return raw
    prefix=f'results/research/DEV04_cloud/{archive.REG}/'
    candidates=[]
    entries=[e for e in tree['tree'] if e['type']=='blob' and e['path'].startswith(prefix+args.case) and e['path'].endswith('/receipt.json')]
    completed=[e for e in entries if e['path']==prefix+args.case+'/receipt.json']
    # Complete archive dominates partials by registered rank; avoid downloading every older receipt.
    for entry in completed or entries:
        path=entry['path']
        if entry['type']!='blob' or not path.startswith(prefix) or not path.endswith('/receipt.json'):continue
        raw=pinned_bytes(path)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=entry['sha']:raise ValueError('Pinned receipt Git blob differs')
        receipt=json.loads(raw)
        validate_receipt(receipt,args.case,expected[args.case])
        candidates.append(((receipt['case_status']=='completed',receipt['stage_rank'],receipt['update']),path,receipt,raw,blob))
    if not candidates:
        print(json.dumps(dict(case=args.case,status='no_durable_archive_yet',training_restarted=False)));return
    _,path,receipt,receipt_raw,blob=max(candidates,key=lambda row:row[0]);folder_name=path.rsplit('/',2)[1]
    dest=ROOT/f'results/research/DEV04_recovery/{RUN}';original=dest/'originals'/folder_name;zipped=original/'original.zip'
    raw=zipped.read_bytes() if zipped.exists() else pinned_bytes(path.rsplit('/',1)[0]+'/raw.zip')
    if len(raw)!=receipt['zip_bytes'] or hashlib.sha256(raw).hexdigest()!=receipt['zip_sha256']:raise ValueError('Original archive size/hash differs')
    archive.old.immutable(zipped,raw);archive.old.immutable(original/'receipt.json',receipt_raw)
    target=dest/'cohort' if receipt['case_status']=='completed' else ROOT.parent/'.cognition/dev04-partials'/str(RUN)/folder_name
    archive.unpack(raw,target,args.case,PLAN)
    anchor=dict(archive_git_commit=commit,receipt_git_blob=blob,zip_sha256=receipt['zip_sha256'],case=args.case,
        stage=receipt['stage'],update=receipt['update'],case_status=receipt['case_status'],extracted_destination=str(target),
        tree_snapshot_sha256=hashlib.sha256(args.tree_snapshot.read_bytes()).hexdigest())
    anchor_path=original/'recovery_receipt.json'
    if not anchor_path.exists():archive.old.immutable(anchor_path,(json.dumps(anchor,indent=2)+'\n').encode())
    print(json.dumps(dict(case=args.case,status=receipt['case_status'],stage=receipt['stage'],update=receipt['update'],scientific_task3_complete=False)))


if __name__=='__main__':main()
