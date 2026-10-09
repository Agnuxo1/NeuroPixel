"""Serial read-only replay within one free scientific slot; no retraining or selection."""
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dev04_execution_registry import PLAN,RUN,SOURCE,RECOVERY_RUN,RECOVERY_SOURCE,FAILED_STARTS,expected_execution,validate_complete_attempt
from readout_snapshot import load_snapshot
spec=importlib.util.spec_from_file_location('transport',ROOT/'neuropixel/research/qtrain_budget_transport.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
REPO='Agnuxo1/NeuroPixel'
API='https://api.github.com/repos/'+REPO+'/'
SUPERSEDED_OBSERVER_RUN=37872427770
SUPERSEDED_OBSERVER_SOURCE='125feea39a8c940a7a911b453f11b6e0db0c3fd6'

def get(url):return json.loads(transport.fetch(url))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def blob(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
def immutable(path,value):transport.immutable(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())

def free_scientific_slot(jobs,observer):
    if len(jobs)!=3 or {r['name'] for r in jobs}!=FAILED_STARTS or len({r['id'] for r in jobs})!=3:raise ValueError('Different/duplicate recovery execution jobs')
    if any(r['run_id']!=RECOVERY_RUN for r in jobs):raise ValueError('Foreign recovery workflow job')
    if any(r['status']=='completed' and r['conclusion']!='success' for r in jobs):raise ValueError('Recovery failed; preserve before verification')
    if observer['status']!='completed':return False
    if observer['conclusion']=='success':raise ValueError('Old verifier already succeeded; use its evidence rather than repeat')
    if observer['conclusion'] not in ('cancelled','failure'):raise ValueError('Unexpected old observation disposition')
    return sum(r['status']!='completed' for r in jobs)<=1

def await_scientific_slot(case=None):
    # The old observer must be terminal; one serial replay plus <=1 live recovery is <=2.
    start=time.monotonic()
    while True:
        states=[get(API+'actions/runs/'+str(r)) for r in (RUN,RECOVERY_RUN)]
        observer=get(API+'actions/runs/'+str(SUPERSEDED_OBSERVER_RUN))
        if (states[0]['head_sha']!=SOURCE or states[1]['head_sha']!=RECOVERY_SOURCE or states[0]['status']!='completed'
            or any(r['run_attempt']!=1 for r in states) or observer['head_sha']!=SUPERSEDED_OBSERVER_SOURCE):raise ValueError('Science/source/attempt or old observation differs')
        jobs=get(API+'actions/runs/'+str(RECOVERY_RUN)+'/jobs?per_page=100')['jobs']
        admitted=free_scientific_slot(jobs,observer)
        if case in FAILED_STARTS:admitted=admitted and next(r for r in jobs if r['name']==case)['status']=='completed'
        if admitted:return dict(scientific_runs=states,superseded_observer=observer,recovery_jobs=jobs,case=case,active_scientific_recovery_jobs=sum(r['status']!='completed' for r in jobs),serial_replay_slots=1)
        print(json.dumps({'event':'waiting_safe_scientific_slot_or_case','case':case,'old_observer_status':observer['status'],
            'active_scientific_recovery_jobs':sum(r['status']!='completed' for r in jobs)}),flush=True)
        if time.monotonic()-start>5*3600:raise TimeoutError('Bounded observation timed out; never restart scientific jobs')
        time.sleep(60)

def ensure_no_closed_replay_repeated():
    branch=os.environ['GITHUB_REF_NAME'];commit=get(API+'git/ref/heads/'+branch)['object']['sha']
    tree=get(API+'git/trees/'+commit+'?recursive=1')
    if tree.get('truncated'):raise ValueError('Incomplete replay-evidence tree')
    prefix=f'results/research/DEV04_replay_cloud/{RUN}/{SUPERSEDED_OBSERVER_RUN}/'
    if any(r['type']=='blob' and r['path'].startswith(prefix) for r in tree['tree']):raise ValueError('Old closed numerical proofs exist; do not repeat')

def main():
    states=await_scientific_slot();ensure_no_closed_replay_repeated()
    plan_path=ROOT/'docs/research/DEV04_execution_plan.json'
    if sha(plan_path.read_bytes())!=PLAN:raise ValueError('Original frozen plan differs')
    plan=json.loads(plan_path.read_bytes())
    for name,digest in plan['source_sha256'].items():
        if sha((ROOT/name).read_bytes())!=digest:raise ValueError('Frozen governing file changed '+name)
    work=ROOT/'results/research/DEV04_readonly_verification';work.mkdir(parents=True,exist_ok=True)
    immutable(work/'initial_safe_scientific_admission.json',states)
    records=[]
    # Availability-only scheduling: the one case still training comes last.
    # No result values or scores enter the ordering; all twelve cases remain mandatory.
    for conf in sorted(plan['runs'],key=lambda c:c['run_id']=='DEV04_partition101_seed403'):
        case=conf['run_id'];branch='research/dev04-evidence-2026-10-08/'+case
        admission=await_scientific_slot(case);immutable(work/'admissions'/(case+'.json'),admission)
        commit_sha=get(API+'git/ref/heads/'+branch)['object']['sha']
        commit=get(API+'git/commits/'+commit_sha);tree=get(API+'git/trees/'+commit['tree']['sha']+'?recursive=1')
        snapshot=work/'snapshots'/(case+'.json');immutable(snapshot,dict(repository=REPO,commit=commit,tree=tree))
        load_snapshot(snapshot,REPO)
        run,source=expected_execution(case)
        attempt_path=f'results/research/DEV04_cloud/NP-DEV04-20261008/attempts/{run}/{case}/attempt_receipt.json'
        entry=next((x for x in tree['tree'] if x['path']==attempt_path and x['type']=='blob'),None)
        if entry is None:raise ValueError('Missing final execution receipt for '+case)
        raw=transport.fetch(f'https://raw.githubusercontent.com/{REPO}/{commit_sha}/{attempt_path}')
        if blob(raw)!=entry['sha']:raise ValueError('Attempt Git blob differs')
        attempt=json.loads(raw);validate_complete_attempt(attempt,case)
        immutable(work/'attempts'/(case+'.json'),dict(receipt=attempt,git_commit=commit_sha,git_blob=entry['sha'],sha256=sha(raw)))
        subprocess.run([sys.executable,'scripts/recover_DEV04_snapshot.py','--tree-snapshot',str(snapshot),'--case',case],cwd=ROOT,check=True)
        folder=ROOT/f'results/research/DEV04_recovery/{RUN}/cohort'/case
        receipt_path=work/(case+'.json')
        result=subprocess.run([sys.executable,'scripts/replay_DEV04.py','--case',case,'--case-folder',str(folder),'--output',str(receipt_path)],cwd=ROOT)
        if receipt_path.exists():receipt=json.loads(receipt_path.read_bytes())
        else:receipt=dict(status='replay_failed_before_receipt',case=case,issues=['Replay failed before complete receipt'],checks=0,new_training_updates=0,test_scored=False,scientific_task3_complete=False)
        original=ROOT/f'results/research/DEV04_recovery/{RUN}/originals'/case
        receipt.update(scientific_run_id=run,scientific_source_commit=source,frozen_recipe_source_commit=SOURCE,
            verification_run_id=os.environ['GITHUB_RUN_ID'],verification_source_commit=os.environ['GITHUB_SHA'],verification_returncode=result.returncode,
            original_zip_sha256=sha((original/'original.zip').read_bytes()),archive_git_commit=commit_sha,
            tree_snapshot_sha256=sha(snapshot.read_bytes()),plan_sha256=PLAN,external_replication=False,
            superseded_observer_run=SUPERSEDED_OBSERVER_RUN,closed_replay_cases_repeated=0,
            admission_sha256=sha((work/'admissions'/(case+'.json')).read_bytes()))
        raw=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode();receipt_path.write_bytes(raw)
        remote=f'results/research/DEV04_replay_cloud/{RUN}/{os.environ["GITHUB_RUN_ID"]}/{case}/replay.json'
        transport.publish_immutable(remote,raw)
        records.append(dict(case=case,status=receipt['status'],returncode=result.returncode,checks=receipt['checks'],
            individual_decisions_replayed=receipt.get('individual_decisions_replayed',0),archive_path=remote,sha256=sha(raw)))
        print(json.dumps(records[-1]),flush=True)
        if result.returncode:break
    missing=sorted({x['run_id'] for x in plan['runs']}-{x['case'] for x in records})
    final=dict(plan_sha256=PLAN,frozen_recipe_source_commit=SOURCE,initial_scientific_run_id=RUN,recovery_scientific_run_id=RECOVERY_RUN,
        verification_run_id=os.environ['GITHUB_RUN_ID'],verification_source_commit=os.environ['GITHUB_SHA'],records=records,missing=missing,
        expected_cases=12,expected_body_fits=12,expected_head_fits=24,new_training_updates=0,test_scored=False,external_replication=False,scientific_task3_complete=False,
        superseded_observer_run=SUPERSEDED_OBSERVER_RUN,closed_replay_cases_repeated=0,maximum_scientific_executors=2)
    transport.publish_immutable(f'results/research/DEV04_replay_cloud/{RUN}/{os.environ["GITHUB_RUN_ID"]}/cohort_receipt.json',(json.dumps(final,indent=2)+'\n').encode())
    if missing or any(x['returncode'] for x in records):raise SystemExit(1)

if __name__=='__main__':main()
