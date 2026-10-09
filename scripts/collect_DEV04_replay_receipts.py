"""Pin replay receipts to Git blobs and require all scientific jobs before closure."""
import argparse
import datetime
import hashlib
import json
import math
import pathlib
from collect_READ03_replay_receipts import get,immutable
from dev04_execution_registry import PLAN,RUN,SOURCE,RECOVERY_RUN,RECOVERY_SOURCE,FAILED_STARTS,expected_execution
from readout_snapshot import load_snapshot
from summarize_DEV04 import archived_case,validate_proof

ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO='Agnuxo1/NeuroPixel'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def blob(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()

def validate_terminal_snapshot(terminal,expected,verification_run,verification_source):
    runs=terminal['runs'];jobs=terminal['jobs']
    registered=((RUN,SOURCE,'.github/workflows/dev04-experiment.yml'),
        (RECOVERY_RUN,RECOVERY_SOURCE,'.github/workflows/dev04-operational-recovery.yml'),
        (verification_run,verification_source,'.github/workflows/dev04-readonly-verification.yml'))
    for run_id,source,path in registered:
        record=runs[str(run_id)]
        if (record['id']!=run_id or record['head_sha']!=source or record['status']!='completed'
            or record['path']!=path or record['run_attempt']!=1):raise ValueError('Exact single-attempt scientific/verification workflow not terminal')
        if record['conclusion']!=('failure' if run_id==RUN else 'success'):raise ValueError('Scientific execution disposition changed')
        rows=jobs[str(run_id)]['jobs']
        if any(row['run_id']!=run_id for row in rows):raise ValueError('Foreign job belongs to different workflow run')
        if len({row['id'] for row in rows})!=len(rows):raise ValueError('Duplicate scientific job identifier')
    initial_rows=jobs[str(RUN)]['jobs'];recovery_rows=jobs[str(RECOVERY_RUN)]['jobs']
    initial={r['name']:r for r in initial_rows};recovery={r['name']:r for r in recovery_rows}
    if len(initial_rows)!=12 or len(recovery_rows)!=3 or set(initial)!=expected or set(recovery)!=FAILED_STARTS:raise ValueError('Different/missing/duplicate execution cohort jobs')
    for case,row in initial.items():
        if row['status']!='completed' or row['conclusion']!=('failure' if case in FAILED_STARTS else 'success'):raise ValueError('Initial execution disposition changed')
    if any(r['status']!='completed' or r['conclusion']!='success' for r in recovery.values()):raise ValueError('Recovery case missing success')
    verify_jobs=jobs[str(verification_run)]['jobs']
    if (len(verify_jobs)!=1 or verify_jobs[0]['name']!='readonly-verification'
        or verify_jobs[0]['status']!='completed' or verify_jobs[0]['conclusion']!='success'):raise ValueError('Serial verification job missing terminal success')
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--tree-snapshot',type=pathlib.Path,required=True);p.add_argument('--runs-snapshot',type=pathlib.Path)
    p.add_argument('--verification-run',type=int,required=True);p.add_argument('--verification-source',required=True);a=p.parse_args()
    snapshot=load_snapshot(a.tree_snapshot,REPO);commit=snapshot['commit']['sha'];tree=snapshot['tree'];entries={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    from research_DEV04 import load_plan
    plan=load_plan(ROOT/'docs/research/DEV04_execution_plan.json',PLAN);expected={c['run_id'] for c in plan['runs']};configs={c['run_id']:c for c in plan['runs']}
    prefix=f'results/research/DEV04_replay_cloud/{RUN}/{a.verification_run}/'
    review=ROOT/f'results/research/DEV04_review/{RUN}';dest=review/'pinned_replay_receipts';records=[];issues=[]
    for case in sorted(expected):
        path=prefix+case+'/replay.json'
        if path not in entries:continue
        raw=get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}')
        if blob(raw)!=entries[path]['sha']:raise ValueError('Pinned replay Git blob differs')
        receipt=json.loads(raw);run,source=expected_execution(case)
        if (receipt.get('case')!=case or receipt.get('scientific_run_id')!=run or receipt.get('scientific_source_commit')!=source
            or str(receipt.get('verification_run_id'))!=str(a.verification_run) or receipt.get('verification_source_commit')!=a.verification_source
            or receipt.get('frozen_recipe_source_commit')!=SOURCE or receipt.get('plan_sha256')!=PLAN):raise ValueError('Replay execution/recipe/case identity differs')
        verified=(receipt.get('status')=='verified_available_readonly_DEV04_state_and_endpoints' and receipt.get('verification_returncode')==0
            and receipt.get('issues')==[] and receipt.get('individual_decisions_replayed')==811008 and math.isfinite(receipt.get('max_mean_nll_error',float('inf'))) and 0<=receipt.get('max_mean_nll_error',1)<=1e-4
            and receipt.get('new_training_updates')==0 and receipt.get('test_scored') is False and receipt.get('external_replication') is False)
        if not verified:issues.append(dict(case=case,status=receipt.get('status'),issues=receipt.get('issues'),returncode=receipt.get('verification_returncode')))
        immutable(dest/case/'replay.json',raw)
        anchor=dict(archive_git_commit=commit,git_blob=entries[path]['sha'],remote_path=path,sha256=sha(raw),bytes=len(raw))
        if not (dest/case/'git_anchor.json').exists():immutable(dest/case/'git_anchor.json',(json.dumps(anchor,indent=2)+'\n').encode())
        if verified:
            original=ROOT/f'results/research/DEV04_recovery/{RUN}/originals'/case
            base=original.parent.parent/'cohort'/case
            try:
                manifest=archived_case(case,configs[case],base,original)
                validate_proof(receipt,case,original,manifest,base,a.verification_run,a.verification_source)
            except (ValueError,FileNotFoundError) as error:
                verified=False;issues.append(dict(case=case,status='original_archive_and_replay_binding_not_verified',reason=str(error)))
        records.append(dict(case=case,verified=verified,checks=receipt.get('checks',0),decisions=receipt.get('individual_decisions_replayed',0),max_mean_nll_error=receipt.get('max_mean_nll_error',1),**anchor))
    missing=sorted(expected-{x['case'] for x in records});now=datetime.datetime.now(datetime.timezone.utc)
    collection=dict(observed_utc=now.isoformat(),status='issues_found' if issues else 'verified_complete_internal_endpoint_replay' if not missing else 'verified_available_partial_endpoint_replay',
        verification_run_id=a.verification_run,verification_source_commit=a.verification_source,records=records,missing=missing,issues=issues,
        verified_cases=sum(r['verified'] for r in records),checks=sum(r['checks'] for r in records),individual_decisions_replayed=sum(r['decisions'] for r in records),
        tree_snapshot_sha256=sha(a.tree_snapshot.read_bytes()),new_training_updates=0,cohort_effects_estimated=False,scientific_task3_complete=False)
    immutable(dest/('collection_'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json'),(json.dumps(collection,indent=2)+'\n').encode())
    if not missing and not issues and a.runs_snapshot:
        terminal=json.loads(a.runs_snapshot.read_bytes());validate_terminal_snapshot(terminal,expected,a.verification_run,a.verification_source)
        path=prefix+'cohort_receipt.json'
        if path not in entries:raise ValueError('Final verification cohort receipt absent')
        raw=get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}')
        if blob(raw)!=entries[path]['sha']:raise ValueError('Final cohort Git blob differs')
        final=json.loads(raw)
        if (final['missing'] or final['plan_sha256']!=PLAN or final['verification_source_commit']!=a.verification_source
            or str(final['verification_run_id'])!=str(a.verification_run) or len(final['records'])!=12
            or {r['case'] for r in final['records']}!=expected or any(r['returncode'] for r in final['records'])):raise ValueError('Final replay receipt incomplete')
        closed=dict(status='verified_complete_cohort_and_terminal_success',initial_scientific_run_id=RUN,recovery_scientific_run_id=RECOVERY_RUN,
            verification_run_id=a.verification_run,verification_source_commit=a.verification_source,verified_cases=12,body_fits=12,head_fits=24,
            checks=collection['checks'],individual_decisions_replayed=collection['individual_decisions_replayed'],tree_snapshot_sha256=collection['tree_snapshot_sha256'],
            runs_snapshot_sha256=sha(a.runs_snapshot.read_bytes()),cohort_receipt_sha256=sha(raw),cohort_git_blob=entries[path]['sha'],
            original_startup_failures_preserved=3,successful_training_cases_repeated=0,original_H1_unchanged=True,test_scored=False,external_replication=False,scientific_task3_complete=False)
        immutable(review/'final_receipts_verification.json',(json.dumps(closed,indent=2)+'\n').encode())
    print(json.dumps({k:collection[k] for k in ('status','verified_cases','checks','individual_decisions_replayed','missing','issues')}))
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
