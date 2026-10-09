"""Full-cohort completion guard linking all originals, replay blobs and actual jobs."""
import argparse,hashlib,json,math,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from research_GLOB05 import load_plan,sha,expected_cases
from collect_GLOB05_snapshot import PLAN,RUN,SOURCE
from neuropixel.research.glob05_transport import blob,digest,old

def validate_jobs(snapshot):
    r=snapshot['run'];jobs=snapshot['jobs']['jobs']
    if (r['id']!=RUN or r['head_sha']!=SOURCE or r['status']!='completed' or r['conclusion']!='success'
        or r['run_attempt']!=1 or r['path']!='.github/workflows/glob05-operational-execution.yml'):raise ValueError('Exact terminal original workflow success required')
    if len(jobs)!=12 or {j['name'] for j in jobs}!=set(expected_cases()) or len({j['id'] for j in jobs})!=12:raise ValueError('All twelve unique jobs required')
    if any(j['run_id']!=RUN or j['status']!='completed' or j['conclusion']!='success' for j in jobs):raise ValueError('Foreign/incomplete/failed scientific job')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-snapshot',type=pathlib.Path,required=True);a=p.parse_args();snapshot=json.loads(a.run_snapshot.read_bytes());validate_jobs(snapshot)
    plan=load_plan(ROOT/'docs/research/GLOB05_execution_plan.json',PLAN)
    review=ROOT/f'results/research/GLOB05_review/{RUN}';base=ROOT/f'results/research/GLOB05_recovery/{RUN}';cases=[]
    for case in expected_cases():
        pinned=review/'pinned_replay_receipts'/case;original=base/'originals'/case
        required=[pinned/'replay.json',pinned/'attempt.json',pinned/'git_anchor.json',pinned/'verification_receipt.json',original/'receipt.json',original/'original.zip']
        if any(not x.exists() for x in required):raise ValueError('Missing full scientific/proof case '+case)
        proof=json.loads(required[0].read_bytes());attempt=json.loads(required[1].read_bytes());anchor=json.loads(required[2].read_bytes());receipt=json.loads(required[4].read_bytes())
        if (sha(required[0])!=anchor['proof_sha256'] or blob(required[0].read_bytes())!=anchor['proof_git_blob']
            or sha(required[1])!=anchor['attempt_sha256'] or blob(required[1].read_bytes())!=anchor['attempt_git_blob']
            or sha(required[4])!=anchor['receipt_sha256'] or blob(required[4].read_bytes())!=anchor['receipt_git_blob']
            or sha(required[5])!=anchor['original_zip_sha256'] or sha(required[5])!=receipt['zip_sha256']):raise ValueError('Pinned original/proof/attempt bytes differ')
        if (proof['status']!='verified_readonly_GLOB05_control_replay' or proof['case']!=case or proof['issues']
            or proof['plan_sha256']!=PLAN or str(proof['scientific_run_id'])!=str(RUN) or proof['scientific_source_commit']!=SOURCE
            or proof['individual_decisions_replayed']!=294912 or not math.isfinite(proof['max_mean_nll_error']) or not 0<=proof['max_mean_nll_error']<=1e-4
            or proof['new_training_updates']!=0 or proof['new_body_initializations']!=0 or proof['old_head_fits_repeated']!=0
            or proof['original_test_scored'] is not False):raise ValueError('Full fixed numerical proof differs')
        if attempt['head_training_completed'] is not True or attempt['numerical_replay_verified'] is not True:raise ValueError('Complete attempt required')
        with zipfile.ZipFile(required[5]) as z:
            manifest=json.loads(z.read('archive_manifest.json'))
            if digest(z.read('execution_plan.json'))!=PLAN or manifest['case_run_id']!=case or manifest['case_status']!='completed':raise ValueError('Original complete scientific archive required')
            if {case+'/'+n:h for n,h in proof['input_files_sha256'].items()}!=manifest['files']:raise ValueError('Replay not bound to every original file')
            for name,h in manifest['files'].items():
                if digest(z.read(name))!=h or sha(base/'cohort'/name)!=h:raise ValueError('Scientific input bytes changed')
        cases.append(dict(case=case,checks=proof['checks'],decisions=proof['individual_decisions_replayed'],max_mean_nll_error=proof['max_mean_nll_error'],original_zip_sha256=receipt['zip_sha256'],proof_sha256=sha(required[0])))
    out=dict(status='verified_complete_GLOB05_cohort_and_terminal_success',plan_sha256=PLAN,scientific_run_id=RUN,scientific_source_commit=SOURCE,
        verified_cases=12,new_head_fits=12,new_body_initializations=0,old_head_fits_repeated=0,cases=cases,checks=sum(r['checks'] for r in cases),
        individual_decisions_replayed=sum(r['decisions'] for r in cases),max_mean_nll_error=max(r['max_mean_nll_error'] for r in cases),
        original_DEV04_precision_target='failed_preserved',original_test_scored=False,external_replication=False,scientific_task3_complete=False,run_snapshot_sha256=sha(a.run_snapshot))
    if out['individual_decisions_replayed']!=3538944:raise ValueError('Complete fixed decision count required')
    old.immutable(review/'final_receipts_verification.json',(json.dumps(out,indent=2,allow_nan=False)+'\n').encode());print(json.dumps({k:out[k] for k in ('status','verified_cases','checks','individual_decisions_replayed','max_mean_nll_error')}))

if __name__=='__main__':main()
