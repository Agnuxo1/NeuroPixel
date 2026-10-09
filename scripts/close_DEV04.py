"""Scoped scientific closure; never silently closes broad task3 or H1."""
import datetime
import hashlib
import json
import pathlib
import sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dev04_execution_registry import PLAN,RUN
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    review=ROOT/f'results/research/DEV04_review/{RUN}';analysis=review/'analysis'
    needed=[review/'final_receipts_verification.json',analysis/'summary.json',analysis/'analysis_manifest.json',analysis/'report_recount.json',analysis/'report_manifest.json',ROOT/'docs/research/DEV04_results.md']
    if any(not p.exists() for p in needed):raise ValueError('Complete proofs, analysis and report required before scoped closure')
    final=json.loads(needed[0].read_bytes());summary=json.loads(needed[1].read_bytes());manifest=json.loads(needed[2].read_bytes());recount=json.loads(needed[3].read_bytes());report=json.loads(needed[4].read_bytes())
    if final['status']!='verified_complete_cohort_and_terminal_success' or final['verified_cases']!=12 or final['individual_decisions_replayed']!=9732096 or summary['plan_sha256']!=PLAN or summary['body_fits']!=12 or summary['head_fits']!=24:raise ValueError('Missing complete fixed cohort')
    for name,digest in manifest['inputs_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Original scientific/proof input changed')
    for name,digest in manifest['outputs_sha256'].items():
        if sha(analysis/name)!=digest:raise ValueError('Statistical artifact changed')
    if recount['issues'] or recount['checks']!=41 or recount['summary_sha256']!=sha(needed[1]) or report['summary_sha256']!=sha(needed[1]):raise ValueError('Orthogonal recount/report binding differs')
    for name,digest in report['outputs'].items():
        if name=='plot_status':continue
        portable=name.replace('\\','/')
        relative=pathlib.PurePosixPath(portable)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe report output path')
        path=ROOT/relative if portable.startswith('docs/') else analysis/relative
        if sha(path)!=digest:raise ValueError('Reported output changed')
    plan_path=ROOT/'docs/research/DEV04_execution_plan.json';plan=json.loads(plan_path.read_bytes())
    if sha(plan_path)!=PLAN or any(sha(ROOT/n)!=h for n,h in plan['source_sha256'].items()):raise ValueError('Frozen governing recipe changed')
    receipt=dict(status='closed_exact_DEV04_internal_recipe_with_registered_outcomes',closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=PLAN,
        source_governors_unchanged=43,body_fits=12,head_fits=24,initializations=12,policy_means_for_interval=3,individual_decisions_replayed=9732096,
        competence_gate_passed=summary['competence_gate_all_twelve'],precision_target_passed=summary['primary']['precision_target_passed'],primary=summary['primary'],
        original_failure_attempts_preserved=3,successful_training_cases_repeated=0,external_replication=False,test_scored=False,
        original_H1='not_supported_under_original_recipe',scientific_task3_complete=False,programme_complete=False,
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in needed})
    path=review/'closure_receipt.json'
    if path.exists():
        prior=json.loads(path.read_bytes())
        if prior['artifact_sha256']!=receipt['artifact_sha256']:raise ValueError('Preserve different prior closure')
        print(json.dumps(prior));return
    path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
