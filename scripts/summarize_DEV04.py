"""Registered DEV04 estimator; hard stop on any missing scientific/replay proof."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import sys
import zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dev04_execution_registry import PLAN,RUN,SOURCE,RECOVERY_RUN,validate_receipt,expected_execution
MODES=('query_attention','local_query')
T95_DF2=4.302652729696142

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def registered_summary(rows):
    expected={(s,400+4*j+k,m) for j,s in enumerate((101,102,103)) for k in range(4) for m in MODES}
    actual=[(r['partition_seed'],r['init_seed'],r['mode']) for r in rows]
    if len(actual)!=24 or set(actual)!=expected:raise ValueError('Complete unique 12-body/24-head cohort required; no estimates')
    if any(not all(math.isfinite(r[k]) and 0<=r[k]<=1 for k in ('probe_joint','validation_joint','validation_binding')) for r in rows):raise ValueError('Finite valid accuracies required')
    cells={key:r for key,r in zip(actual,rows)};pairs=[]
    for j,policy in enumerate((101,102,103)):
        for seed in range(400+4*j,404+4*j):
            a=cells[policy,seed,'query_attention'];b=cells[policy,seed,'local_query']
            pairs.append(dict(partition_seed=policy,init_seed=seed,attention_joint_pp=100*a['validation_joint'],local_joint_pp=100*b['validation_joint'],difference_pp=100*(a['validation_joint']-b['validation_joint']),attention_probe_pp=100*a['probe_joint'],attention_gate=a['probe_joint']>=.95 and a['validation_joint']>=.9))
    policies=[dict(partition_seed=s,mean_difference_pp=statistics.mean(r['difference_pp'] for r in pairs if r['partition_seed']==s),initializations=4) for s in (101,102,103)]
    values=[r['mean_difference_pp'] for r in policies];mean=statistics.mean(values);half=T95_DF2*statistics.stdev(values)/math.sqrt(3)
    # Independent analytic Student-df2 CDF, no SciPy/version dependent quantile call.
    if abs(.5+T95_DF2/(2*math.sqrt(2+T95_DF2*T95_DF2))-.975)>1e-12:raise ValueError('Critical quantile analytic check failed')
    return dict(pairs=pairs,policies=policies,primary=dict(mean_pp=mean,t95_pp=[mean-half,mean+half],half_width_pp=half,n_policy_means=3,df=2,t95_critical=T95_DF2,half_width_target_pp=5,precision_target_passed=half<=5 and mean-half>0),competence_gate_all_twelve=all(r['attention_gate'] for r in pairs),attention_gate_passed_cases=sum(r['attention_gate'] for r in pairs),body_fits=12,head_fits=24,independent_external_replications=0)

def archived_case(case,config,base,original):
    receipt=json.loads((original/'receipt.json').read_bytes());validate_receipt(receipt,case,config)
    zpath=original/'original.zip'
    if receipt['case_status']!='completed' or sha(zpath)!=receipt['zip_sha256'] or zpath.stat().st_size!=receipt['zip_bytes']:raise ValueError('Original complete archive identity differs')
    with zipfile.ZipFile(zpath) as z:
        manifest=json.loads(z.read('archive_manifest.json'))
        if manifest['config']!=config or manifest['case_status']!='completed' or manifest['plan_sha256']!=PLAN or hashlib.sha256(z.read('execution_plan.json')).hexdigest()!=PLAN:raise ValueError('Archived recipe differs')
        if set(z.namelist())!=set(manifest['files'])|{'execution_plan.json','archive_manifest.json'}:raise ValueError('Archive inventory differs')
        for name,digest in manifest['files'].items():
            path=base.parent/name
            if hashlib.sha256(z.read(name)).hexdigest()!=digest or not path.exists() or sha(path)!=digest:raise ValueError('Analyzed scientific input differs '+name)
        return manifest

def validate_proof(proof,case,original,manifest,base,verification_run,verification_source):
    science_run,science_source=expected_execution(case)
    if (proof.get('case')!=case or proof.get('status')!='verified_available_readonly_DEV04_state_and_endpoints'
        or proof.get('scientific_run_id')!=science_run or proof.get('scientific_source_commit')!=science_source
        or proof.get('checks',0)<=0
        or proof.get('issues')!=[] or proof.get('verification_returncode')!=0 or proof.get('individual_decisions_replayed')!=811008
        or proof.get('plan_sha256')!=PLAN or not math.isfinite(proof.get('max_mean_nll_error',float('inf'))) or not 0<=proof.get('max_mean_nll_error',1)<=1e-4
        or proof.get('new_training_updates')!=0 or proof.get('test_scored') is not False or proof.get('external_replication') is not False
        or str(proof.get('verification_run_id'))!=str(verification_run) or proof.get('verification_source_commit')!=verification_source
        or proof.get('original_zip_sha256')!=sha(original/'original.zip') or proof.get('frozen_recipe_source_commit')!=SOURCE
        or set(proof.get('body_endpoints',[]))!={8192,16384} or set(proof.get('head_endpoints',{}))!=set(MODES)
        or any(set(v)!={1024,4096,8192} for v in proof.get('head_endpoints',{}).values())):raise ValueError('Incomplete/different neural replay proof')
    hashes={}
    for name,digest in proof.get('input_files_sha256',{}).items():
        parts=name.replace('\\','/').split('/'+case+'/',1)
        if len(parts)!=2:raise ValueError('Replay input outside scientific case')
        rel=case+'/'+parts[1]
        if rel in hashes:raise ValueError('Duplicate replay input')
        hashes[rel]=digest
    if hashes!=manifest['files']:raise ValueError('Replay not bound to every original case file')

def main():
    p=argparse.ArgumentParser();p.add_argument('--verification-run',type=int,required=True);p.add_argument('--verification-source',required=True);a=p.parse_args()
    plan_path=ROOT/'docs/research/DEV04_execution_plan.json'
    if sha(plan_path)!=PLAN:raise ValueError('Original registered plan differs')
    plan=json.loads(plan_path.read_bytes())
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Scientific governing source changed')
    review=ROOT/f'results/research/DEV04_review/{RUN}';final_path=review/'final_receipts_verification.json'
    if not final_path.exists():raise ValueError('Full cohort final execution/replay proof missing; no effects')
    final=json.loads(final_path.read_bytes())
    if (final.get('status')!='verified_complete_cohort_and_terminal_success' or final.get('initial_scientific_run_id')!=RUN
        or final.get('recovery_scientific_run_id')!=RECOVERY_RUN or final.get('verification_run_id')!=a.verification_run
        or final.get('verification_source_commit')!=a.verification_source or final.get('verified_cases')!=12):raise ValueError('Different/incomplete terminal verification')
    base=ROOT/f'results/research/DEV04_recovery/{RUN}/cohort';rows=[];parent_rows=[];inputs={str(final_path.relative_to(ROOT)):sha(final_path)}
    for conf in plan['runs']:
        case=conf['run_id'];folder=base/case;original=base.parent/'originals'/case
        manifest=archived_case(case,conf,folder,original)
        proof_path=review/'pinned_replay_receipts'/case/'replay.json';anchor_path=proof_path.with_name('git_anchor.json')
        if not proof_path.exists() or not anchor_path.exists():raise ValueError('Missing pinned endpoint proof')
        proof=json.loads(proof_path.read_bytes());anchor=json.loads(anchor_path.read_bytes())
        if anchor['sha256']!=sha(proof_path):raise ValueError('Pinned replay receipt changed')
        validate_proof(proof,case,original,manifest,folder,a.verification_run,a.verification_source)
        body=json.loads((folder/'body/result.json').read_bytes());parent=body['evaluations'][-1]['panels']
        parent_rows.append(dict(case=case,partition_seed=conf['partition_seed'],init_seed=conf['init_seed'],probe_joint=parent['probe']['metrics']['matched']['joint_query_binding'],validation_joint=parent['validation']['metrics']['matched']['joint_query_binding'],parameter_count=body['parameter_count']))
        for mode in MODES:
            path=folder/mode/'result.json';record=json.loads(path.read_bytes())
            endpoints=record['evaluations']
            if [r['update'] for r in endpoints]!=[1024,4096,8192] or record['config']['updates']!=8192 or record['new_backbone_updates']!=0 or record['test_accessed'] is not False:raise ValueError('Different head endpoint/recipe/scope')
            probe=endpoints[-1]['panels']['probe']['metrics'];val=endpoints[-1]['panels']['validation']['metrics']
            rows.append(dict(case=case,partition_seed=conf['partition_seed'],init_seed=conf['init_seed'],mode=mode,probe_joint=probe['joint_query_binding'],validation_joint=val['joint_query_binding'],validation_binding=val['binding'],nominal_parameters=record['head_nominal_parameters'],active_parameters=record['effective_gradient_parameters']))
            inputs[str(path.relative_to(ROOT))]=sha(path)
        for path in (proof_path,anchor_path,original/'receipt.json',original/'original.zip',folder/'body/result.json'):inputs[str(path.relative_to(ROOT))]=sha(path)
    summary=registered_summary(rows)
    summary.update(plan_sha256=PLAN,initial_scientific_run_id=RUN,recovery_scientific_run_id=RECOVERY_RUN,verification_run_id=a.verification_run,verification_source_commit=a.verification_source,
        primary_contrast=plan['analysis']['primary'],scope='Selected competent condition, new bodies and partitions of shared exposed development universe; exploratory unadjusted/untruncated t95 df2. Heads/masks/examples/endpoints/retries are not independent replications.',H1='not_supported_under_original_recipe',test_scored=False,external_replication=False,scientific_task3_complete=False,parents=parent_rows)
    out=review/'analysis';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    for name,data in [('heads.csv',rows),('pairs.csv',summary['pairs']),('policies.csv',summary['policies']),('parents.csv',parent_rows)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.csv','.json') and p.name!='analysis_manifest.json'}
    (out/'analysis_manifest.json').write_text(json.dumps(dict(source_sha256=sha(pathlib.Path(__file__)),inputs_sha256=inputs,outputs_sha256=outputs),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='complete_registered_DEV04_internal_analysis',primary=summary['primary'],competence_gate_all_twelve=summary['competence_gate_all_twelve'],scientific_task3_complete=False)))

if __name__=='__main__':main()
