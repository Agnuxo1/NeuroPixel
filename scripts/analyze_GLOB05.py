"""Fixed grouped paired contrast; no incomplete-cohort estimates or precision repair."""
import argparse,csv,hashlib,json,math,pathlib,statistics,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from research_GLOB05 import load_plan,load_inventory,sha,expected_cases

def contrast_vectors(cells):
    expected={(case,mode) for case in expected_cases() for mode in ('attention','gated_uniform_global','local')}
    if set(cells)!=expected or any(not math.isfinite(v) or not 0<=v<=1 for v in cells.values()):raise ValueError('Complete unique finite12body/three-reader cells required')
    return {name:[statistics.mean(100*(cells[c,a]-cells[c,b]) for c in expected_cases()[j*4:(j+1)*4]) for j in range(3)]
        for name,a,b in [('attention_minus_active_global','attention','gated_uniform_global'),('active_global_minus_local','gated_uniform_global','local')]}

def interval(values):
    if len(values)!=3 or not all(math.isfinite(v) for v in values):raise ValueError('Exactly three policy means required')
    mean=statistics.mean(values);half=4.302652729696142*statistics.stdev(values)/math.sqrt(3)
    return dict(policy_means_pp=values,mean_pp=mean,t95_pp=[mean-half,mean+half],half_width_pp=half,df=2,
        lower_bound_positive=mean-half>0,descriptive_new_contrast_precision_target_passed=half<=5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=pathlib.Path,default=ROOT/'docs/research/GLOB05_execution_plan.json');p.add_argument('--expected-plan-sha256',required=True)
    p.add_argument('--cohort',type=pathlib.Path,required=True);p.add_argument('--review',type=pathlib.Path,required=True);a=p.parse_args()
    plan=load_plan(a.plan,a.expected_plan_sha256);inventory=load_inventory(ROOT/plan['parent_inventory_path'],plan['parent_inventory_sha256'])
    final=a.review/'final_receipts_verification.json'
    if not final.exists():raise ValueError('Full12originalarchives/replays/terminalsuccess required before effects')
    f=json.loads(final.read_bytes())
    if f['status']!='verified_complete_GLOB05_cohort_and_terminal_success' or f['verified_cases']!=12 or f['plan_sha256']!=a.expected_plan_sha256:raise ValueError('Different/incomplete full-cohort proof')
    cells={};rows=[];inputs={final.as_posix():sha(final)}
    for parent in inventory['parents']:
        case=parent['case'];folder=a.cohort/case;path=folder/'result.json';proofpath=a.review/'pinned_replay_receipts'/case/'replay.json'
        if not path.exists() or not proofpath.exists():raise ValueError('Missing fixed-case scientific original or proof')
        record=json.loads(path.read_bytes());proof=json.loads(proofpath.read_bytes());config=next(r for r in plan['runs'] if r['parent_case']==case)
        if (record['status']!='completed' or record['config']!=config or record['plan_sha256']!=a.expected_plan_sha256
            or record['frozen_body_digest']!=parent['frozen_body_digest'] or record['new_backbone_updates']!=0
            or record['effective_gradient_parameters']!=3168 or record['head_nominal_parameters']!=3168
            or record['test_accessed'] is not False or record['original_DEV04_precision_reinterpreted'] is not False):raise ValueError('Analyzed exact matched-control recipe differs')
        if (proof['status']!='verified_readonly_GLOB05_control_replay' or proof['case']!=case or proof['issues']
            or proof['individual_decisions_replayed']!=294912 or proof['plan_sha256']!=a.expected_plan_sha256
            or proof['max_mean_nll_error']>1e-4 or proof['new_training_updates']!=0):raise ValueError('Complete numerical proof required')
        for name,digest in proof['input_files_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Analyzed inputs differ from exact numerical proof')
        original=ROOT/'results/research/DEV04_recovery/37845945011/cohort'/case
        for mode,label,expected in [('query_attention','attention',parent['closed_attention_result_sha256']),('local_query','local',parent['closed_local_result_sha256'])]:
            oldpath=original/mode/'result.json'
            if sha(oldpath)!=expected:raise ValueError('Preserve closed original comparison')
            old=json.loads(oldpath.read_bytes());cells[case,label]=old['evaluations'][-1]['panels']['validation']['metrics']['joint_query_binding'];inputs[oldpath.as_posix()]=sha(oldpath)
        last=record['evaluations'][-1]['panels'];cells[case,'gated_uniform_global']=last['validation']['metrics']['joint_query_binding']
        rows.append(dict(case=case,partition_seed=parent['config']['partition_seed'],init_seed=parent['config']['init_seed'],
            attention_joint=cells[case,'attention'],active_global_joint=cells[case,'gated_uniform_global'],local_joint=cells[case,'local'],
            global_probe_joint=last['probe']['metrics']['joint_query_binding'],global_gate=last['probe']['metrics']['joint_query_binding']>=.95 and cells[case,'gated_uniform_global']>=.9))
        inputs[path.as_posix()]=sha(path);inputs[proofpath.as_posix()]=sha(proofpath)
    effects={name:interval(v) for name,v in contrast_vectors(cells).items()}
    result=dict(status='complete_registered_GLOB05_internal_analysis',plan_sha256=a.expected_plan_sha256,new_heads=12,new_body_initializations=0,old_head_trainings_repeated=0,
        effects=effects,global_gate_all_twelve=all(r['global_gate'] for r in rows),original_DEV04_precision_target='failed_preserved',
        original_H1='not_supported_under_original_recipe',original_test_scored=False,external_replication=False,scientific_task3_complete=False,
        interpretation='Development comparison of per-cell query selection against query-gated global uniform moments. Shared frozen bodies/labels/initial head tensors/batches/masks/update budgets; functional space and arithmetic differ. Three overlapping-policy means; exploratory df2 unadjusted/untruncated intervals; no equivalence inference.')
    out=a.review/'analysis';out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    with (out/'cases.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (out/'analysis_manifest.json').write_text(json.dumps(dict(source_sha256=sha(pathlib.Path(__file__)),inputs_sha256=inputs,outputs_sha256={n:sha(out/n) for n in ('summary.json','cases.csv')}),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=result['status'],effects=effects,global_gate_all_twelve=result['global_gate_all_twelve'],original_DEV04_precision_target='failed_preserved')))

if __name__=='__main__':main()
