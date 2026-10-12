"""Descriptive archived curves/resources/secondary metrics, not new contrasts."""
import datetime
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    review = ROOT/'results/research/OPT03_QTRAIN_U16_review/37771588722'
    final = json.loads((review/'final_receipts_verification.json').read_text(encoding='utf-8'))
    summary = json.loads((review/'analysis/summary.json').read_text(encoding='utf-8'))
    assert final['status'] == 'verified_final_cohort_receipts_and_workflow_success'
    assert not final['issues'] and summary['replay_decisions'] == 2654208
    plan = json.loads((ROOT/'docs/research/OPT03_QTRAIN_U16_plan.json').read_text(encoding='utf-8'))
    rows, inputs = [], {}
    for conf in plan['runs']:
        path = ROOT/f"results/research/OPT03_QTRAIN_U16_recovery/37771588722/cohort/{conf['run_id']}/result.json"
        result = json.loads(path.read_text(encoding='utf-8'))
        curve = result['curve']
        validation = result['evaluations'][-1]['panels']['validation']['metrics']
        rows.append(dict(case=conf['run_id'], late_CE_before=statistics.mean(w['answer_loss_mean'] for w in curve[-16:-8]),
                         late_CE_last=statistics.mean(w['answer_loss_mean'] for w in curve[-8:]),
                         recorded_window_elapsed_seconds=curve[-1]['elapsed_training_seconds'],
                         minimum_recorded_available_ram_gib=min(w['available_ram_gib'] for w in result['resources']),
                         validation_nominal_binding_pct=100*validation['matched']['binding'],
                         validation_dense_joint_pct=100*validation['dense']['joint_query_binding'],
                         validation_matched_per_role_pct=[100*v for v in validation['matched']['per_role']],
                         joint_at12288_matched_pct=100*result['evaluations'][0]['panels']['validation']['metrics']['matched']['joint_query_binding']))
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=rows,
                  scope='Supplemental descriptive curves/resources/secondary endpoints; no new formal contrast or gate',
                  recorded_window_elapsed_seconds_sum=sum(r['recorded_window_elapsed_seconds'] for r in rows),
                  minimum_available_ram_gib=min(r['minimum_recorded_available_ram_gib'] for r in rows),
                  time_scope='Window timer includes sampling, update, witness hashing and bookkeeping; excludes most checkpoint/evaluation/archive time. Not complete compute or energy.',
                  new_trainings=0, new_model_evaluations=0, energy_joules=None,
                  input_sha256=inputs, generator_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    dest = review/'descriptive_diagnostics.json'
    dest.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:output[k] for k in ('recorded_window_elapsed_seconds_sum', 'minimum_available_ram_gib', 'energy_joules')}))


if __name__ == '__main__':
    main()
