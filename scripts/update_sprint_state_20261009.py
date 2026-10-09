"""Factual milestone only; no scientific inputs or thresholds changed."""
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
now = datetime.now(timezone.utc).isoformat()
p = root / 'docs/research/programme_20261008.json'
data = json.loads(p.read_text(encoding='utf-8-sig'))
t = next(x for x in data['tasks'] if x['number'] == 3)
t.update(status='closed_bounded_investigation_with_mixed_outcomes', phase='DEV04 competence passed; original precision failed; GLOB05 genuine information control closed; proceed historical audit', assessment='docs/research/TASK3_assessment_20261009.md', all_positive_targets_achieved=False, closed_utc=now)
t['next_after_readout_closure']['primary_effects_established'] = True
g = t['active_global_control_preparation']
g.update(status='closed_exact_GLOB05_internal_control_recipe', recovered_cases=12, terminal_success_jobs=12, neural_checks=3120, exact_decisions=3538944, replay_max_mean_nll_error=0.0, report='docs/research/GLOB05_results.md', closure_receipt='results/research/GLOB05_review/37887762153/closure_receipt.json', original_DEV04_precision_target_passed=False, last_verified_utc=now)
t4 = next(x for x in data['tasks'] if x['number'] == 4)
t4.update(status='in_progress', phase='read_only_existing_cloud_claims_and_source_audit', started_utc=now)
data.update(updated_utc=now, programme_complete=False)
p.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
entry = f'\n\n## GLOB05 cerrado y transición acotada — {now}\nDoce jobs run37887762153/sourcee4b82b28 terminalsuccess. Todos originales/proofs fijados:3120checks/3538944decisiones exactas/maxNLL0/issues0;52recountchecks. Atención-global+6.278483pp IC[4.107922,8.449044],semiancho2.170561; global-local descriptivo+9.965515pp IC[5.868360,14.062670];globalgateall12false. DEV04precision5.657614>5 sigueFAILED; Gov58/old43 intactos,0oldrefits/0newbodies/test0/external0. Scopedclosure/report preservados. JEVexit0connectedprovenancejev aconseja cierre investigación amplia acotada con resultados mixtos, no todas metas positivas: TASK3_assessment_20261009.md. Task4 inicia auditoría de informes históricos7–15; objetivos4–8/fullgoal pendientes. Recursos científicos propios0activos;deadline13:49:31Z.\n'
cp = root / 'coord/recovery/status-audit-20261007/checkpoint.md'
cp.write_bytes(cp.read_bytes().rstrip(b'\r\n')+entry.encode('utf-8'))
print(json.dumps({'updated_utc':now,'task3':t['status'],'task4':t4['status'],'programme_complete':False}))
