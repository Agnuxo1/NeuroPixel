"""Resolve precise historical report references without inferring absent artifacts."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research/COMPLETION20261012'
original=json.loads((OUT/'study_traceability_map.json').read_text(encoding='utf-8'))
tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',original['base_commit']],cwd=ROOT,text=True).splitlines()
plans={'OPT03_D':'docs/research/OPT03_execution_plan.json'}
checked=[]
for item in original['studies']:
    report=ROOT/item['claim_source_report']
    text=report.read_text(encoding='utf-8')
    study=item['study']
    prefixes=[f'results/research/{study}']
    if study in ['04','05','06_A06_1']:
        prefixes=[f'results/research/{study}_']
    if study=='TASK4_historical_capabilities_audit_20261009':
        prefixes=['results/research/TASK4_review/']+[f'results/research/{n:02d}_cloud_runs/' for n in range(7,16)]+['results/research/09_offline_runs/']
    if study=='CONTINUATIONS_RECONCILED_20261008':
        prefixes=['coord/recovery/reconciliation-20261008/','results/research/06_']
    data=[p for p in tracked if any(p.lower().startswith(prefix.lower()) for prefix in prefixes)]
    cited=sorted(set(re.findall(r'(?<![A-Za-z0-9_/])(?:results|docs|coord|scripts|neuropixel)/[A-Za-z0-9_./-]+',text)))
    cited=[p.rstrip('./') for p in cited]
    refs=[{'path':p,'exists':(ROOT/p).exists()} for p in cited]
    if study in plans:
        path=ROOT/plans[study]
        plan=json.loads(path.read_text(encoding='utf-8'))
        item['frozen_plans']=[{'path':plans[study],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'registration':plan.get('local_registration_id'),'source_governors':plan.get('source_sha256',{})}]
    item['data_checkpoints_predictions_logs_and_failed_attempts']=data
    item['saved_recounts']=[p for p in data if p.endswith('.json') and any(x in Path(p).name.lower() for x in ['summary','audit','receipt','analysis'])]
    item['documented_path_references']=refs
    item['missing_dependencies']=[p for p in refs if not p['exists']]
    if not data:
        item['missing_dependencies'].append({'path':prefixes,'reason':'No study raw directory in the examined publication tree; report existence is not raw-data availability.'})
    if not item['frozen_plans'] and study not in ['TASK4_historical_capabilities_audit_20261009','CONTINUATIONS_RECONCILED_20261008']:
        item['missing_dependencies'].append({'reason':'No matched frozen plan in the examined tree.'})
    checked.append(item)
path=OUT/'study_traceability_map_checked.json'
assert not path.exists(),path
path.write_text(json.dumps({'base_commit':original['base_commit'],'studies':checked,'scope':'Exact directory prefixes and documented file references; absent original data remain absent. Protocol/source/run/recount links are provenance, not a new neural replay.'},indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
summary={'studies':len(checked),'reports_with_missing_dependencies':[r['study'] for r in checked if r['missing_dependencies']]}
(OUT/'traceability_dependencies.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(summary))
