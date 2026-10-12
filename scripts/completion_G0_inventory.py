"""Freeze bounded-memory artifact identities, Git provenance and study traceability."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import hashlib
import json
import platform
import re
import subprocess
import sys
import psutil

ROOT=Path(__file__).resolve().parents[1]
BASE='d8721a59d36d127642a5664c0c41e6091296ba21'
OUT=ROOT/'results/research/COMPLETION20261012'
def git(*args,check=True):
    return subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,text=True,check=check)
def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()
def write(name,data):
    path=OUT/name
    assert not path.exists(),path
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')

blob_audit=read('results/research/COMPLETION20261012/original_blob_audit.json')
assert blob_audit['registered']==1797 and blob_audit['verified']==1797 and not blob_audit['issues']
manifest=read('docs/research/publication_raw_manifest.json')
origins=defaultdict(list)
for item in manifest['raw_entries']:
    origins[item['origin_commit']].append(item)
provenance=[]
for origin,entries in origins.items():
    available=git('cat-file','-e',origin+'^{commit}',check=False).returncode==0
    record={'origin_commit':origin,'registered_blobs':len(entries),'commit_object_available':available}
    if available:
        objects=set(git('ls-tree','-r','--format=%(objectname)',origin).stdout.splitlines())
        absent=[e['path'] for e in entries if e['sha'] not in objects]
        record.update(origin_tree_contains_all_blobs=not absent,absent_origin_blobs=absent,
                      is_ancestor_of_examined_main=git('merge-base','--is-ancestor',origin,BASE,check=False).returncode==0,
                      commit_metadata=git('show','-s','--format=%H %T %aI',origin).stdout.strip())
    else:
        record['missing_dependency']='Original commit object unavailable locally; archived blob identity verified, commit lineage remains unverified.'
    provenance.append(record)
    print(json.dumps({'origin_checked':len(provenance),'origins':len(origins)}),flush=True)
write('origin_provenance.json',{'base_commit':BASE,'origins':provenance,'scope':'Git object provenance is separate from physical replication or third-party authorship.'})

paths=git('ls-tree','-r','--name-only',BASE).stdout.splitlines()
registered={e['path'] for e in manifest['raw_entries']}
extras=[]
for relative in paths:
    if relative in registered or not relative.startswith(('results/','docs/research/','docs/animations/','coord/')):
        continue
    path=ROOT/relative
    extras.append({'path':relative,'sha256':digest(path),'bytes':path.stat().st_size,'base_git_blob':git('rev-parse',BASE+':'+relative).stdout.strip()})
write('additional_artifact_inventory.json',{'base_commit':BASE,'registered_scope':1797,'additional_files':extras,'count':len(extras),'notes':'Additional file identity only; not automatically covered by the raw publication manifest.'})

reports=sorted((ROOT/'docs/research').glob('*results.md'))
reports.extend([ROOT/'docs/research/TASK4_historical_capabilities_audit_20261009.md',ROOT/'docs/research/CONTINUATIONS_RECONCILED_20261008.md'])
result_paths=[p for p in paths if p.startswith('results/research/')]
raw_by_digest={r['sha256']:r['path'] for r in blob_audit['records']}
mapping=[]
for report in reports:
    relative=report.relative_to(ROOT).as_posix()
    text=report.read_text(encoding='utf-8')
    study=report.stem.removesuffix('_results')
    prefix=study.replace('_MODE','_mode').replace('_ROLE','_role')
    plans=list((ROOT/'docs/research').glob(prefix+'*plan.json'))
    if study=='06_A06_1':
        plans=[ROOT/'docs/research/06_execution_plan.json']
    plan_records=[]
    for plan in plans:
        value=json.loads(plan.read_text(encoding='utf-8'))
        governors=value.get('source_sha256',value.get('implementation_sha256',{}))
        if not isinstance(governors,dict):
            governors={}
        sources=[]
        for source,expected in governors.items():
            path=ROOT/source
            sources.append({'source':source,'expected_sha256':expected,
                            'current_source_matches':path.is_file() and digest(path)==expected,
                            'indexed_archived_source_with_same_digest':raw_by_digest.get(expected)})
        plan_records.append({'path':plan.relative_to(ROOT).as_posix(),'sha256':digest(plan),'registration':value.get('registration',value.get('registration_id',value.get('protocol_id'))),'source_governors':sources})
    protocols=[p.relative_to(ROOT).as_posix() for p in (ROOT/'docs/research').glob(prefix+'*protocol*.md')]
    evidence=[p for p in result_paths if study.lower() in p.lower() or prefix.lower() in p.lower()]
    if study in ['TASK4_historical_capabilities_audit_20261009','CONTINUATIONS_RECONCILED_20261008']:
        evidence=[p for p in result_paths if any(x in p for x in ['TASK4_review/','06_A06_1_recovery/','cloud_runs/'])]
    summaries=[p for p in evidence if any(x in Path(p).name.lower() for x in ['summary','audit','receipt','analysis']) and p.endswith('.json')]
    mapping.append({'study':study,'claim_source_report':relative,'report_sha256':digest(report),
                    'protocols':protocols,'frozen_plans':plan_records,
                    'documented_execution_ids':sorted(set(re.findall(r'(?<!\d)\d{9,12}(?!\d)',text))),
                    'data_checkpoints_predictions_logs_and_failed_attempts':evidence,
                    'saved_recounts':summaries,
                    'missing_dependencies':(['No uniquely matched frozen plan; recover through the cited historical archive before replay.'] if not plans else [])+(['No uniquely matched result directory; do not infer a completed execution.'] if not evidence else []),
                    'scope':'Historical report and archived evidence pointers only; new capability claims require claims_ledger.'})
write('study_traceability_map.json',{'base_commit':BASE,'studies':mapping,'source_policy':'Current governor mismatches remain explicit. Archived studies are not reopened or normalized to later source versions.'})

device=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
memory=psutil.virtual_memory()
system={'utc':datetime.now(timezone.utc).isoformat(),'base_commit':BASE,'base_tree':git('rev-parse',BASE+'^{tree}').stdout.strip(),
        'inventory_commit':git('rev-parse','HEAD').stdout.strip(),'checkout':str(ROOT),'platform':platform.platform(),
        'inventory_python':sys.version,'inventory_interpreter':sys.executable,'cpu':platform.processor(),'logical_cpus':psutil.cpu_count(),
        'ram_total_gib':memory.total/2**30,'ram_available_gib':memory.available/2**30,'gpu_driver_device_query':device.stdout.strip(),'gpu_query_exit':device.returncode,
        'kit_environment_receipt':'docs/execplan/evidence/P0.3_environment.txt','kit_python':'E:/Proyecto-concursos/NeuroPixel-env-completion/Scripts/python.exe'}
write('frozen_system_inventory.json',system)
release=read('results/research/COMPLETION20261012/release_audit_before.json')
unexpected=[r for r in provenance if r.get('absent_origin_blobs')]
summary={'status':'frozen_inventory_with_explicit_dependencies' if not unexpected else 'unexplained_origin_discrepancy',
         'indexed_original_blobs':1797,'verified_original_blobs':1797,'unexplained_original_discrepancies':0,
         'origin_commits':len(origins),'origin_commit_objects_available':sum(r['commit_object_available'] for r in provenance),
         'missing_origin_commit_dependencies':[r['origin_commit'] for r in provenance if not r['commit_object_available']],
         'additional_files':len(extras),'mapped_study_reports':len(mapping),'origin_tree_discrepancies':unexpected,
         'release_issues_preserved':release['issues'],
         'preserved_limitations':['Ten VIS07 T16/T32 continuous failures remain at frozen tolerance.','A06.1 and cloud continuation discrepancies remain separate.','No historical cohort reopened, no checkpoint loaded, no new measurement.'],
         'scientific_replication':False}
write('G0_inventory_summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ['missing_origin_commit_dependencies','origin_tree_discrepancies']},indent=2))
assert not unexpected,unexpected
