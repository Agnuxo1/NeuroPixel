"""Prepare a complete local publication manifest; perform no external operation."""
from pathlib import Path
import json
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from completion_kit.common import read_json, sha256_file, verify_files, write_json

OUT=ROOT/'results/research/COMPLETION20261012'
plan=read_json(ROOT/'docs/research/RENDER11_energy_plan.json')
verify_files(ROOT,plan['source_sha256'])
verify_files(ROOT,plan['input_sha256'])
audit=read_json(OUT/'release_audit_after.json')
assert not audit['issues'] and audit['evidence']['energy']['complete']
paths=set(plan['source_sha256'])|set(plan['input_sha256'])
paths.update(['README.md','docs/research/RENDER11_energy_plan.json','docs/research/RENDER11_results.md','docs/research/RENDER10_results.md','docs/research/RESULTS_INDEX_20261009.md','docs/research/REPRODUCE_20261009.md','docs/research/programme_20261008.json','docs/papers/NeuroPixel_manuscript_20261009.md','docs/papers/NeuroPixel_evidence_reconciled_20261012.md','docs/research/publication_raw_manifest.json'])
for relative in ['results/research/RENDER11_energy/original_20261009A','results/research/RENDER11_energy/recovered_20261009B','docs/archive/completion_20261012/G1_before']:
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/relative).rglob('*') if p.is_file())
paths.update(p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file())
records=[{'path':rel,'sha256':sha256_file(ROOT/rel),'bytes':(ROOT/rel).stat().st_size} for rel in sorted(paths)]
packet={'status':'prepared_owner_publication_required','branch':'research/completion-20261012','checkout':str(ROOT),
        'prepared_from_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'external_operation_performed':False,'publication_or_push_performed':False,'files':records,
        'required_owner_action':'Review the complete local branch and publish the reconciled documents with original/recovered receipts and hashes; public push remains prohibited for this agent.',
        'scope':'G1 evidence reconciliation package only; not the final G12 manuscript or independent reproduction.'}
write_json(OUT/'G1_owner_publication_package.json',packet,exclusive=True)
print(json.dumps({'status':packet['status'],'files':len(records),'external_operation_performed':False}))
