"""Freeze current claim-bearing documents and index every nonstructural source line."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research/COMPLETION20261012'
ARCHIVE=ROOT/'docs/archive/completion_20261012/A18_sources'
assert not ARCHIVE.exists(),'Never replace a frozen source inventory'
mapping=json.loads((OUT/'study_traceability_map_checked.json').read_text(encoding='utf-8'))
sources={row['claim_source_report'] for row in mapping['studies']}
sources.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'docs/papers').glob('*.md'))
sources.update(['README.md','docs/research/RESULTS_INDEX_20261009.md','docs/research/REPRODUCE_20261009.md','docs/research/COMPLETION_G0_inventory_20261012.md'])
ARCHIVE.mkdir(parents=True)
(ARCHIVE/'.gitattributes').write_text('* -text\n',encoding='utf-8',newline='\n')
documents=[]
candidates=[]
for doc_number,relative in enumerate(sorted(sources),1):
    raw=(ROOT/relative).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    copy=ARCHIVE/relative
    copy.parent.mkdir(parents=True,exist_ok=True)
    copy.write_bytes(raw)
    lines=raw.decode('utf-8').splitlines()
    coverage=[]
    in_code=False
    for number,line in enumerate(lines,1):
        text=line.strip()
        identifier=f'S{doc_number:03d}-L{number:04d}'
        if not text:
            coverage.append({'line':number,'classification':'blank'})
            continue
        if text.startswith('```'):
            in_code=not in_code
            coverage.append({'line':number,'classification':'code_fence'})
            continue
        if text.startswith('#'):
            coverage.append({'line':number,'classification':'heading','text':text})
            continue
        if re.fullmatch(r'[| :\-]+',text):
            coverage.append({'line':number,'classification':'table_separator'})
            continue
        kind='code_or_formula' if in_code else ('table_row' if text.startswith('|') else 'narrative_or_list')
        candidate={'id':identifier,'source_path':relative,'source_line':number,'source_sha256':digest,
                   'archived_source_path':copy.relative_to(ROOT).as_posix(),'kind':kind,'source_text':line,
                   'classification_status':'pending_scientific_and_editorial_review','declared_level':None}
        candidates.append(candidate)
        coverage.append({'line':number,'classification':'candidate','candidate_id':identifier})
    assert len(coverage)==len(lines) and len({r['line'] for r in coverage})==len(lines)
    documents.append({'path':relative,'sha256':digest,'bytes':len(raw),'archived_path':copy.relative_to(ROOT).as_posix(),
                      'lines':len(lines),'candidate_count':sum(r['classification']=='candidate' for r in coverage),'coverage':coverage})
manifest={'utc':datetime.now(timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'scope':'Current README, all current manuscripts, every indexed historical result report and new G0/G1 phase documents. Archives are preserved sources, not new current claims.',
          'documents':documents,'source_documents':len(documents),'candidate_source_units':len(candidates),
          'classification_is_complete':False,'levels_assigned':False,
          'limitations':['A source line may contain several claims and must be split during semantic review.','A table row or code fragment may be an editorial item; any exclusion must be justified.','Inventory coverage is not proof of truth, experimental completeness, originality or public preregistration.']}
for name,data in [('claim_source_inventory.json',manifest),('claim_candidates.json',{'schema':1,'candidates':candidates,'levels_assigned':False})]:
    path=OUT/name
    assert not path.exists(),path
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
verification={'source_documents':len(documents),'candidate_source_units':len(candidates),'all_source_bytes_preserved':all(hashlib.sha256((ROOT/r['archived_path']).read_bytes()).hexdigest()==r['sha256'] for r in documents),
              'all_source_lines_accounted':all(len(r['coverage'])==r['lines'] for r in documents),
              'negative_findings_excluded':False,'levels_assigned':False,'scope':'Source-unit coverage and identities only; complete claims ledger still pending.'}
assert verification['all_source_bytes_preserved'] and verification['all_source_lines_accounted']
path=ROOT/'docs/execplan/evidence/A18.1_inventory_verification.json'
assert not path.exists(),path
path.write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(verification,indent=2))
