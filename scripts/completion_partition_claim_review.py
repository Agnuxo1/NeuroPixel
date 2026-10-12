"""Partition every frozen candidate into a bounded review batch without omissions."""
from pathlib import Path
from collections import defaultdict
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research/COMPLETION20261012'
inventory=json.loads((OUT/'claim_candidates.json').read_text(encoding='utf-8'))
groups=defaultdict(list)
for candidate in inventory['candidates']:
    source=candidate['source_path']
    name=Path(source).name
    if name.startswith(('04_','05_','06_A06_1','TASK4_','CONTINUATIONS_')):
        batch='A18.2.H'
    elif name.startswith(('DEV04','GLOB05','OPT03','READ03','SCN06','VIS07')):
        batch='A18.2.D'
    elif name.startswith('RENDER'):
        batch='A18.2.G'
    else:
        batch='A18.2.P'
    groups[batch].append(candidate['id'])
all_ids=[c['id'] for c in inventory['candidates']]
partitioned=[identifier for ids in groups.values() for identifier in ids]
assert len(all_ids)==len(set(all_ids))==len(partitioned)==len(set(partitioned))
assert set(all_ids)==set(partitioned)
record={'scope':'Semantic review and evidence binding are pending. No levels assigned by this partition.',
        'source_candidate_count':len(all_ids),'all_candidates_partitioned_once':True,
        'batches':[{'task':key,'candidate_ids':value,'count':len(value)} for key,value in sorted(groups.items())]}
path=OUT/'claim_review_batches.json'
assert not path.exists(),path
path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({r['task']:r['count'] for r in record['batches']}))
