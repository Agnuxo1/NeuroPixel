"""Archive completed task progress while preserving every original entry."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

folder=Path(__file__).resolve().parent
ledger=json.loads((folder/'tasks.json').read_text(encoding='utf-8-sig'))
plan=folder/'EXECPLAN.md'
text=plan.read_text(encoding='utf-8')
start=text.index('\n## Progreso')
end=text.index('\n## Sorpresas y hallazgos',start)
progress=text[start:end]
for tid in sys.argv[1:]:
    task=next(t for t in ledger['tasks'] if t['id']==tid)
    assert task['status'] in {'done','negative'}, tid
    lines=progress.splitlines(keepends=True)
    entries=[line for line in lines if line.startswith('- [') and (f') {tid}:' in line or f') {tid},' in line)]
    if entries:
        archive=folder/'archive/PROGRESS_ARCHIVE.md'
        with archive.open('a',encoding='utf-8',newline='\n') as f:
            f.write('\n## '+tid+'\n'+''.join(entries).rstrip()+'\n')
        progress=''.join(line for line in lines if line not in entries)
        utc=datetime.now(timezone.utc).isoformat(timespec='seconds')
        progress+=f'- [x] ({utc}) {tid}: {task["status"]}; commit {task["commit"]}; {task.get("note", "criterio verificado")}; detalle en archive/PROGRESS_ARCHIVE.md.\n'
text=text[:start]+progress+text[end:]
plan.write_text(text,encoding='utf-8',newline='\n')
