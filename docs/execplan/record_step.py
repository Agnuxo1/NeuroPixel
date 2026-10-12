"""Update factual plan checkpoints without modifying the task ledger."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument('--task', required=True)
p.add_argument('--next', required=True)
p.add_argument('--done', required=True)
p.add_argument('--remaining', required=True)
p.add_argument('--budget', default='15 minutos')
p.add_argument('--risk', default='RAM >=8 GiB antes de carga neural; procesos ajenos intactos.')
p.add_argument('--attempt', type=int, default=1)
a = p.parse_args()
folder = Path(__file__).resolve().parent
ledger = json.loads((folder/'tasks.json').read_text(encoding='utf-8-sig'))
task = next(t for t in ledger['tasks'] if t['id']==a.task)
utc = datetime.now(timezone.utc).isoformat(timespec='seconds')
plan_path = folder/'EXECPLAN.md'
text = plan_path.read_text(encoding='utf-8')
start = text.index('- Actualizado (UTC):')
end = text.index('\n## Propósito',start)
state = f'''- Actualizado (UTC): {utc}
- Tarea en curso o siguiente: {a.task}, {task['title']}; intento {a.attempt}; presupuesto {a.budget}.
- Criterio de cierre: {task['acceptance']}
- Hecho en esta tarea: {a.done}
- Falta en esta tarea: {a.remaining}
- Siguiente comando exacto: {a.next}
- Trabajos en segundo plano: ninguno propio salvo lo anotado en Progreso.
- Dudas o riesgos abiertos: {a.risk}
'''
text=text[:start]+state+text[end:]
heading='\n## Sorpresas y hallazgos'
entry=f'- [ ] ({utc}) {a.task}, intento {a.attempt}: hecho {a.done}; falta {a.remaining}. Siguiente: `{a.next}`.\n'
text=text.replace(heading,'\n'+entry+heading,1)
plan_path.write_text(text,encoding='utf-8')
print(f'Checkpoint {a.task} {utc}')
