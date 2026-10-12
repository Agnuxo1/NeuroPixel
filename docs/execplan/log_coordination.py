"""Append a concise factual message to the own and shared coordination boards."""
from pathlib import Path
from datetime import datetime, timezone
import argparse

p=argparse.ArgumentParser()
p.add_argument('--message',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
utc=datetime.now(timezone.utc).isoformat(timespec='seconds')
entry=f'\n### {utc} — CODEX — NP-COMPLETION-20261012\nTipo: HITO / CONTINUIDAD\nMensaje: {a.message}\n'
for base in [root,Path('D:/PROJECTS/196_NeuroPixel')]:
    with (base/'coordinacion/TABLON.md').open('a',encoding='utf-8',newline='\n') as f:
        f.write(entry)
print(utc)
