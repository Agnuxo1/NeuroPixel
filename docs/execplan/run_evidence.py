"""Run a bounded command and preserve output and exit status inside the plan."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--timeout',type=int,default=300)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
root=Path(__file__).resolve().parents[2]
out=(root/a.out).resolve()
assert out.is_relative_to(root) and not out.exists(), str(out)
env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TEMP='D:/PROJECTS/.cognition/neuropixel-completion-20261012/tmp',TMP='D:/PROJECTS/.cognition/neuropixel-completion-20261012/tmp')
Path(env['TEMP']).mkdir(parents=True,exist_ok=True)
started=datetime.now(timezone.utc).isoformat()
with out.open('w',encoding='utf-8') as f:
    f.write('UTC: '+started+'\nCOMMAND: '+subprocess.list2cmdline(command)+'\n\n')
    f.flush()
    try:
        result=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
        code=result.returncode
    except subprocess.TimeoutExpired:
        code=124
        f.write('\nTIMEOUT\n')
    f.write(f'\nEXIT_CODE: {code}\n')
print(json.dumps({'output':str(out),'exit_code':code}))
raise SystemExit(code)
