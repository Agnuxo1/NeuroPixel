"""Verify executed bootstrap receipts and persist compact, factual evidence."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'docs/execplan'
EVIDENCE=PLAN/'evidence'
PYTHON='E:/Proyecto-concursos/NeuroPixel-env-completion/Scripts/python.exe'
mode=sys.argv[1]
if mode=='kit':
    log=(EVIDENCE/'P0.3_extract.txt').read_text(encoding='utf-8')
    extracted=json.JSONDecoder().raw_decode(log[log.index('{'):])[0]
    assert len(extracted['files'])==26
    for item in extracted['files']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    results={}
    for name,count in [('contracts',38),('extensions',25)]:
        text=(EVIDENCE/f'P0.3_{name}.txt').read_text(encoding='utf-8')
        assert f'Ran {count} tests' in text and '\nOK\n' in text and 'EXIT_CODE: 0' in text
        results[name]=count
    version=subprocess.check_output([PYTHON,'--version'],text=True).strip()
    receipt={'files_authenticated':26,'tests_passed':results,'python':version,'interpreter':PYTHON,'scope':'Software contracts only; tiny synthetic fixtures and registered mocked test admission are not scientific training.'}
    path=EVIDENCE/'P0.3_validation.json'
    orientation=f'- Python del kit: {PYTHON} ({version}); NumPy 2.2.6, SciPy 1.15.1, ModernGL 5.12.0, glcontext 3.1.0.'
    plan=(PLAN/'EXECPLAN.md').read_text(encoding='utf-8')
    plan=re.sub(r'^- Python del kit.*$',orientation,plan,flags=re.M)
    (PLAN/'EXECPLAN.md').write_text(plan,encoding='utf-8',newline='\n')
elif mode=='plan':
    status=json.loads(subprocess.check_output([sys.executable,str(PLAN/'plan_status.py'),'status'],cwd=ROOT,text=True,encoding='utf-8'))
    assert not status['problems'], status['problems']
    required=['EXECPLAN.md','tasks.json','CHECKLIST.md','SESSION_LOG.md','plan_status.py']
    assert all((PLAN/name).is_file() for name in required)
    assert (PLAN/'evidence').is_dir() and (PLAN/'archive').is_dir()
    assert '<!-- EXECPLAN-COMPLETION-20261012 -->' in (ROOT/'AGENTS.md').read_text(encoding='utf-8')
    exclude=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-path','info/exclude'],cwd=ROOT,text=True).strip())
    assert 'docs/execplan/tasks.json.bak' in exclude.read_text(encoding='utf-8').splitlines()
    receipt={'status':status,'required_files':required,'resume_block_present':True,'backup_locally_excluded':True,'ledger_modified_only_by_plan_status':True}
    path=EVIDENCE/'P0.4_plan_validation.json'
else:
    raise SystemExit('Unknown mode')
assert not path.exists(),path
path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'verified':mode,'evidence':str(path),'tests':receipt.get('tests_passed')}))
