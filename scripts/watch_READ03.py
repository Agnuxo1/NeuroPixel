"""Recover immutable readout partials/results from the exact live job; no restart."""
import hashlib
import importlib.util
import json
import pathlib
import time
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
RUN=37831051538
SOURCE='a19c0000b4175984c14cd4eaf08392ceffd9e917'
PLAN='d273b8284d9a387b49266dc4ec62cfad76aefaa11c5f82c7ab6a97ba0e431ea0'
REPO='Agnuxo1/NeuroPixel'
BRANCH='research/frozen-readout-development-2026-10-08'
REG='NP-READ03-20261008'
DEST=ROOT/f'results/research/READ03_recovery/{RUN}'
STATE=ROOT/'coord/recovery/status-audit-20261007/READ03_live_observation.json'
spec=importlib.util.spec_from_file_location('readout_unpacker',ROOT/'neuropixel/research/readout_transport.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
expected={x['run_id'] for x in json.loads((ROOT/'docs/research/READ03_execution_plan.json').read_text())['runs']}


def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'NeuroPixel-exact-readout-observer','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=30) as r:return r.read()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n');tmp.replace(path)


def main():
    recovered=set();complete=set()
    api=f'https://api.github.com/repos/{REPO}/'
    while True:
        terminal=False
        try:
            run=json.loads(get(api+f'actions/runs/{RUN}'))
            if run['head_sha']!=SOURCE:raise ValueError('Exact scientific workflow source differs')
            terminal=run['status']=='completed'
            commit=json.loads(get(api+'git/ref/heads/'+BRANCH))['object']['sha']
            tree=json.loads(get(api+f'git/trees/{commit}?recursive=1'))
            if tree.get('truncated'):raise ValueError('Archive tree incomplete')
            prefix=f'results/research/READ03_cloud/{REG}/'
            entries={e['path']:e for e in tree['tree'] if e['type']=='blob'}
            for path in sorted(p for p in entries if p.startswith(prefix) and p.endswith('/receipt.json')):
                folder=path.rsplit('/',2)[1]
                if folder in recovered:continue
                raw=get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}')
                if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=entries[path]['sha']:raise ValueError('Receipt Git blob differs')
                receipt=json.loads(raw);case=receipt['case_run_id']
                if (case not in expected or receipt['source_commit']!=SOURCE or str(receipt['workflow_run_id'])!=str(RUN)
                    or receipt['plan_sha256']!=PLAN or receipt['registration_id']!=REG):raise ValueError('Archive belongs to another fit/recipe/attempt')
                zipped=get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path.rsplit("/",1)[0]}/raw.zip')
                if len(zipped)!=receipt['zip_bytes'] or hashlib.sha256(zipped).hexdigest()!=receipt['zip_sha256']:raise ValueError('ZIP hash/size differs')
                original=DEST/'originals'/folder;original.mkdir(parents=True,exist_ok=True)
                transport.old.immutable(original/'original.zip',zipped);transport.old.immutable(original/'receipt.json',raw)
                target=DEST/'cohort' if receipt['case_status']=='completed' else DEST/'partials'/folder
                manifest=transport.unpack(zipped,target,case,PLAN)
                save(original/'recovery_receipt.json',dict(archive_git_commit=commit,receipt_git_blob=entries[path]['sha'],
                    zip_sha256=receipt['zip_sha256'],case=case,last_update=receipt['last_update'],case_status=receipt['case_status']))
                recovered.add(folder)
                if receipt['case_status']=='completed':complete.add(case)
                print(json.dumps(dict(case=case,status=receipt['case_status'],update=receipt['last_update'],
                                      complete_fits_recovered=len(complete),scientific_task3_complete=False)),flush=True)
            save(STATE,dict(run_id=RUN,source_commit=SOURCE,status=run['status'],conclusion=run['conclusion'],
                 archive_git_commit=commit,complete_fits_recovered=sorted(complete),expected_fits=36,
                 terminal_state_established=terminal,experiment_restarted=False,scientific_task3_complete=False))
        except Exception as error:
            save(STATE.with_name('READ03_observation_error.json'),dict(error_type=type(error).__name__,
                 terminal_state_established=False,experiment_restarted=False))
            terminal=False
        if terminal:return
        time.sleep(300)


if __name__=='__main__':main()
