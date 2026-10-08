"""Archive each completed case durably in its isolated Git branch, no Actions cache/artifact."""
import argparse,base64,hashlib,json,os,pathlib,urllib.error,urllib.request,zipfile
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--plan',type=pathlib.Path,required=True);a=ap.parse_args()
    token=os.environ['QTRAIN_ARCHIVE_TOKEN'];repo=os.environ['GITHUB_REPOSITORY'];branch=os.environ['GITHUB_REF_NAME'];jobrun=os.environ['GITHUB_RUN_ID'];source=os.environ['GITHUB_SHA']
    case=a.root/a.run_id;result=json.loads((case/'result.json').read_text())
    if result['status']!='completed':raise ValueError('only completed cases are archived as completed')
    plan_hash=hashlib.sha256(a.plan.read_bytes()).hexdigest()
    if result['plan_sha256']!=plan_hash:raise ValueError('case/plan hash differs')
    files=list(sorted(case.rglob('*')))+list(sorted((a.root/'datasets').glob('*')))+[a.plan]
    manifest={'source_commit':source,'workflow_run_id':jobrun,'case_run_id':a.run_id,'plan_sha256':plan_hash,'files':{str(p.relative_to(a.root)) if p.is_relative_to(a.root) else 'execution_plan.json':{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files if p.is_file()},'scope':'Development only, no final test; one completed case and shared datasets'}
    archive=a.root.parent/(a.run_id+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.is_file():z.write(p,str(p.relative_to(a.root)) if p.is_relative_to(a.root) else 'execution_plan.json')
        z.writestr('archive_manifest.json',json.dumps(manifest,indent=2)+'\n')
    raw=archive.read_bytes()
    if len(raw)>25*1024**2:raise ValueError('unexpected single-case archive size')
    receipt={'source_commit':source,'workflow_run_id':jobrun,'case_run_id':a.run_id,'plan_sha256':plan_hash,'zip_sha256':hashlib.sha256(raw).hexdigest(),'zip_bytes':len(raw),'files':len(manifest['files']),'status':'archived_completed_case_pending_independent_audit'}
    for name,body in [('raw.zip',raw),('receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())]:
        path=f'results/research/OPT03_QTRAIN_cloud/{jobrun}/{a.run_id}/{name}';url=f'https://api.github.com/repos/{repo}/contents/{path}'
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json'}
        try:
            request=urllib.request.Request(url+'?ref='+branch,headers=headers)
            with urllib.request.urlopen(request,timeout=60) as response:existing=json.load(response)
        except urllib.error.HTTPError as error:
            if error.code!=404:raise
            existing=None
        if existing is not None:
            if existing['sha']!=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest():raise ValueError('preserve existing different archive')
            continue
        payload={'branch':branch,'message':'Archive QTRAIN completed case '+a.run_id,'content':base64.b64encode(body).decode()}
        request=urllib.request.Request(url,data=json.dumps(payload).encode(),method='PUT',headers=headers)
        with urllib.request.urlopen(request,timeout=60) as response:
            if response.status!=201:raise ValueError('Git archive not created')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
