"""Isolated read-only cloud verifier; does not launch or restart scientific training."""
import base64,datetime,json,os,pathlib,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1];SCI_RUN=37728384380;SCI_SOURCE='6a5af5b68aaa9173edf46658dd8f9d16845fbad6'
def publish(path,body):
    repo=os.environ['GITHUB_REPOSITORY'];branch=os.environ['GITHUB_REF_NAME'];token=os.environ['VERIFICATION_ARCHIVE_TOKEN']
    data={'message':'Archive QTRAIN read-only verification','branch':branch,'content':base64.b64encode(body).decode()}
    request=urllib.request.Request('https://api.github.com/repos/'+repo+'/contents/'+path,data=json.dumps(data).encode(),method='PUT',headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=60) as response:
        if response.status!=201:raise ValueError('verification Git receipt not created')
def main():
    plan=json.loads((ROOT/'docs/research/OPT03_QTRAIN_plan.json').read_text());base=ROOT/'results/research/OPT03_QTRAIN_recovery'/str(SCI_RUN)/'cohort';out=ROOT/'results/research/QTRAIN_readonly_verification';out.mkdir(parents=True,exist_ok=True);verified=[];missing=[]
    with (out/'collection.log').open('w',encoding='utf-8') as log:
        observer=subprocess.Popen([sys.executable,'scripts/watch_OPT03_QTRAIN.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            for conf in plan['runs']:
                name=conf['run_id']
                while not (base/name/'result.json').exists() or not (base/'case_manifests'/(name+'.json')).exists():
                    if observer.poll() is not None:break
                    time.sleep(20)
                if not (base/name/'result.json').exists():missing.append(name);continue
                p=out/(name+'.json');proc=subprocess.run([sys.executable,'scripts/replay_OPT03_QTRAIN.py','--root',str(base),'--run-id',name,'--output',str(p)],cwd=ROOT)
                if not p.exists():raise ValueError('replay ended without a factual receipt')
                record=json.loads(p.read_text());record.update(scientific_run_id=SCI_RUN,scientific_source_commit=SCI_SOURCE,verification_source_commit=os.environ['GITHUB_SHA'],verification_workflow_run_id=os.environ['GITHUB_RUN_ID'],verification_returncode=proc.returncode,independent_external_scientific_replication=False)
                body=(json.dumps(record,indent=2)+'\n').encode();p.write_bytes(body)
                archive_path=f"results/research/QTRAIN_replay_cloud/{SCI_RUN}/{os.environ['GITHUB_RUN_ID']}/{name}/replay.json";publish(archive_path,body);verified.append({'case':name,'status':record['status'],'returncode':proc.returncode,'archive_path':archive_path});print(json.dumps(verified[-1]),flush=True)
        finally:
            if observer.poll() is None:observer.terminate();observer.wait(timeout=10)
    receipt={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scientific_run_id':SCI_RUN,'scientific_source_commit':SCI_SOURCE,'verification_workflow_run_id':os.environ['GITHUB_RUN_ID'],'verification_source_commit':os.environ['GITHUB_SHA'],'verified_cases':verified,'missing':missing,'expected_cases':12,'new_trainings':0,'test_sampled':False,'complete_scientific_task3':False,'scope':'Internal read-only endpoint replay; separate executor is not independent scientific replication'}
    publish(f"results/research/QTRAIN_replay_cloud/{SCI_RUN}/{os.environ['GITHUB_RUN_ID']}/cohort_receipt.json",(json.dumps(receipt,indent=2)+'\n').encode());print(json.dumps(receipt),flush=True)
    if missing or any(x['returncode'] for x in verified):raise SystemExit(1)
if __name__=='__main__':main()
