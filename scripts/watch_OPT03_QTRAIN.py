"""Read-only workflow observer and per-case exact Git recovery; never restarts training."""
import datetime,hashlib,json,pathlib,time,urllib.error,urllib.request,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];REPO='Agnuxo1/NeuroPixel';RUN=37728384380;JOB=113151663191
SOURCE='6a5af5b68aaa9173edf46658dd8f9d16845fbad6';BRANCH='research/query-supervision-development-2026-10-08';PLAN='20610e354288b48eeb7316367f61da5173fb2a124a01effa93e7d40e61bf3fb5'
DEST=ROOT/'results/research/OPT03_QTRAIN_recovery'/str(RUN);STATE=ROOT/'coord/recovery/status-audit-20261007/QTRAIN_live_observation.json'
def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'NeuroPixel-scientific-observer','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=30) as r:return r.read()
def jget(path):return json.loads(fetch('https://api.github.com/repos/'+REPO+'/'+path))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n');tmp.replace(path)
def put_immutable(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=data:raise ValueError('preserve existing different evidence '+str(path))
    if not path.exists():path.write_bytes(data)
def recover(meta_path,git_sha,tree):
    raw=fetch(f'https://raw.githubusercontent.com/{REPO}/{git_sha}/{meta_path}');receipt=json.loads(raw)
    if receipt['source_commit']!=SOURCE or receipt['workflow_run_id']!=str(RUN) or receipt['plan_sha256']!=PLAN:raise ValueError('case provenance differs')
    blob=tree[meta_path]['sha']
    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=blob:raise ValueError('receipt Git blob differs')
    folder=meta_path.rsplit('/',1)[0];case=folder.rsplit('/',1)[1];original=DEST/'originals'/case
    archive=fetch(f'https://raw.githubusercontent.com/{REPO}/{git_sha}/{folder}/raw.zip')
    if len(archive)!=receipt['zip_bytes'] or hashlib.sha256(archive).hexdigest()!=receipt['zip_sha256']:raise ValueError('case ZIP differs')
    put_immutable(original/'original.zip',archive);put_immutable(original/'receipt.json',raw)
    completed=receipt['status']=='archived_completed_case_pending_independent_audit';base=DEST/'cohort' if completed else DEST/'partial'/case
    with zipfile.ZipFile(original/'original.zip') as z:
        if sum(x.file_size for x in z.infolist())>100*1024**2:raise ValueError('unexpected ZIP expansion size')
        for item in z.infolist():
            if item.is_dir():continue
            if ((item.external_attr>>16)&0o170000)==0o120000:raise ValueError('ZIP symlink unsupported')
            relative=('case_manifests/'+case+'.json') if item.filename=='archive_manifest.json' else item.filename
            p=(base/relative).resolve();p.relative_to(base.resolve());put_immutable(p,z.read(item))
    evidence={'status':'recovered_case_pending_independent_audit','source_commit':SOURCE,'archive_git_commit':git_sha,'workflow_run_id':RUN,'case':case,'case_completed':completed,'zip_sha256':receipt['zip_sha256'],'complete_scientific_task3':False};save(original/'recovery_receipt.json',evidence);return evidence

previous=None;recovered={}
while True:
    terminal=False
    try:
        run=jget(f'actions/runs/{RUN}')
        if run['head_sha']!=SOURCE:raise ValueError('workflow source differs')
        terminal=run['status']=='completed';sha=jget('git/ref/heads/'+BRANCH)['object']['sha'];data=jget(f'git/trees/{sha}?recursive=1');tree={x['path']:x for x in data['tree'] if x['type']=='blob'}
        prefix=f'results/research/OPT03_QTRAIN_cloud/{RUN}/'
        candidates=sorted(p for p in tree if p.startswith(prefix) and p.endswith('/receipt.json'))
        for p in candidates:
            case=p.rsplit('/',2)[1]
            if case not in recovered:
                evidence=recover(p,sha,tree);recovered[case]=evidence;print(json.dumps(evidence),flush=True)
        complete=[k for k,v in recovered.items() if v['case_completed']];partial=[k for k,v in recovered.items() if not v['case_completed']]
        row={'observed_utc':utc(),'run_id':RUN,'job_id':JOB,'source_commit':SOURCE,'status':run['status'],'conclusion':run['conclusion'],'archive_git_head':sha,'complete_cases_recovered':complete,'partial_cases_recovered':partial,'expected_cases':12,'terminal_state_established':terminal,'observation_error':None,'complete_scientific_task3':False};save(STATE,row)
        key=(run['status'],run['conclusion'],len(complete),len(partial))
        if key!=previous:print(json.dumps(row),flush=True);previous=key
        if terminal:
            save(DEST/'recovery_summary.json',row);break
    except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,ValueError,KeyError,OSError) as error:
        save(STATE.with_name('QTRAIN_observation_error.json'),{'observed_utc':utc(),'run_id':RUN,'error_type':type(error).__name__,'http_status':getattr(error,'code',None),'terminal_state_established':terminal,'action':'Retry same handle/recovery; never restart a scientific training'})
    time.sleep(600)
