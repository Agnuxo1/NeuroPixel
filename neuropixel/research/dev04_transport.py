"""Immutable fresh-body/head case archives, excluding regenerable latent caches."""
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import urllib.error
import urllib.request
import zipfile

ROOT=pathlib.Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('old_archive',ROOT/'neuropixel/research/qtrain_budget_transport.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
REG='NP-DEV04-20261008'


def select_case_archive_branch(case):
    """Per-case evidence refs avoid concurrent Git Contents HEAD collisions."""
    if not case.startswith('DEV04_partition') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in case):raise ValueError('Invalid archive case')
    repository=os.environ['GITHUB_REPOSITORY'];source=os.environ['GITHUB_SHA']
    branch='research/dev04-evidence-2026-10-08/'+case
    api=f'https://api.github.com/repos/{repository}/'
    try:json.loads(old.fetch(api+'git/ref/heads/'+branch))
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        body=json.dumps({'ref':'refs/heads/'+branch,'sha':source}).encode()
        token=os.environ['QTRAIN_U16_ARCHIVE_TOKEN']
        request=urllib.request.Request(api+'git/refs',data=body,method='POST',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(request,timeout=60) as response:
            if response.status!=201:raise ValueError('Independent evidence ref not created')
    os.environ['GITHUB_REF_NAME']=branch
    return branch


def digest(raw):return hashlib.sha256(raw).hexdigest()


def make_archive(root,case,plan_path):
    folder=root/case;done=(folder/'result.json').exists()
    if done:stage,update,rank='complete',8192,3
    else:
        stage,update,rank='body',0,0
        for index,name in enumerate(('body','query_attention','local_query')):
            p=folder/name/'progress.json'
            if p.exists():
                record=json.loads(p.read_text(encoding='utf-8'));stage,update,rank=name,record['update'],index
    paths=[p for p in sorted(folder.rglob('*')) if p.is_file() and 'cache' not in p.relative_to(folder).parts and p.suffix!='.tmp']
    if not paths:raise ValueError('No durable case evidence')
    raw_plan=plan_path.read_bytes();plan=json.loads(raw_plan)
    if plan.get('registration_id')!=REG or plan.get('status')!='frozen_before_scientific_training':raise ValueError('Frozen DEV04 plan required for archival')
    conf=next(row for row in plan['runs'] if row['run_id']==case)
    files={p.relative_to(root).as_posix():digest(p.read_bytes()) for p in paths}
    manifest=dict(registration_id=REG,case_run_id=case,config=conf,plan_sha256=digest(raw_plan),
        case_status='completed' if done else 'partial',stage=stage,update=update,stage_rank=rank,files=files)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in [('archive_manifest.json',(json.dumps(manifest,indent=2)+'\n').encode()),('execution_plan.json',raw_plan)]+[(p.relative_to(root).as_posix(),p.read_bytes()) for p in paths]:
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
    raw=buffer.getvalue()
    if len(raw)>25_000_000:raise ValueError('Registered case archive exceeds bounded Git transport')
    receipt=dict(**{k:v for k,v in manifest.items() if k!='files'},zip_sha256=digest(raw),zip_bytes=len(raw),
        source_commit=os.environ.get('GITHUB_SHA','fixture'),workflow_run_id=os.environ.get('GITHUB_RUN_ID','fixture'),
        historical_checkpoint_loaded=False,new_body_initialization=True,test_scored=False,archive_branch=os.environ.get('GITHUB_REF_NAME','fixture'))
    return raw,receipt


def publish_case(root,case,plan_path):
    raw,receipt=make_archive(root,case,plan_path)
    suffix='' if receipt['case_status']=='completed' else '_'+receipt['stage']+'_u'+str(receipt['update'])
    prefix=f'results/research/DEV04_cloud/{REG}/{case}{suffix}/'
    old.publish_immutable(prefix+'raw.zip',raw)
    old.publish_immutable(prefix+'receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())


def unpack(raw,root,case,plan_hash):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist();manifest=json.loads(z.read('archive_manifest.json'))
        if (len(names)!=len(set(names)) or manifest['case_run_id']!=case or manifest['registration_id']!=REG
            or manifest['plan_sha256']!=plan_hash or digest(z.read('execution_plan.json'))!=plan_hash):raise ValueError('Archive case/recipe/paths differ')
        if set(names)!=set(manifest['files'])|{'archive_manifest.json','execution_plan.json'}:raise ValueError('Archive inventory differs')
        if sum(i.file_size for i in z.infolist())>150_000_000:raise ValueError('Archive expansion exceeds bound')
        folder=root/case;closed=(folder/'result.json').exists()
        # Validate every member before writing any file or accepting a closed result.
        contents=[]
        for name,expected in manifest['files'].items():
            parts=pathlib.PurePosixPath(name);info=z.getinfo(name)
            if (parts.is_absolute() or '..' in parts.parts or '\\' in name or not name.startswith(case+'/')
                or 'cache' in parts.parts or info.file_size>30_000_000 or (info.external_attr>>16)&0o170000==0o120000):raise ValueError('Unsafe archive member')
            body=z.read(name)
            if digest(body)!=expected:raise ValueError('Archive member hash differs')
            target=root.joinpath(*parts.parts)
            if closed and manifest['case_status']=='completed' and (not target.exists() or digest(target.read_bytes())!=expected):raise ValueError('Preserve different closed DEV04 case')
            contents.append((target,body))
        if closed:return manifest
        local_rank=(-1,-1)
        for index,stage in enumerate(('body','query_attention','local_query')):
            progress=folder/stage/'progress.json'
            if progress.exists():
                prior=json.loads(progress.read_text(encoding='utf-8'))
                if prior['plan_sha256']!=plan_hash:raise ValueError('Local DEV04 partial recipe differs')
                local_rank=max(local_rank,(index,prior['update']))
        if manifest['case_status']!='completed' and local_rank>=(manifest['stage_rank'],manifest['update']):return manifest
        for target,body in contents:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
        return manifest


def recover(root,expected,plan_hash):
    repo=os.environ['GITHUB_REPOSITORY'];branch=os.environ['GITHUB_REF_NAME'];api=f'https://api.github.com/repos/{repo}/'
    commit=json.loads(old.fetch(api+'git/ref/heads/'+branch))['object']['sha']
    tree=json.loads(old.fetch(api+f'git/trees/{commit}?recursive=1'))
    if tree.get('truncated'):raise ValueError('Incomplete DEV04 recovery tree')
    candidates={};prefix=f'results/research/DEV04_cloud/{REG}/'
    for row in tree['tree']:
        path=row['path']
        if row['type']!='blob' or not path.startswith(prefix) or not path.endswith('/receipt.json'):continue
        raw=old.fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}')
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=row['sha']:raise ValueError('Receipt Git blob differs')
        receipt=json.loads(raw);case=receipt['case_run_id']
        if case not in expected or receipt['plan_sha256']!=plan_hash or receipt['registration_id']!=REG:raise ValueError('Different DEV04 recipe/case')
        rank=(receipt['case_status']=='completed',receipt['stage_rank'],receipt['update'])
        if case not in candidates or rank>candidates[case][0]:candidates[case]=(rank,path,receipt)
    for case,(_,path,receipt) in candidates.items():
        raw=old.fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{path.rsplit("/",1)[0]}/raw.zip')
        if len(raw)!=receipt['zip_bytes'] or digest(raw)!=receipt['zip_sha256']:raise ValueError('Recovered DEV04 ZIP differs')
        saved=root.parent/'recovered_originals'/path.rsplit('/',2)[1]/'original.zip';old.immutable(saved,raw)
        unpack(raw,root,case,plan_hash)
    return list(candidates)
