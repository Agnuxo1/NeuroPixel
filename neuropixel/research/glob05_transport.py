"""Bounded immutable GLOB05 archives; original DEV04 archives are read-only parents."""
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import zipfile

ROOT=pathlib.Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('glob05_old_transport',ROOT/'neuropixel/research/qtrain_budget_transport.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
REG='NP-GLOB05-20261009'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def blob(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()

def select_case_archive_branch(case):
    if case not in {f'DEV04_partition{s}_seed{400+4*j+k}' for j,s in enumerate((101,102,103)) for k in range(4)}:raise ValueError('Unregistered parent case')
    branch='research/glob05-evidence-2026-10-09/'+case
    # Branches are explicitly pre-created through the connector before allocating runners.
    record=json.loads(old.fetch(f'https://api.github.com/repos/{os.environ["GITHUB_REPOSITORY"]}/git/ref/heads/'+branch))
    if record['ref']!='refs/heads/'+branch:raise ValueError('Evidence branch identity differs')
    os.environ['GITHUB_REF_NAME']=branch;return branch

def recover_parent(descriptor,destination):
    case=descriptor['case'];commit=descriptor['archive_git_commit'];repo='Agnuxo1/NeuroPixel'
    original=destination.parent/'originals'/case;path=original/'original.zip'
    raw=path.read_bytes() if path.exists() else old.fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{descriptor["remote_zip_path"]}')
    if len(raw)!=descriptor['original_zip_bytes'] or digest(raw)!=descriptor['original_zip_sha256']:raise ValueError('Original DEV04 parent ZIP differs')
    old.immutable(path,raw)
    from neuropixel.research import dev04_transport as parent_archive
    parent_plan='2484f3c5917e474ca3755bcdbbaca7d322077615ba47d51653116042d980f20a'
    manifest=parent_archive.unpack(raw,destination,case,parent_plan)
    folder=destination/case
    if manifest['case_status']!='completed' or manifest['config']!=descriptor['config']:raise ValueError('Closed registered parent required')
    for p,expected in [(folder/'body/checkpoint.pt',descriptor['body_checkpoint_sha256']),
        (folder/'query_attention/result.json',descriptor['closed_attention_result_sha256']),
        (folder/'local_query/result.json',descriptor['closed_local_result_sha256'])]+[(folder/'datasets'/f'{n}.npz',h) for n,h in descriptor['dataset_sha256'].items()]:
        if digest(p.read_bytes())!=expected:raise ValueError('Immutable parent input differs '+str(p))
    return folder

def make_archive(root,case,plan_path):
    folder=root/case;done=(folder/'result.json').exists();record=json.loads((folder/('result.json' if done else 'progress.json')).read_bytes())
    raw_plan=plan_path.read_bytes();plan=json.loads(raw_plan)
    if plan['registration_id']!=REG or plan['status']!='frozen_before_scientific_training' or record['plan_sha256']!=digest(raw_plan):raise ValueError('Exact frozen GLOB05 recipe required')
    paths=[p for p in sorted(folder.rglob('*')) if p.is_file() and 'cache' not in p.relative_to(folder).parts and p.suffix!='.tmp']
    files={p.relative_to(root).as_posix():digest(p.read_bytes()) for p in paths}
    manifest=dict(registration_id=REG,case_run_id=case,config=record['config'],parent=record['parent'],plan_sha256=digest(raw_plan),
        case_status='completed' if done else 'partial',update=record['config']['updates'] if done else record['update'],files=files,
        new_body_initializations=0,new_backbone_updates=0,closed_head_trainings_repeated=0,test_accessed=False)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in [('archive_manifest.json',(json.dumps(manifest,indent=2)+'\n').encode()),('execution_plan.json',raw_plan)]+[(p.relative_to(root).as_posix(),p.read_bytes()) for p in paths]:
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
    raw=buffer.getvalue()
    if len(raw)>25_000_000:raise ValueError('Bounded head archive exceeded')
    receipt={k:v for k,v in manifest.items() if k!='files'}
    receipt.update(zip_sha256=digest(raw),zip_bytes=len(raw),source_commit=os.environ.get('GITHUB_SHA','fixture'),
        workflow_run_id=os.environ.get('GITHUB_RUN_ID','fixture'),archive_branch=os.environ.get('GITHUB_REF_NAME','fixture'))
    return raw,receipt

def publish_case(root,case,plan_path):
    raw,receipt=make_archive(root,case,plan_path);suffix='' if receipt['case_status']=='completed' else '_u'+str(receipt['update'])
    prefix=f'results/research/GLOB05_cloud/{REG}/{case}{suffix}/'
    old.publish_immutable(prefix+'raw.zip',raw);old.publish_immutable(prefix+'receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())

def unpack(raw,root,case,plan_hash):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist();manifest=json.loads(z.read('archive_manifest.json'))
        if (len(names)!=len(set(names)) or manifest['registration_id']!=REG or manifest['case_run_id']!=case
            or manifest['plan_sha256']!=plan_hash or digest(z.read('execution_plan.json'))!=plan_hash):raise ValueError('Archive identity/recipe differs')
        if set(names)!=set(manifest['files'])|{'archive_manifest.json','execution_plan.json'}:raise ValueError('Complete archive inventory required')
        if sum(i.file_size for i in z.infolist())>100_000_000:raise ValueError('Bounded archive expansion exceeded')
        contents=[]
        for name,expected in manifest['files'].items():
            parts=pathlib.PurePosixPath(name);info=z.getinfo(name)
            if (parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name or not name.startswith(case+'/')
                or 'cache' in parts.parts or info.file_size>30_000_000 or (info.external_attr>>16)&0o170000==0o120000):raise ValueError('Unsafe archive member')
            target=root.joinpath(*parts.parts);target.resolve().relative_to(root.resolve());body=z.read(name)
            if digest(body)!=expected:raise ValueError('Archive member changed')
            contents.append((target,body))
        folder=root/case;closed=(folder/'result.json').exists()
        if closed:
            if manifest['case_status']=='completed' and any(not p.exists() or digest(p.read_bytes())!=manifest['files'][p.relative_to(root).as_posix()] for p,_ in contents):raise ValueError('Preserve different closed control')
            return manifest
        progress=folder/'progress.json'
        if progress.exists():
            prior=json.loads(progress.read_bytes())
            if prior['plan_sha256']!=plan_hash:raise ValueError('Local partial recipe differs')
            if manifest['case_status']!='completed' and prior['update']>=manifest['update']:return manifest
        for target,body in contents:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
    return manifest

def recover(root,case,plan_hash):
    repo=os.environ['GITHUB_REPOSITORY'];branch=os.environ['GITHUB_REF_NAME'];api=f'https://api.github.com/repos/{repo}/'
    commit=json.loads(old.fetch(api+'git/ref/heads/'+branch))['object']['sha'];tree=json.loads(old.fetch(api+'git/trees/'+commit+'?recursive=1'))
    if tree.get('truncated'):raise ValueError('Incomplete recovery tree')
    prefix=f'results/research/GLOB05_cloud/{REG}/';entries={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    candidates=[p for p in entries if p.startswith(prefix+case) and p.endswith('/receipt.json')]
    complete=[p for p in candidates if p==prefix+case+'/receipt.json'];rows=[]
    for path in complete or candidates:
        raw=old.fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}')
        if blob(raw)!=entries[path]['sha']:raise ValueError('Pinned receipt blob differs')
        r=json.loads(raw)
        if r['case_run_id']!=case or r['plan_sha256']!=plan_hash or r['registration_id']!=REG:raise ValueError('Different case/recipe in evidence namespace')
        rows.append(((r['case_status']=='completed',r['update']),path,r,raw))
    if not rows:return False
    _,path,r,receipt_raw=max(rows,key=lambda row:row[0]);zpath=path.rsplit('/',1)[0]+'/raw.zip';raw=old.fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{zpath}')
    if len(raw)!=r['zip_bytes'] or digest(raw)!=r['zip_sha256'] or blob(raw)!=entries[zpath]['sha']:raise ValueError('Original archive size/hash/blob differs')
    original=root.parent/'recovered_originals'/path.rsplit('/',2)[1];old.immutable(original/'original.zip',raw);old.immutable(original/'receipt.json',receipt_raw)
    unpack(raw,root,case,plan_hash);return True
