"""Immutable per-fit/partial archives and recovery, no Torch or training actions."""
import hashlib
import importlib.util
import json
import os
import pathlib
import zipfile

ROOT=pathlib.Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('old_immutable_transport',ROOT/'neuropixel/research/qtrain_budget_transport.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
REGISTRATION='NP-READ03-20261008'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def publish_fit(root,parent,mode,plan_path):
    folder=root/parent/mode
    completed=(folder/'result.json').exists()
    record=json.loads((folder/('result.json' if completed else 'progress.json')).read_text(encoding='utf-8'))
    update=record['config']['updates'] if completed else record['update']
    case=parent+'__'+mode
    archive_name=case if completed else case+f'_partial_u{update}'
    paths=[p for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix!='.tmp']
    files={str(p.relative_to(root)).replace('\\','/'):sha(p.read_bytes()) for p in paths}
    plan_raw=plan_path.read_bytes()
    manifest=dict(registration_id=REGISTRATION,case_run_id=case,parent_case=parent,mode=mode,
                  case_status='completed' if completed else 'partial',last_update=update,
                  plan_sha256=sha(plan_raw),files=files)
    archive=root.parent/'archives'/archive_name/'raw.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in [('archive_manifest.json',(json.dumps(manifest,indent=2)+'\n').encode()),
                         ('execution_plan.json',plan_raw)]+[(str(p.relative_to(root)).replace('\\','/'),p.read_bytes()) for p in paths]:
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,raw)
    raw=archive.read_bytes()
    if len(raw)>25_000_000:
        raise ValueError('Fit archive exceeded bounded Git transport size')
    receipt=dict(**{k:manifest[k] for k in manifest if k!='files'},zip_sha256=sha(raw),zip_bytes=len(raw),
                 source_commit=os.environ['GITHUB_SHA'],workflow_run_id=os.environ['GITHUB_RUN_ID'],
                 new_backbone_updates=0,test_accessed=False)
    prefix=f'results/research/READ03_cloud/{REGISTRATION}/{archive_name}/'
    old.publish_immutable(prefix+'raw.zip',raw)
    old.publish_immutable(prefix+'receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())


def unpack(raw,root,expected_case,expected_plan):
    import io
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):
            raise ValueError('Duplicate archive paths')
        manifest=json.loads(archive.read('archive_manifest.json'))
        if manifest['case_run_id']!=expected_case or manifest['plan_sha256']!=expected_plan or manifest['registration_id']!=REGISTRATION:
            raise ValueError('Archive recipe/case differs')
        if sha(archive.read('execution_plan.json'))!=expected_plan:
            raise ValueError('Embedded frozen plan differs')
        prefix=manifest['parent_case']+'/'+manifest['mode']+'/'
        if manifest['mode'] not in ('query_attention','uniform_global','local_query') or expected_case!=manifest['parent_case']+'__'+manifest['mode']:
            raise ValueError('Parent/mode case binding differs')
        if set(names)!=set(manifest['files'])|{'archive_manifest.json','execution_plan.json'}:
            raise ValueError('Manifest inventory differs')
        for name in manifest['files']:
            parts=pathlib.PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name or not name.startswith(prefix):
                raise ValueError('Archive path outside declared fit')
        folder=root/manifest['parent_case']/manifest['mode']
        if (folder/'result.json').exists():
            if manifest['case_status']!='completed':return manifest
            for name,digest in manifest['files'].items():
                path=root/pathlib.PurePosixPath(name)
                if not path.exists() or sha(path.read_bytes())!=digest:
                    raise ValueError('Preserve existing different closed fit')
            return manifest
        if (folder/'progress.json').exists():
            progress=json.loads((folder/'progress.json').read_text())
            if progress['plan_sha256']!=expected_plan:raise ValueError('Local partial recipe differs')
            if progress['update']>=manifest['last_update'] and manifest['case_status']!='completed':return manifest
        if sum(archive.getinfo(n).file_size for n in names)>100_000_000:
            raise ValueError('Uncompressed archive exceeds bound')
        for name,digest in manifest['files'].items():
            parts=pathlib.PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name or not name.startswith(prefix):
                raise ValueError('Archive path outside declared fit')
            info=archive.getinfo(name)
            if info.file_size>30_000_000 or (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Oversize or symlink archive member')
            data=archive.read(name)
            if sha(data)!=digest:
                raise ValueError('Archive file hash differs')
            target=root.joinpath(*parts.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
    return manifest


def recover(root,expected,plan_hash):
    repo=os.environ['GITHUB_REPOSITORY'];branch=os.environ['GITHUB_REF_NAME']
    api=f'https://api.github.com/repos/{repo}/'
    commit=json.loads(old.fetch(api+'git/ref/heads/'+branch))['object']['sha']
    tree=json.loads(old.fetch(api+f'git/trees/{commit}?recursive=1'))
    if tree.get('truncated'):raise ValueError('Truncated archive tree')
    prefix=f'results/research/READ03_cloud/{REGISTRATION}/';candidates={}
    for entry in tree['tree']:
        path=entry['path']
        if entry['type']!='blob' or not path.startswith(prefix) or not path.endswith('/receipt.json'):continue
        raw=old.fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}')
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=entry['sha']:raise ValueError('Receipt blob differs')
        receipt=json.loads(raw);case=receipt['case_run_id']
        if case not in expected or receipt['registration_id']!=REGISTRATION or receipt['plan_sha256']!=plan_hash:raise ValueError('Different registered fit')
        rank=(receipt['case_status']=='completed',receipt['last_update'])
        if case not in candidates or rank>candidates[case][0]:candidates[case]=(rank,path,receipt)
    for case,(_,path,receipt) in candidates.items():
        raw=old.fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{path.rsplit("/",1)[0]}/raw.zip')
        if len(raw)!=receipt['zip_bytes'] or sha(raw)!=receipt['zip_sha256']:raise ValueError('Fit archive differs')
        saved=root.parent/'recovered_originals'/path.rsplit('/',2)[1]/'raw.zip'
        old.immutable(saved,raw)
        unpack(raw,root,case,plan_hash)
    return list(candidates)
