"""Immutable parent recovery and per-case Git archival for one registered recipe."""
import base64
import hashlib
import json
import os
import pathlib
import urllib.error
import urllib.request
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url):
    headers = {'User-Agent': 'NeuroPixel-registered-budget-continuation'}
    token = os.environ.get('QTRAIN_U16_ARCHIVE_TOKEN')
    if token and url.startswith('https://api.github.com/'):
        headers['Authorization'] = 'Bearer ' + token
        headers['Accept'] = 'application/vnd.github+json'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return response.read()


def immutable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != body:
        raise ValueError('Preserve existing different evidence: ' + str(path))
    if not path.exists():
        path.write_bytes(body)


def unpack(archive, destination, case):
    with zipfile.ZipFile(archive) as source:
        if sum(x.file_size for x in source.infolist()) > 100 * 1024**2:
            raise ValueError('Unexpected archive expansion')
        manifest = json.loads(source.read('archive_manifest.json'))
        for item in source.infolist():
            if item.is_dir():
                continue
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Archive symlink refused')
            name = 'case_manifests/' + case + '.json' if item.filename == 'archive_manifest.json' else item.filename
            path = (destination / name).resolve()
            path.relative_to(destination.resolve())
            if item.filename != 'archive_manifest.json':
                metadata = manifest['files'].get(item.filename)
                body = source.read(item)
                if metadata is None or len(body) != metadata['bytes'] or hashlib.sha256(body).hexdigest() != metadata['sha256']:
                    raise ValueError('Archive file hash differs')
                immutable(path, body)
            else:
                immutable(path, source.read(item))
    for name, metadata in manifest['files'].items():
        path = (destination / name).resolve()
        path.relative_to(destination.resolve())
        if not path.exists() or path.stat().st_size != metadata['bytes'] or sha(path) != metadata['sha256']:
            raise ValueError('Incomplete extracted manifest')
    return manifest


def recover_parent(inventory, descriptor, destination):
    case = descriptor['parent_case']
    manifest_path = destination / 'case_manifests' / (case + '.json')
    if not manifest_path.exists():
        archive = destination.parent / 'parent_originals' / case / 'original.zip'
        if archive.exists():
            body = archive.read_bytes()
        else:
            body = fetch('https://raw.githubusercontent.com/Agnuxo1/NeuroPixel/'
                         + inventory['parent_archive_git_commit'] + '/' + descriptor['remote_zip_path'])
        if len(body) != descriptor['zip_bytes'] or hashlib.sha256(body).hexdigest() != descriptor['zip_sha256']:
            raise ValueError('Frozen parent ZIP differs')
        immutable(archive, body)
        manifest = unpack(archive, destination, case)
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['source_commit'] != inventory['parent_source_commit']
            or str(manifest['workflow_run_id']) != str(inventory['parent_scientific_run'])
            or manifest['case_run_id'] != case or manifest['case_status'] != 'completed'
            or manifest['plan_sha256'] != inventory['parent_plan_sha256']):
        raise ValueError('Parent provenance differs')
    for suffix, digest in [('checkpoint.pt', descriptor['checkpoint_sha256']),
                           ('result.json', descriptor['result_sha256'])]:
        if sha(destination / case / suffix) != digest:
            raise ValueError('Immutable parent input differs')
    return destination / case / 'checkpoint.pt'


def publish_case(root, case, plan_path, registration, allow_partial=False):
    folder = root / case
    completed = (folder / 'result.json').exists()
    if not completed and not allow_partial:
        raise ValueError('Explicit partial archival required')
    record = json.loads((folder / ('result.json' if completed else 'progress.json')).read_text(encoding='utf-8'))
    if record['plan_sha256'] != sha(plan_path):
        raise ValueError('Archive plan differs')
    update = record['last_update'] if completed else record['completed_updates']
    files = [p for p in sorted(folder.rglob('*')) if p.is_file()]
    files += [p for p in sorted((root / 'datasets').glob('*')) if p.is_file()] + [plan_path]
    metadata = {str(p.relative_to(root)) if p.is_relative_to(root) else 'execution_plan.json':
                {'sha256': sha(p), 'bytes': p.stat().st_size} for p in files}
    manifest = {'registration_id': registration, 'source_commit': os.environ['GITHUB_SHA'],
                'workflow_run_id': os.environ['GITHUB_RUN_ID'], 'case_run_id': case,
                'parent_case': record['parent_case'], 'plan_sha256': sha(plan_path),
                'case_status': 'completed' if completed else 'partial_not_completed',
                'last_update': update, 'files': metadata}
    suffix = '' if completed else '_partial_u' + str(update)
    archive = root.parent / (case + suffix + '.zip')
    if archive.exists():
        with zipfile.ZipFile(archive) as old:
            if json.loads(old.read('archive_manifest.json')) != manifest:
                raise ValueError('Preserve different existing archive')
    else:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
            for path in files:
                output.write(path, str(path.relative_to(root)) if path.is_relative_to(root) else 'execution_plan.json')
            output.writestr('archive_manifest.json', json.dumps(manifest, indent=2) + '\n')
    body = archive.read_bytes()
    if len(body) > 25 * 1024**2:
        raise ValueError('Unexpected case archive size')
    receipt = {key: manifest[key] for key in ('registration_id', 'source_commit', 'workflow_run_id',
                                             'case_run_id', 'parent_case', 'plan_sha256', 'case_status', 'last_update')}
    receipt.update(zip_sha256=hashlib.sha256(body).hexdigest(), zip_bytes=len(body),
                   test_accessed=False, scientific_task3_complete=False)
    base = f'results/research/OPT03_QTRAIN_U16_cloud/{registration}/{case}{suffix}/'
    for name, value in [('raw.zip', body), ('receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())]:
        publish_immutable(base + name, value)
    return receipt


def publish_immutable(path, body):
    repository = os.environ['GITHUB_REPOSITORY']
    branch = os.environ['GITHUB_REF_NAME']
    token = os.environ['QTRAIN_U16_ARCHIVE_TOKEN']
    url = f'https://api.github.com/repos/{repository}/contents/{path}'
    try:
        existing = json.loads(fetch(url + '?ref=' + branch))
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        existing = None
    blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
    if existing is not None:
        if existing['sha'] != blob:
            raise ValueError('Preserve existing different published evidence')
        return
    payload = {'branch': branch, 'message': 'Archive registered QTRAIN-U16 evidence',
               'content': base64.b64encode(body).decode()}
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
               'Content-Type': 'application/json'}
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method='PUT', headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 201:
            raise ValueError('Git archive not created')


def recover_existing(root, cases, registration, expected_plan):
    """Recover completed or latest durable partial extensions before any update."""
    repository = os.environ['GITHUB_REPOSITORY']
    branch = os.environ['GITHUB_REF_NAME']
    api = f'https://api.github.com/repos/{repository}/'
    commit = json.loads(fetch(api + 'git/ref/heads/' + branch))['object']['sha']
    tree = json.loads(fetch(api + f'git/trees/{commit}?recursive=1'))
    if tree.get('truncated'):
        raise ValueError('Incomplete recovery tree')
    entries = {x['path']: x for x in tree['tree'] if x['type'] == 'blob'}
    prefix = f'results/research/OPT03_QTRAIN_U16_cloud/{registration}/'
    candidates = {}
    for path in entries:
        if not path.startswith(prefix) or not path.endswith('/receipt.json'):
            continue
        raw = fetch(f'https://raw.githubusercontent.com/{repository}/{commit}/{path}')
        if hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != entries[path]['sha']:
            raise ValueError('Recovery receipt Git blob differs')
        record = json.loads(raw)
        if record['registration_id'] != registration or record['plan_sha256'] != expected_plan or record['case_run_id'] not in cases:
            raise ValueError('Different recipe in recovery namespace')
        case = record['case_run_id']
        rank = (record['case_status'] == 'completed', record['last_update'])
        if case not in candidates or rank > candidates[case][0]:
            candidates[case] = (rank, path, record)
    recovered = []
    for case, (_, path, record) in candidates.items():
        archive_path = path.rsplit('/', 1)[0] + '/raw.zip'
        body = fetch(f'https://raw.githubusercontent.com/{repository}/{commit}/{archive_path}')
        if len(body) != record['zip_bytes'] or hashlib.sha256(body).hexdigest() != record['zip_sha256']:
            raise ValueError('Recovered extension archive differs')
        archive = root.parent / 'recovered_originals' / path.rsplit('/', 2)[1] / 'original.zip'
        immutable(archive, body)
        manifest = unpack(archive, root, case)
        if manifest['plan_sha256'] != expected_plan or manifest['registration_id'] != registration:
            raise ValueError('Recovered manifest differs')
        recovered.append({'case': case, 'status': record['case_status'], 'update': record['last_update'],
                          'archive_git_commit': commit, 'zip_sha256': record['zip_sha256']})
    return recovered
