"""Read-only live observer; recover immutable complete/partial archives, never restart."""
import datetime
import hashlib
import importlib.util
import json
import pathlib
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = 'Agnuxo1/NeuroPixel'
RUN = 37771588722
SOURCE = 'adab1f3dc0ded9da3c42a3e7e5adcb96cdaba67e'
BRANCH = 'research/query-budget-continuation-2026-10-08'
REGISTRATION = 'NP-OPT03-QTRAIN-U16-20261008'
PLAN = 'a4093bb8d559404b9f68587b753f8cea180a1e75d33363a04c1f2c7ebcb22dc0'
DEST = ROOT / 'results/research/OPT03_QTRAIN_U16_recovery' / str(RUN)
STATE = ROOT / 'coord/recovery/status-audit-20261007/U16_live_observation.json'
spec = importlib.util.spec_from_file_location('readonly_archive_transport', ROOT / 'neuropixel/research/qtrain_budget_transport.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
expected = {x['run_id'] for x in json.loads((ROOT / 'docs/research/OPT03_QTRAIN_U16_plan.json').read_text(encoding='utf-8'))['runs']}


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'NeuroPixel-U16-readonly-observer',
                                                  'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def api(path):
    return json.loads(get('https://api.github.com/repos/' + REPO + '/' + path))


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
    temp.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


previous, recovered = None, {}
while True:
    terminal = False
    try:
        run = api(f'actions/runs/{RUN}')
        if run['head_sha'] != SOURCE:
            raise ValueError('Scientific workflow source differs')
        terminal = run['status'] == 'completed'
        commit = api('git/ref/heads/' + BRANCH)['object']['sha']
        data = api(f'git/trees/{commit}?recursive=1')
        if data.get('truncated'):
            raise ValueError('Archive tree truncated')
        tree = {x['path']: x for x in data['tree'] if x['type'] == 'blob'}
        prefix = f'results/research/OPT03_QTRAIN_U16_cloud/{REGISTRATION}/'
        for path in sorted(p for p in tree if p.startswith(prefix) and p.endswith('/receipt.json')):
            folder = path.rsplit('/', 2)[1]
            if folder in recovered:
                continue
            raw = get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}')
            if hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != tree[path]['sha']:
                raise ValueError('Receipt Git blob differs')
            record = json.loads(raw)
            if (record['registration_id'] != REGISTRATION or record['plan_sha256'] != PLAN
                    or record['case_run_id'] not in expected or record['source_commit'] != SOURCE
                    or str(record['workflow_run_id']) != str(RUN)):
                raise ValueError('Different scientific recipe or attempt')
            zip_path = path.rsplit('/', 1)[0] + '/raw.zip'
            body = get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{zip_path}')
            if len(body) != record['zip_bytes'] or hashlib.sha256(body).hexdigest() != record['zip_sha256']:
                raise ValueError('Archive ZIP differs')
            original = DEST / 'originals' / folder
            transport.immutable(original / 'original.zip', body)
            transport.immutable(original / 'receipt.json', raw)
            completed = record['case_status'] == 'completed'
            destination = DEST / 'cohort' if completed else DEST / 'partials' / folder
            manifest = transport.unpack(original / 'original.zip', destination, record['case_run_id'])
            if manifest['plan_sha256'] != PLAN or manifest['case_status'] != record['case_status']:
                raise ValueError('Extracted manifest identity differs')
            evidence = {'case': record['case_run_id'], 'archive_folder': folder,
                        'status': record['case_status'], 'last_update': record['last_update'],
                        'archive_git_commit': commit, 'zip_sha256': record['zip_sha256'],
                        'complete_scientific_task3': False}
            save(original / 'recovery_receipt.json', evidence)
            recovered[folder] = evidence
            print(json.dumps(evidence), flush=True)
        complete = sorted({x['case'] for x in recovered.values() if x['status'] == 'completed'})
        partial_steps = {case: max((x['last_update'] for x in recovered.values() if x['case'] == case), default=8192)
                         for case in expected}
        row = {'observed_utc': utc(), 'run_id': RUN, 'source_commit': SOURCE,
               'status': run['status'], 'conclusion': run['conclusion'], 'archive_git_head': commit,
               'complete_cases_recovered': complete, 'latest_durable_steps': partial_steps,
               'expected_cases': 12, 'terminal_state_established': terminal,
               'observation_error': None, 'complete_scientific_task3': False}
        save(STATE, row)
        key = (run['status'], run['conclusion'], len(complete), tuple(sorted(partial_steps.items())))
        if key != previous:
            print(json.dumps(row), flush=True)
            previous = key
        if terminal:
            save(DEST / 'recovery_summary.json', row)
            break
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as error:
        save(STATE.with_name('U16_observation_error.json'), {'utc': utc(), 'run_id': RUN,
             'error_type': type(error).__name__, 'http_status': getattr(error, 'code', None),
             'terminal_state_established': terminal, 'action': 'Retry same handle; never restart training from polling failure'})
    time.sleep(600)
