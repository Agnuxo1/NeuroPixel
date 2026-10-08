"""Separate read-only executor: verify published continuations, never train/restart."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = 37771588722
SOURCE = 'adab1f3dc0ded9da3c42a3e7e5adcb96cdaba67e'
spec = importlib.util.spec_from_file_location('readonly_publish', ROOT / 'neuropixel/research/qtrain_budget_transport.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def main():
    plan = json.loads((ROOT / 'docs/research/OPT03_QTRAIN_U16_plan.json').read_text(encoding='utf-8'))
    base = ROOT / 'results/research/OPT03_QTRAIN_U16_recovery' / str(RUN) / 'cohort'
    output = ROOT / 'results/research/U16_readonly_verification'
    output.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    with (output / 'collection.log').open('w', encoding='utf-8') as log:
        observer = subprocess.Popen([sys.executable, 'scripts/watch_OPT03_QTRAIN_U16.py'], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
        try:
            for config in plan['runs']:
                case = config['run_id']
                while not (base / case / 'result.json').exists() or not (base / 'case_manifests' / (case + '.json')).exists():
                    if observer.poll() is not None:
                        break
                    time.sleep(20)
                if not (base / case / 'result.json').exists():
                    missing.append(case)
                    continue
                receipt_path = output / (case + '.json')
                process = subprocess.run([sys.executable, 'scripts/replay_OPT03_QTRAIN_U16.py',
                    '--root', str(base), '--run-id', case, '--output', str(receipt_path)], cwd=ROOT)
                if not receipt_path.exists():
                    raise ValueError('Read-only replay ended without receipt')
                receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                receipt.update(scientific_run_id=RUN, scientific_source_commit=SOURCE,
                               verification_source_commit=os.environ['GITHUB_SHA'],
                               verification_run_id=os.environ['GITHUB_RUN_ID'],
                               verification_returncode=process.returncode,
                               independent_external_replication=False)
                body = (json.dumps(receipt, indent=2) + '\n').encode()
                receipt_path.write_bytes(body)
                path = f"results/research/U16_replay_cloud/{RUN}/{os.environ['GITHUB_RUN_ID']}/{case}/replay.json"
                transport.publish_immutable(path, body)
                record = {'case': case, 'status': receipt['status'], 'returncode': process.returncode,
                          'archive_path': path}
                records.append(record)
                print(json.dumps(record), flush=True)
        finally:
            if observer.poll() is None:
                observer.terminate()
                observer.wait(timeout=10)
    receipt = {'scientific_run_id': RUN, 'scientific_source_commit': SOURCE,
               'verification_run_id': os.environ['GITHUB_RUN_ID'],
               'verification_source_commit': os.environ['GITHUB_SHA'], 'verified_cases': records,
               'missing': missing, 'expected_cases': 12, 'new_updates': 0, 'test_accessed': False,
               'independent_external_replication': False, 'scientific_task3_complete': False}
    transport.publish_immutable(f"results/research/U16_replay_cloud/{RUN}/{os.environ['GITHUB_RUN_ID']}/cohort_receipt.json",
                                (json.dumps(receipt, indent=2) + '\n').encode())
    if missing or any(row['returncode'] for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
