"""Separate immutable read-only executor; never starts scientific training."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
SCIENCE_RUN=37831051538
SCIENCE_SOURCE='a19c0000b4175984c14cd4eaf08392ceffd9e917'
PLAN_SHA='d273b8284d9a387b49266dc4ec62cfad76aefaa11c5f82c7ab6a97ba0e431ea0'
spec=importlib.util.spec_from_file_location('archive',ROOT/'neuropixel/research/qtrain_budget_transport.py')
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)


def main():
    observer_path=ROOT/'scripts/watch_READ03.py'
    if not observer_path.is_file():raise ValueError('Required exact-run recovery observer missing; no replay started')
    plan=json.loads((ROOT/'docs/research/READ03_execution_plan.json').read_text(encoding='utf-8'))
    parents=json.loads((ROOT/'docs/research/READ03_parent_inventory.json').read_text(encoding='utf-8'))
    archive.recover_existing(ROOT/'results/research/OPT03_QTRAIN_U16_recovery/37771588722/cohort',
        {row['parent_case'] for row in parents['parents']},'NP-OPT03-QTRAIN-U16-20261008',parents['parent_plan_sha256'])
    base=ROOT/f'results/research/READ03_recovery/{SCIENCE_RUN}/cohort'
    out=ROOT/'results/research/READ03_readonly_verification';out.mkdir(parents=True,exist_ok=True)
    records,missing=[],[]
    with (out/'observer.log').open('w',encoding='utf-8') as log:
        observer=subprocess.Popen([sys.executable,str(observer_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            for conf in plan['runs']:
                case=conf['run_id'];folder=base/conf['parent_case']/conf['mode']
                while not (folder/'result.json').exists():
                    if observer.poll() is not None:break
                    time.sleep(20)
                if not (folder/'result.json').exists():missing.append(case);continue
                receipt_path=out/(case+'.json')
                process=subprocess.run([sys.executable,'scripts/replay_READ03.py','--fit-id',case,'--root',str(base),'--output',str(receipt_path)],cwd=ROOT)
                if not receipt_path.exists():
                    receipt=dict(status='replay_failed_before_receipt',cases=[case],checks=0,issues=['Replay terminated before complete audit receipt'],
                        new_training_updates=0,new_backbone_updates=0,test_accessed=False,independent_external_replication=False,scientific_task3_complete=False)
                else:receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
                receipt.update(scientific_run_id=SCIENCE_RUN,scientific_source_commit=SCIENCE_SOURCE,
                    verification_run_id=os.environ['GITHUB_RUN_ID'],verification_source_commit=os.environ['GITHUB_SHA'],
                    verification_returncode=process.returncode,independent_external_replication=False)
                raw=(json.dumps(receipt,indent=2)+'\n').encode();receipt_path.write_bytes(raw)
                path=f'results/research/READ03_replay_cloud/{SCIENCE_RUN}/{os.environ["GITHUB_RUN_ID"]}/{case}/replay.json'
                archive.publish_immutable(path,raw)
                records.append(dict(case=case,status=receipt['status'],returncode=process.returncode,archive_path=path))
                print(json.dumps(records[-1]),flush=True)
                # An implementation/verification fault is evidence; do not blanket-repeat all cases.
                if process.returncode:break
        finally:
            if observer.poll() is None:observer.terminate();observer.wait(timeout=10)
    available={r['case'] for r in records}
    missing=sorted({r['run_id'] for r in plan['runs']}-available)
    receipt=dict(scientific_run_id=SCIENCE_RUN,scientific_source_commit=SCIENCE_SOURCE,plan_sha256=PLAN_SHA,
        verification_run_id=os.environ['GITHUB_RUN_ID'],verification_source_commit=os.environ['GITHUB_SHA'],
        verified_fits=records,missing=missing,expected_fits=36,new_training_updates=0,new_backbone_updates=0,
        test_accessed=False,independent_external_replication=False,scientific_task3_complete=False)
    archive.publish_immutable(f'results/research/READ03_replay_cloud/{SCIENCE_RUN}/{os.environ["GITHUB_RUN_ID"]}/cohort_receipt.json',
        (json.dumps(receipt,indent=2)+'\n').encode())
    if missing or any(r['returncode'] for r in records):raise SystemExit(1)


if __name__=='__main__':main()
