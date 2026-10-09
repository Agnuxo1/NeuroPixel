"""Reject false completion caused by wrong attempts, foreign jobs or missing shards."""
import copy
import json
import pathlib
import unittest
from collect_DEV04_replay_receipts import validate_terminal_snapshot
from dev04_execution_registry import RUN,SOURCE,RECOVERY_RUN,RECOVERY_SOURCE,FAILED_STARTS

ROOT=pathlib.Path(__file__).resolve().parents[1]
EXPECTED={r['run_id'] for r in json.loads((ROOT/'docs/research/DEV04_execution_plan.json').read_bytes())['runs']}
VERIFY=37872427770
VERIFY_SOURCE='125feea39a8c940a7a911b453f11b6e0db0c3fd6'

class Contracts(unittest.TestCase):
    def fixture(self):
        runs={str(r):dict(id=r,head_sha=s,status='completed',conclusion='failure' if r==RUN else 'success',path=p,run_attempt=1)
            for r,s,p in [(RUN,SOURCE,'.github/workflows/dev04-experiment.yml'),(RECOVERY_RUN,RECOVERY_SOURCE,'.github/workflows/dev04-operational-recovery.yml'),(VERIFY,VERIFY_SOURCE,'.github/workflows/dev04-readonly-verification.yml')]}
        jobs={str(RUN):dict(jobs=[dict(id=i,run_id=RUN,name=c,status='completed',conclusion='failure' if c in FAILED_STARTS else 'success') for i,c in enumerate(sorted(EXPECTED),1)]),
            str(RECOVERY_RUN):dict(jobs=[dict(id=100+i,run_id=RECOVERY_RUN,name=c,status='completed',conclusion='success') for i,c in enumerate(sorted(FAILED_STARTS),1)]),
            str(VERIFY):dict(jobs=[dict(id=200,run_id=VERIFY,name='readonly-verification',status='completed',conclusion='success')])}
        return dict(runs=runs,jobs=jobs)

    def validate(self,d):return validate_terminal_snapshot(d,EXPECTED,VERIFY,VERIFY_SOURCE)

    def test_nine_original_successes_plus_three_exact_recoveries(self):
        self.assertTrue(self.validate(self.fixture()))

    def test_running_or_failed_recovery_rejected(self):
        for key,value in [('status','in_progress'),('conclusion','failure')]:
            d=self.fixture();d['runs'][str(RECOVERY_RUN)][key]=value
            with self.assertRaises(ValueError):self.validate(d)

    def test_original_retried_or_reclassified_as_success_rejected(self):
        for key,value in [('run_attempt',2),('conclusion','success')]:
            d=self.fixture();d['runs'][str(RUN)][key]=value
            with self.assertRaises(ValueError):self.validate(d)

    def test_foreign_source_or_wrong_workflow_rejected(self):
        for key,value in [('head_sha',SOURCE),('path','.github/workflows/dev04-experiment.yml')]:
            d=self.fixture();d['runs'][str(VERIFY)][key]=value
            with self.assertRaises(ValueError):self.validate(d)

    def test_missing_duplicate_or_foreign_job_rejected(self):
        for change in ('missing','duplicate','foreign','sameid'):
            d=self.fixture();rows=d['jobs'][str(RUN)]['jobs']
            if change=='missing':rows.pop()
            elif change=='duplicate':rows.append(copy.deepcopy(rows[0]))
            elif change=='foreign':rows[0]['run_id']=RECOVERY_RUN
            else:rows[0]['id']=rows[1]['id']
            with self.assertRaises(ValueError):self.validate(d)

    def test_successful_case_cannot_count_as_recovered_failure(self):
        d=self.fixture();d['jobs'][str(RECOVERY_RUN)]['jobs'][0]['name']='DEV04_partition101_seed401'
        with self.assertRaises(ValueError):self.validate(d)

    def test_wrong_or_incomplete_verifier_job_rejected(self):
        for key,value in [('name','other-workflow'),('status','in_progress'),('conclusion','failure')]:
            d=self.fixture();d['jobs'][str(VERIFY)]['jobs'][0][key]=value
            with self.assertRaises(ValueError):self.validate(d)

if __name__=='__main__':unittest.main()
