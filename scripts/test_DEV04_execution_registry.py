"""Reject recipe drift, false retries and incomplete execution provenance."""
import copy
import json
import pathlib
import unittest
from dev04_execution_registry import PLAN,SOURCE,RUN,RECOVERY_SOURCE,RECOVERY_RUN,validate_receipt,validate_complete_attempt

ROOT=pathlib.Path(__file__).resolve().parents[1]
PLAN_DOC=json.loads((ROOT/'docs/research/DEV04_execution_plan.json').read_bytes())

class Contracts(unittest.TestCase):
    def receipt(self,case):
        config=next(x for x in PLAN_DOC['runs'] if x['run_id']==case)
        failed=case.endswith(('seed400','seed402','seed403'))
        return dict(registration_id='NP-DEV04-20261008',case_run_id=case,config=config,plan_sha256=PLAN,
            workflow_run_id=str(RECOVERY_RUN if failed else RUN),source_commit=RECOVERY_SOURCE if failed else SOURCE,
            historical_checkpoint_loaded=False,new_body_initialization=True,test_scored=False,case_status='completed',
            archive_branch='research/dev04-evidence-2026-10-08/'+case)

    def test_all_twelve_registered_executions(self):
        for config in PLAN_DOC['runs']:
            r=self.receipt(config['run_id']);validate_receipt(r,r['case_run_id'],config)

    def test_successful_case_retraining_under_recovery_run_rejected(self):
        r=self.receipt('DEV04_partition101_seed401');r.update(workflow_run_id=str(RECOVERY_RUN),source_commit=RECOVERY_SOURCE)
        with self.assertRaises(ValueError):validate_receipt(r,r['case_run_id'],r['config'])

    def test_zero_update_initial_failure_cannot_be_labeled_completed(self):
        r=self.receipt('DEV04_partition101_seed400');r.update(workflow_run_id=str(RUN),source_commit=SOURCE)
        with self.assertRaises(ValueError):validate_receipt(r,r['case_run_id'],r['config'])

    def test_sources_recipe_case_and_test_changes_rejected(self):
        original=self.receipt('DEV04_partition101_seed400')
        for key,value in [('source_commit',SOURCE),('plan_sha256','0'*64),('case_run_id','DEV04_partition101_seed401'),('test_scored',True),('historical_checkpoint_loaded',True),('archive_branch','main'),('new_body_initialization',False)]:
            r=copy.deepcopy(original);r[key]=value
            with self.assertRaises(ValueError):validate_receipt(r,original['case_run_id'],original['config'])

    def test_original_success_attempt_does_not_require_new_fields(self):
        case='DEV04_partition101_seed401'
        validate_complete_attempt(dict(workflow_run_id=str(RUN),source_commit=SOURCE,case_run_id=case,completed=True,durable_evidence_available=True,historical_checkpoint_loaded=False,test_scored=False),case)

    def test_recovery_attempt_requires_frozen_recipe_and_completed_evidence(self):
        case='DEV04_partition101_seed400'
        original=dict(workflow_run_id=str(RECOVERY_RUN),execution_source_commit=RECOVERY_SOURCE,frozen_recipe_source_commit=SOURCE,
            initial_workflow_run_id=RUN,case_run_id=case,completed=True,durable_evidence_available=True,plan_sha256=PLAN,
            scientific_governing_files_changed=False,historical_checkpoint_loaded=False,test_scored=False)
        validate_complete_attempt(original,case)
        for key,value in [('completed',False),('durable_evidence_available',False),('scientific_governing_files_changed',True),('frozen_recipe_source_commit',RECOVERY_SOURCE),('initial_workflow_run_id',RECOVERY_RUN)]:
            r=dict(original);r[key]=value
            with self.assertRaises(ValueError):validate_complete_attempt(r,case)

if __name__=='__main__':unittest.main()
