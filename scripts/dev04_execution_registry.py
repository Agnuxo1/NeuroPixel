"""Operational provenance for unchanged DEV04 recipe; retries are not new replicas."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN='2484f3c5917e474ca3755bcdbbaca7d322077615ba47d51653116042d980f20a'
SOURCE='b9431e9b927ba5b6539a172d4b27fe6199776462'
RUN=37845945011
RECOVERY_RUN=37871777801
RECOVERY_SOURCE='81cf94e8f2a089b0d695f5fd465c6daaff50c6f1'
FAILED_STARTS={'DEV04_partition101_seed400','DEV04_partition101_seed402','DEV04_partition101_seed403'}

def expected_execution(case):
    if case in FAILED_STARTS:return RECOVERY_RUN,RECOVERY_SOURCE
    return RUN,SOURCE

def validate_receipt(receipt,case,config):
    run,source=expected_execution(case)
    # Initial failed cases may have older partial archives, but no finished fits.
    allowed={(str(run),source)}
    if receipt.get('case_status')!='completed':allowed.add((str(RUN),SOURCE))
    if ((str(receipt.get('workflow_run_id')),receipt.get('source_commit')) not in allowed
        or receipt.get('case_run_id')!=case or receipt.get('config')!=config
        or receipt.get('registration_id')!='NP-DEV04-20261008'
        or receipt.get('plan_sha256')!=PLAN or receipt.get('historical_checkpoint_loaded') is not False
        or receipt.get('new_body_initialization') is not True or receipt.get('test_scored') is not False
        or receipt.get('archive_branch')!='research/dev04-evidence-2026-10-08/'+case):
        raise ValueError('Unregistered execution/recipe/case/lineage')
    return run,source

def validate_complete_attempt(receipt,case):
    run,source=expected_execution(case)
    source_key='execution_source_commit' if case in FAILED_STARTS else 'source_commit'
    if (str(receipt.get('workflow_run_id'))!=str(run) or receipt.get(source_key)!=source
        or receipt.get('case_run_id')!=case or receipt.get('completed') is not True
        or receipt.get('durable_evidence_available') is not True
        or receipt.get('historical_checkpoint_loaded') is not False or receipt.get('test_scored') is not False):
        raise ValueError('Missing/different completed execution attempt')
    if case in FAILED_STARTS and (receipt.get('plan_sha256')!=PLAN or receipt.get('frozen_recipe_source_commit')!=SOURCE
        or receipt.get('initial_workflow_run_id')!=RUN or receipt.get('scientific_governing_files_changed') is not False):
        raise ValueError('Recovery scientific recipe/source differs')

