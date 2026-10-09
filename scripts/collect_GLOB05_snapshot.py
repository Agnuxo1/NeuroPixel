"""Pin immutable new-head archives, numerical proofs and attempt receipts; no effects."""
import argparse,hashlib,json,math,pathlib,subprocess,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from research_GLOB05 import load_plan,load_inventory,sha
from readout_snapshot import load_snapshot
from neuropixel.research import glob05_transport as archive
PLAN='b26f17cec8a98fda6dbd5aa128d67501fd2273bf07718300c1c865391985e0b2'
RUN=37887762153
SOURCE='e4b82b280ece658ef14ea6d7e215e249edb06af3'
REPO='Agnuxo1/NeuroPixel'

def main():
    p=argparse.ArgumentParser();p.add_argument('--tree-snapshot',type=pathlib.Path,required=True);p.add_argument('--case',required=True);p.add_argument('--git-object-cache',type=pathlib.Path);a=p.parse_args()
    plan=load_plan(ROOT/'docs/research/GLOB05_execution_plan.json',PLAN);inventory=load_inventory(ROOT/plan['parent_inventory_path'],plan['parent_inventory_sha256'])
    descriptor=next(r for r in inventory['parents'] if r['case']==a.case);config=next(r for r in plan['runs'] if r['parent_case']==a.case)
    snapshot=load_snapshot(a.tree_snapshot,REPO);commit=snapshot['commit']['sha'];entries={r['path']:r for r in snapshot['tree']['tree'] if r['type']=='blob'}
    def get(path):
        if path not in entries:raise ValueError('Missing pinned entry '+path)
        if a.git_object_cache:
            raw=subprocess.check_output(['git','-c','safe.directory='+a.git_object_cache.resolve().as_posix(),'--git-dir='+str(a.git_object_cache.resolve()),'cat-file','blob',entries[path]['sha']])
        else:raw=archive.old.fetch(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}')
        if archive.blob(raw)!=entries[path]['sha']:raise ValueError('Pinned blob bytes differ')
        return raw
    prefix=f'results/research/GLOB05_cloud/{archive.REG}/';receiptpath=prefix+a.case+'/receipt.json'
    if receiptpath not in entries:
        print(json.dumps(dict(case=a.case,status='no_complete_new_head_archive_yet',training_restarted=False)));return
    raw_receipt=get(receiptpath);r=json.loads(raw_receipt)
    expected_parent=dict(parent_case=a.case,original_zip_sha256=descriptor['original_zip_sha256'],body_checkpoint_sha256=descriptor['body_checkpoint_sha256'],frozen_body_digest=descriptor['frozen_body_digest'],parent_inventory_sha256=plan['parent_inventory_sha256'])
    if (r['registration_id']!=archive.REG or r['case_run_id']!=a.case or r['config']!=config or r['parent']!=expected_parent
        or r['plan_sha256']!=PLAN or r['case_status']!='completed' or r['update']!=8192
        or str(r['workflow_run_id'])!=str(RUN) or r['source_commit']!=SOURCE
        or r['new_body_initializations']!=0 or r['new_backbone_updates']!=0 or r['closed_head_trainings_repeated']!=0
        or r['test_accessed'] is not False or r['archive_branch']!='research/glob05-evidence-2026-10-09/'+a.case):raise ValueError('New-head recipe/source/run/parent differs')
    base=ROOT/f'results/research/GLOB05_recovery/{RUN}';original=base/'originals'/a.case;raw_zip=get(prefix+a.case+'/raw.zip')
    if len(raw_zip)!=r['zip_bytes'] or archive.digest(raw_zip)!=r['zip_sha256']:raise ValueError('Scientific ZIP size/hash differs')
    archive.old.immutable(original/'original.zip',raw_zip);archive.old.immutable(original/'receipt.json',raw_receipt)
    manifest=archive.unpack(raw_zip,base/'cohort',a.case,PLAN)
    folder=base/'cohort'/a.case
    with zipfile.ZipFile(original/'original.zip') as z:
        for name,h in manifest['files'].items():
            if archive.digest(z.read(name))!=h or sha(base/'cohort'/name)!=h:raise ValueError('Original/extracted scientific bytes differ')
    for name,h in descriptor['dataset_sha256'].items():
        if sha(folder/'datasets'/f'{name}.npz')!=h:raise ValueError('Exact old dataset changed')
    record=json.loads((folder/'result.json').read_bytes())
    if (record['config']!=config or record['parent']!=expected_parent or record['frozen_body_digest']!=descriptor['frozen_body_digest']
        or record['head_nominal_parameters']!=3168 or record['effective_gradient_parameters']!=3168
        or record['new_backbone_updates']!=0 or record['original_DEV04_precision_reinterpreted'] is not False
        or record['test_accessed'] is not False or [e['update'] for e in record['evaluations']]!=[1024,4096,8192]):raise ValueError('Closed new-control contents differ')
    old=ROOT/'results/research/DEV04_recovery/37845945011/cohort'/a.case/'query_attention/result.json'
    if sha(old)!=descriptor['closed_attention_result_sha256']:raise ValueError('Old frozen attention baseline changed')
    attention=json.loads(old.read_bytes())
    if [w['batch_witness_sha256'] for w in record['curve']]!=[w['batch_witness_sha256'] for w in attention['curve']]:raise ValueError('Matched batch/mask streams differ')
    proofpath=prefix+'proofs/'+str(RUN)+'/'+a.case+'/replay.json';attemptpath=prefix+'attempts/'+str(RUN)+'/'+a.case+'/attempt_receipt.json'
    if proofpath not in entries or attemptpath not in entries:
        print(json.dumps(dict(case=a.case,status='complete_archive_recovered_final_replay_or_attempt_not_yet_available',effects_estimated=False)));return
    raw_proof=get(proofpath);proof=json.loads(raw_proof);raw_attempt=get(attemptpath);attempt=json.loads(raw_attempt)
    if (proof['status']!='verified_readonly_GLOB05_control_replay' or proof['case']!=a.case or proof['case_run_id']!=a.case
        or proof['issues'] or proof['individual_decisions_replayed']!=294912 or proof['plan_sha256']!=PLAN
        or str(proof['scientific_run_id'])!=str(RUN) or proof['scientific_source_commit']!=SOURCE
        or proof['new_training_updates']!=0 or proof['new_body_initializations']!=0 or proof['old_head_fits_repeated']!=0
        or proof['original_test_scored'] is not False or proof['original_DEV04_precision_reinterpreted'] is not False
        or not math.isfinite(proof['max_mean_nll_error']) or not 0<=proof['max_mean_nll_error']<=1e-4):raise ValueError('Complete exact numerical proof differs')
    if {a.case+'/'+name:h for name,h in proof['input_files_sha256'].items()}!=manifest['files']:raise ValueError('Numerical proof not bound to every original input')
    if (attempt['source_commit']!=SOURCE or str(attempt['workflow_run_id'])!=str(RUN) or attempt['case_run_id']!=a.case
        or attempt['head_training_completed'] is not True or attempt['numerical_replay_verified'] is not True
        or attempt['new_body_initializations']!=0 or attempt['old_head_fits_repeated']!=0
        or attempt['original_test_scored'] is not False or attempt['original_DEV04_precision_reinterpreted'] is not False):raise ValueError('Complete exact attempt receipt differs')
    review=ROOT/f'results/research/GLOB05_review/{RUN}';dest=review/'pinned_replay_receipts'/a.case
    archive.old.immutable(dest/'replay.json',raw_proof);archive.old.immutable(dest/'attempt.json',raw_attempt)
    anchor=dict(archive_git_commit=commit,proof_git_blob=entries[proofpath]['sha'],proof_sha256=archive.digest(raw_proof),attempt_git_blob=entries[attemptpath]['sha'],attempt_sha256=archive.digest(raw_attempt),
        original_zip_sha256=r['zip_sha256'],receipt_git_blob=entries[receiptpath]['sha'],receipt_sha256=archive.digest(raw_receipt),tree_snapshot_sha256=sha(a.tree_snapshot))
    if not (dest/'git_anchor.json').exists():archive.old.immutable(dest/'git_anchor.json',(json.dumps(anchor,indent=2)+'\n').encode())
    summary=dict(case=a.case,status='verified_original_GLOB05_archive_parent_streams_and_numeric_replay',checks=proof['checks'],individual_decisions_replayed=proof['individual_decisions_replayed'],max_mean_nll_error=proof['max_mean_nll_error'],files_verified=len(manifest['files']),scientific_effects_estimated=False,original_DEV04_precision_reinterpreted=False,scientific_task3_complete=False)
    archive.old.immutable(dest/'verification_receipt.json',(json.dumps(summary,indent=2)+'\n').encode());print(json.dumps(summary))

if __name__=='__main__':main()
