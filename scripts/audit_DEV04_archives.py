"""Read-only archive/lineage/dataset audit; no Torch or partial-cohort effects."""
import hashlib
import json
import pathlib
import sys
import zipfile

ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=37845945011
PLAN='2484f3c5917e474ca3755bcdbbaca7d322077615ba47d51653116042d980f20a'
sys.path.insert(0,str(ROOT/'scripts'))
from dev04_execution_registry import validate_receipt


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_compositions(path,panel):
    import numpy as np
    from neuropixel.research.OPT03_role_views import parse
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!={'canvas','target','roles'}:raise ValueError('Native dataset inventory differs')
        x,y,roles=(data[k] for k in ('canvas','target','roles'))
        if x.dtype!=np.int64 or y.dtype!=np.int64 or roles.dtype!=np.int64:raise ValueError('Native dataset dtype differs')
        expected=4096 if panel in ('head_train','validation') else 2048
        if x.shape!=(expected,8,8) or y.shape!=(expected,) or roles.shape!=y.shape:raise ValueError('Native dataset shape differs')
        if not np.array_equal(np.bincount(roles,minlength=4),np.full(4,expected//4)):raise ValueError('Balanced roles required')
        meta=parse(x,y,roles)
        compositions={tuple(map(int,(a-5,b-17,c-5))) for a,b,c in meta['semantic_triple']}
        if panel=='head_train':
            if not np.array_equal(roles,np.tile(np.arange(4),1024)):raise ValueError('Four-query context grouping differs')
            facts=x.copy();facts[:,7,6]=0
            if not np.all(facts.reshape(1024,4,8,8)==facts[::4,None]):raise ValueError('Heads must see four queries of identical visible contexts')
        return compositions


def main():
    plan_path=ROOT/'docs/research/DEV04_execution_plan.json'
    if sha(plan_path)!=PLAN:raise ValueError('Exact plan differs')
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen governing source changed')
    expected={r['run_id']:r for r in plan['runs']};originals=ROOT/f'results/research/DEV04_recovery/{RUN}/originals'
    checks=0;issues=[];cases=[];datasets_checked=0
    def check(value,label):
        nonlocal checks
        checks+=1
        if not bool(value):issues.append(label)
    for folder in sorted(originals.iterdir() if originals.exists() else []):
        if not (folder/'recovery_receipt.json').exists():continue
        receipt_path=folder/'receipt.json';archive=folder/'original.zip';receipt_raw=receipt_path.read_bytes();receipt=json.loads(receipt_raw)
        anchor=json.loads((folder/'recovery_receipt.json').read_text(encoding='utf-8'));case=receipt['case_run_id'];conf=expected[case]
        check(receipt['plan_sha256']==PLAN and receipt['config']==conf,folder.name+' config/recipe')
        validate_receipt(receipt,case,conf);check(True,folder.name+' registered execution source/run')
        check(hashlib.sha1(b'blob '+str(len(receipt_raw)).encode()+b'\0'+receipt_raw).hexdigest()==anchor['receipt_git_blob'],folder.name+' pinned Git receipt')
        check(sha(archive)==receipt['zip_sha256'] and archive.stat().st_size==receipt['zip_bytes'],folder.name+' original ZIP')
        target=pathlib.Path(anchor['extracted_destination'])
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('archive_manifest.json'))
            check(hashlib.sha256(z.read('execution_plan.json')).hexdigest()==PLAN,folder.name+' embedded plan')
            check(set(z.namelist())==set(manifest['files'])|{'archive_manifest.json','execution_plan.json'},folder.name+' complete archive inventory')
            for name,digest in manifest['files'].items():
                check(hashlib.sha256(z.read(name)).hexdigest()==digest and sha(target/name)==digest,folder.name+' raw/extracted '+name)
        base=target/case;partition=json.loads((base/'partition.json').read_text(encoding='utf-8'))
        check(partition==plan['partitions'][str(conf['partition_seed'])],folder.name+' prospectively admitted partition')
        progress=json.loads((base/'body/progress.json').read_text(encoding='utf-8'))
        check(progress['birth_model_digest']==plan['initial_body_model_digests'][case] and progress['new_initialization'] is True and progress['historical_checkpoint_loaded'] is False,folder.name+' fresh birth lineage')
        check(progress['config']==conf and progress['plan_sha256']==PLAN and sha(base/'body/checkpoint.pt')==progress['checkpoint_sha256'],folder.name+' actual body checkpoint identity')
        memberships={}
        for panel in ('head_train','probe','validation'):
            try:memberships[panel]=dataset_compositions(base/'datasets'/f'{panel}.npz',panel);checks+=1;datasets_checked+=1
            except (ValueError,KeyError) as error:issues.append(folder.name+' dataset '+panel+' '+str(error))
        if len(memberships)==3:
            check(not ((memberships['head_train']|memberships['probe'])&memberships['validation']),folder.name+' observed train/validation composition separation')
            check(len(memberships['head_train']|memberships['probe'])<=924 and len(memberships['validation'])<=132,folder.name+' observed membership bounds')
        cases.append(dict(case=case,stage=receipt['stage'],update=receipt['update'],status=receipt['case_status'],original_zip_sha256=receipt['zip_sha256']))
    out=dict(status='verified_available_archive_and_dataset_controls' if not issues else 'issues_found',checks=checks,issues=issues,
        recovered_archives=cases,dataset_panels_checked=datasets_checked,nn_checkpoint_replay_completed=False,
        original_test_membership_recomputed_by_this_auditor=False,cohort_effects_estimated=False,scientific_task3_complete=False)
    path=ROOT/f'results/research/DEV04_review/{RUN}/archive_dataset_audit.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:out[k] for k in ('status','checks','issues','dataset_panels_checked','nn_checkpoint_replay_completed')}))
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
