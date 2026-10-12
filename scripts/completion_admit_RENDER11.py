"""Admit an existing complete immutable recovery, with no physical execution."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from completion_kit.common import read_json, sha256_file, verify_files, write_json, canonical_bytes
from completion_kit.energy import merge_receipts, row_key

SOURCE=Path('D:/PROJECTS/196_NeuroPixel_int')
REL='results/research/RENDER11_energy/recovered_20261009B'
DEST=ROOT/REL
ORIGINAL=ROOT/'results/research/RENDER11_energy/original_20261009A/receipt.json'
PLAN=ROOT/'docs/research/RENDER11_energy_plan.json'
plan=read_json(PLAN)
original=read_json(ORIGINAL)
source_receipt=SOURCE/REL/'receipt.json'
recovery=read_json(source_receipt)
original_digest=sha256_file(ORIGINAL)
assert sha256_file(PLAN)==original['plan_sha256']==recovery['plan_sha256']
assert recovery['recovery_original_receipt_sha256']==original_digest
assert recovery['completed_rows_repeated']==0
verify_files(ROOT,plan['source_sha256'])
verify_files(ROOT,plan['input_sha256'])
merged=merge_receipts(original,recovery,plan)
assert merged['complete'] and merged['unique_rows']==105 and merged['original_rows']==103
old={row_key(r):canonical_bytes(r) for r in original['rows']}
new={row_key(r):canonical_bytes(r) for r in recovery['rows']}
assert all(new[k]==v for k,v in old.items())
assert len(set(new)-set(old))==2

commit=subprocess.check_output(['git','-C',str(SOURCE),'log','-1','--format=%H','--',REL+'/receipt.json'],text=True).strip()
assert commit
blob=subprocess.check_output(['git','-C',str(SOURCE),'show',commit+':'+REL+'/receipt.json'])
assert blob==source_receipt.read_bytes(),'Live saved receipt differs from its Git archive'
files=[]
for source in sorted((SOURCE/REL).rglob('*')):
    if source.is_file():
        files.append({'path':source.relative_to(SOURCE/REL).as_posix(),'sha256':sha256_file(source),'bytes':source.stat().st_size})
assert not DEST.exists(),'Use a fresh recovery directory; never overwrite evidence'
shutil.copytree(SOURCE/REL,DEST)
for item in files:
    assert sha256_file(DEST/item['path'])==item['sha256']
assert sha256_file(ORIGINAL)==original_digest
receipt={'status':'admitted_existing_complete_registered_recovery','source_checkout':str(SOURCE),'source_commit':commit,
         'original_receipt_sha256':original_digest,'recovered_receipt_sha256':sha256_file(DEST/'receipt.json'),
         'original_rows_preserved':103,'unique_registered_rows':105,'new_originally_missing_keys':[list(k) for k in sorted(set(new)-set(old))],
         'old_row_canonical_bytes_verified':True,'source_and_input_governors_verified':True,'original_raw_receipt_bytes_unchanged':True,
         'new_measurements_this_task':0,'closed_keys_remeasured_this_task':0,'original_recovery_recorded_repeats':recovery['completed_rows_repeated'],
         'recorded_hardware':recovery.get('hardware'),'recorded_software':recovery.get('software'),
         'saved_files':files,'physical_gpu_executed_this_task':False,
         'scope':'Saved internal physical evidence admission under guide §6.2; not a new scientific or independent replication; GPU-board energy only.'}
write_json(ROOT/'results/research/COMPLETION20261012/energy_recovery_admission.json',receipt,exclusive=True)
print(json.dumps({k:v for k,v in receipt.items() if k not in ['saved_files','recorded_hardware','recorded_software']},indent=2))
