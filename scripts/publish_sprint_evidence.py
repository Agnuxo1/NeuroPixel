"""Archive a completed/failed bounded sprint case immutably; never trains."""
import argparse,hashlib,io,json,os,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neuropixel.research.qtrain_budget_transport import publish_immutable
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--registration',required=True);p.add_argument('--branch',required=True);a=p.parse_args()
    if not a.registration.isalnum():raise ValueError('Unsafe registration')
    os.environ['GITHUB_REF_NAME']=a.branch
    files={str(f.relative_to(a.root)).replace('\\','/'):f.read_bytes() for f in a.root.rglob('*') if f.is_file() and f.name!='archive_manifest.json'}
    manifest=dict(registration=a.registration,run_id=os.environ['GITHUB_RUN_ID'],source_commit=os.environ['GITHUB_SHA'],files={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in files.items()})
    files['archive_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for k,v in sorted(files.items()):z.writestr(k,v)
    body=out.getvalue();base=f'results/research/{a.registration}_cloud/'+os.environ['GITHUB_RUN_ID']+'/'
    publish_immutable(base+'raw.zip',body)
    receipt=dict(registration=a.registration,workflow_run_id=os.environ['GITHUB_RUN_ID'],source_commit=os.environ['GITHUB_SHA'],files=len(files),zip_bytes=len(body),zip_sha256=hashlib.sha256(body).hexdigest(),remote_zip_path=base+'raw.zip',archive_branch=a.branch)
    publish_immutable(base+'receipt.json',(json.dumps(receipt,indent=2)+'\n').encode());print(json.dumps(receipt))
if __name__=='__main__':main()
