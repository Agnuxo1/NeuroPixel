"""Supplemental audit of exact Git plan; preserve the failed local-byte comparison."""
import hashlib,importlib.util,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
LOCAL_HASH='bf54be3c4c8e30885b330e67e92195622e28c85a49e317358d106699d40008df'
EXECUTED_HASH='f1caada1af01d0078877b55800e3e7928e7bc8832993b8c189e0ffae2cdadb67'
REVIEW=ROOT/'results/research/OPT03_root_review/37696760073'

def main():
    local=(ROOT/'docs/research/OPT03_execution_plan.json').read_bytes()
    executed=(REVIEW/'execution_plan_source_commit.json').read_bytes()
    if hashlib.sha256(local).hexdigest()!=LOCAL_HASH or hashlib.sha256(executed).hexdigest()!=EXECUTED_HASH:raise ValueError('known original plan identity differs')
    if local.replace(b'\r\n',b'\n')!=executed or json.loads(local)!=json.loads(executed):raise ValueError('change beyond line endings; supplemental audit forbidden')
    original_failure=json.loads((REVIEW/'metadata_audit.json').read_text())
    if original_failure['status']!='issues_found' or not any('plan hash mismatch' in x for x in original_failure['issues']):raise ValueError('preserved original failure missing')
    if '--plan' not in sys.argv:raise ValueError('exact executed --plan path required')
    specified=pathlib.Path(sys.argv[sys.argv.index('--plan')+1]).resolve()
    if specified!=(REVIEW/'execution_plan_source_commit.json').resolve():raise ValueError('arbitrary replacement plan forbidden')
    s=importlib.util.spec_from_file_location('executed_plan_audit',ROOT/'scripts/audit_OPT03_checkpoint_results.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.PLAN_SHA256=EXECUTED_HASH
    m.main()

if __name__=='__main__':main()
