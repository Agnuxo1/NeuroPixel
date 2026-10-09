"""Operational recovery only: frozen math, reuse rows, skip every completed key."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    original=ROOT/'results/research/RENDER10_benchmark/original_20261009A';record=json.loads((original/'receipt.json').read_bytes());plan=json.loads((ROOT/'docs/research/RENDER10_benchmark_plan.json').read_bytes());script=ROOT/'scripts/benchmark_RENDER10.py'
    assert record['status']=='failed_preserved' and all(c['passed'] for c in record['checks'])
    assert hashlib.sha256(script.read_bytes()).hexdigest()==plan['source_sha256']['scripts/benchmark_RENDER10.py']
    keys=[(r['case'],r['block'],r['method']) for r in record['rows']];assert len(keys)==len(set(keys))
    source=script.read_text();source=source.replace('rows=[];cases=[];checks=[];energy_supported=True',f'rows=json.loads(Path({str(original/"receipt.json")!r}).read_bytes())["rows"];cases=[];checks=[];energy_supported=True')
    source=source.replace('for method in order:\n                    ram();','for method in order:\n                    if any(r["case"]==case["spec"]["name"] and r["block"]==block and r["method"]==method for r in rows):continue\n                    ram();')
    source=source.replace("np.savez_compressed(a.output/f'case{number}_inputs.npz',**arrays)",f"with np.load(Path({str(original)!r})/f'case{{number}}_inputs.npz',allow_pickle=False) as previous:\n                if set(arrays)!=set(previous.files) or any(not np.array_equal(arrays[k],previous[k]) for k in arrays):raise ValueError('Original registered inputs changed')\n            np.savez_compressed(a.output/f'case{{number}}_inputs.npz',**arrays)")
    source=source.replace("result['technical_repeats_are_not_scientific_replicas']=True","result['technical_repeats_are_not_scientific_replicas']=True;result['recovery_original_receipt_sha256']="+repr(hashlib.sha256((original/'receipt.json').read_bytes()).hexdigest())+";result['completed_rows_repeated']=0;result['operational_interruption_qualified']=True")
    sys.argv=[str(script),'--output',str(ROOT/'results/research/RENDER10_benchmark/recovered_20261009B')]
    exec(compile(source,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
if __name__=='__main__':main()
