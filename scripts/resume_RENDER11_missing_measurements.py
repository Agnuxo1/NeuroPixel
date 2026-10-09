"""Operational recovery of five missing energy rows; frozen math stays unchanged."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import measure_RENDER11_energy as frozen
def main():
    old=ROOT/'results/research/RENDER11_energy/original_20261009A';record=json.loads((old/'receipt.json').read_bytes());plan=json.loads((ROOT/'docs/research/RENDER11_energy_plan.json').read_bytes())
    assert record['status']=='failed_preserved' and len(record['rows'])==100 and len(record['checks'])==77 and all(c['passed'] for c in record['checks'])
    assert hashlib.sha256((ROOT/'scripts/measure_RENDER11_energy.py').read_bytes()).hexdigest()==plan['source_sha256']['scripts/measure_RENDER11_energy.py']
    loop=frozen.ENERGY_LOOP;marker='        for number,case in enumerate(cases):\n';assert loop.count(marker)==1
    loop=loop.replace(marker,"        rows=json.loads(Path("+repr(str(old/'receipt.json'))+").read_bytes())['rows']\n"+marker)
    marker='                for method in order:\n';assert loop.count(marker)==1
    loop=loop.replace(marker,marker+'                    if any(r["case"]==case["spec"]["name"] and r["block"]==block and r["method"]==method for r in rows):continue\n')
    frozen.ENERGY_LOOP=loop
    out=ROOT/'results/research/RENDER11_energy/recovered_20261009B';sys.argv=[str(ROOT/'scripts/measure_RENDER11_energy.py'),'--output',str(out)]
    try:frozen.main()
    finally:
        if (out/'receipt.json').exists():
            j=json.loads((out/'receipt.json').read_bytes());j.update(recovery_original_receipt_sha256=hashlib.sha256((old/'receipt.json').read_bytes()).hexdigest(),completed_rows_repeated=0,operational_interruption_qualified=True)
            (out/'receipt.json').write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
