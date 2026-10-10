"""Operational recovery of the two missing RGB8 block-2 energy rows; frozen math stays unchanged."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import measure_RENDER11_energy as frozen
EXPECTED_MISSING={('RGB8_retina_C16',2,'vector_render'),('RGB8_retina_C16',2,'render')}
TOTAL_ROWS=105
def main():
    # Guard: la cohorte ya esta recuperada y verificada (commit 87c60227); relanzar sobrescribiria el recibo.
    if (ROOT/'results/research/RENDER11_energy/recovered_20261009B/receipt.json').exists():
        raise SystemExit('Recuperacion ya verificada (commit 87c60227): no se relanza')
    old=ROOT/'results/research/RENDER11_energy/original_20261009A';record=json.loads((old/'receipt.json').read_bytes());plan=json.loads((ROOT/'docs/research/RENDER11_energy_plan.json').read_bytes())
    orig={(r['case'],r['block'],r['method']):json.dumps(r,sort_keys=True) for r in record['rows']}
    assert record['status']=='failed_preserved' and len(record['rows'])==TOTAL_ROWS-len(EXPECTED_MISSING) and len(orig)==len(record['rows']) and len(record['checks'])==77 and all(c['passed'] for c in record['checks'])
    assert not (set(orig)&EXPECTED_MISSING)
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
    # Cohorte completa antes de cualquier analisis: 105 filas unicas, las 103 originales intactas y solo las 2 que faltaban.
    recovered=json.loads((out/'receipt.json').read_bytes())
    new={(r['case'],r['block'],r['method']):json.dumps(r,sort_keys=True) for r in recovered['rows']}
    assert len(recovered['rows'])==TOTAL_ROWS and len(new)==TOTAL_ROWS, 'cohorte incompleta o con duplicados'
    assert set(new)==set(orig)|EXPECTED_MISSING, 'filas inesperadas'
    assert all(new[k]==v for k,v in orig.items()), 'una fila original cambio'
if __name__=='__main__':main()
