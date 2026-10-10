"""Complete-cohort admission and independent arithmetic recount of device energy."""
import hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    # Por defecto, el recibo original; para la cohorte recuperada: python summarize_RENDER11_energy.py <carpeta>
    folder=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/research/RENDER11_energy/original_20261009A'
    record=json.loads((folder/'receipt.json').read_bytes());plan=json.loads((ROOT/'docs/research/RENDER11_energy_plan.json').read_bytes())
    assert record['status']=='completed_all_registered_RENDER11_energy_workloads'
    assert len(record['checks'])==77 and all(c['passed'] for c in record['checks'])
    assert record['plan_sha256']==hashlib.sha256((ROOT/'docs/research/RENDER11_energy_plan.json').read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    methods=['cuda_eager','cuda_graph','scalar_render','vector_render','render'];expected={(c['name'],b,m) for c in plan['cases'] for b in range(3) for m in methods};keys=[(r['case'],r['block'],r['method']) for r in record['rows']]
    assert len(keys)==105 and len(set(keys))==105 and set(keys)==expected
    for row in record['rows']:
        count=row['inference_calls'];gross=row['gpu_gross_energy_j'];elapsed=row['active_seconds']
        assert count>=1 and elapsed>=3 and gross>0 and row['available_ram_gib']>=8
        assert row['temperature_start_c']<83 and row['temperature_end_c']<83
        assert abs(row['gpu_gross_j_per_call']-gross/count)<1e-10
        low=(gross-max(row['idle_before_w'],row['idle_after_w'])*elapsed)/count;high=(gross-min(row['idle_before_w'],row['idle_after_w'])*elapsed)/count
        assert abs(row['baseline_adjusted_j_per_call_low']-low)<1e-10 and abs(row['baseline_adjusted_j_per_call_high']-high)<1e-10
    tables=[]
    for case in plan['cases']:
        for method in methods:
            rows=[r for r in record['rows'] if r['case']==case['name'] and r['method']==method]
            values=[r['gpu_gross_j_per_call'] for r in rows]
            tables.append(dict(case=case['name'],method=method,gross_j_median=statistics.median(values),gross_j_min=min(values),gross_j_max=max(values),baseline_low_median=statistics.median(r['baseline_adjusted_j_per_call_low'] for r in rows),baseline_high_median=statistics.median(r['baseline_adjusted_j_per_call_high'] for r in rows),calls=sum(r['inference_calls'] for r in rows)))
    result=dict(status='verified_complete_energy_cohort',rows=105,parity_checks=77,table=tables,min_ram_gib=min(r['available_ram_gib'] for r in record['rows']),max_temperature_c=max(r['temperature_end_c'] for r in record['rows']),total_active_board_energy_j=sum(r['gpu_gross_energy_j'] for r in record['rows']),total_complete_inference_calls=sum(r['inference_calls'] for r in record['rows']),technical_blocks_not_independent_replications=True,wall_energy_measured=False)
    (folder/'independent_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    text=['# Sustained GPU device energy: all registered cases','', 'Actual RTX 3090 board counters were measured over three-second active blocks, bracketed by one-second idle observations. All 77 output gates passed; all 105 registered rows and original source hashes were admitted. Three blocks are technical observations, not independent scientific replications.','', '| Workload | Method | Gross J per call, median [min,max] | Baseline-adjusted range, median J per call | Calls in three blocks |','|---|---|---:|---:|---:|']
    labels={'render':'OpenGL 4.6 compute','scalar_render':'scalar fragment','vector_render':'vector fragment','cuda_eager':'CUDA eager','cuda_graph':'CUDA Graph'}
    for row in tables:text.append(f"| {row['case']} | {labels[row['method']]} | {row['gross_j_median']:.6f} [{row['gross_j_min']:.6f},{row['gross_j_max']:.6f}] | [{row['baseline_low_median']:.6f},{row['baseline_high_median']:.6f}] | {row['calls']} |")
    text.extend(['','Gross energy includes board background and idle consumption. The baseline range uses the observed before/after idle rates; it is an estimate, not a confidence interval or unique causal attribution. CPU, RAM, PSU and whole-system wall energy are excluded. Resident inference includes seeding, sixteen updates and the complete decoder, but excludes training, compilation, transfers and publication. No universal energy advantage follows.','',f"Minimum recorded available RAM: {result['min_ram_gib']:.3f} GiB; maximum ending GPU temperature: {result['max_temperature_c']} °C. Original rows and fixed protocol are retained in `results/research/RENDER11_energy/original_20261009A` and `docs/research/RENDER11_energy_plan.json`."])
    (ROOT/'docs/research/RENDER11_results.md').write_text('\n'.join(text)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ['status','rows','parity_checks','min_ram_gib','max_temperature_c','total_active_board_energy_j','total_complete_inference_calls']}))
if __name__=='__main__':main()
