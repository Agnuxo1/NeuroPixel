"""New sustained-energy endpoint; reuse immutable R10 mathematics and gates."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TIMING_MARKER='        for number,case in enumerate(cases):\n'
FINAL_MARKER="        result=record();result['status']='completed_all_registered_RENDER10_workloads'"
ENERGY_LOOP='''        if not energy_supported:raise RuntimeError('Actual energy counter unavailable')
        for number,case in enumerate(cases):
            methods=['cuda_eager']+(['cuda_graph'] if case['graph'] is not None else [])+['scalar_render','vector_render','render']
            if len(methods)!=5:raise RuntimeError('Registered CUDA graph comparator unavailable')
            for block in range(plan['energy_technical_blocks']):
                order=methods if block%2==0 else list(reversed(methods))
                for method in order:
                    ram();torch.cuda.synchronize();ctx.finish()
                    temperature0=pynvml.nvmlDeviceGetTemperature(device,pynvml.NVML_TEMPERATURE_GPU)
                    if temperature0>=83:raise RuntimeError('GPU temperature guard')
                    idle0=energy();idle_start=time.perf_counter();time.sleep(plan['energy_idle_seconds']);idle1=energy();idle_elapsed=time.perf_counter()-idle_start
                    before_w=(idle1-idle0)/idle_elapsed
                    e0=energy();start=time.perf_counter();count=0
                    while time.perf_counter()-start<plan['energy_active_seconds']:
                        ram()
                        if method=='cuda_eager':case['tensor']();torch.cuda.synchronize()
                        elif method=='cuda_graph':case['graph'].replay();torch.cuda.synchronize()
                        else:
                            (case['render'] if method=='render' else case['scalar_render'] if method=='scalar_render' else case['vector_render'])();ctx.finish()
                        count+=1
                    elapsed=time.perf_counter()-start;e1=energy();gross=e1-e0
                    idle2=energy();idle_start=time.perf_counter();time.sleep(plan['energy_idle_seconds']);idle3=energy();idle_elapsed2=time.perf_counter()-idle_start
                    after_w=(idle3-idle2)/idle_elapsed2
                    if count<1 or gross<=0:raise RuntimeError('Unresolved or reset actual energy counter')
                    temperature1=pynvml.nvmlDeviceGetTemperature(device,pynvml.NVML_TEMPERATURE_GPU)
                    rows.append(dict(case=case['spec']['name'],block=block,method=method,inference_calls=count,active_seconds=elapsed,gpu_gross_energy_j=gross,gpu_gross_j_per_call=gross/count,idle_before_w=before_w,idle_after_w=after_w,baseline_adjusted_j_per_call_low=(gross-max(before_w,after_w)*elapsed)/count,baseline_adjusted_j_per_call_high=(gross-min(before_w,after_w)*elapsed)/count,temperature_start_c=temperature0,temperature_end_c=temperature1,available_ram_gib=ram(),scope='Whole GPU board; baseline range is an estimate, not a confidence interval or complete system energy'))
                    record()
                    if temperature1>=83:raise RuntimeError('GPU temperature guard')
            print(json.dumps(dict(case=case['spec']['name'],energy_blocks=plan['energy_technical_blocks'],methods=methods)),flush=True)
'''
def main():
    script=ROOT/'scripts/benchmark_RENDER10.py';source=script.read_text(encoding='utf-8')
    plan_path=ROOT/'docs/research/RENDER11_energy_plan.json';plan=json.loads(plan_path.read_bytes())
    assert hashlib.sha256(script.read_bytes()).hexdigest()==plan['source_sha256']['scripts/benchmark_RENDER10.py']
    assert source.count(TIMING_MARKER)==1 and source.count(FINAL_MARKER)==1
    start=source.index(TIMING_MARKER);end=source.index(FINAL_MARKER)
    source=source[:start]+ENERGY_LOOP+source[end:]
    source=source.replace("ROOT/'docs/research/RENDER10_benchmark_plan.json'","ROOT/'docs/research/RENDER11_energy_plan.json'")
    source=source.replace('completed_all_registered_RENDER10_workloads','completed_all_registered_RENDER11_energy_workloads')
    marker="np.savez_compressed(a.output/f'case{number}_inputs.npz',**arrays)"
    original=ROOT/'results/research/RENDER09_benchmark/original_20261009A'
    replacement=f"with np.load(Path({str(original)!r})/f'case{{number}}_inputs.npz',allow_pickle=False) as previous:\n                if set(arrays)!=set(previous.files) or any(not np.array_equal(arrays[k],previous[k]) for k in arrays):raise ValueError('Registered input bytes changed')\n            "+marker
    assert source.count(marker)==1;source=source.replace(marker,replacement)
    for name,digest in plan['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    exec(compile(source,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
if __name__=='__main__':main()
