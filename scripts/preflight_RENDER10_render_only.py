"""OpenGL4.6 parity against immutable already-captured CUDA arrays; Torch-free."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    import psutil
    def ram():
        v=psutil.virtual_memory().available/2**30
        if v<8:raise RuntimeError('RAM8 refused')
        return v
    start_ram=ram();import numpy as np,moderngl
    from neuropixel.rendering.nca_gl_compute import ComputeGLNCA,shared_memory_limit
    ram();out=ROOT/'results/research/RENDER10_preflight/render_only_20261009A';out.mkdir(exist_ok=False);source=ROOT/'results/research/RENDER09_benchmark/original_20261009A';plan=json.loads((ROOT/'docs/research/RENDER09_benchmark_plan.json').read_bytes());ctx=moderngl.create_standalone_context(require=460);checks=[];rows=[]
    def save(status,error=None):
        j=dict(status=status,checks=checks,rows=rows,hardware={k:ctx.info[k] for k in ['GL_VENDOR','GL_RENDERER','GL_VERSION']},actual_shared_memory_limit=shared_memory_limit(ctx),available_ram_start_gib=start_ram,available_ram_end_gib=psutil.virtual_memory().available/2**30,torch_imported='torch' in sys.modules,training_updates=0,source_comparison='Exact immutable inputs/nativeCUDAarrays from RENDER09; no altered workload or precision',error=error)
        (out/'receipt.json').write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8');return j
    try:
        for number,spec in enumerate(plan['cases']):
            ram();input_path=source/f'case{number}_inputs.npz';reference_path=source/f'case{number}_parity.npz'
            with np.load(input_path,allow_pickle=False) as z:
                weights={k.removeprefix('weight__'):z[k].copy() for k in z.files if k.startswith('weight__')};dictionary=z['dictionary'].copy();ids=z['ids'].copy();present=z['present'].copy();rgb=z['rgb'].copy()
            gl=ComputeGLNCA(ctx,spec['width'],spec['height'],weights);gl.install_readout(weights['read.weight'],weights['read.bias'],dictionary)
            if spec['pixel']:gl.install_retina(weights);gl.initialize_image(rgb)
            else:gl.initialize(ids,present)
            for _ in range(16):gl.step()
            state=gl.state();logits=gl.logits(True)
            with np.load(reference_path,allow_pickle=False) as ref:
                ds=np.abs(state-ref['native_state']);dz=np.abs(logits-ref['native_logits']);checks.extend([dict(case=spec['name'],name='state',passed=bool(np.all(ds<=1e-4+1e-5*np.abs(ref['native_state']))),maxabs=float(ds.max())),dict(case=spec['name'],name='all_logits',passed=bool(np.all(dz<=1e-4+1e-5*np.abs(ref['native_logits']))),maxabs=float(dz.max())),dict(case=spec['name'],name='all_decisions',passed=bool(np.array_equal(logits.argmax(-1),ref['native_logits'].argmax(-1))))])
            np.savez_compressed(out/f'case{number}_compute.npz',state=state,logits=logits);rows.append(dict(case=spec['name'],input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),metadata=gl.metadata()));gl.release();save('running');print(json.dumps(dict(case=spec['name'],checks_passed=all(c['passed'] for c in checks))),flush=True)
        j=save('verified_all_seven_GL46_compute_workloads' if all(c['passed'] for c in checks) else 'failed_preserved');print(json.dumps(dict(status=j['status'],checks=len(checks),torch_imported=j['torch_imported'])))
    except Exception as e:save('failed_preserved',dict(type=type(e).__name__,message=str(e)));raise
    finally:ctx.release()
if __name__=='__main__':main()
