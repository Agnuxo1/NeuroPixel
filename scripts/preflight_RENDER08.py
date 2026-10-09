"""Bounded engineering parity; no training or scientific speed claim."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--software',action='store_true');a=p.parse_args()
    import psutil
    def ram():
        v=psutil.virtual_memory().available/2**30
        if v<8:raise RuntimeError('RAM8 refused')
        return v
    source_files=['neuropixel/rendering/nca_gl.py','scripts/preflight_RENDER08.py','neuropixel/model.py']
    initial_source={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in source_files}
    start_ram=ram();import numpy as np,torch,moderngl
    from neuropixel.model import NeuroPixel
    from neuropixel.rendering.nca_gl import GLNCA,numpy_update,pack,unpack
    ram();torch.set_num_threads(1 if a.software else 2);torch.set_num_interop_threads(1 if a.software else 2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device='cpu' if a.software else 'cuda'
    if not a.software and not torch.cuda.is_available():raise RuntimeError('Actual CUDA device required')
    ctx=moderngl.create_standalone_context(require=330,**({'backend':'egl'} if a.software else {}))
    info={k:ctx.info[k] for k in ['GL_VENDOR','GL_RENDERER','GL_VERSION','GL_MAX_TEXTURE_SIZE']}
    if not a.software and 'NVIDIA' not in info['GL_VENDOR']:raise RuntimeError('Physical NVIDIA render context required')
    a.output.mkdir(parents=True,exist_ok=False);checks=[];rows=[];arrays={};start=time.perf_counter()
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    try:
        specs=[(4,3,8,1,1),(5,3,7,5,3),(16,8,32,8,8),(48,16,128,10,8)]
        for case,(c,cid,hidden,w,h) in enumerate(specs):
            ram();rng=np.random.default_rng(84100+case);torch.manual_seed(84100+case)
            m=NeuroPixel(7,(0,0),c_id=cid,c=c,hidden=hidden,steps=4,fire_rate=1).eval()
            with torch.no_grad():m.f2.weight.copy_(torch.from_numpy(rng.normal(0,.01,size=tuple(m.f2.weight.shape)).astype(np.float32)));m.f2.bias.copy_(torch.from_numpy(rng.normal(0,.01,size=c).astype(np.float32)))
            weights={k:v.detach().cpu().numpy().copy() for k,v in m.state_dict().items()};gl=GLNCA(ctx,w,h,weights)
            canvas=torch.from_numpy(rng.integers(0,7,size=(1,h,w),dtype=np.int64));ids=m.dictionary().detach()[canvas][0].numpy().transpose(2,0,1).copy();present=(canvas[0].numpy()!=0).astype(np.float32)
            for mode in ['full','checkerboard']:
                gl.initialize(ids,present);model=m.to(device)
                with torch.inference_mode():
                    t_ids=torch.from_numpy(ids[None]).to(device);t_present=torch.from_numpy(present[None,None]).to(device);state=model.seed(t_ids)*t_present
                    check(f'{case}/{mode}/seed',np.allclose(gl.state(),state[0].cpu().numpy(),atol=1e-5,rtol=1e-5))
                    oracle=state[0].cpu().numpy().astype(np.float64)
                    for step in range(1,5):
                        fire=np.ones((h,w),np.float32) if mode=='full' else ((np.indices((h,w)).sum(0)+step)%2).astype(np.float32)
                        gl.step(fire)
                        if a.software:ctx.finish()
                        features=torch.cat([state,model.perceive(state),t_ids],1);delta=model.f2(torch.relu(model.f1(features)));state=state+delta*torch.from_numpy(fire[None,None]).to(device)
                        oracle=numpy_update(oracle,ids,weights,fire);actual=gl.state();native=state[0].cpu().numpy();error=float(np.max(np.abs(actual-native)))
                        allowance=1e-4+1e-5*np.abs(native);check(f'{case}/{mode}/step{step}/native',np.all(np.abs(actual-native)<=allowance),dict(max_abs=error,max_ratio=float(np.max(np.abs(actual-native)/allowance))))
                        check(f'{case}/{mode}/step{step}/oracle',np.all(np.abs(actual-oracle)<=1e-4+1e-5*np.abs(oracle)))
                        arrays[f'case{case}_{mode}_state{step}']=actual
                    # Explicitly clear the previous checker mask; None means ordinary eval.
                    gl.step(None);state=state+model.f2(torch.relu(model.f1(torch.cat([state,model.perceive(state),t_ids],1))))
                    check(f'{case}/{mode}/mask_reset',np.allclose(gl.state(),state[0].cpu().numpy(),atol=1e-4,rtol=1e-5))
                    raw=model.lens_logits(state)[0].cpu().numpy();r=gl.state().transpose(1,2,0).astype(np.float64)@weights['read.weight'].astype(np.float64).T+weights['read.bias'].astype(np.float64);ours=r@model.dictionary().detach().cpu().numpy().astype(np.float64).T
                    check(f'{case}/{mode}/complete_logits',np.allclose(ours,raw,atol=1e-4,rtol=1e-5));ours[...,0]=-10000;raw[...,0]=-10000;check(f'{case}/{mode}/all_decisions',np.array_equal(ours.argmax(-1),raw.argmax(-1)))
                    check(f'{case}/{mode}/pack_roundtrip',np.array_equal(unpack(pack(native).tobytes(),c,h,w),native))
                    rows.append(dict(case=case,channels=c,identity=cid,hidden=hidden,width=w,height=h,mode=mode,shader=gl.metadata()))
                model=m.cpu()
            gl.release();ram()
        check('source_unchanged_during_fixture',initial_source=={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in source_files})
    finally:
        ctx.release()
        np.savez_compressed(a.output/'states.npz',**arrays)
        result=dict(status='passed' if checks and all(x['passed'] for x in checks) else 'failed',checks=checks,rows=rows,graphics=info,physical_gpu=not a.software,tensor_device=device,torch_version=torch.__version__,python=sys.version,available_ram_start_gib=start_ram,available_ram_end_gib=psutil.virtual_memory().available/2**30,wall_seconds=time.perf_counter()-start,source_sha256={str(k):hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in ['neuropixel/rendering/nca_gl.py','scripts/preflight_RENDER08.py','neuropixel/model.py']},trained_models_loaded=0,training_updates=0,scientific_speed_benchmark=False)
        (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],checks=len(checks),physical_gpu=result['physical_gpu'],graphics=info)))
    if result['status']!='passed':raise RuntimeError('Keep original preflight discrepancies')
if __name__=='__main__':main()
