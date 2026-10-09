"""Same-device native CUDA eager/CUDA graph versus float framebuffer rendering.

No training, accuracy or universal architecture claim. Registered complete
parity gate precedes all timings. Energy is measured at the whole GPU device.
"""
import argparse,hashlib,json,os,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import psutil
    def ram():
        v=psutil.virtual_memory().available/2**30
        if v<8:raise RuntimeError('RAM8 refused')
        return v
    ram();plan_path=ROOT/'docs/research/RENDER09_benchmark_plan.json';plan=json.loads(plan_path.read_bytes())
    for k,h in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=h:raise ValueError('Frozen benchmark source changed '+k)
    import numpy as np,torch,moderngl,pynvml
    from neuropixel.model import NeuroPixel,Retina
    from neuropixel.rendering.nca_gl import GLNCA as ScalarGLNCA
    from neuropixel.rendering.nca_gl_vector import VectorGLNCA as GLNCA
    ram();torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    ctx=moderngl.create_standalone_context(require=330);pynvml.nvmlInit();device=pynvml.nvmlDeviceGetHandleByIndex(0)
    if 'NVIDIA' not in ctx.info['GL_VENDOR']:raise RuntimeError('Physical NVIDIA render required')
    if pynvml.nvmlDeviceGetCount()!=1 or pynvml.nvmlDeviceGetName(device) not in ctx.info['GL_RENDERER']:raise RuntimeError('One verified shared physical GPU required')
    a.output.mkdir(parents=True,exist_ok=False);rows=[];cases=[];checks=[];energy_supported=True
    try:pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
    except pynvml.NVMLError:energy_supported=False
    def energy():return pynvml.nvmlDeviceGetTotalEnergyConsumption(device)/1000 if energy_supported else None
    def record():
        result=dict(status='running',plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),checks=checks,rows=rows,cases=[dict(spec=c['spec'],setup_seconds=c['setup'],graph_available=c['graph'] is not None,graph_failure=c['graph_error'],renderer=c['gl'].metadata()) for c in cases],hardware=dict(cuda_device=torch.cuda.get_device_name(0),gl_vendor=ctx.info['GL_VENDOR'],gl_renderer=ctx.info['GL_RENDERER'],gl_version=ctx.info['GL_VERSION'],driver=pynvml.nvmlSystemGetDriverVersion()),software=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,moderngl=moderngl.__version__),gpu_energy_counter_supported=energy_supported,energy_scope='Whole GPU device including background/idle; excludes CPU, RAM, PSU and wall energy',new_trainings=0,scientific_independent_replicas=0)
        (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result
    try:
        for number,spec in enumerate(plan['cases']):
            ram();t0=time.perf_counter();c,cid,hidden,w,h=spec['c'],spec['cid'],spec['hidden'],spec['width'],spec['height'];rng=np.random.default_rng(85200+number);torch.manual_seed(85200+number)
            m=NeuroPixel(spec['vocab'],(0,0),c_id=cid,c=c,hidden=hidden,steps=16,fire_rate=1,retina=spec['pixel']).eval()
            if spec['pixel']:m.retina=Retina(cid,w=8)
            with torch.no_grad():m.f2.weight.copy_(torch.from_numpy(rng.normal(0,.01,size=tuple(m.f2.weight.shape)).astype(np.float32)));m.f2.bias.copy_(torch.from_numpy(rng.normal(0,.01,size=c).astype(np.float32)))
            weights={k:v.detach().numpy().copy() for k,v in m.state_dict().items()};dictionary=m.dictionary().detach().numpy().copy();gl=GLNCA(ctx,w,h,weights);gl.install_readout(weights['read.weight'],weights['read.bias'],dictionary);scalar_gl=ScalarGLNCA(ctx,w,h,weights);scalar_gl.install_readout(weights['read.weight'],weights['read.bias'],dictionary)
            present=(rng.uniform(size=(h,w))>.3).astype(np.float32);ids=rng.uniform(-.1,.1,size=(cid,h,w)).astype(np.float32)*present[None];rgb=rng.uniform(0,1,size=(3,h,w)).astype(np.float32)
            if spec['pixel']:gl.install_retina(weights);gl.initialize_image(rgb);scalar_gl.install_retina(weights);scalar_gl.initialize_image(rgb);present[:]=1
            else:gl.initialize(ids,present);scalar_gl.initialize(ids,present)
            arrays={**{'weight__'+k:v for k,v in weights.items()},'dictionary':dictionary,'ids':ids,'present':present,'rgb':rgb}
            np.savez_compressed(a.output/f'case{number}_inputs.npz',**arrays)
            model=m.cuda();t_ids=torch.from_numpy(ids[None]).cuda();t_present=torch.from_numpy(present[None,None]).cuda();t_rgb=torch.from_numpy(rgb[None]).cuda()
            # Default arguments bind each case; references remain valid after loop advances.
            def tensor(model=model,t_ids=t_ids,t_present=t_present,t_rgb=t_rgb,pixel=spec['pixel']):
                with torch.inference_mode():
                    identity=model.retina(t_rgb) if pixel else t_ids;state=model.seed(identity)*t_present
                    for _ in range(16):state=state+model.f2(torch.relu(model.f1(torch.cat([state,model.perceive(state),identity],1))))
                    logits=model.lens_logits(state);logits[...,0]=-10000
                    return state,logits
            def render(gl=gl,pixel=spec['pixel']):
                if pixel:
                    texture=gl.rgb
                    for stage in gl.retina_passes:texture=stage.draw(texture)
                    gl.active_ids=texture
                gl.restart()
                for _ in range(16):gl.step()
                return gl.readout_texture(True)
            def scalar_render(gl=scalar_gl,pixel=spec['pixel']):
                if pixel:
                    texture=gl.rgb
                    for stage in gl.retina_passes:texture=stage.draw(texture)
                    gl.active_ids=texture
                gl.restart()
                for _ in range(16):gl.step()
                return gl.readout_texture(True)
            graph=None;graph_error=None;graph_output=None
            warmup_stream=torch.cuda.Stream();warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):tensor()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize();ctx.finish()
            try:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):graph_output=tensor()
                graph.replay();torch.cuda.synchronize()
            except Exception as e:graph=None;graph_error=type(e).__name__
            native=tensor();torch.cuda.synchronize();render();ctx.finish();gs=gl.state();gz=gl.logits(True);ts=native[0][0].cpu().numpy();tz=native[1][0].cpu().numpy()
            def check(label,ok,detail=None):checks.append(dict(case=spec['name'],name=label,passed=bool(ok),detail=detail))
            check('same native state',np.all(np.abs(gs-ts)<=1e-4+1e-5*np.abs(ts)),float(np.max(np.abs(gs-ts))))
            check('same complete scanner logits',np.all(np.abs(gz-tz)<=1e-4+1e-5*np.abs(tz)),float(np.max(np.abs(gz-tz))))
            check('same all categorical scanner decisions',np.array_equal(gz.argmax(-1),tz.argmax(-1)))
            scalar_render();ctx.finish();ss=scalar_gl.state();sz=scalar_gl.logits(True)
            check('scalar same native state',np.all(np.abs(ss-ts)<=1e-4+1e-5*np.abs(ts)))
            check('scalar same scanner logits',np.all(np.abs(sz-tz)<=1e-4+1e-5*np.abs(tz)))
            check('scalar same decisions',np.array_equal(sz.argmax(-1),tz.argmax(-1)))
            if graph is not None:
                graph.replay();torch.cuda.synchronize();check('graph/eager states exact',torch.equal(graph_output[0],native[0]));check('graph/eager logits exact',torch.equal(graph_output[1],native[1]))
            np.savez_compressed(a.output/f'case{number}_parity.npz',render_state=gs,native_state=ts,render_logits=gz,native_logits=tz)
            cases.append(dict(spec=spec,gl=gl,tensor=tensor,render=render,graph=graph,graph_output=graph_output,graph_error=graph_error,scalar_gl=scalar_gl,scalar_render=scalar_render,setup=time.perf_counter()-t0));record()
        if not all(x['passed'] for x in checks):raise RuntimeError('Entire parity gate failed; no speed measurements admitted')
        for number,case in enumerate(cases):
            ram();methods=['cuda_eager']+(['cuda_graph'] if case['graph'] is not None else [])+['scalar_render','render']
            for block in range(plan['technical_blocks']):
                order=methods if block%2==0 else list(reversed(methods))
                for method in order:
                    ram();torch.cuda.synchronize();ctx.finish();time.sleep(plan['idle_bracket_seconds']);eb0=energy();bt0=time.perf_counter();time.sleep(plan['idle_bracket_seconds']);bt=time.perf_counter()-bt0;eb1=energy()
                    idle_w=None if eb0 is None else (eb1-eb0)/bt
                    e0=energy();host=time.perf_counter();device_ms=None
                    if method in ['render','scalar_render']:
                        query=ctx.query(time=True)
                        with query:
                            for _ in range(plan['inner_repetitions']):(case['render'] if method=='render' else case['scalar_render'])()
                        ctx.finish();device_ms=query.elapsed/1e6
                    else:
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                        for _ in range(plan['inner_repetitions']):
                            if method=='cuda_graph':case['graph'].replay()
                            else:case['tensor']()
                        end.record();torch.cuda.synchronize();device_ms=begin.elapsed_time(end)
                    host_s=time.perf_counter()-host;e1=energy();gross=None if e0 is None else e1-e0
                    rows.append(dict(case=case['spec']['name'],block=block,method=method,repetitions=plan['inner_repetitions'],host_seconds=host_s,gpu_device_ms=device_ms,gpu_gross_energy_j=gross,idle_gpu_w=idle_w,idle_subtracted_gpu_energy_j=None if gross is None else gross-idle_w*host_s,energy_scope='whole device',available_ram_gib=ram()));record()
            print(json.dumps(dict(case=case['spec']['name'],blocks=plan['technical_blocks'],methods=methods)),flush=True)
        result=record();result['status']='completed_all_registered_RENDER09_workloads';result['all_parity_passed']=True;result['technical_repeats_are_not_scientific_replicas']=True
        for k,h in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=h:raise ValueError('Source changed during measurements')
        (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    except BaseException as e:
        result=record();result.update(status='failed_preserved',error_type=type(e).__name__,error_message=str(e));(a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');raise
    finally:
        for case in cases:case['gl'].release();case['scalar_gl'].release()
        ctx.release();pynvml.nvmlShutdown()
if __name__=='__main__':main()
