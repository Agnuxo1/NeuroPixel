"""Actual trained-model pixel render frames, fixed sample and per-seed fidelity."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
def main():
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 admission refused')
    import numpy as np,torch,moderngl
    import torch.nn.functional as F
    from neuropixel.research.vis07_models import make
    from neuropixel.rendering.nca_gl_vector import VectorGLNCA
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    root=Path('D:/PROJECTS/NP-VIS07-20261009');exports=root/'replay_original/export';out=ROOT/'results/research/VIS07_learned_gpu';out.mkdir(exist_ok=False);ctx=moderngl.create_standalone_context(require=330)
    if 'NVIDIA' not in ctx.info['GL_VENDOR']:raise RuntimeError('Actual NVIDIA GPU required')
    checks=[];rows=[]
    with np.load(root/'main_original/official_test_data.npz',allow_pickle=False) as z:images=z['image'][:64].copy();gold=z['target'][:64].copy()
    for seed in range(240,245):
        if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 before learned weights')
        descriptor=json.loads((exports/f'weights_seed{seed}.json').read_bytes());weights={k:np.asarray(v['values'],np.float32).reshape(v['shape']) for k,v in descriptor['weights'].items()};model=make('nca',seed);model.load_state_dict({'core.'+k:torch.from_numpy(v.copy()) for k,v in weights.items()});model.cuda().eval();gl=VectorGLNCA(ctx,8,8,weights);gl.install_retina(weights);gl.install_readout(weights['read.weight'],weights['read.bias'],weights['embed.weight'])
        with torch.inference_mode():native=model(torch.from_numpy(images).cuda()).cpu().numpy()
        actual=[]
        for image in images:
            gl.initialize_image(image)
            for _ in range(8):gl.step()
            actual.append(gl.logits(True)[4,4].copy())
        actual=np.stack(actual);pred=actual.argmax(-1);ref=native.argmax(-1);nll=F.cross_entropy(torch.from_numpy(actual),torch.from_numpy(gold),reduction='none').numpy();nnll=F.cross_entropy(torch.from_numpy(native),torch.from_numpy(gold),reduction='none').numpy();error=np.abs(actual-native);limit=1e-4+1e-5*np.abs(native)
        with np.load(root/'main_original/final'/f'seed{seed}'/'nca/base.npz',allow_pickle=False) as original:cpu_ref=original['logits'][:64].copy();cpu_pred=original['prediction'][:64].copy()
        check=dict(seed=seed,categorical_exact=bool(np.array_equal(pred,ref)),cuda_original_CPU_decisions_exact=bool(np.array_equal(ref,cpu_pred)),mean_nll_error=float(abs(nll.astype(np.float64).mean()-nnll.astype(np.float64).mean())),raw_logit_tolerance_passed=bool(np.all(error<=limit)),max_raw_logit_error=float(error.max()),max_raw_logit_ratio=float((error/limit).max()),cuda_original_CPU_max_raw_logit_error=float(np.max(np.abs(native-cpu_ref))));checks.append(check);np.savez_compressed(out/f'test64_seed{seed}.npz',render_logits=actual,cuda_logits=native,original_cpu_logits=cpu_ref,target=gold)
        if seed==240:
            # Fixed class0 first original TRAIN example, chosen before GPU inspection.
            gallery=json.loads((exports/'gallery_train.json').read_bytes());sample=next(x for x in gallery['samples'] if x['label']==0);gray=np.asarray(sample['pixels'],np.float32).reshape(8,8)/16;image=np.repeat(gray[None],3,axis=0);gl.initialize_image(image);states=[];logits=[]
            for step in range(9):
                if step:gl.step()
                states.append(gl.state());logits.append(gl.logits(True))
            np.savez_compressed(out/'actual_rendered_train0_seed240_frames.npz',image=image,states=np.stack(states),logits=np.stack(logits),steps=np.arange(9,dtype=np.int64),label=np.int64(0),original_train_index=np.int64(sample['original_train_index']))
        gl.release();print(json.dumps(check),flush=True)
    result=dict(status='completed_fixed_learned_GPU_fidelity_capture',checks=checks,subset='first64 official TEST rows, every seed, fixed before execution',frame_example='first original TRAIN class0, registered seed240',source_weight_hashes={f'weights_seed{s}.json':hashlib.sha256((exports/f'weights_seed{s}.json').read_bytes()).hexdigest() for s in range(240,245)},hardware={k:ctx.info[k] for k in ['GL_VENDOR','GL_RENDERER','GL_VERSION']},training_updates=0,tolerance_unchanged=True,all_decisions_exact=all(c['categorical_exact'] for c in checks),all_mean_nll_within_1e4=all(c['mean_nll_error']<=1e-4 for c in checks),all_pointwise_logit_checks_passed=all(c['raw_logit_tolerance_passed'] for c in checks))
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');ctx.release()
if __name__=='__main__':main()
