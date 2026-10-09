"""Bounded actual-GPU HTTP integration gate; run only under GPU FIFO admission."""
import argparse,hashlib,json,os,re,subprocess,sys,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='2'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--software',action='store_true');args=parser.parse_args()
    expected_mode='software_shader_inference' if args.software else 'actual_GPU_inference'
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 before live viewer admission')
    import numpy as np
    out=ROOT/'results/research/VIS07_live_viewer'/('software_20261009A' if args.software else 'live_20261009A');out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(ROOT/'scripts/live_VIS07_viewer.py'),'--port','8768','--duration','90','--output',str(out/'server')]
    if args.software:command.append('--software')
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    checks=[];base='http://127.0.0.1:8768';token=None
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    def request(route,data=None):
        req=urllib.request.Request(base+route,data=None if data is None else json.dumps(data).encode(),headers={} if data is None else {'Content-Type':'application/json','X-NeuroPixel-Token':token})
        with urllib.request.urlopen(req,timeout=10) as r:return json.load(r)
    try:
        line=process.stdout.readline()
        if not line:raise RuntimeError('Live viewer failed before serving: '+process.stderr.read()[-700:])
        ready=json.loads(line);check('declared actual execution ready',ready['mode']==expected_mode and ready['checks']==27)
        html=urllib.request.urlopen(base,timeout=5).read().decode();token=re.search("const token='([^']+)'",html).group(1)
        info=request('/api/info');check('explicit live mode',info['mode']==expected_mode);check('ten original TRAIN images',len(info['samples'])==10)
        request('/api/reset',{'sample':0})
        with np.load(ROOT/'results/research/VIS07_learned_gpu/actual_rendered_train0_seed240_frames.npz',allow_pickle=False) as reference:
            for step in range(9):
                frame=request('/api/frame') if step==0 else request('/api/step',{})
                check(f'step{step} declared inference mode',frame['neural_computation_in_this_session'] and frame['gpu_inference_in_this_session']==(not args.software) and frame['step']==step)
                state=np.asarray(frame['state'],np.float32).reshape(16,8,8);logits=np.asarray(frame['logits'],np.float32).reshape(8,8,11)
                for name,a,b in [('state',state,reference['states'][step]),('logits',logits,reference['logits'][step])]:check(f'step{step} {name}',np.all(np.abs(a-b)<=1e-4+1e-5*np.abs(b)))
                check(f'step{step} all decisions',np.array_equal(logits.argmax(-1),reference['logits'][step].argmax(-1)))
        check('depth cap',request('/api/step',{})['step']==8);check('reset',request('/api/reset',{'sample':0})['step']==0)
        # UI input controls: all ten fixed TRAIN samples initialize, without target tokens.
        for label in range(10):check(f'reset registered TRAIN digit{label}',request('/api/reset',{'sample':label})['step']==0)
        request('/api/close',{});process.wait(timeout=10);check('clean resource release',process.returncode==0)
        result=dict(status='verified' if all(c['passed'] for c in checks) else 'failed',checks=checks,new_training=0,mode=expected_mode,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scripts/live_VIS07_viewer.py','scripts/verify_live_VIS07_viewer.py','docs/viewer/index.html']})
        (out/'http_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],checks=len(checks))))
        if result['status']!='verified':raise RuntimeError('Live integration gate failed')
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error),checks=checks),indent=2)+'\n',encoding='utf-8');raise
    finally:
        if process.poll() is None:
            if token:
                try:request('/api/close',{})
                except Exception:pass
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
if __name__=='__main__':main()
