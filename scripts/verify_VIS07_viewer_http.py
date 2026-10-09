"""Validate archive viewer transport, reset/depth bounds and admission controls."""
import argparse,json,re,urllib.request,urllib.error
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8767');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    html=urllib.request.urlopen(args.url,timeout=5).read().decode();token=re.search("const token='([^']+)'",html).group(1);checks=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    def request(route,data=None,session_token=token):
        req=urllib.request.Request(args.url+route,data=None if data is None else json.dumps(data).encode(),headers={} if data is None else {'Content-Type':'application/json','X-NeuroPixel-Token':session_token})
        with urllib.request.urlopen(req,timeout=10) as r:return json.load(r)
    info=request('/api/info');check('explicit archive replay',info['mode']=='archive_replay');check('only registered archive sample',len(info['samples'])==1 and info['samples'][0]['label']==0)
    request('/api/reset',{'sample':0})
    with np.load(ROOT/'results/research/VIS07_learned_gpu/actual_rendered_train0_seed240_frames.npz',allow_pickle=False) as reference:
        for step in range(9):
            frame=request('/api/frame') if step==0 else request('/api/step',{})
            check(f'step{step} index',frame['step']==step)
            state=np.asarray(frame['state'],np.float32).reshape(16,8,8);logits=np.asarray(frame['logits'],np.float32).reshape(8,8,11)
            check(f'step{step} exact raw channels',np.array_equal(state,reference['states'][step]));check(f'step{step} exact complete logits',np.array_equal(logits,reference['logits'][step]));check(f'step{step} no new GPU inference',frame['gpu_inference_in_this_session'] is False)
            check(f'step{step} RMS all raw channels',abs(frame['rms']-np.sqrt(np.mean(state.astype(np.float64)**2)))<1e-12)
    check('depth8 cap',request('/api/step',{})['step']==8)
    try:request('/api/step',{},'invalid');check('reject foreign session',False)
    except urllib.error.HTTPError as e:check('reject foreign session',e.code==403)
    frame=request('/api/reset',{'sample':0});check('reset to original T0',frame['step']==0)
    result=dict(status='verified' if all(c['passed'] for c in checks) else 'failed',mode='archive_replay',checks=checks,new_training=0,new_GPU_inference=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],checks=len(checks))))
    if result['status']!='verified':raise SystemExit(1)
if __name__=='__main__':main()
