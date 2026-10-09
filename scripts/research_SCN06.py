"""Prospective native scanner witnesses on the unchanged DEV04 core."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 admission refused')
    plan=ROOT/'docs/research/SCN06_execution_plan.json';j=json.loads(plan.read_bytes())
    for name,h in j['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source differs '+name)
    if (a.output/'result.json').exists():raise ValueError('Closed experiment cannot repeat')
    import numpy as np,torch
    from neuropixel.model import NeuroPixel
    from neuropixel.scanner import lens
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(246)
    start=time.perf_counter();cpu=time.process_time();a.output.mkdir(parents=True,exist_ok=True);arrays={};checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise ValueError(name)
    def model(coupled):
        m=NeuroPixel(4,(0,0),c_id=3,c=4,hidden=8,steps=2,fire_rate=1).double().eval()
        for v in m.parameters():v.data.zero_()
        m.embed.weight.data[1:]=torch.eye(3,dtype=torch.float64);m.read.weight.data[:,:3]=torch.eye(3,dtype=torch.float64)
        if coupled:m.f1.weight.data[0,3,0,0]=1;m.f2.weight.data[1,0,0,0]=1
        return m
    with torch.no_grad():
        for label,coupled in [('coupled',True),('inactive',False)]:
            m=model(coupled);state=torch.tensor([[1,0,0,0],[1,0,0,2]],dtype=torch.float64)[:,:,None,None]
            canvas=torch.zeros((2,1,1),dtype=torch.long)
            out=m(canvas,trace=True,lens_every=1,hook=lambda t,s:state.clone() if t==1 else s)
            frames=out['frames'];flat=frames.reshape(6,4,1,1);raw=m.lens_logits(flat).reshape(2,3,1,1,4);masked=raw.clone();masked[...,0]=-10000
            word,conf=lens(m,flat);word=word.reshape(2,3,1,1);conf=conf.reshape(2,3,1,1)
            arrays[label+'_frames']=frames.numpy();arrays[label+'_raw_logits']=raw.numpy();arrays[label+'_masked_logits']=masked.numpy();arrays[label+'_word']=word.numpy();arrays[label+'_confidence']=conf.numpy();arrays[label+'_pre_update_lens']=out['lens'].numpy();arrays[label+'_final_logits']=out['logits'].numpy()
            check(label+' initial raw logits identical',torch.equal(raw[0,1],raw[1,1]))
            check(label+' scanner at intervention identical',torch.equal(word[0,1],word[1,1]) and torch.equal(conf[0,1],conf[1,1]))
            check(label+' pre-update timing',torch.equal(out['lens'][:,0],raw[:,0]) and torch.equal(out['lens'][:,1],raw[:,1]))
            check(label+' next token contrast',bool(word[0,2].item()!=word[1,2].item())==coupled)
            arrays[label+'_canvas']=canvas.numpy();check(label+' future input all PAD',torch.count_nonzero(canvas).item()==0)
            for k,v in m.state_dict().items():arrays[label+'_weight__'+k]=v.numpy()
            for k,v in m.named_buffers():arrays[label+'_weight__'+k]=v.numpy()
        m=model(False)
        states=torch.tensor([[np.log(.6),np.log(.3),np.log(.1),0],[np.log(.6),np.log(.1),np.log(.3),0]],dtype=torch.float64)[:,:,None,None]
        word,conf=lens(m,states);raw=m.lens_logits(states);arrays['compression_states']=states.numpy();arrays['compression_raw_logits']=raw.numpy();arrays['compression_word']=word.numpy();arrays['compression_confidence']=conf.numpy()
        check('top summary compresses distinct vectors',torch.equal(word[0],word[1]) and abs(float(conf[0]-conf[1]))<=1e-12 and not torch.equal(raw[0],raw[1]))
        states=torch.tensor([[-1,-2,-3,0],[-20000,-20001,-20002,0]],dtype=torch.float64)[:,:,None,None];word,conf=lens(m,states);raw=m.lens_logits(states)
        arrays['pad_states']=states.numpy();arrays['pad_raw_logits']=raw.numpy();arrays['pad_word']=word.numpy();arrays['pad_confidence']=conf.numpy()
        check('finite sentinel has documented extreme limit',word[:,0,0].tolist()==[1,0])
    check('RAM8 finish',psutil.virtual_memory().available/2**30>=8)
    np.savez_compressed(a.output/'native.npz',**arrays)
    result=dict(status='verified_native_SCN06_witnesses',source_commit=os.environ.get('GITHUB_SHA'),workflow_run_id=os.environ.get('GITHUB_RUN_ID'),plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),checks=checks,training_updates=0,learned_forward_calls=0,constructed_models=3,wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,available_ram_gib=psutil.virtual_memory().available/2**30,native_npz_sha256=hashlib.sha256((a.output/'native.npz').read_bytes()).hexdigest(),limits=['Assigned weights/states; no learned semantic causal mechanism or reachable-manifold claim','Top summary differs from full probability vector; finite PAD policy retained','No visual task or external replication'])
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],checks=len(checks))))
if __name__=='__main__':main()
