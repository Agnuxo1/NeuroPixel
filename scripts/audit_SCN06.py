"""Independent NumPy arithmetic for constructed native scanner arrays."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise ValueError(name)
    with np.load(a.input/'native.npz',allow_pickle=False) as z:
        for label,coupled in [('coupled',True),('inactive',False)]:
            states=np.zeros((2,3,4,1,1),dtype=np.float64);states[:,1,:,0,0]=[[1,0,0,0],[1,0,0,2]];states[:,2]=states[:,1]
            if coupled:states[:,2,1,0,0]+=np.maximum(0,states[:,1,3,0,0])
            check(label+' whole retained native state',np.array_equal(states,z[label+'_frames']))
            raw=np.zeros((2,3,1,1,4));raw[...,1:]=np.moveaxis(states[:,:,:3],2,-1)
            check(label+' full raw logits independent',np.array_equal(raw,z[label+'_raw_logits']))
            raw[...,0]=-10000;p=np.exp(raw-raw.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            check(label+' categorical summary independent',np.array_equal(p.argmax(-1),z[label+'_word']))
            check(label+' probability summary independent',np.allclose(p.max(-1),z[label+'_confidence'],atol=1e-14,rtol=0))
            check(label+' complete final native answer',np.array_equal(raw[:,-1,0,0],z[label+'_final_logits']))
        for label in ['compression','pad']:
            raw=z[label+'_raw_logits'].copy();raw[...,0]=-10000;p=np.exp(raw-raw.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            check(label+' argmax',np.array_equal(p.argmax(-1),z[label+'_word']))
            check(label+' confidence',np.allclose(p.max(-1),z[label+'_confidence'],atol=1e-14,rtol=0))
    result=dict(status='verified_independent_SCN06_saved_array_arithmetic',checks=checks,input_sha256=hashlib.sha256((a.input/'native.npz').read_bytes()).hexdigest(),issues=[],model_loads=0,training_updates=0)
    (a.input/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],checks=len(checks))))
if __name__=='__main__':main()
