"""Preserved fixed tolerance: distinguish same-host operator and archival drift."""
import argparse,hashlib,io,json,os,sys,urllib.request,zipfile,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 refused')
    import numpy as np,torch
    import torch.nn.functional as F
    from neuropixel.research.vis07_models import make
    from neuropixel.research.vis07_data import parse,pixels,TEST_SHA
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);a.output.mkdir(parents=True,exist_ok=False)
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Agnuxo1/NeuroPixel/a783bb77d8e9a73a1c6f3fb624562ffde242ebd7/results/research/VIS07MAIN_cloud/37908785698/raw.zip',timeout=60).read(128*1024**2);assert hashlib.sha256(raw).hexdigest()=='6677eb3c5d8e04fad5d6ed2947c707bc5ab0761a12f654b4b5972cf9bd648ca3';rows=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        test=parse(z.read('original_uci.zip'),'optdigits.tes',TEST_SHA);image,_=pixels(test[:128]);image=torch.from_numpy(image)
        for seed in range(240,245):
            cp=torch.load(io.BytesIO(z.read(f'cases/seed{seed}/nca/best.pt')),map_location='cpu',weights_only=True);model=make('nca',seed);model.load_state_dict(cp['model']);core=model.core.eval();w=cp['model']
            with np.load(io.BytesIO(z.read(f'final/seed{seed}/nca/held_depth_controls.npz')),allow_pickle=False) as original,torch.inference_mode():
                # Same-host original Module path versus separately indexed functional path.
                native_ids=core.retina(image);ns=core.seed(native_ids);fi=image
                for ix,d in [(0,1),(2,2),(4,4),(6,8)]:fi=F.relu(F.conv2d(fi,w[f'core.retina.net.{ix}.weight'],w[f'core.retina.net.{ix}.bias'],padding=d,dilation=d))
                fi=F.conv2d(fi,w['core.retina.net.8.weight'],w['core.retina.net.8.bias']);fs=F.conv2d(fi,w['core.seed.weight'],w['core.seed.bias']);arrays={}
                for depth in range(33):
                    if depth in [0,4,8,16,32]:
                        nl=core.read(ns[:,:,4,4])@core.dictionary().T;nl[:,0]=-10000;fl=F.linear(fs[:,:,4,4],w['core.read.weight'],w['core.read.bias'])@w['core.embed.weight'].T;fl[:,0]=-10000;n=nl.numpy().copy();f=fl.numpy().copy();ref=original[f'logits_T{depth}'][:128];allowance=1e-4+1e-5*np.abs(ref)
                        rows.append(dict(seed=seed,depth=depth,native_functional_same_host_maxabs=float(np.max(np.abs(n-f))),native_functional_same_host_exact=bool(np.array_equal(n,f)),archived_native_maxabs=float(np.max(np.abs(n-ref))),archived_native_max_error_over_frozen_tolerance=float(np.max(np.abs(n-ref)/allowance)),archived_native_changed_decisions=int(np.sum(n.argmax(-1)!=ref.argmax(-1))),state_rms=float(np.sqrt(np.mean(ns.numpy().astype(np.float64)**2)))))
                        arrays[f'native_logits_T{depth}']=n;arrays[f'functional_logits_T{depth}']=f;arrays[f'archived_logits_T{depth}']=ref
                    if depth<32:
                        ns=ns+core.f2(torch.relu(core.f1(torch.cat([ns,core.perceive(ns),native_ids],1))));per=F.conv2d(fs,w['core.perceive.weight'],padding=1,groups=16);h=F.relu(F.conv2d(torch.cat([fs,per,fi],1),w['core.f1.weight'],w['core.f1.bias']));fs=fs+F.conv2d(h,w['core.f2.weight'],w['core.f2.bias'])
                np.savez_compressed(a.output/f'seed{seed}.npz',**arrays)
    cpu='unknown'
    if Path('/proc/cpuinfo').exists():cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
    result=dict(status='completed_fixed_tolerance_depth_diagnostic',rows=rows,subset='first128 original official test rows, fixed before execution',cpu_model=cpu,torch=torch.__version__,python=sys.version,training_updates=0,tolerance_unchanged=True,interpretation='Same-host equality tests implementation consistency; original-host differences remain recorded. Finite growth/categorical agreement do not establish asymptotic stability.',source_commit=os.environ.get('GITHUB_SHA'))
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],same_host_all_exact=all(r['native_functional_same_host_exact'] for r in rows),archival_violations=sum(r['archived_native_max_error_over_frozen_tolerance']>1 for r in rows))))
if __name__=='__main__':main()
