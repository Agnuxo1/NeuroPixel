"""Actual selected-model CPU trace for publication; no generated fake neural image."""
import argparse,hashlib,io,json,os,sys,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import psutil
    if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 admission refused')
    import numpy as np,torch
    from neuropixel.research.vis07_models import make
    from neuropixel.research.vis07_data import development,pixels
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);a.output.mkdir(parents=True,exist_ok=False)
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Agnuxo1/NeuroPixel/a783bb77d8e9a73a1c6f3fb624562ffde242ebd7/results/research/VIS07MAIN_cloud/37908785698/raw.zip',timeout=60).read(128*1024**2);assert hashlib.sha256(raw).hexdigest()=='6677eb3c5d8e04fad5d6ed2947c707bc5ab0761a12f654b4b5972cf9bd648ca3'
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        cp_raw=z.read('cases/seed240/nca/best.pt');cp=torch.load(io.BytesIO(cp_raw),map_location='cpu',weights_only=True);model=make('nca',240);model.load_state_dict(cp['model']);model.eval();train,ids,_=development(z.read('original_uci.zip'));index=int(next(i for i in ids if train[i,-1]==0));images,targets=pixels(train[[index]])
        with torch.inference_mode():
            core=model.core;result=core(torch.zeros((1,8,8),dtype=torch.long),rgb=torch.from_numpy(images),cam=torch.ones((1,8,8),dtype=torch.bool),trace=True,steps=8);frames=result['frames'][0].numpy();logits=core.lens_logits(result['frames'][0]).numpy();logits[...,0]=-10000
        np.savez_compressed(a.output/'actual_train0_seed240_frames.npz',image=images[0],states=frames,logits=logits,steps=np.arange(9),label=np.int64(0),original_train_index=np.int64(index))
        receipt=dict(status='actual_selected_model_native_trace',seed=240,model_checkpoint_sha256=hashlib.sha256(cp_raw).hexdigest(),source_archive_sha256=hashlib.sha256(raw).hexdigest(),image_source='first original TRAIN class0 row, fixed before execution',original_train_index=index,steps=list(range(9)),predictions_center=(logits[:,4,4].argmax(-1)-1).tolist(),execution='CPU PyTorch native, not GPU capture',raw_trace_sha256=hashlib.sha256((a.output/'actual_train0_seed240_frames.npz').read_bytes()).hexdigest(),training_updates=0,scope='Finite registered8-step trajectory; RGB visual projection is not identified biological circuitry',source_commit=os.environ.get('GITHUB_SHA'))
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
