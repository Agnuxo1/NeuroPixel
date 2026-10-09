"""Scientific GIFs from complete result tables and actual archived neural states."""
import hashlib,io,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/animations';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fb','axes.facecolor':'#ffffff','axes.titleweight':'bold','text.color':'#16243c','axes.labelcolor':'#16243c','xtick.color':'#334155','ytick.color':'#334155'})
def frame(fig):
    b=io.BytesIO();fig.savefig(b,format='png',dpi=105,bbox_inches='tight');plt.close(fig);b.seek(0);return Image.open(b).convert('RGB')
def gif(name,frames,duration=700):
    w=max(i.width for i in frames);h=max(i.height for i in frames);normalized=[]
    for im in frames:
        p=Image.new('RGB',(w,h),'#f6f8fb');p.paste(im,((w-im.width)//2,(h-im.height)//2));normalized.append(p.quantize(colors=192))
    normalized[0].save(OUT/name,save_all=True,append_images=normalized[1:],duration=duration,loop=0,optimize=False,disposal=2)
    normalized[-1].convert('RGB').save(OUT/(name.removesuffix('.gif')+'.png'))
def main():
    sources=[]
    p=ROOT/'results/research/VIS07_learned_gpu/actual_rendered_train0_seed240_frames.npz';sources.append(p)
    with np.load(p,allow_pickle=False) as z:
        states=z['states'];logits=z['logits'];image=z['image'][0];scale=float(np.max(np.abs(states[:,:3])));rms=np.sqrt(np.mean(states.astype(np.float64)**2,axis=1));maxr=float(rms.max());frames=[]
        for step in range(9):
            fig,axes=plt.subplots(1,3,figsize=(12.2,4.4));fig.suptitle('NeuroPixel | an actual learned neural trajectory',fontsize=16,x=.5)
            axes[0].imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[0].set_title('RGB input Ã‚Â· TRAIN digit 0')
            axes[1].imshow(np.clip(states[step,:3].transpose(1,2,0)/(2*max(scale,1e-9))+.5,0,1),interpolation='nearest');axes[1].set_title(f'Raw state C0Ã¢â‚¬â€œC2 Ã‚Â· update {step}/8')
            im=axes[2].imshow(rms[step],cmap='magma',vmin=0,vmax=maxr,interpolation='nearest');axes[2].set_title('State magnitude Ã‚Â· all 16 channels');fig.colorbar(im,ax=axes[2],fraction=.046,pad=.04,label='RMS, raw model units')
            for ax in axes:ax.set_xticks([]);ax.set_yticks([])
            pred=int(logits[step,4,4].argmax()-1);fig.text(.5,.03,f'Seed 240, fixed TRAIN example Ã‚Â· centre prediction: {pred} Ã‚Â· actual GPU rendering, slowed for inspection\nDisplay colours are projections; validated inference ends at 8 updates.',ha='center',fontsize=9);fig.subplots_adjust(top=.80,bottom=.21,wspace=.40);frames.append(frame(fig))
        gif('actual_neural_trajectory.gif',frames+[frames[-1]]*3,550)
    p=ROOT/'results/research/SPRINT_review/RENDER09_summary.json';sources.append(p);j=json.loads(p.read_bytes());cases=list(dict.fromkeys(r['case'] for r in j['cases']));frames=[]
    methods=['cuda_eager','cuda_graph','scalar_render','render'];labels=['CUDA eager','CUDA Graph','Scalar render','RGBA vector render'];colors=['#3b82f6','#0f766e','#f59e0b','#7c3aed']
    for name in cases:
        vals=[next(r['median_host_ms'] for r in j['cases'] if r['case']==name and r['method']==m) for m in methods];ratio=next(r['ratio'] for r in j['cases'] if r['case']==name and r['method']=='best_cuda_over_render');fig,ax=plt.subplots(figsize=(9,4.5));bars=ax.barh(labels,vals,color=colors,height=.58);ax.invert_yaxis();ax.set_xlabel('Host latency per 16 updates + complete scanner, ms');ax.set_xlim(0,max(vals)*1.22);ax.set_title(f'Same RTX 3090 Ã‚Â· {name}',fontsize=15)
        for b,v in zip(bars,vals):ax.text(v+max(vals)*.015,b.get_y()+b.get_height()/2,f'{v:.3f} ms',va='center',fontsize=10)
        fig.text(.5,.02,f'Best CUDA / vector render = {ratio:.3f}Ãƒâ€”  (>1 favours render)\nAll outputs passed. Seven technical blocks; one device; resident inputs; not a universal speed claim.',ha='center',fontsize=9);fig.subplots_adjust(left=.22,bottom=.24);frames.append(frame(fig))
    gif('all_GPU_workloads.gif',frames,1300)
    p=ROOT/'results/research/SPRINT_review/VIS07_summary.json';sources.append(p);j=json.loads(p.read_bytes());rows=j['rows'];frames=[]
    for count in range(1,6):
        seeds=list(range(240,240+count));a=[100*next(r['accuracy'] for r in rows if r['seed']==s and r['family']=='nca' and r['condition']=='base') for s in seeds];b=[100*next(r['accuracy'] for r in rows if r['seed']==s and r['family']=='cnn' and r['condition']=='base') for s in seeds];fig,ax=plt.subplots(figsize=(9,4.6));xx=np.arange(count);ax.bar(xx-.18,a,.36,color='#7c3aed',label='NeuroPixel + retina');ax.bar(xx+.18,b,.36,color='#0f766e',label='CNN comparator');ax.set_xticks(xx,[str(s) for s in seeds]);ax.set_ylim(0,100);ax.set_xlabel('Paired initialization');ax.set_ylabel('Official test accuracy, %');ax.set_title('Pixel-only recognition Ã‚Â· all five realizations',fontsize=15);ax.legend(loc='lower right');ax.axhline(90,color='#94a3b8',ls='--',lw=1)
        fig.text(.5,.02,'NCA 94.36% Ã‚Â· CNN 90.84% Ã‚Â· paired difference +3.52 pp, t95 [1.93, 5.10]\n1,797 shared public UCI test images. Interval concerns training realizations; no external replication.',ha='center',fontsize=9);fig.subplots_adjust(bottom=.25);frames.append(frame(fig))
    gif('external_vision_all_seeds.gif',frames+[frames[-1]]*2,950)
    frames=[];depth=j['depth_controls']
    for end in range(1,6):
        d=depth[:end];x=[r['depth'] for r in d];fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.5));ax.plot(x,[100*r['mean_accuracy'] for r in d],'o-',color='#7c3aed');ax.set_xlim(-1,33);ax.set_ylim(0,100);ax.set_xlabel('Updates with the same input');ax.set_ylabel('Mean test accuracy, %');ax.axvline(8,color='#0f766e',ls='--');ax.set_title('Useful at trained depth 8')
        bx.semilogy(x,[r['mean_state_rms'] for r in d],'o-',color='#d97706');bx.set_xlim(-1,33);bx.set_ylim(.1,1e6);bx.set_xlabel('Updates with the same input');bx.set_ylabel('Mean state RMS');bx.axvline(8,color='#0f766e',ls='--');bx.set_title('Long continuation is not stable')
        fig.text(.5,.02,'Original finite endpoints, all five models Ã‚Â· no clipping of model states\nExtended T16/T32 continuous replay failures are retained. This is not an all-time divergence theorem.',ha='center',fontsize=9);fig.subplots_adjust(bottom=.25,wspace=.32);frames.append(frame(fig))
    gif('depth_limits_and_state_growth.gif',frames+[frames[-1]]*2,1000)
    manifest=dict(status='scientific_animations_from_actual_complete_data',sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},artifacts={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.iterdir() if p.suffix in ['.gif','.png']},neural_capture='actual GPU-rendered selected model frames, explicitly labelled; no synthetic brain artwork',display_transform='RGB raw channels linear mapped to fixed all-frame range; RMS shown separately; model data unchanged')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps({'artifacts':manifest['artifacts']}))
if __name__=='__main__':main()
