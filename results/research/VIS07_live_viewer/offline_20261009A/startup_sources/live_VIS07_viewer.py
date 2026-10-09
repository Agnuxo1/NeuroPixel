"""Loopback viewer: actual GPU computation or explicitly labelled archive replay."""
import argparse,hashlib,json,secrets,sys,time
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

class Engine:
    def __init__(self,offline):
        import numpy as np
        self.np=np;self.offline=offline;self.step_index=0;self.context=None;self.gl=None
        self.archive=ROOT/'results/research/VIS07_learned_gpu/actual_rendered_train0_seed240_frames.npz'
        with np.load(self.archive,allow_pickle=False) as z:
            self.reference_states=z['states'].copy();self.reference_logits=z['logits'].copy();self.reference_image=z['image'][0].copy()
        self.colour_range=[float(self.reference_states[:,:3].min()),float(self.reference_states[:,:3].max())]
        self.gallery=json.loads((ROOT/'data/vis07_models/gallery_train.json').read_bytes())['samples']
        self.samples=self.gallery[:1] if offline else self.gallery
        assert self.samples[0]['label']==0
        self.checks=[]
        if not offline:
            import psutil
            if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 admission refused')
            import moderngl
            from neuropixel.rendering.nca_gl_vector import VectorGLNCA
            descriptor=json.loads((ROOT/'data/vis07_models/weights_seed240.json').read_bytes())
            weights={k:np.asarray(v['values'],np.float32).reshape(v['shape']) for k,v in descriptor['weights'].items()}
            self.context=moderngl.create_standalone_context(require=330)
            if 'NVIDIA' not in self.context.info['GL_VENDOR']:raise RuntimeError('Actual NVIDIA GPU required')
            self.gl=VectorGLNCA(self.context,8,8,weights);self.gl.install_retina(weights);self.gl.install_readout(weights['read.weight'],weights['read.bias'],weights['embed.weight'])
            self.reset(0)
            for step in range(9):
                if step:self.advance()
                state,logits=self.arrays()
                for name,a,b in [('state',state,self.reference_states[step]),('logits',logits,self.reference_logits[step])]:
                    self.checks.append(dict(step=step,name=name,passed=bool(np.all(np.abs(a-b)<=1e-4+1e-5*np.abs(b)))))
                self.checks.append(dict(step=step,name='decisions',passed=bool(np.array_equal(logits.argmax(-1),self.reference_logits[step].argmax(-1)))))
            if not all(c['passed'] for c in self.checks):raise RuntimeError('Original trained trajectory gate failed')
        self.reset(0)
    def reset(self,label):
        np=self.np;sample=next((s for s in self.samples if s['label']==label),None)
        if sample is None:raise ValueError('Unknown registered TRAIN sample')
        self.image=np.asarray(sample['pixels'],np.float32).reshape(8,8)/16;self.label=label;self.step_index=0
        if self.offline:assert np.array_equal(self.image,self.reference_image)
        else:
            self.guard();self.gl.initialize_image(np.repeat(self.image[None],3,axis=0))
    def guard(self):
        if not self.offline:
            import psutil
            if psutil.virtual_memory().available/2**30<8:raise RuntimeError('RAM8 runtime refused')
    def advance(self):
        if self.step_index<8:
            if not self.offline:self.guard();self.gl.step()
            self.step_index+=1
    def arrays(self):
        if self.offline:return self.reference_states[self.step_index],self.reference_logits[self.step_index]
        self.guard();return self.gl.state(),self.gl.logits(True)
    def frame(self):
        np=self.np;state,logits=self.arrays()
        return dict(step=self.step_index,state=state.reshape(-1).tolist(),logits=logits.reshape(-1).tolist(),image=self.image.reshape(-1).tolist(),colour_range=self.colour_range,rms=float(np.sqrt(np.mean(state.astype(np.float64)**2))),minimum=float(state.min()),maximum=float(state.max()),gpu_inference_in_this_session=not self.offline)
    def info(self):
        return dict(mode='archive_replay' if self.offline else 'actual_GPU_inference',mode_label='REPRODUCCIÓN: estados GPU archivados. No se ejecuta la red en esta sesión.' if self.offline else 'EN VIVO: cada paso ejecuta la red entrenada mediante renderizado GPU.',scope='VIS07 · modelo seleccionado seed240 · ejemplo TRAIN fijo · ocho pasos · sin nuevo entrenamiento ni nueva precisión de test',samples=[dict(label=s['label'],original_train_index=s['original_train_index']) for s in self.samples])
    def close(self):
        if self.gl:self.gl.release()
        if self.context:self.context.release()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline-replay',action='store_true');parser.add_argument('--port',type=int,default=8767);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if not 1024<=args.port<=65535:raise ValueError('Invalid local port')
    args.output.mkdir(parents=True,exist_ok=False);token=secrets.token_urlsafe(24);engine=Engine(args.offline_replay)
    names=['scripts/live_VIS07_viewer.py','docs/viewer/index.html','data/vis07_models/weights_seed240.json','data/vis07_models/gallery_train.json','neuropixel/rendering/nca_gl.py','neuropixel/rendering/nca_gl_vector.py',engine.archive.relative_to(ROOT).as_posix()]
    receipt=dict(status='ready',mode=engine.info()['mode'],new_training=0,checks=engine.checks,gpu_inference_in_this_session=not args.offline_replay,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},hardware=None if args.offline_replay else {k:engine.context.info[k] for k in ['GL_VENDOR','GL_RENDERER','GL_VERSION']},python=sys.version,opened_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,body,kind='application/json'):
            data=body.encode('utf-8') if isinstance(body,str) else json.dumps(body).encode('utf-8');self.send_response(status);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                if self.path=='/':self.send(200,(ROOT/'docs/viewer/index.html').read_text(encoding='utf-8').replace('__TOKEN__',token),'text/html')
                elif self.path=='/api/info':self.send(200,engine.info())
                elif self.path=='/api/frame':self.send(200,engine.frame())
                else:self.send(404,dict(error='Unknown route'))
            except Exception as e:self.send(503,dict(error=str(e)))
        def do_POST(self):
            if self.headers.get('X-NeuroPixel-Token')!=token or self.headers.get('Content-Type')!='application/json':self.send(403,dict(error='Local session token required'));return
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{args.port}'):self.send(403,dict(error='Local origin required'));return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length>1024:raise ValueError('Request too large')
                body=json.loads(self.rfile.read(length))
                if self.path=='/api/step':engine.advance()
                elif self.path=='/api/reset':engine.reset(int(body['sample']))
                else:self.send(404,dict(error='Unknown route'));return
                self.send(200,engine.frame())
            except Exception as e:self.send(503,dict(error=str(e)))
    server=HTTPServer(('127.0.0.1',args.port),Handler);print(json.dumps(dict(url=f'http://127.0.0.1:{args.port}',mode=receipt['mode'],checks=len(engine.checks))),flush=True)
    try:server.serve_forever()
    finally:server.server_close();engine.close()
if __name__=='__main__':main()
