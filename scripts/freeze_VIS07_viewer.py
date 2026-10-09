"""Prospective viewer integration plan, unchanged archived trajectory limits."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    names=['scripts/live_VIS07_viewer.py','scripts/verify_live_VIS07_viewer.py','docs/viewer/index.html','docs/research/VIS07_viewer_protocol.md','neuropixel/rendering/nca_gl.py','neuropixel/rendering/nca_gl_vector.py','data/vis07_models/weights_seed240.json','data/vis07_models/gallery_train.json','results/research/VIS07_learned_gpu/actual_rendered_train0_seed240_frames.npz']
    plan=dict(registration='NP-VIS07-VIEWER-INTEGRATION-20261009',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names},depths=list(range(9)),selected_seed=240,reference_example='first original TRAIN class0',continuous_rule='abs(actual-reference)<=1e-4+1e-5*abs(reference)',categorical_rule='all scanner decisions exact',endpoints=['physical NVIDIA GPU','CPU/Mesa software renderer'],new_training=0,independent_external_replication=False)
    path=ROOT/'docs/research/VIS07_viewer_plan.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    for name in names:
        if name.endswith('.py'):compile((ROOT/name).read_text(encoding='utf-8'),name,'exec')
    print(json.dumps(dict(plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),new_training=0,new_inference=0)))
if __name__=='__main__':main()
