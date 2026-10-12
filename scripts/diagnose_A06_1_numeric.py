"""Post-audit diagnostic of the four cross-platform numeric replay exceptions."""
import json,os,sys
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from neuropixel.research.ablations import training_model,evaluation_view
from neuropixel.research.experiment import evaluate

torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
r=Path(__file__).resolve().parents[1]/'results/research/06_A06_1_recovery';root=r/'archive'
c=json.loads((root/'core/ablation_records.json').read_text())
with np.load(root/'core/datasets/split0_test_n4096_seed61002.npz') as a:ds=tuple(torch.from_numpy(a[k].copy()) for k in ('canvas','target','roles'))
results=[]
for row in c['records']:
    if row['config']['panel']!='optimization':continue
    model=training_model(row['config']);model.load_state_dict(torch.load(root/'core'/row['config']['run_id']/'weights.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
    p=root/'core/evaluations'/row['evaluation_case']['evaluation_id']/'final_predictions.npz'
    with np.load(p) as a:old={k:a[k] for k in a.files}
    for backend in (True,False):
        torch.backends.mkldnn.enabled=backend
        _,arr=evaluate(evaluation_view(model,row['config'],row['evaluation_case'],None),ds,torch.device('cpu'),intervals=False)
        results.append({'run':row['config']['run_id'],'mkldnn_enabled':backend,'changed_predictions':int(np.sum(arr['prediction']!=old['prediction'])),
                        'maximum_abs_nll_error':float(np.max(np.abs(arr['nll']-old['nll']))),
                        'mean_nll_difference':float(arr['nll'].astype(np.float64).mean()-old['nll'].astype(np.float64).mean()),
                        'maximum_abs_confidence_error':float(np.max(np.abs(arr['confidence']-old['confidence'])))})
        print(json.dumps(results[-1]),flush=True)
out={'scope':'Exploratory post-audit diagnostic, preserving original tolerance failure; no training',
     'results':results,'cause':'Kernel/platform rounding is a plausible explanation, not an isolated causal finding',
     'primary_decision_replay':all(x['changed_predictions']==0 for x in results),
     'original_threshold':1e-4,'original_failed_receipt':'audit_replay.json',
     'classification':'Exact decisions with non-bit-exact continuous outputs; no retrospective relaxation of tolerance'}
(r/'numeric_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
