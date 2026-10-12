"""Check a scoped forward patch without importing Torch or claiming runtime parity."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FROZEN=Path('D:/PROJECTS/NeuroPixel-Frozen-d8721a59')
BASE='d8721a59d36d127642a5664c0c41e6091296ba21'
EXPECTED='564045fff799185e15b921873225d344014a4d4bcae6df6deacaf42f6e4bd701'
head=subprocess.check_output(['git','-C',str(FROZEN),'rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C',str(FROZEN),'status','--short'],text=True).strip()
assert head==BASE and not status
old=(FROZEN/'neuropixel/model.py').read_bytes()
new=(ROOT/'neuropixel/model.py').read_bytes()
assert hashlib.sha256(old).hexdigest()==EXPECTED
assert new!=old
old_tree=ast.parse(old.decode('utf-8'))
new_tree=ast.parse(new.decode('utf-8'))
def remove_forward(tree):
    tree=copy.deepcopy(tree)
    classes=[c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='NeuroPixel']
    assert len(classes)==1
    forwards=[f for f in classes[0].body if isinstance(f,ast.FunctionDef) and f.name=='forward']
    assert len(forwards)==1
    classes[0].body.remove(forwards[0])
    return ast.dump(tree,include_attributes=False)
assert remove_forward(old_tree)==remove_forward(new_tree)
record={'status':'scoped_AST_patch_verified_runtime_pending','frozen_checkout':str(FROZEN),'frozen_commit':head,'frozen_status_clean':True,
        'original_model_sha256':hashlib.sha256(old).hexdigest(),'new_model_sha256':hashlib.sha256(new).hexdigest(),
        'only_NeuroPixel_forward_AST_changed':True,'new_source_syntax_valid':True,
        'Torch_imported':False,'runtime_tests_executed':False,'positive_depth_runtime_parity_established':False,
        'remaining_gates':['G2.2 real Torch and CPU regressions','G2.3 affected physical backend checks'],
        'scope':'Source identity and patch scope only; does not close G2 runtime criteria.'}
path=ROOT/'docs/execplan/evidence/G2.1_patch_scope.json'
assert not path.exists(),path
path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record,indent=2))
