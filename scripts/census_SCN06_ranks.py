"""Retrospective matrix census only: no Torch import, model load or forward."""
import hashlib,json,os,sys
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    plan_path=ROOT/'docs/research/SCN06_rank_execution_plan.json';plan=json.loads(plan_path.read_bytes());out=ROOT/'results/research/SCN06_rank_census'
    if (out/'receipt.json').exists():raise ValueError('Closed census cannot repeat')
    for name,h in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    for path,h in plan['input_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    out.mkdir(exist_ok=True,parents=True);rows=[];arrays={};checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    for seed in [40,41,42,43,44]:
        root=Path(plan['input_root'])
        with np.load(root/f'native/gates/seed{seed}.npz',allow_pickle=False) as z:
            read=z['weight__read.weight'].astype(np.float64);embed=z['weight__embed.weight'].astype(np.float64);rgb=z['weight__g_rgb'].astype(np.float64);mask=z['weight__g_mask']
            dictionary=np.concatenate([np.where(mask,rgb,embed[:,:3]),embed[:,3:]],1);dictionary[0]=0
            for k in ['embed.weight','read.weight','read.bias','g_rgb','g_mask']:check(f'{seed} unchanged stored parameter {k}',np.array_equal(z['weight__'+k],z['weight_after__'+k]))
            for name,matrix in [('read',read),('nonpad',dictionary[1:]@read)]:
                u,s,vh=np.linalg.svd(matrix,full_matrices=True);threshold=np.finfo(np.float64).eps*max(matrix.shape)*s[0];rank=int(np.sum(s>threshold));nullity=matrix.shape[1]-rank;basis=vh[rank:].T;projector=basis@basis.T
                residual=float(np.max(np.abs(matrix@projector)));symmetry=float(np.max(np.abs(projector-projector.T)));idempotence=float(np.max(np.abs(projector@projector-projector)))
                check(f'{seed}/{name} kernel residual',residual<=1e-10*max(1,float(np.max(np.abs(matrix)))))
                check(f'{seed}/{name} projector symmetry',symmetry<=1e-10);check(f'{seed}/{name} projector idempotence',idempotence<=1e-10);check(f'{seed}/{name} projector trace',abs(float(np.trace(projector))-nullity)<=1e-10)
                check(f'{seed}/{name} ambient dimension bound',rank<=16 and nullity>=32)
                rows.append(dict(seed=seed,matrix=name,shape=list(matrix.shape),rank=rank,nullity=nullity,threshold=threshold,sensitivity_ranks={str(f):int(np.sum(s>threshold*f)) for f in [.1,1,10,100]},singular_values=s.tolist(),kernel_residual_maxabs=residual,projector_symmetry_maxabs=symmetry,projector_idempotence_maxabs=idempotence,projector_trace=float(np.trace(projector))))
                arrays[f'seed{seed}_{name}_matrix']=matrix;arrays[f'seed{seed}_{name}_projector']=projector;arrays[f'seed{seed}_{name}_spectrum']=s
    np.savez_compressed(out/'matrices.npz',**arrays)
    result=dict(status='verified_retrospective_SCN06_ambient_matrix_census' if all(x['passed'] for x in checks) else 'failed',checks=checks,rows=rows,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),input_sha256=plan['input_sha256'],numpy_version=np.__version__,model_loads=0,learned_forward_calls=0,optimization_steps=0,external_replication=False,scope='Ambient instantaneous linear decoder matrices. Does not identify reachable learned states, causal semantic directions or multitime observability.')
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],checks=len(checks),rows=[{k:x[k] for k in ['seed','matrix','rank','nullity']} for x in rows])))
if __name__=='__main__':main()
