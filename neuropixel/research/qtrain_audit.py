"""Independent integer/schema control checks, no model factory or training imports."""
import numpy as np

def native_values(canvas):
    x=np.asarray(canvas)
    if x.ndim!=3 or x.shape[1:]!=(8,8) or x.dtype!=np.int64:raise ValueError('native canvas type/shape')
    values=np.empty((len(x),4),dtype=np.int64)
    for n,scene in enumerate(x):
        rows=[]
        for k in range(4):
            hits=np.argwhere(scene[:7,:7]==k+1)
            if len(hits)!=1:raise ValueError('non-query role key ambiguity')
            r,c=hits[0];rows.append(int(r));values[n,k]=scene[r,c+1]
        if len(set(rows))!=4:raise ValueError('role rows are not distinct')
    if np.any(values[:,0]==values[:,2]):raise ValueError('noun fillers must differ')
    for k,allowed in enumerate((range(5,17),range(17,27),range(5,17),range(27,35))):
        if not np.isin(values[:,k],list(allowed)).all():raise ValueError('filler class incorrect')
    return values

def check_witness(data,policy):
    issues=[]
    try:
        base=data['base_canvas'];values=native_values(base);canvas=data['canvas'];roles=data['roles'];target=data['target'];assignment=data['assignment'];indices=data['context_indices']
        if len(base)!=16 or canvas.shape!=(64,8,8) or roles.shape!=(64,) or target.shape!=(64,):return ['batch shape incorrect']
        if not np.array_equal(values,data['fillers']):issues.append('base filler metadata disagrees with native keys')
        if assignment.shape!=(16,) or not np.array_equal(np.bincount(assignment,minlength=4),[4]*4):issues.append('context-role assignment not balanced')
        expected=np.tile(np.arange(4),16) if policy=='paired' else np.repeat(assignment,4) if policy=='repeated' else None
        if expected is None:issues.append('unknown policy');return issues
        if not np.array_equal(roles,expected) or not np.array_equal(np.bincount(roles,minlength=4),[16]*4):issues.append('query role counts/structure incorrect')
        if not np.array_equal(indices,np.repeat(np.arange(16),4)):issues.append('context identity/order incorrect')
        wanted=base[indices].copy();wanted[:,7,6]=roles+1
        if not np.array_equal(canvas,wanted):issues.append('batch modifies non-query context content')
        if not np.array_equal(target,values[indices,roles]):issues.append('query target not native filler')
    except (ValueError,KeyError,IndexError,TypeError) as e:issues.append('invalid witness '+type(e).__name__)
    return issues

def recount_endpoint(data):
    p=np.asarray(data['predictions']);cp=np.asarray(data['query_predictions']);target=np.asarray(data['target']);roles=np.asarray(data['roles']);qt=np.asarray(data['query_target']);qr=np.asarray(data['query_roles'])
    if p.ndim!=2 or p.shape!=cp.shape or p.shape[1]!=len(target):raise ValueError('endpoint shape incorrect')
    if not np.isin(roles,range(4)).all() or any(np.sum(roles==k)!=len(roles)//4 for k in range(4)):raise ValueError('role balance incorrect')
    noun=np.isin(roles,[0,2]);control=~noun
    if not np.array_equal(qr,np.where(noun,2-roles,roles)) or not np.all(qt[noun]!=target[noun]) or not np.array_equal(qt[control],target[control]):raise ValueError('query counterfactual labels incorrect')
    if not np.array_equal(p[:,control],cp[:,control]):raise ValueError('common-mask unchanged controls disagree')
    nll=np.asarray(data['nll']);cnll=np.asarray(data['query_nll'])
    if nll.shape!=p.shape or cnll.shape!=p.shape or not np.isfinite(nll).all() or not np.isfinite(cnll).all() or np.min(nll)<0 or np.min(cnll)<0:raise ValueError('invalid endpoint loss')
    good=p==target[None,:];pair=good&(cp==qt[None,:]);per=[float(good[:,roles==k].mean()) for k in range(4)]
    return {'accuracy':float(good.mean()),'binding':(per[0]+per[2])/2,'joint_query_binding':float(pair[:,noun].mean()),'per_role':per,'mean_nll':float(nll.astype(np.float64).mean()),'mask_seeds':np.asarray(data['mask_seeds']).tolist()}
