"""Native role-key/value counterfactuals that preserve the semantic triple."""
import numpy as np
ROLE_IDS=np.array([1,2,3,4],dtype=np.int64)

def parse(canvas,target=None,roles=None):
    x=np.asarray(canvas)
    if x.ndim!=3 or x.shape[1:]!=(8,8) or x.dtype!=np.int64:raise ValueError('native int64 8x8 canvas required')
    values=np.empty((len(x),4),dtype=np.int64);positions=np.empty((len(x),4,2),dtype=np.int64)
    query=x[:,7,6]-1
    if not np.isin(query,range(4)).all() or not (x[:,7,7]==0).all():raise ValueError('invalid query/output position')
    for n,scene in enumerate(x):
        for role,key in enumerate(ROLE_IDS):
            hits=np.argwhere(scene[:7,:7]==key)
            if len(hits)!=1:raise ValueError('exactly one non-query key per role required')
            row,col=hits[0];positions[n,role]=[row,col];values[n,role]=scene[row,col+1]
        if len(set(positions[n,:,0]))!=4:raise ValueError('native generator requires four distinct rows')
    allowed=[range(5,17),range(17,27),range(5,17),range(27,35)]
    if any(not np.isin(values[:,i],list(pool)).all() for i,pool in enumerate(allowed)):raise ValueError('role/filler category mismatch')
    if np.any(values[:,0]==values[:,2]):raise ValueError('agent and patient must differ')
    gold=values[np.arange(len(x)),query]
    if target is not None and not np.array_equal(gold,target):raise ValueError('native target mismatch')
    if roles is not None and not np.array_equal(query,roles):raise ValueError('native query-role mismatch')
    return {'values':values,'positions':positions,'roles':query,'target':gold,'semantic_triple':values[:,:3].copy()}

def transform(canvas,target,roles,kind):
    if kind not in ('query_flip','pair_relocate','both'):raise ValueError('unknown transform')
    before=parse(canvas,target,roles);x=canvas.copy()
    if kind in ('pair_relocate','both'):
        for n in range(len(x)):
            ra,ca=before['positions'][n,0];rp,cp=before['positions'][n,2]
            first=x[n,ra,ca:ca+2].copy();second=x[n,rp,cp:cp+2].copy()
            x[n,ra,ca:ca+2]=second;x[n,rp,cp:cp+2]=first
    if kind in ('query_flip','both'):
        noun_query=np.isin(roles,[0,2]);new=roles.copy();new[noun_query]=2-new[noun_query]
        x[:,7,6]=ROLE_IDS[new]
    after=parse(x)
    if not np.array_equal(before['semantic_triple'],after['semantic_triple']) or not np.array_equal(before['values'],after['values']):raise ValueError('counterfactual changed composition/split unit')
    return x,after['target'],after['roles']
