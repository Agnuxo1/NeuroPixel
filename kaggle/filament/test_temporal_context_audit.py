from temporal_context_audit import pair_targets, summary

def main():
    f0,f1,f2 = ['2020010%d000000.jpeg'%i for i in (1,2,3)]
    x = pair_targets([f1],[f0,f1,f2],1)[0]
    assert x['previous']['file']==f0 and x['next']['file']==f2
    assert x['previous']['delta_days']==-1 and x['next']['delta_days']==1
    x = pair_targets([f0],[f0],2)[0]
    assert x['nearest_days'] is None and x['previous'] is None and x['next'] is None
    x = pair_targets([f0],[f2],1)[0]
    assert x['nearest_days']==2 and x['next'] is None
    x = pair_targets([f0],[f0.replace('.jpeg','.png')],2)[0]
    assert x['previous'] is None and x['next'] is None
    assert summary([])['images']==0
    print('PASS: self/exact-time exclusion, past/future signs, max gap, empty targets')

if __name__ == '__main__':
    main()
