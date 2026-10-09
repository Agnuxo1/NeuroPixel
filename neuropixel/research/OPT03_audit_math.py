"""Independent arithmetic checks of recorded development classification metrics."""
from __future__ import annotations
import math
from statistics import NormalDist

ROLES=('AGENTE','ACCION','PACIENTE','LUGAR')

def check_metrics(row, expected_n):
    """Return factual violations; do not infer truth from a declared status field."""
    issues=[]
    if not isinstance(row,dict):return ['metrics are not an object']
    def count(x):return type(x) is int and x>=0
    def finite(x):return type(x) in (int,float) and math.isfinite(x)
    n=row.get('n');correct=row.get('correct')
    if not count(n) or n!=expected_n:return ['wrong total n']
    if not count(correct) or correct>n:issues.append('invalid total correct')
    roles=row.get('per_role')
    if not isinstance(roles,dict) or set(roles)!=set(ROLES):return issues+['missing/extra roles']
    accuracies=[];total_correct=0
    for name in ROLES:
        item=roles[name]
        if not isinstance(item,dict):issues.append(name+' not an object');continue
        rn=item.get('n');rk=item.get('correct');acc=item.get('accuracy')
        if not count(rn) or rn!=expected_n//4:issues.append(name+' wrong balanced n');continue
        if not count(rk) or rk>rn:issues.append(name+' invalid correct');continue
        expected=rk/rn
        if not finite(acc) or abs(acc-expected)>1e-12:issues.append(name+' accuracy/count disagreement')
        accuracies.append(expected);total_correct+=rk
        if 'wilson_95' in item:
            z=NormalDist().inv_cdf(.975);den=1+z*z/rn;center=(expected+z*z/(2*rn))/den
            half=z*math.sqrt(expected*(1-expected)/rn+z*z/(4*rn*rn))/den
            ci=item['wilson_95'];wanted=(max(0.,center-half),min(1.,center+half))
            if not isinstance(ci,list) or len(ci)!=2 or any(not finite(x) for x in ci) or any(abs(x-y)>1e-12 for x,y in zip(ci,wanted)):issues.append(name+' Wilson disagreement')
    if len(accuracies)==4:
        if correct!=total_correct:issues.append('total/role correct disagreement')
        wanted={'accuracy':total_correct/n,'macro_all_roles':sum(accuracies)/4,
                'macro_agent_patient_accuracy':(accuracies[0]+accuracies[2])/2}
        for key,value in wanted.items():
            reported=row.get(key)
            if not finite(reported) or abs(reported-value)>1e-12:issues.append(key+' role/count disagreement')
    nll=row.get('cross_entropy')
    if not finite(nll) or nll<0:issues.append('invalid NLL')
    return issues

def check_run_report(row, conf, plan_sha256):
    import json
    issues=[]
    if not isinstance(row,dict):return ['run is not an object']
    canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
    try:
        if canonical(row.get('config'))!=canonical(conf):issues.append('config mismatch')
    except (ValueError,TypeError):issues.append('config not valid finite JSON')
    if row.get('status')!='completed':issues.append('run not complete')
    if row.get('test_accessed') is not False:issues.append('unknown or declared final-test access')
    if row.get('plan_sha256')!=plan_sha256:issues.append('plan hash mismatch')
    endpoints=row.get('evaluations')
    if not isinstance(endpoints,list) or [r.get('update') for r in endpoints if isinstance(r,dict)]!=[1024,4096,8192]:return issues+['wrong endpoints']
    for endpoint in endpoints:
        if endpoint.get('test_accessed') is not False:issues.append('endpoint test exposure')
        for panel,n in (('probe',2048),('validation',4096)):
            issues.extend(f"{endpoint['update']} {panel}: {x}" for x in check_metrics(endpoint.get(panel),n))
    if not check_metrics(endpoints[-1].get('probe'),2048) and not check_metrics(endpoints[-1].get('validation'),4096):
        gate=endpoints[-1]['probe']['macro_agent_patient_accuracy']>=.95 and endpoints[-1]['validation']['macro_agent_patient_accuracy']>=.90
        if row.get('competence_gate_passed') is not gate:issues.append('reported competence gate disagreement')
    resources=row.get('resources')
    if not isinstance(resources,list) or not resources:issues.append('resource evidence missing')
    else:
        for sample in resources:
            ram=sample.get('available_ram_gib') if isinstance(sample,dict) else None
            if type(ram) not in (int,float) or not math.isfinite(ram) or ram<8:issues.append('RAM gate violation or unknown sample')
    curve=row.get('curve')
    if not isinstance(curve,list) or [r.get('update') for r in curve if isinstance(r,dict)]!=list(range(128,8193,128)):issues.append('missing/duplicate/reordered curve windows')
    elif any(not {'answer_loss_mean','school_loss_mean','total_loss_mean','last_gradient_norm','elapsed_training_seconds'}<=r.keys() for r in curve):issues.append('missing curve measurements')
    elif any(type(v) not in (int,float) or not math.isfinite(v) for r in curve for k,v in r.items() if k!='update'):issues.append('nonfinite/non-numeric curve')
    elif any(abs(r['total_loss_mean']-(r['answer_loss_mean']+.3*r['school_loss_mean']))>8*1.1920928955078125e-7*max(1,abs(r['total_loss_mean']),abs(r['answer_loss_mean'])+.3*abs(r['school_loss_mean'])) for r in curve):issues.append('answer/school/total objective disagreement beyond float32 rounding bound')
    if isinstance(curve,list) and all(isinstance(r,dict) and all(type(r.get(k)) in (int,float) and math.isfinite(r[k]) for k in ('answer_loss_mean','school_loss_mean','total_loss_mean','last_gradient_norm','elapsed_training_seconds')) for r in curve):
        if any(r.get(k,-1)<0 for r in curve for k in ('answer_loss_mean','school_loss_mean','total_loss_mean','last_gradient_norm','elapsed_training_seconds')):issues.append('negative objective/norm/time')
        times=[r['elapsed_training_seconds'] for r in curve]
        if times!=sorted(times):issues.append('nonmonotonic training time')
    return issues
