/* Item15 elementary JSON recount, frozen before native execution.
   Scope: scalar report from 14 assigned trajectories, not binary NPZ access.
   Native complete-array verification belongs to the separate NumPy auditor. */
function auditItem15Report(report) {
  const issues = []; let checks = 0, numeric = 0, maxAbs = 0, maxScaled = 0;
  function check(ok, where, actual, expected) {
    checks++; if (!ok) issues.push({where,actual,expected});
  }
  function close(actual, expected, where) {
    numeric++; const delta = Math.abs(actual-expected);
    maxAbs=Math.max(maxAbs,delta); maxScaled=Math.max(maxScaled,delta/(1+Math.abs(expected)));
    check(typeof actual==="number" && Number.isFinite(actual) &&
          delta <= 1e-10 + 1e-12*Math.abs(expected),where,actual,expected);
  }
  const names=["identity","decay","expansion","hidden_drift","two_cycle",
    "nonnormal_transient","forced_contraction"], times=[0,1,2,3,4,8,16,32,64,128,256];
  const norm=v=>Math.sqrt(v.reduce((s,x)=>s+x*x,0));
  const sub=(x,y)=>x.map((v,i)=>v-y[i]);
  function matrix(kind) {
    let diagonal = kind==="identity" || kind==="hidden_drift" ? 1 :
      kind==="expansion" ? 1.05 : .5;
    const A=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>i===j?diagonal:0));
    const b=[0,0,0,0];
    if(kind==="hidden_drift") b[3]=1;
    if(kind==="forced_contraction") b[2]=1;
    if(kind==="two_cycle") {A[2][2]=0;A[3][3]=0;A[2][3]=1;A[3][2]=1;}
    if(kind==="nonnormal_transient") {A[2][2]=.9;A[3][3]=.9;A[2][3]=4;}
    return {A,b};
  }
  function next(s,A,b) {return A.map((row,i)=>row.reduce((z,v,j)=>z+v*s[j],b[i]));}
  const expectedRows=[], expectedSummaries=[];
  for(const kind of names) {
    const {A,b}=matrix(kind);
    const base=kind==="forced_contraction"?[0,0,0,0]:[0,0,1,0];
    const pair=base.map((v,i)=>v+(i===3?1/16:0));
    const states=[[base],[pair]];
    for(let t=1;t<=256;t++) for(let j=0;j<2;j++)
      states[j].push(next(states[j][t-1],A,b));
    const gains=Array.from({length:257},(_,t)=>norm(sub(states[1][t],states[0][t]))*16);
    const peak=Math.max(...gains), peakTime=gains.indexOf(peak);
    for(let j=0;j<2;j++) for(const t of times) {
      const v=states[j][t], z=v[2];
      // PAD is fixed -1e4, other non-target logits are zero.
      const nll=Math.log1p(2*Math.exp(-z)+Math.exp(-10000-z));
      expectedRows.push({case:kind,trajectory:j===0?"base":"perturbed",time:t,
        state_l2:norm(v),step_difference_l2:t===0?null:norm(sub(v,states[j][t-1])),
        fixedpoint_residual_l2:norm(sub(next(v,A,b),v)),pair_gain:gains[t],
        prediction:z>0?2:1,token2_nll:nll});
    }
    const rho=kind==="identity" || kind==="hidden_drift" || kind==="two_cycle"?1:
      kind==="expansion"?1.05:kind==="nonnormal_transient"?.9:.5;
    const op=kind==="nonnormal_transient"?Math.sqrt((16+2*.9*.9+Math.sqrt(256+64*.9*.9))/2):rho;
    expectedSummaries.push({case:kind,spectral_radius:rho,operator_2_norm:op,
      peak_pair_gain:peak,first_peak_pair_time:peakTime,final_pair_gain:gains[256],
      maximum_state_l2_by_trajectory:states.map(tr=>Math.max(...tr.map(norm))),
      final_state_l2_by_trajectory:states.map(tr=>norm(tr[256])),
      final_residual_l2_by_trajectory:states.map(tr=>norm(sub(next(tr[256],A,b),tr[256])))});
  }
  check(report.item===15,"item",report.item,15);
  check(report.status==="verified","status",report.status,"verified");
  check(Array.isArray(report.rows),"rows_array",typeof report.rows,"array");
  const rows=Array.isArray(report.rows)?report.rows:[];
  check(rows.length===154,"rows_length",rows.length,154);
  for(let i=0;i<expectedRows.length;i++) {
    const e=expectedRows[i],o=rows[i];
    check(Boolean(o),"row_present."+i,Boolean(o),true);
    if(!o)continue;
    for(const k of ["case","trajectory","time","prediction"]) check(o[k]===e[k],i+"."+k,o[k],e[k]);
    for(const k of ["state_l2","fixedpoint_residual_l2","pair_gain","token2_nll"]) close(o[k],e[k],i+"."+k);
    if(e.step_difference_l2===null)check(o.step_difference_l2===null,i+".step_difference_l2",o.step_difference_l2,null);
    else close(o.step_difference_l2,e.step_difference_l2,i+".step_difference_l2");
  }
  check(Array.isArray(report.case_summaries) && report.case_summaries.length===7,
    "case_summary_count",report.case_summaries?.length,7);
  for(let i=0;i<7;i++) {
    const e=expectedSummaries[i],o=report.case_summaries?.[i];
    if(!o){check(false,"case_summary_present."+i,false,true);continue;}
    for(const k of ["case","first_peak_pair_time"])check(o[k]===e[k],"summary."+i+"."+k,o[k],e[k]);
    for(const k of ["spectral_radius","operator_2_norm","peak_pair_gain","final_pair_gain"])
      close(o[k],e[k],"summary."+i+"."+k);
    for(const k of ["maximum_state_l2_by_trajectory","final_state_l2_by_trajectory","final_residual_l2_by_trajectory"]) {
      check(Array.isArray(o[k]) && o[k].length===2,"summary."+i+"."+k+".length",o[k]?.length,2);
      for(let j=0;j<2;j++)close(o[k]?.[j],e[k][j],"summary."+i+"."+k+"."+j);
    }
    check(Number.isFinite(o.maximum_autograd_absolute_error) && o.maximum_autograd_absolute_error<=1e-12,
      "summary."+i+".native_autograd_tolerance",o.maximum_autograd_absolute_error,"<=1e-12");
    check(Number.isFinite(o.maximum_jacobian_absolute_error) && o.maximum_jacobian_absolute_error<=1e-9,
      "summary."+i+".native_fd_tolerance",o.maximum_jacobian_absolute_error,"<=1e-9");
  }
  check(report.optimization_steps===0,"optimization_steps",report.optimization_steps,0);
  check(report.checkpoints_loaded===0,"checkpoints_loaded",report.checkpoints_loaded,0);
  check(Array.isArray(report.models_unchanged) && report.models_unchanged.length===7 &&
    report.models_unchanged.every(x=>x===true),"models_unchanged",report.models_unchanged,"7 true");
  return {schema_version:1,item:15,status:issues.length?"discrepancy":"verified",checks,
    numeric_comparisons:numeric,absolute_tolerance:1e-10,relative_tolerance:1e-12,
    maximum_absolute_difference:maxAbs,maximum_scaled_difference:maxScaled,issues,
    scope:"Independent elementary JSON scalar recount; binary NPZ not read by this function",
    expected_rows:expectedRows,expected_summaries:expectedSummaries};
}
