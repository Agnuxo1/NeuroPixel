/* Frozen elementary item15B summary recount. No binary deserialization or inference.
 * Receives the immutable JSON artifacts; verifies arithmetic and cross-file counts.
 * Raw tensor calculations and checkpoint loading remain separate audited processes.
 */
function recountItem15Learned(input) {
  let checks = 0, numericChecks = 0;
  const issues = [], summaries = [];
  const check = (ok, label) => { checks++; if (!ok) issues.push(label); };
  const near = (a,b,label) => {
    numericChecks++;
    check(typeof a==="number" && typeof b==="number" && Number.isFinite(a) && Number.isFinite(b)
          && Math.abs(a-b)<=1e-9+1e-12*Math.abs(b), label);
  };
  const equal = (a,b,label) => check(JSON.stringify(a)===JSON.stringify(b),label);
  const {plan,native,audit,binding,recovery,gates,seedRecords}=input;
  check(plan.item===15 && plan.phase==="preflight" && plan.status==="frozen","frozen phase");
  check(native.status==="verified" && audit.status==="verified" && binding.status==="verified","component statuses");
  check(audit.issues.length===0 && binding.issues.length===0 && audit.checks>0 && binding.checks>0,"independent checks");
  const r=audit.results;
  check(native.source_commit===r.source_commit && binding.source_commit===r.source_commit,"source linkage");
  check(native.plan_sha256===r.plan_sha256 && binding.plan_sha256===r.plan_sha256,"plan linkage");
  check(recovery.status==="verified" && recovery.recovered.length===22 && recovery.binary_files_deserialized===0,"recovery");
  check(recovery.recovered.reduce((a,x)=>a+x.bytes,0)===1777007,"original byte sum");
  check(native.checks.length===5 && native.checks.every(x=>x.passed===true),"five native policy checks");
  check(native.unique_checkpoints===5 && native.optimization_steps===0 && binding.checkpoint_deserializations===5,"five distinct checkpoints and no training");
  check(binding.checkpoint_weights_exact && binding.configuration_exact && binding.constructor_buffers_zero,"checkpoint binding assertions");
  check(gates.length===5 && r.gates.length===5,"all gate records");
  const seeds=[40,41,42,43,44];
  equal(r.gates.map(x=>x.seed),seeds,"gate seed order");
  let passed=true;
  for(let i=0;i<5;i++){
    const g=gates[i], q=r.gates[i];
    check(g.seed===seeds[i],"gate identity "+i);
    for(const k of ["passed","finite_logits_and_selected_initial_states","logits_match","predictions_match","prediction_mismatches","weights_and_buffers_unchanged","rng_unchanged"])
      equal(g[k],q[k],"gate field "+i+" "+k);
    for(const k of ["maximum_absolute_logit_error","maximum_tolerance_ratio"])
      if(g[k]===null || q[k]===null)equal(g[k],q[k],"gate null "+i+" "+k);
      else near(g[k],q[k],"gate number "+i+" "+k);
    check(g.passed===(g.finite_logits_and_selected_initial_states && g.logits_match && g.predictions_match),"gate conjunction "+i);
    if(g.passed)check(g.maximum_tolerance_ratio<=1 && g.prediction_mismatches===0,"passing gate boundary "+i);
    passed=passed && g.passed;
  }
  check(typeof passed==="boolean" && native.compatibility_gate_passed===passed
        && native.extended_panel_executed===passed && r.compatibility_gate_passed===passed
        && r.extended_panel_executed===passed,"all-five conditional decision");
  check(r.seeds.length===(passed?5:0) && seedRecords.length===(passed?5:0),"conditional seed records");
  check(native.checkpoints_loaded===(passed?10:5),"native repeated restoration count");
  check(binding.native_array_files_checked===(passed?10:5)
        && Object.keys(binding.native_sha256).length===(passed?22:12),"binding conditional inventory");
  let totalObserved=0,totalCompleted=0,totalGuarded=0,pairCount=0;
  for(let i=0;i<r.seeds.length;i++){
    const rec=r.seeds[i], original=seedRecords[i], summary={seed:seeds[i],base_start_rms:[],base_last_rms:[],pair_peak_gains:[],pair_last_gains:[],guard_times:[]};
    check(rec.seed===seeds[i] && original.seed===seeds[i],"extended seed identity "+i);
    check(rec.trajectories.length===8 && original.trajectories.length===8 && rec.pairs.length===4,"extended inventory "+i);
    let done=0,guarded=0;
    for(let j=0;j<rec.trajectories.length;j++){
      const t=rec.trajectories[j], old=original.trajectories[j], key=i+"/"+j;
      check(t.row_index===Math.floor(j/2) && t.branch===(j%2?"perturbed":"base"),"trajectory order "+key);
      for(const k of ["row_index","branch","observed_count","last_time","corresponding_total_update","stop_reason"])
        equal(t[k],old[k],"trajectory linkage "+key+" "+k);
      check(Number.isInteger(t.observed_count) && t.observed_count>=1 && t.observed_count<=257,"prefix range "+key);
      check(t.last_time===t.observed_count-1 && t.corresponding_total_update===16+t.last_time,"time arithmetic "+key);
      if(t.stop_reason==="completed"){done++;check(t.last_time===256,"completed horizon "+key);}
      else {guarded++;check(["max_abs_state","nonfinite_state"].includes(t.stop_reason),"guard reason "+key);summary.guard_times.push(t.last_time);}
      if(t.stop_reason==="max_abs_state")check(t.last_state_maxabs>1e12,"first guard threshold "+key);
      for(const end of ["first","last"]){
        const l2=t[end+"_state_l2"], rms=t[end+"_state_rms"], maxabs=t[end+"_state_maxabs"];
        if(l2!==null && rms!==null)near(l2,rms*Math.sqrt(3840),"norm relation "+key+" "+end);
        if(rms!==null && maxabs!==null)check(rms<=maxabs+1e-9+1e-12*Math.abs(maxabs),"RMS maximum bound "+key+" "+end);
      }
      if(t.peak_observed_state_rms!==null){
        check(Number.isInteger(t.first_peak_state_time) && t.first_peak_state_time>=0 && t.first_peak_state_time<=t.last_time,"state peak time "+key);
        if(t.first_state_rms!==null)check(t.peak_observed_state_rms>=t.first_state_rms,"state peak covers first "+key);
        if(t.last_state_rms!==null)check(t.peak_observed_state_rms>=t.last_state_rms,"state peak covers last "+key);
      }
      if(j%2===0){summary.base_start_rms.push(t.first_state_rms);summary.base_last_rms.push(t.last_state_rms);}
      totalObserved+=t.observed_count;
    }
    check(rec.completed_trajectories===done && rec.guarded_trajectories===guarded && done+guarded===8,"seed completion counts "+i);
    totalCompleted+=done;totalGuarded+=guarded;
    for(let row=0;row<4;row++){
      const p=rec.pairs[row], key=i+"/"+row, profile=p.gain_by_time;
      check(p.row_index===row && p.role===row && p.group_index===0,"pair related-query identity "+key);
      check(profile.length===257 && Number.isInteger(p.last_shared_time) && p.last_shared_time>=0 && p.last_shared_time<=256,"pair horizon "+key);
      check(p.common_finite_states===p.last_shared_time+1,"pair prefix count "+key);
      check(profile.every((x,t)=>t<=p.last_shared_time?typeof x==="number" && Number.isFinite(x) && x>=0:x===null),"pair censorship "+key);
      const observed=profile.slice(0,p.last_shared_time+1), maximum=Math.max(...observed), first=observed.indexOf(maximum);
      near(profile[0],1,"initial gain "+key);
      near(p.peak_observed_gain,maximum,"peak gain "+key);
      check(p.first_peak_time===first,"first peak gain time "+key);
      near(p.last_shared_gain,observed[observed.length-1],"last gain "+key);
      near(p.last_shared_rms,p.last_shared_gain*p.actual_delta_rms,"pair RMS/gain relation "+key);
      check(p.actual_delta_rms>0 && p.nominal_amplitude>=1e-4,"actual perturbation "+key);
      const t=rec.trajectories[2*row];
      near(p.nominal_amplitude,1e-4*Math.max(1,t.first_state_rms),"nominal RMS rule "+key);
      summary.pair_peak_gains.push(maximum);summary.pair_last_gains.push(p.last_shared_gain);
      pairCount++;
    }
    summaries.push(summary);
  }
  check(totalObserved===r.observed_states,"observed state count");
  check(totalCompleted+totalGuarded===r.trajectories && r.trajectories===(passed?40:0),"trajectory total");
  check(pairCount===r.pairs && pairCount===(passed?20:0),"pair total");
  return {schema_version:1,item:15,status:issues.length?"failed":"verified",checks,numeric_checks:numericChecks,issues,
    source_commit:r.source_commit,plan_sha256:r.plan_sha256,compatibility_gate_passed:passed,
    extended_panel_executed:passed,observed_states:totalObserved,completed_trajectories:totalCompleted,
    guarded_trajectories:totalGuarded,pairs:pairCount,summaries,
    limitations:["Elementary recount of authenticated JSON summaries; no raw tensors or checkpoint deserialization.",
      "Internal cross-check; not external replication or a new scientific sampling design.",
      "Four queries share one exposed scene and each checkpoint; no population confidence interval.",
      "Finite directional and censored observations do not establish universal or infinite-time stability."]};
}
