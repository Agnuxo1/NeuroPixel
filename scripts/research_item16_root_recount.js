function recountItem16(doc) {
  let checks=0, numeric=0; const issues=[];
  function check(ok, name, detail) { checks++; if(!ok) issues.push({name,detail}); }
  function near(value, expected, tolerance, name) {
    numeric++; check(Number.isFinite(value)&&Math.abs(value-expected)<=tolerance, name, {value,expected,tolerance});
  }
  function canonical(v) {
    if(Array.isArray(v))return v.map(canonical);
    if(v && typeof v==="object")return Object.fromEntries(Object.keys(v).sort().map(k=>[k,canonical(v[k])]));
    return v;
  }
  function same(a,b,name) {check(JSON.stringify(canonical(a))===JSON.stringify(canonical(b)),name);}
  const {plan,native,audit,recovery}=doc;
  check(plan.item===16&&plan.phase==="probe"&&plan.status==="frozen","frozen_plan");
  check(native.item===16&&native.status==="verified","native_status");
  check(audit.item===16&&audit.status==="verified"&&audit.issues.length===0&&audit.checks>0,"audit_status");
  check(recovery.item===16&&recovery.status==="verified","recovery_status");
  same(native.recipe,plan.evidence_recipe.scanner,"native_recipe");
  same(native.input_sha256,plan.inputs.scanner.files_sha256,"native_inputs");
  same(recovery.inputs,plan.inputs.scanner,"recovery_inputs");
  check(recovery.recovered.length===12,"twelve_recovered_inputs");
  near(recovery.recovered.reduce((s,x)=>s+x.bytes,0),1578221,0,"recovered_byte_total");
  for(const x of recovery.recovered) {
    check(x.sha256===plan.inputs.scanner.files_sha256[x.path],"input_sha256_"+x.path);
    near(x.bytes,plan.inputs.scanner.files_bytes[x.path],0,"input_bytes_"+x.path);
  }
  const names=["source_and_inputs_unchanged","parameter_buffer_and_rng_invariants",
    "identical_immediate_full_projections","hidden_to_visible_causal_witness",
    "unused_hidden_negative_control","native_preupdate_lens_and_posthook_frames",
    "finite_pad_policy_and_legacy_difference","top_label_confidence_compression",
    "learned_rank_dimension_bounds","learned_kernel_projector_witnesses"];
  same(native.checks.map(x=>x.name),names,"native_check_inventory");
  check(native.checks.every(x=>x.passed===true),"native_checks_passed");
  near(native.learned_forward_calls,0,0,"no_learned_forward");
  near(native.optimization_steps,0,0,"no_optimization");
  const c=native.constructed;
  same(c.rules,["hidden_to_visible","hidden_unused"],"constructed_rules");
  same(c.immediate_labels,[[1,1],[1,1]],"equal_current_labels");
  same(c.future_labels,[[1,2],[1,1]],"coupled_and_unused_future_labels");
  same(c.pad_legacy_labels,[0,0],"legacy_pad_labels");
  same(c.pad_scanner_labels,[1,0],"finite_pad_policy_labels");
  same(c.compression_labels,[1,1],"compression_labels");
  for(const [i,x] of c.compression_confidence.entries())near(x,0.6,1e-12,"compression_confidence_"+i);
  const q=plan.evidence_recipe.scanner.constructed, V=q.vocab,d=q.c_id,C=q.state_channels,H=q.hidden;
  const count=V*d+d*C+C+18*C+(3*C+d)*H+H+H*C+C+C*d+d;
  near(c.constructed_parameter_count_each,count,0,"analytical_parameter_count");
  near(c.native_forward_calls,3,0,"three_batched_constructed_forward_calls");
  near(c.learned_forward_calls,0,0,"constructed_record_has_no_learned_forward");
  same(native.learned_ranks.map(x=>x.seed),[40,41,42,43,44],"all_five_learned_seeds");
  let minNull=48, maxResidual=0, maxProjectorError=0;
  for(const row of native.learned_ranks) for(const name of ["read","nonpad"]) {
    const r=row[name], tag=row.seed+"_"+name;
    near(r.rows,name==="read"?16:36,0,"matrix_rows_"+tag);
    near(r.columns,48,0,"matrix_columns_"+tag);
    check(Number.isInteger(r.rank)&&r.rank>=0&&r.rank<=16,"rank_bound_"+tag);
    near(r.rank+r.nullity,48,0,"rank_nullity_"+tag);
    check(r.nullity>=32,"nullity_lower_bound_"+tag);
    check(Number.isFinite(r.threshold)&&r.threshold>=0,"rank_threshold_"+tag);
    check(Number.isFinite(r.minimum_computed_singular_threshold_margin)&&r.minimum_computed_singular_threshold_margin>=0,
      "recorded_threshold_margin_"+tag);
    const tol=plan.evidence_recipe.scanner.projector_absolute_tolerance;
    check(Number.isFinite(r.matrix_maxabs)&&r.matrix_maxabs>=0,"matrix_magnitude_"+tag);
    check(Number.isFinite(r.kernel_residual_maxabs)&&r.kernel_residual_maxabs>=0&&r.kernel_residual_maxabs<=tol*Math.max(1,r.matrix_maxabs),
      "kernel_residual_bound_"+tag);
    near(r.projector_symmetry_maxabs,0,tol,"projector_symmetry_"+tag);
    near(r.projector_idempotence_maxabs,0,tol,"projector_idempotence_"+tag);
    near(r.projector_trace,r.nullity,tol,"projector_trace_"+tag);
    minNull=Math.min(minNull,r.nullity); maxResidual=Math.max(maxResidual,r.kernel_residual_maxabs);
    maxProjectorError=Math.max(maxProjectorError,r.projector_symmetry_maxabs,r.projector_idempotence_maxabs,
      Math.abs(r.projector_trace-r.nullity));
  }
  for(const row of native.learned_ranks)check(row.nonpad.rank<=row.read.rank,"composite_rank_bound_"+row.seed);
  check(Object.keys(native.artifacts.constructed.arrays).length===76,"constructed_array_inventory");
  check(Object.keys(native.artifacts.learned_ranks.arrays).length===18,"rank_array_inventory");
  return {schema_version:1,item:16,status:issues.length?"failed":"verified",checks,numeric_comparisons:numeric,
    issues,summary:{constructed_parameter_count_each:count,constructed_trajectories:4,
      native_causal_frame_slots:12,immediate_intervention_states:4,post_intervention_updates:4,
      learned_matrices:5,matrix_decompositions:10,minimum_reported_nullity:minNull,
      maximum_reported_kernel_residual:maxResidual,maximum_reported_projector_error:maxProjectorError},
    limits:["Pure JSON arithmetic and inventory recount; no binary array read, native replay or SVD is performed here.",
      "The separate frozen auditor checks raw arrays and independently reconstructs the matrices.",
      "Internal verification is not external replication or semantic causal attribution."]};
}
