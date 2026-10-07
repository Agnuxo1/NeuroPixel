/* Independent JavaScript recount of item13 saved JSON evidence.
 * Pure arithmetic/visible-canvas checks; not a Torch inference reimplementation.
 */
function auditItem13Native(native) {
  const issues=[]; let checks=0,numericComparisons=0,maxAbsoluteDifference=0;
  function check(ok,name,detail){checks++;if(!ok)issues.push({name,detail});}
  function canonical(value){
    if(Array.isArray(value))return value.map(canonical);
    if(value!==null&&typeof value==="object")return Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])]));
    return value;
  }
  function same(a,b,name){check(JSON.stringify(canonical(a))===JSON.stringify(canonical(b)),name);}
  function close(a,b,name,tolerance=1e-10){
    numericComparisons++;const difference=Math.abs(a-b);
    if(Number.isFinite(difference))maxAbsoluteDifference=Math.max(maxAbsoluteDifference,difference);
    check(Number.isFinite(a)&&Number.isFinite(b)&&difference<=tolerance*Math.max(1,Math.abs(a),Math.abs(b)),name,{a,b,difference});
  }
  const names=["base","new_agent","new_patient","two_new","irrelevant_old","irrelevant_new"];
  check(native.item===13&&native.status==="verified","native status");
  same(native.label_census.conditions,names,"condition inventory");
  const rows=native.label_census.examples;
  check(rows.length===768,"example count");
  const seen=new Set(),bases=new Map(),counts=[];
  for(const r of rows){
    const id=[r.scene,r.query,r.condition].join("|");
    check(!seen.has(id),"unique scene/query/condition");seen.add(id);
    check(Number.isInteger(r.scene)&&r.scene>=0&&r.scene<32&&r.query>=1&&r.query<=4&&names.includes(r.condition),"row coordinates");
    const c=r.canvas, roleFillers=new Map();
    check(c.length===8&&c.every(x=>x.length===8),"canvas dimensions");
    check(c[7][6]===r.query&&c[7][7]===0,"query and output");
    for(let row=0;row<7;row++)for(let col=0;col<8;col++){
      const t=c[row][col];
      if(t>=1&&t<=4){
        check(!roleFillers.has(t)&&col<7&&c[row][col+1]>=5,"unique immediate pair");
        roleFillers.set(t,c[row][col+1]);
      }
    }
    check(roleFillers.size===4,"four roles");
    const gold=roleFillers.get(r.query);
    check(gold===r.gold&&r.symbolic===gold,"visible gold");
    check(r.constant_35===35,"literal constant");
    const gate=c.some(row=>row.includes(35))?35:gold;
    check(gate===r.presence_gate_with_symbolic_fallback,"visible presence gate");
    if(r.condition==="base")bases.set([r.scene,r.query].join("|"),r);
  }
  check(bases.size===128,"base count");
  for(const r of rows){
    const b=bases.get([r.scene,r.query].join("|"));
    check(r.base_gold===b.gold,"base gold provenance");
    const diffs=[];
    for(let i=0;i<8;i++)for(let j=0;j<8;j++)if(r.canvas[i][j]!==b.canvas[i][j])diffs.push([i,j]);
    const expectedChanges=r.condition==="base"?0:r.condition==="two_new"?2:1;
    check(diffs.length===expectedChanges,"transformation cardinality");
    if(r.condition.startsWith("irrelevant_")){
      same(diffs[0],r.added_unpaired_cell,"distractor location");
      check(b.canvas[diffs[0][0]][diffs[0][1]]===0&&diffs[0][0]<7,"blank nonquery distractor");
      check(r.gold===b.gold,"irrelevant gold invariant");
      check(r.canvas[diffs[0][0]][diffs[0][1]]===(r.condition==="irrelevant_new"?35:5+r.scene%12),"distractor token");
    }
  }
  for(const condition of names)for(const subset of ["all","binding","new_target","old_target"]){
    const eligible=rows.filter(r=>r.condition===condition&&
      (subset==="all"||subset==="binding"&&[1,3].includes(r.query)||
       subset==="new_target"&&r.gold>=35||subset==="old_target"&&r.gold<35));
    const values={condition,subset,n:eligible.length};
    for(const field of ["constant_35","presence_gate_with_symbolic_fallback","symbolic"])
      values[field+"_correct"]=eligible.reduce((a,r)=>a+Number(r[field]===r.gold),0);
    counts.push(values);
  }
  same(counts,native.label_census.counts,"all census summaries");
  const original=native.transformed_split_census.original;
  check(original.train.length===1056&&original.test.length===264,"old split lengths");
  const sets={train:new Set(original.train.map(JSON.stringify)),test:new Set(original.test.map(JSON.stringify))};
  check(sets.train.size===1056&&sets.test.size===264,"unique old triples");
  check([...sets.train].every(k=>!sets.test.has(k)),"old triples disjoint");
  check(new Set([...sets.train,...sets.test]).size===1320,"old union");
  for(const triple of [...original.train,...original.test]){
    const [a,b,p]=triple;
    check(triple.length===3&&Number.isInteger(a)&&Number.isInteger(b)&&Number.isInteger(p)&&
      a>=0&&a<12&&p>=0&&p<12&&b>=0&&b<10&&a!==p,"ordinal triple domain");
  }
  const projected=[];
  for(const kind of ["agent","patient","both"]){
    function keys(values){return new Set(values.map(([a,b,p])=>JSON.stringify([
      kind==="agent"||kind==="both"?35:a,b,kind==="both"?36:kind==="patient"?35:p])));}
    const train=keys(original.train),test=keys(original.test),shared=[...train].filter(k=>test.has(k));
    const actual=native.transformed_split_census.projections.find(r=>r.replacement===kind);
    check(actual.train_unique===train.size&&actual.test_unique===test.size&&actual.shared_unique===shared.length,"projected cardinality");
    same([...shared].sort(),actual.shared_keys.map(JSON.stringify).sort(),"projected intersections");
    projected.push({replacement:kind,train_unique:train.size,test_unique:test.size,shared_unique:shared.length});
  }
  function lse(values){const m=Math.max(...values);return m+Math.log(values.reduce((a,x)=>a+Math.exp(x-m),0));}
  function argmax(values){let k=0;for(let i=1;i<values.length;i++)if(values[i]>values[k])k=i;return k;}
  for(const f of native.softmax_fixtures)for(let i=0;i<2;i++){
    const old=f.old_logits[i],expanded=f.new_logits[i],oldZ=lse(old),newZ=lse(expanded),y=f.targets[i];
    same(expanded.slice(0,old.length),old,"softmax old logits");
    const oldLP=old.map(x=>x-oldZ),newLP=expanded.map(x=>x-newZ);
    for(let j=0;j<old.length;j++)close(oldLP[j],f.old_log_probabilities[i][j],"old log probability");
    for(let j=0;j<expanded.length;j++)close(newLP[j],f.new_log_probabilities[i][j],"new log probability");
    close(newZ-oldZ,f.log_normalizer_increase[i],"log normalizer increase");
    close(oldLP[y]-newLP[y],f.observed_nll_increase[i],"observed NLL delta");
    close(newLP.slice(old.length).reduce((a,x)=>a+Math.exp(x),0),f.new_class_mass[i],"new class mass");
    check(argmax(old)===f.old_argmax[i]&&argmax(expanded)===f.new_argmax[i],"full argmax");
  }
  for(const f of native.joint_renaming.records){
    const pi=f.permutation_old_to_new;
    same(pi,[0,1,2,3,4,6,5,8,7],"declared noun permutation");
    same(f.old_canvas.map(c=>c.map(row=>row.map(id=>pi[id]))),f.new_canvas,"renamed input");
    let max=0;
    for(let i=0;i<f.old_logits.length;i++)for(let j=0;j<pi.length;j++){
      const diff=Math.abs(f.old_logits[i][j]-f.new_logits[i][pi[j]]);
      max=Math.max(max,diff);
      close(f.old_logits[i][j],f.new_logits[i][pi[j]],"aligned renamed logit",1e-12);
    }
    close(max,f.max_aligned_logit_difference,"reported maximum renaming difference",1e-12);
    if(f.family==="neuropixel")same(f.old_state,f.new_state,"NP renamed state exact");
    else check(f.old_state===null&&f.new_state===null,"TF has no exposed grid state");
  }
  check(native.model_training_executed===false&&native.torch_threads===1&&native.torch_interop_threads===1,"native scope and threads");
  check(native.checks.length===8&&native.checks.every(c=>c.passed===true),"native declared checks");
  return {schema_version:1,item:13,status:issues.length?"failed":"verified",checks,numericComparisons,maxAbsoluteDifference,issues,
    label_counts:counts,projected_split_counts:projected,
    limitation:"Recounts saved JSON and checks representation-aligned outputs. Does not rerun Torch inference or establish trained competence."};
}
