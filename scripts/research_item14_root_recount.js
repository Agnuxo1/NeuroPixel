// Independent bounded V8 reconstruction of the item14 summary, without Torch.
// Input is the archived producer JSON. This does not decode the binary archive;
// the separately authored NumPy auditor reconstructs every saved tensor.
function auditItem14Report(report) {
 const models=["input_copier","state_holder","spatial_average"],lesions=["none","center","ring","all"],sources=["held","removed","changed"],cues=[2,3,4,5,6,7];
 const issues=[];let checks=0,numeric=0,maxError=0;
 function check(ok,msg){checks++;if(!ok)issues.push(msg);}
 function eq(a,b,p){numeric++;const e=Math.abs(a-b);maxError=Math.max(maxError,e);check(Number.isFinite(a)&&Number.isFinite(b)&&e<=1e-12,p);}
 const grid=token=>Array.from({length:9},()=>Array.from({length:8},(_,k)=>token!==0&&k===token?1:0));
 const copy=x=>x.map(a=>a.slice());
 function advance(m,s,token){if(m===0)return grid(token);if(m===1)return copy(s);const n=grid(0);for(let r=0;r<3;r++)for(let c=0;c<3;c++)for(let ch=0;ch<8;ch++){let v=0;for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++)if(r+dr>=0&&r+dr<3&&c+dc>=0&&c+dc<3)v+=s[(r+dr)*3+c+dc][ch]/9;n[r*3+c][ch]=v;}return n;}
 function read(s){const a=s[4].slice();a[0]=-10000;return a;}
 function argmax(a){let j=0;for(let i=1;i<a.length;i++)if(a[i]>a[j])j=i;return j;}
 function nll(a,y){const m=Math.max(...a);return m+Math.log(a.reduce((z,x)=>z+Math.exp(x-m),0))-a[y];}
 const traces={};const pre={};const expected=[];
 for(let mi=0;mi<3;mi++)for(let ci=0;ci<6;ci++){let s=grid(cues[ci]);for(let t=0;t<2;t++)s=advance(mi,s,cues[ci]);pre[mi+","+ci]=copy(s);for(let li=0;li<4;li++)for(let si=0;si<3;si++){let d=copy(s);for(let pos=0;pos<9;pos++)if(li===3||(li===1&&pos===4)||(li===2&&pos!==4))d[pos].fill(0);const out=[copy(d)],token=si===0?cues[ci]:si===1?0:cues[(ci+1)%6];for(let t=0;t<8;t++){d=advance(mi,d,token);out.push(copy(d));}traces[[mi,li,si,ci].join(",")]=out;}}
 for(let mi=0;mi<3;mi++)for(let li=0;li<4;li++)for(let si=0;si<3;si++)for(let t=0;t<9;t++){
 let p=0,im=0,late=0,sham=0,lost=0,recovered=0,survived=0,changed=0,ce=0,sse=0;
 for(let ci=0;ci<6;ci++){const tr=traces[[mi,li,si,ci].join(",")],ref=traces[[mi,0,si,ci].join(",")],pc=argmax(read(pre[mi+","+ci]))===cues[ci],ic=argmax(read(tr[0]))===cues[ci],lc=argmax(read(tr[t]))===cues[ci];
 p+=+pc;im+=+ic;late+=+lc;sham+=+(argmax(read(ref[t]))===cues[ci]);lost+=+(pc&&!ic);recovered+=+(pc&&!ic&&lc);survived+=+(pc&&ic&&lc);changed+=+(argmax(read(tr[t]))===cues[(ci+1)%6]);ce+=nll(read(tr[t]),cues[ci]);for(let pos=0;pos<9;pos++)for(let ch=0;ch<8;ch++)sse+=(tr[t][pos][ch]-ref[t][pos][ch])**2;
 }
 expected.push({model:models[mi],lesion:lesions[li],source:sources[si],recovery_step:t,n:6,pre_correct:p,immediate_correct:im,late_correct:late,matched_sham_correct:sham,initially_correct_then_lost:lost,lost_then_recovered:recovered,survived_and_still_correct:survived,late_matches_changed_cue:changed,mean_nll_original_cue:ce/6,rms_state_difference_from_time_matched_sham:Math.sqrt(sse/(6*9*8))});
 }
 check(report.item===14&&report.status==="verified","report identity/status");
 check(report.rows?.length===expected.length,"row inventory");
 for(let i=0;i<expected.length;i++){const r=report.rows?.[i],e=expected[i];check(r&&Object.keys(r).sort().join("|")===Object.keys(e).sort().join("|"),"row fields "+i);if(!r)continue;for(const k of Object.keys(e))if(typeof e[k]==="number")eq(r[k],e[k],"row "+i+"."+k);else check(r[k]===e[k],"row "+i+"."+k);}
 check(report.native_equivalence?.length===12,"native equivalence inventory");
 for(const r of report.native_equivalence||[]){eq(r.maximum_state_error,0,"native state "+r.model+" "+r.lesion);eq(r.maximum_logit_error,0,"native logits "+r.model+" "+r.lesion);}
 check(report.models_unchanged?.length===3&&report.models_unchanged.every(x=>x.unchanged===true),"weights unchanged producer assertions");
 check(report.checks?.length===8&&report.checks.every(x=>x.passed===true),"producer checks");
 return {schema_version:1,item:14,status:issues.length?"failed":"verified",checks,numeric_comparisons:numeric,maximum_absolute_difference:maxError,issues,expected_summary_rows:324,constructed_trajectories:216,scope:"Independent elementary-loop reconstruction of archived JSON summary; native equivalence and unchanged assertions checked against report, with actual saved witnesses separately audited in NumPy."};
}
