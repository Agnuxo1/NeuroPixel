"""Analyze item-6 core ablations and growth without model selection."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import mean, stdev
TCRIT_DF1_95=12.706204736432095
def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def binding(row): return float(row["metrics"]["macro_agent_patient_accuracy"])
def summarize(values):
    values=[float(x) for x in values]; out={"values":values,"mean":mean(values),"range":[min(values),max(values)]}
    if len(values)>1:
        sd=stdev(values); half=TCRIT_DF1_95*sd/(len(values)**0.5) if len(values)==2 else None
        out.update(sample_sd=sd,t95_df1=[out["mean"]-half,out["mean"]+half] if half is not None else None)
    else: out.update(sample_sd=None,t95_df1=None)
    return out
def factorial_cells(records,seed):
    cells={}
    for row in records:
        c,case=row["config"],row["evaluation_case"]
        if c["panel"]=="factorial" and c["init_seed"]==seed and case["kind"]=="clean":
            cells[(int(c["variant"]["tied"]),int(c["school_weight"]>0),int(c["variant"]["reinject"]))]=binding(row)
    if len(cells)!=8: raise ValueError(f"seed {seed} missing factorial cells")
    return cells
def effects(cells):
    avg=lambda x:sum(x)/len(x); main={}
    for name,axis in (("tying",0),("school",1),("reinjection",2)):
        main[name]=avg([v for k,v in cells.items() if k[axis]])-avg([v for k,v in cells.items() if not k[axis]])
    pair={}
    for name,a,b,r in (("tying_x_school",0,1,2),("tying_x_reinjection",0,2,1),("school_x_reinjection",1,2,0)):
        diffs=[]
        for rv in (0,1):
            def y(av,bv):
                key=[0,0,0]; key[a]=av; key[b]=bv; key[r]=rv
                return cells[tuple(key)]
            diffs.append((y(1,1)-y(0,1))-(y(1,0)-y(0,0)))
        pair[name]=avg(diffs)
    y=lambda a,b,c:cells[(a,b,c)]
    return {"main":main,"pairwise":pair,"three_way":(y(1,1,1)-y(0,1,1)-y(1,0,1)+y(0,0,1))-(y(1,1,0)-y(0,1,0)-y(1,0,0)+y(0,0,0))}
def find_record(records,seed,panel,kind="clean",school=0.0,steps=None,updates=None):
    hits=[]
    for row in records:
        c,case=row["config"],row["evaluation_case"]
        if c["init_seed"]!=seed or c["panel"]!=panel or case["kind"]!=kind: continue
        if not c["variant"]["tied"] or not c["variant"]["reinject"] or c["school_weight"]!=school: continue
        if steps is not None and c["steps"]!=steps: continue
        if updates is not None and c["updates"]!=updates: continue
        hits.append(row)
    if len(hits)!=1: raise ValueError(f"expected one row, got {len(hits)}")
    return hits[0]
def core_analysis(core):
    records=core["records"]; seeds=(20,21); by_seed={}
    for seed in seeds:
        cells=factorial_cells(records,seed); base=find_record(records,seed,"factorial",school=0.0); base_clean=binding(base); depth={}
        for t in (1,4):
            trained=find_record(records,seed,"recurrence",school=0.0,steps=t)
            dep=[r for r in records if r["config"]["init_seed"]==seed and r["config"]["panel"]=="factorial" and r["config"]["variant"]["tied"] and r["config"]["variant"]["reinject"] and r["config"]["school_weight"]==0.0 and r["evaluation_case"]["kind"]=="deployment_truncation" and r["evaluation_case"]["steps_override"]==t]
            if len(dep)!=1: raise ValueError("missing deployment truncation")
            depth[str(t)]={"trained_minus_T16":binding(trained)-base_clean,"same_T16_weights_evalT_minus_eval16":binding(dep[0])-base_clean,"trained":binding(trained),"truncated":binding(dep[0]),"T16":base_clean}
        dtc=find_record(records,seed,"damage",kind="clean",school=0.0); dtl=find_record(records,seed,"damage",kind="fixed_lesion",school=0.0); ctl=find_record(records,seed,"factorial",kind="fixed_lesion",school=0.0)
        q00,q01,q10,q11=base_clean,binding(ctl),binding(dtc),binding(dtl)
        damage={"clean_train_clean_eval":q00,"clean_train_lesion_eval":q01,"damage_train_clean_eval":q10,"damage_train_lesion_eval":q11,"train_effect_clean":q10-q00,"train_effect_lesion":q11-q01,"interaction":(q11-q10)-(q01-q00)}
        budget={}
        for school in (0.0,0.3):
            short=find_record(records,seed,"factorial",school=school,updates=1024); long=find_record(records,seed,"optimization",school=school,updates=8192)
            budget[str(school)]={"short":binding(short),"long":binding(long),"long_minus_short":binding(long)-binding(short)}
        budget["school_x_budget"]=budget["0.3"]["long_minus_short"]-budget["0.0"]["long_minus_short"]
        by_seed[str(seed)]={"factorial_cells":{"".join(map(str,k)):v for k,v in cells.items()},"effects":effects(cells),"depth":depth,"damage":damage,"budget":budget}
    agg={"main_effects":{},"pairwise_interactions":{},"three_way":{},"depth":{},"damage":{},"budget":{}}
    for k in ("tying","school","reinjection"): agg["main_effects"][k]=summarize([by_seed[str(s)]["effects"]["main"][k] for s in seeds])
    for k in ("tying_x_school","tying_x_reinjection","school_x_reinjection"): agg["pairwise_interactions"][k]=summarize([by_seed[str(s)]["effects"]["pairwise"][k] for s in seeds])
    agg["three_way"]=summarize([by_seed[str(s)]["effects"]["three_way"] for s in seeds])
    for t in ("1","4"): agg["depth"][t]={k:summarize([by_seed[str(s)]["depth"][t][k] for s in seeds]) for k in ("trained_minus_T16","same_T16_weights_evalT_minus_eval16")}
    for k in ("train_effect_clean","train_effect_lesion","interaction"): agg["damage"][k]=summarize([by_seed[str(s)]["damage"][k] for s in seeds])
    for school in ("0.0","0.3"): agg["budget"][school]=summarize([by_seed[str(s)]["budget"][school]["long_minus_short"] for s in seeds])
    agg["budget"]["school_x_budget"]=summarize([by_seed[str(s)]["budget"]["school_x_budget"] for s in seeds])
    clean=[r for r in records if r["evaluation_case"]["kind"]=="clean"]
    return {"by_seed":by_seed,"aggregate":agg,"budget_limited_runs":sum(bool(r["optimization_budget_limited"]) for r in clean),"clean_runs":len(clean)}
def growth_analysis(growth):
    seeds=("20","21"); out={"by_seed":{},"aggregate":{}}
    for seed in seeds:
        result=growth["results"][seed]; row={"single_final_expert":result["single_final_expert"]["metrics"]["macro_agent_patient_accuracy"]}
        for bank in ("fixed","duplicate","novelty","resonance"):
            row[bank]={"expert_count":result[bank]["expert_count"],"routers":{r:result[bank]["metrics"][r]["macro_agent_patient_accuracy"] for r in ("legacy","random","uniform","learned")},"oracle":result[bank]["hard_selection_oracle"]["binding_accuracy"]}
        row["novelty_actions"]=[x["action"] for x in growth["training"][seed]["novelty_stages"]]; row["resonance_actions"]=[x["action"] for x in growth["training"][seed]["resonance_stages"]]; out["by_seed"][seed]=row
    out["aggregate"]["single_final_expert"]=summarize([out["by_seed"][s]["single_final_expert"] for s in seeds])
    for bank in ("fixed","duplicate","novelty","resonance"):
        out["aggregate"][bank]={"expert_count":[out["by_seed"][s][bank]["expert_count"] for s in seeds],"routers":{r:summarize([out["by_seed"][s][bank]["routers"][r] for s in seeds]) for r in ("legacy","random","uniform","learned")},"oracle":summarize([out["by_seed"][s][bank]["oracle"] for s in seeds])}
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--core",type=Path,required=True); ap.add_argument("--growth",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    core,growth=load(a.core),load(a.growth)
    if core.get("status")!="completed" or growth.get("status")!="completed": raise ValueError("both item-6 panels must be complete")
    result={"item":6,"status":"analyzed","core":core_analysis(core),"growth":growth_analysis(growth),"scope":"exploratory, two initialization seeds, one split; H1 is not reopened"}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":"analyzed","output":str(a.output)},sort_keys=True))
if __name__=="__main__": main()
