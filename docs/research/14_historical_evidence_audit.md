# Item 14 historical damage evidence audit

Read-only review at source `5d9d09fe4c4ba045b3bfae586718f5b1b6c10a6f`, after item13 closure. Only item14 is open. No Python, model, checkpoint, NPZ, workflow, training or inference execution was performed for this review. Existing JSON arithmetic was checked in JavaScript. The historical H1 decision is unchanged.

## What is actually damaged

The original model at source `7da18d1cbd36c40c48228118abb20a7b5ab032f4`, model blob `257982f977cf713d4611b9a32b38eb56adcc6f19`, computes identities once, seeds state once, and concatenates the same identities into every update (lines71–91). A hook then multiplies the updated state by a binary spatial mask. Every selected cell loses all state channels; canvas tokens, identity embeddings and parameters remain available. Thus this intervention permits reconstruction assisted by the intact input. The current PAD correction does not create this assistance; it already existed historically.

Each default forward seeds a fresh state. State activity retained within that forward is distinct from parameters persisted across calls or an explicitly supplied continuation state. A lesion does not erase learned parameters. Survival at the output, recomputation from input and restoration from redundant neighboring state are different mechanisms.

Source anchors at the review commit:

- `scripts/phase3.py`, blob `a73b96df3266b8bb382e7dbb7ba86ad651e965b3`: training lines63–101; lesion104–107; stable111–131; scaling437–467.
- `scripts/battery.py`, blob `6941d329d3ca31888f36281e3447dcb1a30fd21c`: battery damage45–71. Its item13 edits concern vocabulary helpers, not this lesion procedure.
- `neuropixel/research/ablations.py`, blob `5c718a868262a9e4395bad6aa21784a7fd46f0ed`: training damage154–178; evaluation mask242–247; evaluation view250 onward.
- The historical interpretation “repaired to99.4%” is in `docs/FASE3_CONCLUSIONES.md`, blob `a1b85313e67fc7fdbbc698bf5db08d3f2706fe05`, line29, and the current README capability row.

## Historical stable/rest-state panel

Each retained summary contains one initialization, seed0, nine accuracy endpoints, and elapsed seconds. The driver uses2,000 test examples with sample seed5. The fixed configuration trains T16; the rest configuration jointly changes training depth to uniformly sampled12–32 steps and adds damage on probability0.5 of batches, erasing each cell with probability0.3 at a random intermediate time. Both use the occupied-token auxiliary loss0.3. This is not an isolated damage-training ablation.

| Configuration | Clean T8 | T16 | T24 | T32 | T48 | T64 | Lesion post8, T16 | T32 | T48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed16 s0 |99.8%|98.4%|86.25%|67.45%|37.7%|22.7%|90.45%|56.6%|29.3%|
| rest s0 |99.2%|99.6%|99.5%|99.25%|98.2%|95.8%|99.35%|98.8%|97.0%|

The evaluation lesion erases cells with probability0.5 after update8. It is not a guarantee of exactly half the cells. The same seed1 reconstructs the mask for each total depth under the original device RNG. The source saves state dictionaries, but these JSON summaries do not contain actual masks, per-example predictions or immediate post-lesion outputs; checkpoint availability was not independently established in this review. No upward recovery curve follows from the reported later endpoints: all three damaged endpoints decline with increasing total depth. High final accuracy can reflect survival or input-assisted reconstruction without demonstrating durable autonomous memory.

Artifacts:

- `results/phase3/stable_fijo16.json`: blob `dec383f3f6cadfb32179845ba006d9d65009823c`,256bytes, SHA256 `11a1f557f3bf59d731f9df578935145a32c8036149bfde1e9b818a97e9cb5125`.
- `results/phase3/stable_reposo.json`: blob `3b4933065f830519280560f8dc7bd8a1cf45db8b`,254bytes, SHA256 `1206d0219814b5ea2640add9c77d4680903355748f09d4cefaf251bc58cc3f87`.

## Historical battery and scaling endpoints

`results/battery/results.json`, blob `43a6c86c15c99dccdedde40df90beb1e613eeab5`, SHA256 `884bd3be623713eb0592bbdfc4e3e4af954a5260064a242bc61360c002300caf`, contains76 T1 aggregate scores across seven checkpoints: three ordinary NP seeds0–2, one school/lens NP seed0, and three Transformer seeds0–2. Each uses2,000 questions. NP tests four lesion fractions, two post-update times4/8, and endpoints16/24; Transformer tests four fractions after its first encoder layer.

| Checkpoint | Clean T16 | Clean T24 | 50% post4 T16 | 50% post8 T16 | 50% post4 T24 | 50% post8 T24 |
|---|---:|---:|---:|---:|---:|---:|
| NP s0 |93.25%|71.75%|80.4%|83.3%|62.2%|60.05%|
| NP s1 |93.15%|75.65%|83.5%|84.95%|66.3%|68.4%|
| NP s2 |95.35%|58.3%|88.75%|89.1%|57.0%|59.6%|
| NP lens s0 |97.95%|88.6%|92.65%|94.0%|79.25%|81.6%|

All four observed NP checkpoints become less accurate after eight additional updates, including the sham no-damage condition. The two lesion times consume successive masks from the same generator, so time4 and time8 do not share exactly the same realization; T16/T24 do share the mask within each time. The three Transformer clean/half-lesioned endpoints are54.55/30.85%,55.1/31.2%,55.35/27.25%. No immediate lesion loss or per-example recovery transitions are saved here.

Scaling clean/lesioned percentages from the11 inspected JSON summaries are NP5k49.4/53.65, NP16k84.5/78.95, NP30k s0 98.4/88.55, NP108k99.5/94.15, NP236k99.65/93.6, NP30k s1 95.4/84.15; TF13k54.2/25.45, TF44k s0 53.8/28.4, TF163k54.45/27.7, TF811k54.3/33.1, TF44k s1 54.55/30.25. Paths are `results/phase3/scale_{np5k,np14k,np30k,np105k,np230k,np30k_s1,tf12k,tf44k,tf170k,tf800k,tf44k_s1}.json`. These are rounded historical aggregates, not a new replay.

The README99.4% versus Transformer25–33% comparison mixes rest-state NP training with scaling Transformer endpoints. NP is lesioned post-update4 in scaling and has many later recurrent updates with input reinjection. The Transformer is lesioned after its first layer and has a different remaining depth and intact competence. A common erased-cell probability is not equal lesion severity, equal post-damage compute, or an identified memory mechanism.

## Item6 saved evidence

Scientific source `08d0d52edd05da6835e71479f3ba4399fcbeabae`; final raw archive `15e76456bc2b4cce5faec0b08fb5288fe7844547`, root `results/research/06_cloud_runs/37566497890-1/core`. Verified analysis `results/research/06_root_review/37566497890-1/core/core_analysis.json`, blob `676cc25a7cdc8e74541fba0724d61828f3a7a7b1`,1,048,012bytes, SHA256 `2c2ce1e5a584bd635bb6656c6f43c649d9b78ecb8836b4e0b1f0995cfe7c8c6f`.

This is the complete damage factorial: clean/damage training × clean/lesioned evaluation × seeds20/21. All four checkpoints use tying/reinjection, school0,T16,1,024updates,batch64,LR0.003,split0. Damage training applies probability0.5 per batch, cell-erasure probability0.3 after update8, private seed62001. Evaluation applies a shared complete example-indexed mask, seed62002, after update8, and reads only the T16 endpoint.

Each evaluation has4,096 examples,1,024 per role, and2,048 binding questions:

| Training | Evaluation | Seed | Global correct/4096 | Binding correct/2048 | AGENTE/ACCION/PACIENTE/LUGAR correct, each /1024 |
|---|---|---:|---:|---:|---|
| clean | clean |20|671|270|122 /214 /148 /187|
| clean | lesion |20|629|255|118 /195 /137 /179|
| clean | clean |21|970|151|83 /299 /68 /520|
| clean | lesion |21|876|158|91 /289 /67 /429|
| damage | clean |20|625|263|147 /130 /116 /232|
| damage | lesion |20|631|274|149 /127 /125 /230|
| damage | clean |21|944|166|100 /361 /66 /417|
| damage | lesion |21|964|182|107 /360 /75 /422|

Binding means in clean/clean,clean/lesion,damage/clean,damage/lesion order are10.2783203125%,10.0830078125%,10.4736328125%,11.1328125%. The corresponding global means are20.03173828125%,18.37158203125%,19.15283203125%,19.47021484375%. Intact training-probe binding is15.625/9.765625% for clean training and12.109375/10.9375% for damage training. Preserve this weak competence; it prevents interpreting a small robust endpoint as successful learned task repair.

The saved lesion mask has78,392 erased cells out of262,144, fraction0.299041748046875. File `datasets/lesion_keep_10660b453e32e998.npz`,40,319bytes,SHA256 `4687771affe48df3d2774fa1f768cde81abf4305d8cffb1e59b41ba72b66e9c1`. Final dataset `datasets/split0_test_n4096_seed61002.npz`,63,862bytes,SHA256 `8132e5d73f569349bfca30e3ecdfcb9b539252b53a7d2f99cacb040440aeb0d9`. Eight final prediction descriptors are retained under the corresponding `evaluations/{run_id}__clean|lesion/final_predictions.npz` paths in the verified analysis.

Damage-trained minus clean-trained binding under lesion is+0.927734375/+1.171875pp; interaction is+1.26953125/+0.439453125pp. Existing two-seed exploratory intervals include zero. No new interval is calculated here. The saved endpoint arrays could support paired survival/error-transition recounting, but cannot recover missing immediate states or a temporal recovery curve. Such a recount would remain conditional on these already weak checkpoints.

## Bounded native targets

The proposed literal copier, holder and spatial-average controls address distinct mechanisms, without training:

1. Keep one prefix and lesion fixed; separate unchanged, removed and changed future identities, with no reseeding.
2. Preserve pre-lesion, immediate and all later states/logits. Distinguish initially-correct→lost→recovered from never-lost survival and stable wrong answers.
3. Include matched no-lesion trajectories, complete erasure and neighboring redundancy. A changed-input copier tests content dependence.
4. Verify exact native held-input equivalence and before/after parameter-buffer invariance from saved arrays.
5. Report categorical recovery together with NLL and distance from the time-matched sham; a recovered argmax need not mean restored state.
6. Treat all weights and cues as constructed and known. These checks cannot establish learned repair, durable semantics, long-time stability, efficiency or external replication.

Review tooling note: the first JavaScript arithmetic snippet had a syntax error before evaluation; its corrected version checked all eight existing item6 rows and the mask fraction without discrepancies. No underlying result was generated, changed or replaced by that correction.
