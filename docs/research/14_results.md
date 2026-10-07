# Item14 results: assisted repair versus stored memory

## Finding

The measurement now distinguishes input-driven reconstruction, passive retention and categorical recovery from surviving spatial redundancy. Five focused regression methods passed, the three constructed native rules behaved as specified, and an independent NumPy program reconstructed all88 saved arrays without a discrepancy beyond floating-point rounding.

**Learned autonomous repair is still unestablished.** The historical99.35% result is a final accuracy while original input identities remain available; its driver did not retain the answer immediately after damage. The new positive controls use explicitly assigned weights and redundant cue codes. Their success verifies the measurement and possible mechanisms, not that the project's trained models learned those mechanisms.

This closes the feasible investigation of item14. Historical H1 remains closed and not supported. It does not promote a diagnostic result into a Nobel-level discovery.

## 1. Frozen sources and evidence

| Record | Immutable identity |
|---|---|
| Frozen source S | [844f4dfae1cb8909d4f87c0c8cfd96b965590adf](https://github.com/Agnuxo1/NeuroPixel/blob/844f4dfae1cb8909d4f87c0c8cfd96b965590adf/docs/research/14_probe_plan.json) |
| Protocol freeze |2026-10-07T15:34:14+00:00|
| Plan SHA-256 |013863bb2d99277864e2c26e02c2c2d32f689b0dc3b92e6962bd7d3191384447|
| Workflow / job |[37644986366](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37644986366) /112873311188; completed successfully|
| Final raw archive F |[10c694cd299a4114bd447528d22b204d89b00b9e](https://github.com/Agnuxo1/NeuroPixel/blob/10c694cd299a4114bd447528d22b204d89b00b9e/results/research/14_cloud_runs/37644986366-1-probe/archive_manifest.json)|
| Archive manifest |3393bytes; SHA-25623e847b571af3a2f9ae175f3ecce44f232613e90b1dd0038519ef88539580582|
| Archived payload |21files,9,825,102bytes excluding manifest|
| Native trace NPZ |33,729bytes; SHA-2560e898b1b352cc22306c0dbc5fdd37c1ba021c9c0788a84ba55b318685f2a500d|
| Native report |180,658bytes; SHA-25656c143b2035cced57f8eb84ec87cd23aa9f33f270dc96360d8ae5167e189a65a|
| Independent report |183,062bytes; SHA-2566324a5c9829c8a012de5d0a9efd23da50e99e36c915bac1222b6b31f02e91275|

The [protocol](https://github.com/Agnuxo1/NeuroPixel/blob/844f4dfae1cb8909d4f87c0c8cfd96b965590adf/docs/research/14_probe_protocol.md) and23 implementation/source bindings were committed before execution. The raw [native report](https://github.com/Agnuxo1/NeuroPixel/blob/10c694cd299a4114bd447528d22b204d89b00b9e/results/research/14_cloud_runs/37644986366-1-probe/native/report.json), [independent audit](https://github.com/Agnuxo1/NeuroPixel/blob/10c694cd299a4114bd447528d22b204d89b00b9e/results/research/14_cloud_runs/37644986366-1-probe/audit/audit.json), source ZIP, test records and resource logs are retained together. A complete [historical evidence note](https://github.com/Agnuxo1/NeuroPixel/blob/844f4dfae1cb8909d4f87c0c8cfd96b965590adf/docs/research/14_historical_evidence_audit.md) preserves the original source paths and numeric inventories.

## 2. What the historical lesion actually removes

The native update concatenates the recurrent state, local perception of that state and identities obtained from the original canvas. The damage hook runs after an update and multiplies the state by a spatial keep mask. It erases all channels of selected cells. It does not erase the original canvas, the identity input or the model parameters. Consequently the next update can again use information supplied by the original facts.

That mechanism is present in the [historical core](https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/model.py) and in the [current core](https://github.com/Agnuxo1/NeuroPixel/blob/844f4dfae1cb8909d4f87c0c8cfd96b965590adf/neuropixel/model.py). It is not an artifact of the later PAD correction. This source trace supports the presence of an input-assisted route; it does not establish how much a particular learned checkpoint uses it.

A new call to forward starts from a newly seeded state. Calling forward on an empty canvas would therefore test a reset. The streaming function carries a state within a call but adds a new seed at each frame boundary. To isolate continued identity input from new seeding, this point adds a [research-only trace helper](https://github.com/Agnuxo1/NeuroPixel/blob/844f4dfae1cb8909d4f87c0c8cfd96b965590adf/neuropixel/research/repair_trace.py). It seeds once, preserves the state, applies a lesion at an explicit boundary and changes only the identity input during the continuation.

The helper is intentionally limited to deterministic native evaluation with token inputs. The production forward and streaming behavior are unchanged.

## 3. What existing learned results support

### The99.35% endpoint

The stable driver trains a fixed-depth model or a model with both variable update counts and training damage. The second recipe samples12–32 updates and, on probability0.5 of batches, erases each cell with probability0.3 at an intermediate time. It also uses the occupied-token auxiliary loss. Two training changes are therefore coupled.

The saved summaries report one seed per recipe on2000 test examples. At evaluation, the half-erasure probability acts after update8; the reported scores occur at total16,32 or48 updates:

| Historical recipe | Intact T16 | Damaged T16 | Damaged T32 | Damaged T48 |
|---|---:|---:|---:|---:|
| Fixed16, seed0 |98.4%|90.45%|56.6%|29.3%|
| Variable-depth plus damage, seed0 |99.6%|99.35%|98.8%|97.0%|

These figures come from [fixed16](https://github.com/Agnuxo1/NeuroPixel/blob/844f4dfae1cb8909d4f87c0c8cfd96b965590adf/results/phase3/stable_fijo16.json) and [rest-state](https://github.com/Agnuxo1/NeuroPixel/blob/844f4dfae1cb8909d4f87c0c8cfd96b965590adf/results/phase3/stable_reposo.json). The99.35% number rounds to the headline99.4%. It is a strong reported endpoint under that intervention, with source input still available. The later damaged scores decline; the summaries do not contain an immediate-post-lesion reading from which to count lost-and-recovered answers.

The driver saves state dictionaries, but this investigation did not establish availability or reproduce the historical trained checkpoints. The aggregate JSON cannot substitute for missing trajectories, masks and predictions. No claim is made that those models definitely use a shortcut or never repair state.

### Battery and reference comparability

The [battery results](https://github.com/Agnuxo1/NeuroPixel/blob/844f4dfae1cb8909d4f87c0c8cfd96b965590adf/results/battery/results.json) contain76 damage aggregates across four NP checkpoints and three Transformer checkpoints. The NP source uses post-update lesions and endpoints16/24; the Transformer source lesions the first encoder layer. The four NP clean16→24 accuracies are93.25→71.75%,93.15→75.65%,95.35→58.3%, and97.95→88.6%. Additional time therefore already changes the sham result.

The README's historical Transformer25–33% damage range comes from scaling runs, while the99.35% NP value comes from the rest-state recipe. Initial competence, lesion location and remaining computation differ. Equal cell-erasure probability is not equal functional lesion severity or an equal repair opportunity. These reports do not establish a matched advantage in autonomous repair.

### Previously audited item6

The already-observed item6 factorial has two training seeds, clean/damage training and clean/lesioned evaluation. It uses a common example-indexed lesion mask after update8 and scores only the16-update endpoint. The saved mask erases78,392 of262,144cells,29.9041748046875%.

| Training | Evaluation | Binding correct, seed20 /2048 | Seed21 /2048 | Mean binding |
|---|---|---:|---:|---:|
| Clean | Clean |270|151|10.2783203125%|
| Clean | Lesion |255|158|10.0830078125%|
| Damage | Clean |263|166|10.4736328125%|
| Damage | Lesion |274|182|11.1328125%|

The corresponding global means are20.03173828125%,18.37158203125%,19.15283203125% and19.47021484375%. The original source, archive, exact per-role counts and input hashes are recorded in the historical note and [item6 report](https://github.com/Agnuxo1/NeuroPixel/blob/844f4dfae1cb8909d4f87c0c8cfd96b965590adf/docs/research/06_results.md).

A modest damage-training endpoint difference at approximately10–11% binding cannot establish useful high-competence repair. No immediate state series exists in that panel, and a new recount cannot create it. This point did not retrain, reselect or rescore those checkpoints.

## 4. New controlled measurement

All three models are instances of the current native class with the same tiny architecture: V8, eight dictionary channels, eight state channels, hidden width16, a3×3 grid and center readout. Their weights are assigned explicitly in float64. No optimizer is used. Six cue IDs2–7 appear on all nine initial cells; the redundancy is intentional. The target is the original cue's identity, without a role query, semantic composition or novel-meaning task.

| Constructed rule | Exact action on these nonnegative fixtures | Purpose |
|---|---|---|
| Input copier |Next state equals current identity input.|Can rebuild a response from the available source even after all state is erased.|
| State holder |Next state equals the current state.|Retains existing information but cannot move it back to an erased output cell.|
| Spatial average |Next state is its zero-padded3×3 local average.|Can restore a cue-bearing center from surviving neighboring state without further cue input.|

Every case uses two warmup updates. The lesion then erases either no cells, the center, the eight-cell ring or all nine cells. These are exact0/9,1/9,8/9 and9/9 interventions, not a simulation of randomly erasing exactly50%.

The eight continuation updates receive either the original cue, all PAD, or the next cue in a fixed six-cue cycle. The boundary never adds a seed. Original-cue accuracy and following the changed cue are reported separately.

The complete census is3rules×4lesions×3future inputs×6cues =216trajectories. Saving the immediate post-lesion state and all eight later states gives1944endpoints. Aggregating six cues produces324rows. These are deterministic, correlated controls, not216 trained realizations or324 independent scientific experiments.

## 5. Observed controls

The table reports original-cue correct counts out of six. “Immediate” means after lesion with zero further updates. The sham uses the same future input and time as its damaged counterpart.

| Rule / lesion / future | Before | Immediate | After1 | After8 | Sham after8 |
|---|---:|---:|---:|---:|---:|
| Copier / all erased / source held |6|0|6|6|6|
| Copier / all erased / source removed |6|0|0|0|0|
| Copier / all erased / source changed |6|0|0|0|0|
| Holder / none erased / source removed |6|6|6|6|6|
| Holder / center erased / source removed |6|0|0|0|6|
| Spatial average / center erased / source removed |6|0|6|6|6|
| Spatial average / all erased / source removed |6|0|0|0|6|

The copier with a changed source follows the new cue in all six cases after the first update. Its restored answer is therefore dependent on the supplied content. The held-source version supplies a concrete counterexample to the inference “correct after full state erasure implies the episode survived in state.” It does not show that any historical trained NP model implemented this copier.

The holder preserves the cue throughout the tested eight-update horizon when the readout cell survives. Erasing that cell makes it wrong while its no-lesion continuation remains correct; despite surviving information in neighboring cells, this rule performs no repair.

The spatial-average rule loses the correct center answer immediately and then restores it from its neighbors without new cue input. This is genuine categorical recovery for this constructed rule and redundant fixture. It shows that the measurement can detect a positive control. It does not show that such a rule was learned, that arbitrary memories are recoverable, or that the previous tensor is reconstructed exactly.

The matched sham matters. The copier with removed input also loses its correct answer without any lesion. Its failure in the lesioned condition therefore cannot be attributed specifically to inability to repair. Its RMS difference from the same-time sham becomes exactly zero while both are wrong.

### Correct category, state restoration and assigned probability differ

For the spatial-average center-lesion/removed-source case, all six original cues are correct after one and after eight updates. Nevertheless:

| Continuation step | Mean original-cue NLL | RMS difference from same-time sham |
|---|---:|---:|
| Immediately after lesion |1.9459101490553132|0.07129265900852023|
| After1 |1.670893520296367|0.023764219669506745|
| After8 |1.9337284637637409|0.0011027719405024186|

The argmax recovers, but the original-cue NLL moves back toward the seven-non-PAD uniform value, log7≈1.94591. Zero-padding reduces amplitudes in both damaged and sham trajectories. A smaller distance to the sham is not evidence of recovered task information if both states lose a useful signal. This finite witness is not a stability theorem.

### Complete erasure with a common future

For each of the three rules, after all state is erased and the future is blank, every prior cue gives exactly the same output trajectory. The saved total is0correct across162endpoints:3rules×6cues×9readout times. These repeated endpoints are not162 independent episodes.

The native tie choice is token1, outside the six-cue target set. More generally, a single identical prediction could choose one of the six cues and reach1/6 on the balanced set, at a given model and time. This follows from indistinguishability once all episode-specific information is removed. Full-erasure recovery from absent information is not proposed as a requirement for a valid partial-memory repair mechanism.

## 6. Verification and operations

The five test methods and15 additional subcases produced20 successful call records, with no failure, error or skip. There are30 total test events including setup and teardown. XML contains five parent testcases and an aggregate counter of20. These are implementation contracts, not20 learned experiments.

The tests exercise an active untrained native model as well as simple constructed controls: exact held-input forward equivalence, lesion timing, independent snapshots, no extra seed, source changes, input and RNG preservation, and invalid contract rejection. All twelve separately saved held-source model×lesion traces match native forward exactly. All before/after parameter and buffer arrays are equal.

The independent audit performed6176checks on all88arrays and324rows. It imports neither Torch nor the producer or helper. It verifies assigned weights and reconstructs the three rules, padding, prefix, lesions, continuations, logits, argmax, NLL and transitions. The largest state/logit reconstruction difference was2.220446049250313e−16; the NLL difference was4.440892098500626e−16, well within the frozen absolute1e−12 tolerance. Saved direct-native witnesses and before/after weights are audited explicitly.

The frozen elementary-loop V8 recount performed5213checks, including3912numeric comparisons, without an issue; its maximum difference was4.440892098500626e−16. Another4860field comparisons between producer and auditor summaries agreed within that tolerance. Thirteen completion checks verified provenance, statuses, counts, resources and archival conditions. Root independently rehashed19text files and checked the complete22-file Git inventory including the manifest. The source ZIP and NPZ descriptors and sizes were matched; root did not independently decode the ZIP in V8. The separate NumPy program decoded and reconstructed the NPZ.

The worker took14.418820603seconds before final archival. The probe controller took1.338056848seconds; its fresh native and audit processes took1.167610721 and0.165087809seconds. These diagnostic timings are not model cost benchmarks. Numerical, interop and Git thread limits were1, under the aggregate ceiling4. Five supervisor samples had minimum available RAM14.146129608154297GiB; the lowest among all retained observations, including the native endpoint, was14.09469985961914GiB. Every observation stayed above8GiB.

All stages and the provider run completed successfully. There was no learned training, trained checkpoint loading, GPU request or paid computation. Before execution, reviewers improved saved equivalence witnesses and failure retention, aligned the helper import and corrected a count label. Preimages and correction receipts are retained; no failed scientific run or selective rerun occurred.

## 7. Achieved and still missing

| Objective | State after item14 |
|---|---|
| Identify information removed by historical lesions |Verified from source: cell state is erased; original IDs and parameters remain available.|
| Distinguish tolerance, recomputation and categorical recovery |Implemented and verified with pre/immediate/later readouts and matched shams.|
| Change future identity input without resetting or reseeding state |Implemented in the research helper; focused contracts passed.|
| Provide positive and negative native controls |Completed for the three assigned rules and216trajectories.|
| Independently reconstruct all saved tensor evidence |Completed by the separate NumPy audit; summary also reconstructed in V8.|
| Correct the public interpretation of the99.4% headline |README now states input retention, original conditions and comparison limits. Historical result files remain preserved.|
| Demonstrate learned repair in a competent model without answer-bearing input |Unestablished. Requires a competent no-lesion continuation and paired state lesions under a prospectively fixed recipe.|
| Recover diverse held-out episode contents from partial damage |Unestablished; repeated one-hot codes are a diagnostic fixture.|
| Separate redundancy, learned priors and externally repeated input in historical checkpoints |Unestablished; aggregate endpoints do not identify contributions.|
| Establish independent replication or transformative utility |Not established by internal constructed controls and code audits.|

A future learned experiment should first establish correct acquisition and correct source-withheld sham continuation, then retain individual lesion transitions with matched masks and computation. This point does not prescribe another broad run from the already weak checkpoints.

The next point is item15, dynamical stability. It opens only after this investigation's closure is recorded.

## Primary-source context

The focused review read the original [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) article, Model and Experiments1–3: its target enters the training loss and local state rules support pattern regeneration. This motivates distinguishing learned pattern structure from repeatedly supplied input, without transferring the paper's results to NeuroPixel.

[Hopfield1982](https://www.cctbio.ece.umn.edu/wiki/images/4/41/Hopfield_Neural_Networks_and_Physical_Systems_with_Emergent_Collective_Computational_Abilities.pdf), pp2554–2557, provides an explicit connection-based storage and retrieval construction. Its retrieval conditions do not prove convergence of an arbitrary recurrent canvas.

[Neural Turing Machines](https://arxiv.org/pdf/1410.5401), sections3 and4.1/4.3, separates learned procedures from writable episodic contents and tests sequence output without continued presentation of the sequence. It does not establish lesion repair in NeuroPixel. The present causal controls and interpretations are our deductions. This was a focused primary-source review, not exhaustive coverage of all internet documentation.
