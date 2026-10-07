# Item 11 learned-memory controller: implementation contract

Status: authored for prospective review; no compilation, tests, model execution or study outcomes are claimed by this note.

Controller candidate: `scripts/research_memory_study.py`, Git blob `5ea58325d00dec560590961155accdbc81675eb6` (41,732 ASCII bytes). Initial preimage `4a1a5d77d3b3e6af623da5f1ffd983266614f353` and reviewed candidate `a012fcbe956046b52f71af0301c2afda97e7b80d` are retained. The candidate uses strict stream helper `0968eb6216d3d7a07532710f9838aaae75cad801` and pinned-input recovery helper `b793c66eb141b2847fb5f3c1be6836becec9e2f2`. Historical model/driver code is not edited.

## CLI and plans

`python scripts/research_memory_study.py --phase preflight|train|final --plan PLAN --output RUN_ROOT`

`--describe` emits recipe, four pilot configurations and the final-case inventory without importing Torch or NumPy. It is not a study execution.

Plans require schema_version1, item11, status frozen, freeze_utc, phase preflight or study, exact recipe, implementation_sha256, runtime {python,packages,threads:2,interop_threads:1}, and for study inputs.preflight. The latter is a complete immutable archive specification including its own manifest; recovery occurs through research_item11_inputs, not a mutable branch. Exact committed plan bytes and source bindings are checked before numerical imports. Scientific environment/status receipts use the study_ prefix, separate from worker receipts. An already configured interop1 is not set again.

The four pilots use seed69, learning rates0.001/0.003, and1,024 updates. The initial512-update suggestion was prospectively changed to1,024 before any neural execution or outcomes. Main uses five fresh seeds70..74 and2,048 updates, batch32. Constant AdamW learning rates, weight decay0.0001, norm clipping1.0 and answer CE are common. There is no early stopping, adaptive rescue or schedule.

Selection orders mean DEV accuracy at delays3/4 descending, then mean CE ascending, then LR ascending. Both selected pilots must reach accuracy>=0.95 on the known48-case delay2 census. A complete negative preflight preserves all four runs/selection/summary, marks admission false, and completes operationally with exit0. It is a valid scientific negative, not an operational failure. load_selection still rejects admission false before any main/final work. Selection is reauthenticated and raw logits/NLL recounted before train/final reuse.

## Known task census and streams

Vocabulary: PAD0, query1, six answers2..7. Grid3x3, output(2,2), query(2,1). Cue positions are the eight row-major cells except output, including the query cell. Census order is target outer, cue-position inner. This deliberately known48-element population makes no held-out-content or independent-custody claim.

A cue is visible for3 NP updates, each blank/distractor frame for3, and query-only for10. GRU receives one transition per frame. At delay d there are13+3d NP updates and d+2 GRU sequence steps. No equality of computation is claimed. One private PCG64 stream per run seed draws a full saved [updates,32] matrix of uniform census indices, then [updates] uniform delays0/1/2. Delay is common within a minibatch. Exact balance applies to evaluation census, not every32-example random minibatch. Families and pilot rates share the same saved index/delay plans for the same seed. Initialization uses seed; firing uses110000+seed, separately from the index seed100000+seed.

NeuroPixel has29,392 trainable parameters and a per-example state of432 float32 values,1,728 tensor-payload bytes. FrameGRU(e8,d48,hid70) has29,336 parameters and70 values,280 payload bytes. Parameter matching does not match recurrent-state allocation, update count, useful information capacity, allocator/autograd overhead or peak memory.

## Final inventory and causal boundaries

Each checkpoint has16 cases and1,488 nominal rows:

- Normal at delays0,2,4,8,16,32,64: seven cases of48.
- Cue-erased, state-zero and state-swap at delays2 and8: six cases of48.
- Delay8 blank-repeated, same-position distractor and next-position distractor: three cases of288.

All state interventions occur immediately before the query, after the original cue and all gaps. Swap assigns recipient state from the next cyclic answer at the SAME cue position; donor labels and row indices are explicit. The entire case is evaluated as one batch, preserving the bijection.

Interference enumerates every target6 x cue-position8 x distractor-label6 combination independently. It replaces the FIRST blank frame at delay8; total time remains37 NP updates/10 GRU frames. The48 same-label rows are reexposure/reinforcement and are reported separately from240 conflicting-label rows. Next-position means the next of the eight allowed cue cells, cyclically. Blank-repeated uses the same288-row enumeration but presents no distractor: each of48 original histories repeats six times, not six scientific replications. All three arms are evaluated at the same288 batch size.

Final outputs are outside study/runs. A complete ten-run training manifest and root-confirmed pre-final remote archive receipt precede exclusive final_access.json. The gate freezes weights/configuration/summaries; its access boundary concerns model outcomes, since cue content is known by design. A failed consumed access cannot silently retry. Controller does not compute inferential intervals or change criteria.

## Preserved arrays and reconstruction

Each prediction case stores logits float32[N,8], prediction int64[N], target int64[N], float64 NLL computed from the exact saved float32 logits, row_id, full frames[N,F,3,3], NP steps[F], cue/distractor positions, donor/distractor labels, and pre/post-query states. JSON stores exact rows, case metrics, per-label/per-position counts, independent distractor strata and descriptors.

Pre/post-query state arrays have leading batch dimension: NP[N,48,3,3], GRU[N,70]. For zero/swap, additional intervention_before/intervention_after arrays preserve the helper's native state dimensions: NP[N,48,3,3], GRU[1,N,70]. These snapshots straddle the actual state replacement before any query input. Swap also stores donor_row_index. A recount can directly verify zero or indexed permutation without neural inference.

Every run saves config, parameter checkpoint, nonpersistent-buffer metadata, train_indices.npy, training_delays.npy, losses/gradient norms and timings every128 updates, available-RAM boundary samples, and raw known-census predictions at delays2/3/4. Parameters are finite before checkpointing; checkpoint hashes are checked after evaluations. Nonpersistent grounding buffers and constructor metadata are retained because state_dict alone excludes them.

Training-loop times, mean/median update time and total run time are separately labeled; total includes initialization, stream persistence, training and endpoint development evaluations. Sampled MemAvailable is not continuous minimum RAM, RSS or energy. Two numerical threads and an8GiB floor are required; parent worker owns continuous supervision, timeouts and actual remote archive authentication.

This is a research-only state continuation experiment, not a claim that ordinary forward automatically remembers previous calls. The separate auditor must recount competence before interpreting forgetting, distinguish original-target and donor-target accuracy, report the five paired delay8 differences with df4 t uncertainty, and keep finite-horizon, known-census and parameter/state-capacity limitations explicit.

## Focused contract tests (authored, not executed)

`tests/test_memory_study.py` contains eight parent unittest methods: inventory/independent distractors; literal cue-free suffix and donor geometry; private matched streams independent of NumPy globals; selection ties and completed negative admission; semantic NLL/prediction/target corruption after rehash; ten-artifact gate; consumed access before model construction and retry refusal; and one explicit two-row optimizer/weights-only roundtrip fixture for each actual factory. Torch and NumPy requirements may skip only when absent outside the frozen CPU runtime; the admitted environment must run all eight.

Only stream fixtures use seeds110913/110914 (plus global-noise110920..110923); the tiny optimizer fixture uses110931. No scientific seed's stream is generated by these tests. The explicit inputs come from the deliberately known task census and are not represented as hidden final content. The optimizer test restores its prior numerical-thread count.

CLI: `python -m unittest discover -s tests -p test_memory_study.py -v`. No result, compilation success or test pass is asserted here. Source revisions are retained before the first prospective execution. The completed-negative status correction changes no sampling, optimizer, threshold, budget, selection or endpoint.

The final preflight gate hardening requires exactly normal_d2, normal_d3 and normal_d4 development entries per completed main run, with both metadata and prediction descriptors. A prior controller candidate accepted an empty development mapping; root found this in static review before execution. The final eight-method test source is Git blob `288c18f87876c2f54ca77290c1ee9480eaee686d` (20,785 ASCII bytes). Its positive gate now contains real finite NPZ/JSON development fixtures, and its negative witness removes d3 then rehashes the run, manifest and gate, isolating semantic inventory checking from byte-hash checking. Previous controller `38510f1b9a1ecfed8efd9ff4f79ea3aa34fc58b2` and test `cfb5c692cb7076878515d665ba48fb306f73c462` are preserved unexecuted preimages. No model recipe, data stream or inference condition changed.
