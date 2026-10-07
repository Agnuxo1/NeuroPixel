# Item 11B: prospective memory-path diagnostic protocol

## Question and scope

This protocol tests where information can be retained in the current implementation. It is a finite, untrained mechanism diagnostic, preceding any separately admitted learned-memory study. Historical H1 remains closed and unsupported. Items 12–30 remain unopened.

The stored objects must be distinguished: trainable parameters, grounded dictionary buffers, activation within one stream, explicit caller-owned activation, and current inputs. The original `NeuroPixel.forward` constructs its starting state from the current canvas on every call. The existing `np_stream` in `neuropixel/phase3.py` carries state across the frames of one sequence, starting again at its next call. A new research helper exposes state continuation and interventions explicitly while preserving the original recurrence; it is not a production change to either original entry point.

## Prospective expectations

1. With fixed parameters/buffers, evaluation mode and the same current query, separate calls produce the same state and logits irrespective of discarded histories. Mutating a returned state also does not create implicit model storage.
2. Native within-call streaming and the strict helper agree. Dividing the same stream into a prefix and a query, while explicitly passing the prefix state, agrees with the uninterrupted computation.
3. At the boundary immediately before the query, zeroing removes the entire carried activation and produces the query-only computation. Swapping complete batch states follows the donor under the identical query. Save the actual boundary tensors to verify both interventions.
4. An explicit hook can inject a caller-retained state. This is a positive control for an added memory path, not evidence that default calls retain it.
5. Loading parameters alone is insufficient to reconstruct a grounded model because grounded buffers are excluded from its state dictionary. An explicit reconstruction including the original buffers/configuration restores the witness.
6. All values and parameter/buffer/RNG invariants must satisfy the declared checks. A failure is retained and investigated; it is not silently removed from the report.

These expectations concern this implementation and declared fixture. Positive cue dependence is not learned retrieval accuracy, useful memory, indefinite stability or resistance to interference.

## Fixed fixture and parameters

The vocabulary has eight entries: PAD=0, QUERY=1, and six cue labels 2–7. Each canvas is 3 by 3. Cue position is (0,0), query position (2,1), and readout (2,2). All six labels occur in a single batch. The cue receives three local updates, every blank gap three, and the query ten. Blank-frame delays are exactly 0, 4, 16 and 32, giving 13, 25, 61 and 109 local updates.

Use NeuroPixel c_id=16, c=48, hidden=128, retina disabled, no grounding for the primary diagnostic, evaluation mode, CPU float32, and 29,392 trainable parameters. Initialization uses Torch seed 110001. Because the default final update projection starts at zero, activate its weight and bias once with an independent CPU generator seeded 110002 and normal standard deviation 0.01. Preserve parameters and both global/private RNG states. This deterministic intervention is not an optimizer step or a trained checkpoint.

Five separate-call histories are empty, cue only, cue plus four blanks, cue plus sixteen blanks, and cue plus cyclic distractor plus four blanks. The common final query is identical in all cases. Stream conditions at each declared delay include native, strict, explicit chunk continuation, no cue, pre-query zero, pre-query cyclic state swap, and hook injection. The swap mapping is [1,2,3,4,5,0]. The hook adds the saved prefix after query update one, so it has a different update order from initial-state continuation and is not required to equal it.

Native/helper, chunk, zero and history identities are checked exactly where the same operations are expected. Donor permutation comparisons use absolute tolerance 1e-6 and relative tolerance 1e-5 to accommodate execution-order roundoff. All recorded values must be finite. The positive-control difference threshold at delay zero is 1e-10 in state maximum absolute difference; it is an activity witness, not a significance test.

## Contracts and evidence

Run exactly eight collected methods in `tests/test_stream_memory.py`, covering native and stochastic-firing equivalence, differentiable chunk continuation for both families, boundary zero/swap correctness and snapshot ownership, malformed frames/update counts/states/interventions, default-call reset, and grounded checkpoint reconstruction. Nested subtests are recorded separately from collected methods; any skip, failed test, inventory mismatch or changed thread setting fails this operational gate.

Then run `scripts/research_memory_probe.py --phase probe --plan docs/research/11_probe_plan.json --output RUN_ROOT`. Save arrays, dictionaries, parameters, buffers, RNG states, a parameter-only serialization witness, all condition descriptions/checks, runtime, source identity and status. Hash each saved file. A saved-array recount and actual Git/source-archive verification are required before closing this item. They must not rerun models to replace the original outcomes.

## Resource admission and stopping

Use the isolated authorized source/results branches and one standard public ubuntu-24.04 CPU runner. Python and every declared package are pinned by the executable plan. Numerical threads=1, inter-op=1, Git packing=1, aggregate active CPU cap=4; available RAM must be at least 8 GiB at admission and sampled supervision. No other numerical job is admitted concurrently.

Worker budget is 600 seconds TOTAL, including a 180-second final archival reserve. Contract tests and probe each have a 120-second cap within the shared remainder. Workflow limit is 20 minutes, with a deadline of 1,170 seconds from its first step; the earlier deadline always wins. The existing owned-process supervisor samples once per second, reports heartbeats every 60 seconds and attempts partial archival every 300 seconds. Failures, timeouts and bootstrap outcomes are retained; source and final raw archives use ordinary Git commits. No paid/GPU compute or external message is involved.

## Interpretation and primary background

The historical repository curve is separately audited in [11_historical_memory_audit.md](11_historical_memory_audit.md). Its rounded summary is not replaced or treated as independently reproduced by this fixture.

LSTM distinguishes activation-based retention and learned weights, and explicitly studies interference with stored signals [Hochreiter and Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf). Neural Turing Machines reset dynamic state between episodes in their synthetic evaluations and separate input presentation from subsequent copying [Graves et al., 2014](https://arxiv.org/pdf/1410.5401). These motivate explicit episode boundaries and cue removal; this diagnostic is not a replication of their benchmarks. Durable episodic retrieval will require a separate learned task, competent retrieval, controlled delays/interventions, and evidence beyond the finite activity identities here.
