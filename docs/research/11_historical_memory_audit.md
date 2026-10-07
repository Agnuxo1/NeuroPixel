# Item 11: historical memory evidence and counterfactual requirements

Read-only review at 2026-10-07T12:35:26Z of source `b343ff5f06ea980a07263bbb3c598a1d9451643d`. No model, test, training, inference, or job was executed. This inventory distinguishes historical reported results from present source inspection.

## What the historical experiment actually retains

`neuropixel/phase3.py:15-32` implements `np_stream`: it sets a local state `s=None` once per sequence call, adds the seed of each currently visible frame, and carries the resulting tensor into subsequent frames. Facts therefore can influence later query output through recurrent activation even after their input frame disappears. This is a real within-call state pathway; default-call statelessness does not refute its existence. A new stream call initializes state again. `FrameGRU.forward_frames` also starts a new GRU sequence without a supplied prior hidden state.

`MemoryTask.sample` emits four facts in fixed role order, then d empty frames, then only the query. Prior facts are not reintroduced during the empty/query suffix. However, each currently visible fact is available throughout its three updates, and the query throughout ten updates. Fixed presentation order also makes temporal slot informative about role. The historical result does not separate learning role markers from using temporal order.

The default schedule is 4*3 + d*3 + 10 = 22+3d NeuroPixel updates, versus d+5 GRU recurrent sequence steps. Equal blank-frame delay is not equal update count, computation, physical elapsed time, or information-processing opportunity. Here d=8 means 24 blank NeuroPixel updates and 46 total updates.

## Retained numerical artifact and validity limits

The complete `results/phase3/memory.json` is 412 bytes, Git blob `e6ec640fa22e822092dcf83beb1399a89e03c333`.

| Blank frames | NeuroPixel reported accuracy | GRU reported accuracy |
|---:|---:|---:|
| 0 | 0.992 | 0.770 |
| 4 | 0.987 | 0.791 |
| 8 | 0.965 | 0.785 |
| 12 | 0.861 | 0.781 |
| 16 | 0.695 | 0.768 |
| 24 | 0.440 | 0.746 |
| 32 | 0.316 | 0.729 |

It reports 29,824/49,067 parameters and 6,079/565 training seconds. It contains no seed, actual update budget, environment, source binding, per-role counts, raw frames/targets/predictions, checkpoints, loss history, or counterfactual intervention outputs. The inspected driver does not save memory checkpoints or prediction arrays. No independent numerical reproduction is claimed from this summary.

`scripts/phase3.py:187-219` defines one selected seed, batch256, random training delays0..8, AdamW/OneCycleLR, answer CE, and clipping1.0. Evaluation uses1000 test samples and a fresh device-specific generator seeded3 for every delay. Under one runtime/device, that code reuses the same sampled examples across delays; they are paired measurements, not independent populations. The command defaults to20,000 updates but the JSON does not establish which command/budget actually produced it. The CLI selects only the first requested seed. Sampling on CPU versus CUDA is not guaranteed byte-identical.

The README says “99% up to8 blank frames,” which overstates its own d=8 artifact (96.5%). The Spanish conclusions instead round that point to97%. Beyond8 frames the tested delay is outside the driver's training-delay range; this is temporal-horizon extrapolation within the same generator, not novel vocabulary or open-ended episodic competence.

The current effective dictionary projects PAD to zero. The historical model at `7da18d1cbd36c40c48228118abb20a7b5ab032f4` did not apply that projection. Historical blank-token embeddings could therefore be learned constants; they are not necessarily numerically zero, although they do not carry the particular prior episode's cue. A corrected-core rerun is a new investigation, not a byte-exact historical replay.

## Memory objects and prohibited conflations

- **Parameters and expert banks:** persistent learned mappings stored in checkpoints. Changes caused by explicit optimization are parameter learning; item6 bank retention is not episodic activation memory and retains its weak-competence/capacity caveats.
- **Within-call state:** the tensor carried by `np_stream` or the GRU through a sequence. This is the historical memory object.
- **Across-call activation:** absent from the ordinary fixed-weight, evaluation-mode, hook-free `NeuroPixel.forward` API: each call seeds anew from its current input. Returning a state tensor does not automatically reuse it.
- **Caller-supplied state or hooks:** an explicit external carrier. A controlled continuation can test it but must not be credited to an unmodified default-call API.
- **RNG state:** training-mode random firing advances a generator; changing call order may change outputs without content-dependent memory. Use evaluation mode or coupled/replayed RNG states for causal comparisons.
- **Nonpersistent buffers/configuration:** `g_rgb` and `g_mask` affect the effective dictionary but are excluded from state_dict. Checkpoint restoration requires their metadata and all relevant constructor arguments; weight hashes alone do not reconstruct the function.
- **Current input:** repeated reinjection while a cue remains visible is access to that cue, not cue-free retention.

The inspected13 core tests cover forward/backward shapes, initial empty-state behavior and other contracts. Their independent HRR binding test is a separate vector-binding mechanism; it does not establish neural stream retention or resistance to interference. This is not an exhaustive claim about every later research test.

## Bounded counterfactual requirements

1. Build paired histories with distinct cue answers and exactly identical post-cue blank/query suffixes. Retain input hashes and independent gold. Cue-erased and current-query-only conditions must have no answer-bearing token anywhere in that suffix.
2. At a prospectively fixed boundary, continue from the original state, an all-zero state, or the matched other history's state. Preserve suffix, weights, buffers, readout and RNG. A state-swap effect that tracks the other cue is stronger evidence of content carriage than merely observing nonzero activity.
3. Verify an instrumented strict continuation against the unmodified stream on unperturbed sequences before interpreting interventions. Validate frame/step cardinality; the historical zip loop otherwise silently truncates unequal lists.
4. Include a deliberately constructed positive carrier whose readout responds to the carried cue. An untrained zero-final-layer dynamic with an unreachable readout can yield an uninformative null. The positive carrier checks diagnostic sensitivity, not learned capability.
5. Distinguish blank-delay decay from active interference. For distractors, preserve total transition time with a blank-matched comparator, fix whether gold remains the original cue, and report original-cue recall and distractor intrusions separately. Same-position and different-position distractors test distinct pathways.
6. Require near-trained-delay competence before interpreting a decline as forgetting. A system that never reads the cue cannot establish a retention deficit. Finite-horizon success is not indefinite durability.
7. Keep parameters unchanged during retention evaluation. Any optimizer-update/interference study is a separate parameter-memory question; this inventory does not admit one.

A deterministic predictor of only the common suffix returns the same answer for a distinct-target pair. It cannot get both members right, and its pair-mean accuracy is at most one half. This is a logical control bound, not an empirical baseline result. Full or partial cue replay invalidates that interpretation.

## Inspected source identities

| Path | Git blob |
|---|---|
| scripts/phase3.py | a73b96df3266b8bb382e7dbb7ba86ad651e965b3 |
| neuropixel/phase3.py | b826869ad1816f43a182fbcdbcbc47ea2a123dd5 |
| neuropixel/model.py | b66e2f575c6ae937d8cad807b4dd0e48531850f7 |
| neuropixel/task.py | 3e9368c364a884f72931feb3cdc921bc3663d4fe |
| tests/test_core.py | a09bab50ddc3f8d5c495ab4b45effd6217a0a084 |
| README.md | cc6b86440a032e1f6f51e50a8841e50c9fa42a2a |
| docs/FASE3_CONCLUSIONES.md | a1b85313e67fc7fdbbc698bf5db08d3f2706fe05 |
| results/phase3/memory.json | e6ec640fa22e822092dcf83beb1399a89e03c333 |

The attempted read of tests/test_phase3.py returned404; it is not used as proof that no memory tests exist. Default-branch code search returned no useful matches and is not evidence about the complete frozen tree. Existing coordination was read; root owns reservations and any prospective execution. Later programme items remain unopened.
