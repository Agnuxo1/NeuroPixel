# Item14 protocol: assisted repair versus stored memory

## Question and admission decision

This point tests whether the measurement separates reconstruction from a surviving state, tolerance of damage, and recomputation from input that remains available. It does not reopen the historical H1. Previous item6 and item11 competence limitations are already known; no new learned repair benchmark, training search or checkpoint selection is admitted here. Three native models with explicitly assigned rules provide positive and negative controls for the measurement. The archive must label them constructed controls, never trained capabilities.

A final correct answer after a lesion is insufficient to identify recovery. We will preserve the readout before the lesion, immediately after it, and through continued dynamics. The source-removed condition retains the recurrent state and changes only the recurrent input; it does not call a fresh forward on a blank canvas. A full-erasure/common-future condition is an information-loss control, not a requirement that a memory system reconstruct information from nothing.

## Previously observed sources

The following source evidence was inspected before this protocol freeze:

- Current native forward at item14 opening5d9d09fe4c4ba045b3bfae586718f5b1b6c10a6f: initial seed from canvas, fixed dictionary IDs concatenated at every update, post-update hook changing state only.
- Historical stable driver scripts/phase3.py (a73b96df3266b8bb382e7dbb7ba86ad651e965b3): two coupled regimes, fixed16 or training with12–32 updates and stochastic damage. Stable tests retain full input, erase entire state vectors of sampled cells after update8, and score endpoints16/32/48. They have no immediate-post-lesion readout.
- Historical stable JSON: fixed model clean98.4% at16 and90.45% after damage; rest-trained model clean99.6% at16 and99.35% after damage. They each describe one reported seed and2000 examples. This protocol does not recreate their missing per-example traces.
- Battery T1: four NP and three Transformer checkpoints; NP has post-update lesions and extended recurrent time, TF a first-layer activation lesion. Those interventions and initial competencies are different.
- Item6 already-audited binding means range about10–11% for this damage panel. A small endpoint difference there does not establish high-competence repair or autonomous stored-memory recovery.

No historical aggregate is relabelled as newly measured. The separate report will retain exact files, counts, masks and source-version limitations.

## Constructed native census

The same native NeuroPixel class is used for all three rules, with V=8, dictionary/state width8, hidden width16, a3×3 grid, center readout, float64, eval mode, configured fire_rate0.5 (inactive in eval). All model tensors are explicitly assigned, not learned. IDs2–7 are six distinguishable one-hot cue codes; each initial canvas repeats one cue at every one of its nine cells. This redundancy is intentional and makes no role-binding or semantic-composition claim. PAD0 has an exactly zero effective dictionary; token1 is unused as a cue and is the argmax tie choice after all useful logits become zero.

The parameter embedding is I8, effective PAD projection applies, seed and read matrices are identity with zero bias. All other parameters start at zero, followed by the declared assignments:

1. input_copier: two ReLU branches pass nonnegative state and current IDs, with residual update ids minus state. On these fixtures the next state equals the available cue. It supplies an assisted-reconstruction control even after full state erasure.
2. state_holder: zero update. Information in the state survives when its readout cell survives. It has no rule to move surviving neighboring information back into an erased readout cell.
3. spatial_average: two branches pass nonnegative state and its zero-padded3×3 mean, with update mean minus state. Each depthwise channel has two interleaved output filters; the even filter implements the mean. This rule can restore a cue-bearing center from surviving neighbors without further cue input. It ignores IDs during updates. Zero padding changes amplitudes even without lesions, so no exact restoration of a clean state, confidence calibration, asymptotic stability or learned associative memory is claimed.

Every case has exactly two warmup updates and eight continuation updates. Lesions occur after warmup2 and before continuation0. Four boolean keep masks are used: none (0/9 cells erased), center (1/9), ring (8/9) and all (9/9). These are exact masks, not claims about50% random damage.

Three continuation canvases are held original cue, all PAD, and the next cue in the fixed cycle2→3→4→5→6→7→2. There is no seeding at this boundary or thereafter. Cue identity is the reported original target in every condition; following the changed cue is reported separately and is not scored as remembering the original.

The full inventory is3 models ×4 masks ×3 continuation sources ×6 cues =216 trajectories. Each has nine saved continuation states (immediate post-lesion plus steps1–8), giving1944 categorical endpoints. Aggregation over the six cues gives324 rows; these are deterministic repeated controls, not324 independent empirical experiments.

## Measurements and necessary comparisons

Save the seeded state, both warmup states, all pre/post lesion states, every continuation state and its logits. Preserve each model's parameters and buffers before and after the census. For twelve held-source model×lesion combinations, separately execute the unchanged native forward at10 updates with a post2 hook and retain its actual trace/logits. The harness must match this path exactly. The same arithmetic engine is used for that equivalence check; the independent auditor reconstructs the saved arrays separately.

For every model/mask/source/recovery step report n=6, correct before damage, immediately after, at the later step and in the same-source same-time sham. Preserve counts initially correct then lost, lost then recovered, and survived and still correct. Distinguish C→W→C, C→C→C and W→W→C. Also report matches to the changed cue, mean NLL of the original cue and RMS state difference from the time-matched sham. Do not report a conditional recovery fraction without its initially-lost denominator. No confidence intervals, p-values, population effect size or claim of trained competence is planned.

The full-erasure/removed-source control must yield cue-indistinguishable future outputs. Given identical post-erasure state, model and future inputs, deterministic evolution cannot identify an arbitrary prior cue. On a balanced six-cue set, any identical predictor can achieve at most1/6 by always choosing one cue; the present tie rule may perform worse. This is a conditional information argument and a negative control, not a general limitation on recovery from partial redundant information.

## Regression contracts

Five parent unit-test methods cover the new research-only helper: equivalence to active native dynamics; post-update lesion timing and independent snapshots; no reseeding at the continuation boundary; source-switch causality and complete-erasure controls; and explicit rejection of invalid modes, steps, tensors and token IDs. Tests may contain declared subcases. They are software/measurement contracts, not independent scientific trials. The helper is eval-only, token-only, native-class only; camera/retina experiments and stochastic training dynamics are outside this point.

## Independent verification

Two fresh serial processes run the native fixture and an independently authored NumPy audit. The audit imports neither Torch nor the producer/helper; it checks source/plan/array identities and exact inventory, every saved parameter/buffer assignment, all inputs and masks, prefix/damage/continuation arrays, saved direct-native traces, unchanged weights, logits, argmax, NLL and all324 rows. An elementary-loop spatial mean, independently defined from the conceptual rule, is used with absolute tolerance1e-12, relative tolerance0. A separate bounded V8 implementation reconstructs the JSON summary from the rules after archival. Repeated cues are explicitly correlated; these are independent programs on the same evidence, not independent laboratories.

All discrepancies and failed executions must be preserved; any correction after a failure requires a new source/protocol record and fresh run ID rather than overwriting evidence. No outcomes are excluded to improve presentation.

## Execution and source provenance

Only the isolated research source/results branches are used. The new worker is a namespace/scope derivative of the completed item13 worker; underlying item9 supervision and Git transport are unchanged. The frozen plan binds all scientific, test, worker, protocol and correction files by SHA-256. The plan itself is bound by its Git commit/blob and its copied execution-plan hash; it is not made self-referential.

Use one standard public CPU runner, at most20 workflow minutes, with900 worker seconds including180 archival reserve. Contract stage120 seconds; combined probe540 seconds; each native/audit component at most120 seconds in a fresh process. Outer owned-process-group supervision samples each1 second, enforces available RAM>=8GiB, emits heartbeat60 seconds and permits periodic archival300 seconds. Numerical/interop/Git threads1; aggregate activeCPU<=4. No GPU, paid computation, main merge, outreach, billing or access changes. No current numerical task competes with this reservation.

Runtime: Python3.12.14, NumPy2.2.6, SciPy1.15.1, Pillow12.3.0, psutil7.2.2, pytest9.1.1 and Torch2.6.0+cpu. The workflow preserves bootstrap outcomes; complete provider logs are collected. Final archive includes source.zip, execution plan, source guards, test transcript/XML/events, full traces/weights, producer and independent reports, component logs, runtime and resource records. Root verifies archive tree/sizes and available text hashes before closure. No independent binary decoding in V8 is implied.

## Focused primary-source context

Read directly on2026-10-07:

- Mordvintsev et al.2020, [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/), Model and Experiments1–3. Local state-based rules and damaged training states support pattern regeneration in that model. Its target pattern enters the loss rather than being supplied as input at every update; this does not prove storage of arbitrary episodic content.
- Hopfield1982, [original article PDF at University of Minnesota](https://www.cctbio.ece.umn.edu/wiki/images/4/41/Hopfield_Neural_Networks_and_Physical_Systems_with_Emergent_Collective_Computational_Abilities.pdf), pp2554–2557, DOI10.1073/pnas.79.8.2554. Stored patterns and their retrieval conditions are explicit. No convergence theorem is transferred to NeuroPixel. The publisher landing page's cookie redirect was not treated as full-text access.
- Graves et al.2014, [Neural Turing Machines](https://arxiv.org/pdf/1410.5401), sections3 and4.1/4.3. Writable episodic content and sequence output without reintroduced input motivate operational separation of parameters, state and future inputs.

The intervention design and interpretations above are our deductions, not additional claims by those sources. This is a focused review, not exhaustive internet coverage.

## Closure criterion

Close the feasible investigation after tests, native census, independent array and summary audits, historical-source accounting, claim wording corrections and peer review are complete. Closure does not mean autonomous learned repair has been achieved. Items15–30 remain unopened until that separate closure is published.
