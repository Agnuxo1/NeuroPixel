# Item 15 — finite dynamics measured; general stability remains unestablished

## Outcome

Both frozen parts of item15 completed and were independently checked. The constructed native census demonstrates distinct possible behaviors under explicitly assigned weights. The separately frozen retrospective panel measures five actual item9 checkpoints on one already exposed validation scene.

All five learned models reproduced their archived T16 outputs within the prospective tolerance, with every compared token prediction unchanged. All40 conditional trajectories then completed256 additional updates, preserving10,280 observed states. None crossed the operational absolute-state limit1e12 or produced a nonfinite state. Their20 paired final perturbation gains ranged from1.556726 to6.058958, and every trajectory's largest observed state RMS occurred at the final endpoint.

These observations establish finite recorded behavior and successful measurement controls. They do not establish convergence, attraction, useful stable memory, universal stability or asymptotic divergence. The absolute final RMS separation of the learned pairs remained approximately1.56e-4–6.06e-4: amplification of a small perturbation is not automatically a large or chaotic separation. Item15's feasible investigation is complete; its strongest scientific capability claim remains unestablished.

Primary project evidence: [constructed census report](15_census_results.md), [15A final archive](https://github.com/Agnuxo1/NeuroPixel/tree/ff34cc4f44c535e5f4d00276e1e3be194ec70f21/results/research/15_cloud_runs/37649654847-1-probe), and [15B final archive](https://github.com/Agnuxo1/NeuroPixel/tree/205507b71e2f7abb00ce97aa1dbec8321fc0af26/results/research/15_cloud_runs/37654405152-1-preflight). The latter contains full states, original checkpoints, both sets of logits, independent receipts and the source ZIP.

## What was already supported by the repository

The historical rest-state experiment reported output accuracy at depths8,16,24,32,48 and64, using2,000 examples per endpoint and one initialization per configuration. The fixed-depth model had accuracies0.998,0.984,0.8625,0.6745,0.377 and0.227; the rest-trained model reported0.992,0.996,0.995,0.9925,0.982 and0.958. This is useful finite output-retention evidence for those recipes. It is not a measurement of a uniform state bound or a fixed-point residual. The training-depth and damage changes were combined, so their separate contributions are unidentified.

The original summaries lack per-example state trajectories, probabilities, NLL and perturbation responses. The README animation covers24 steps and uses PCA plus normalization/clipping, which cannot preserve raw magnitude as a stability diagnostic. Item11's large finite NLL values likewise do not identify an asymptotic instability or its cause. These historical results concern different tasks and training recipes from item15B and are not directly comparable checkpoints.

The activity panel consists of three T16 configurations, with regularization coefficients0.3,1 and3. All three reported a final active-pixel fraction of1 under a0.05 state threshold; active-update fractions under a0.01 threshold were1,0.7217 and0.4929. Activity used the first1,000 examples and15 adjacent frame differences; accuracy used2,000 examples. There is no coefficient0 configuration among these three summaries. Finite thresholded activity cannot support “never returns to black” for all future time. It also cannot be converted into a physical energy measurement. Cost and physical-energy questions remain separate later items.

The exact original paths, blobs, SHA-256 hashes, values and measurement definitions are in [historical evidence](../../results/research/15_validation/historical_evidence.json). The damage/repair interpretation is documented in [item14](14_results.md).

## Why output, state and stability must be distinguished

For a held input, the native update has the form

F_u(s) = s + W2 ReLU(Ks + W_i u + b1) + b2.

Within a differentiable activation region its state Jacobian is I + W2 diag(active) K. The source imposes no uniform contraction condition, hard norm bound or certified Lyapunov function. Zero initialization of the final update layer yields an identity state map; identity preserves differences and is not attraction to a unique state.

The tied readout observes a projection of state. A constant output therefore need not imply a constant state. Conversely, a state can remain finite over a chosen horizon while its output changes. Four separate questions are relevant:

1. **Finite-horizon bounded observations:** whether the actual retained states remain below a stated threshold.
2. **Convergence:** whether the state tends to a limit as time grows, with a residual approaching zero under explicit assumptions.
3. **Attraction or incremental stability:** whether nearby initial states approach each other, in a specified metric and input regime.
4. **Useful memory:** whether retained information supports a validated task under the intended input/interference conditions.

Passing the first question does not answer the others. A finite perturbation gain is also not a Lyapunov exponent or an operator norm. Its meaning depends on direction, amplitude, reference trajectory, time and arithmetic.

This distinction is consistent with the assumptions of [Miller and Hardt, Stable Recurrent Models](https://arxiv.org/html/1805.10369v4), the explicit contraction constructions of [Revay and Manchester](https://arxiv.org/pdf/1912.10402), and the distinction between training-gradient products and state dynamics in [Pascanu, Mikolov and Bengio](https://proceedings.mlr.press/v28/pascanu13.pdf). These sources motivate the diagnostic; they do not certify this implementation.

## Part A — controlled native counterexamples

The first plan was frozen before execution. It used the unchanged native class in float64 evaluation mode, a1x1 grid, four state channels, an explicit dictionary/readout, two initial states per case and256 retained continuations. We assigned seven affine dynamics on their verified nonnegative invariant region. Their mathematical properties were derived from the assigned maps, not inferred solely from a finite plot.

| Assigned case | Retained numerical result | Supported distinction |
|---|---|---|
| Identity | State norm and perturbation gain remain1; increment is0 | Constant state is neutral retention, not unique attraction |
| Contraction0.5I | Final norm/gain8.636168555e-78 | A particular assigned map can contract |
| Expansion1.05I | Final norm/gain265,742.221922 | Native architecture does not itself forbid expansion |
| Hidden drift | State norm1→256.001953 while logits, prediction and NLL remain constant | Stable readout can conceal changing state |
| Two-cycle | State alternates, norm remains1 and output token alternates | Boundedness does not imply convergence |
| Nonnormal transient | Spectral radius0.9; pair gain peaks15.501662 at step9, then falls to2.198569e-9 at256 | Transient amplification can coexist with eventual decay |
| Forced contraction | State approaches norm2; final measured residual0 | Held forcing changes the limiting state |

The nonnormal example has operator norm4.193171 while spectral radius is0.9. Its discrete block permits the elementary term proportional to t times0.9^(t-1), so eigenvalues alone do not bound every finite amplification. [Trefethen and colleagues' 1992 original report](https://ntrs.nasa.gov/api/citations/19930018499/downloads/19930018499.pdf) supplies related transient-growth context; its continuous-flow setting is an analogy, not a proof of this discrete example. Discrete Lyapunov and local-versus-global distinctions are also explained in [Lemmon's linear-systems notes](https://academicweb.nd.edu/~lemmon/courses/linear-systems/lecture-book/linsys-book-2024.pdf).

The archive contains214 arrays,14 trajectories,3,598 aligned retained states,3,612 native frame snapshots and154 designated summary rows. Twelve native checks and3,532 independent NumPy checks passed. The largest analytic-orbit discrepancy was1.4551915228366852e-11; the largest finite-difference Jacobian discrepancy was9.313227966600834e-11. The autograd Jacobian matched the assigned matrix exactly in this panel.

The frozen root JavaScript recount passed1,667 checks, including826 numerical comparisons. An additional producer/auditor JSON comparison passed1,668 checks, including1,155 numerical comparisons. These are corroborating checks on the same finite census, not thousands of independent scientific experiments. The [full census report](15_census_results.md) retains exact construction and limits.

## Part B — five previously trained checkpoints

### Prospective recipe and historical compatibility

The learned panel was frozen separately at2026-10-07T16:44:15+00:00, after partA was verified and before any checkpoint deserialization or learned continuation for this panel. Scientific source771975b4833bfcf20e06f848253912c699b2152e binds25 files; plan SHA-256 is ff561f7ca71ff76afe61d1593de9decbd1e4447140dd57fb84f4385a2473dd44.

Original source83fe135e301f76bc0c74e30c66bb18e067ca5959 and final archive b64d0e2ba222df76caf72bc9870c7602873e7236 supply22 fixed files totaling1,777,007bytes. They include five checkpoints, seeds40–44, and their validation predictions/configurations/run metadata. Each model has29,856 parameters,37 tokens,48 state channels, identity width16, hidden width128 and a10x8 grid.

No model was retrained. Native inference loaded each checkpoint once for compatibility and again for continuation. The separate binder deserialized each of the five original checkpoints once more, performed no inference, and compared every saved parameter/buffer snapshot with those weights. Ten native restorations and five binder loads still refer to only five distinct learned initializations.

The original study used Python3.12.8 and two numerical threads. This panel uses Python3.12.14, one thread and the same Torch2.6.0+cpu and native core source. Compatibility therefore concerns archived outputs under this runtime change. Historical hidden states were not archived, so their identity cannot be checked retrospectively.

For each checkpoint the first64 validation rows were evaluated together atT16, matching the historical batch size. All64 predictions had to be exact, with finite logits satisfying absolute error<=1e-4+1e-5*abs(reference). All five gates had to be saved before any continuation; a failed gate would have preserved all five comparisons and prohibited all extensions.

| Seed | Maximum absolute logit difference | Largest error / allowed tolerance | Changed predictions out of64 |
|---|---:|---:|---:|
| 40 | 1.907348633e-6 | 0.016137660 | 0 |
| 41 | 1.430511475e-6 | 0.011521109 | 0 |
| 42 | 1.907348633e-6 | 0.014279624 | 0 |
| 43 | 1.907348633e-6 | 0.011192651 | 0 |
| 44 | 2.384185791e-6 | 0.016986707 | 0 |

All gates passed. The320 prediction comparisons are64 reused inputs evaluated under five checkpoints, not320 independently sampled scenes. Exact gate arrays and discrepancies are preserved under native/gates in the final archive.

### State intervention and retained horizon

Indices0,1,2,3 are the four role queries for event0 of validation scene0, confirmed from the archived metadata and literal visible-input parser. They are already exposed, correlated development inputs.

Each base state is the selected T16 state from the same64-example gate forward. The second branch adds a fixed coordinate-checkerboard perturbation, computed with float64 nominal amplitude1e-4*max(1,RMS(base)) and then represented in float32. All selected initial RMS values were below1, so nominal amplitude was1e-4. Actual represented RMS differences ranged from0.0001000005598863193 to0.00010000157231957612.

A post-update1 hook installs the saved base or perturbed state. The same canvas remains available during256 subsequent ordinary native updates in evaluation mode. Diagnostic time0 is the installed T16 state; time256 corresponds to effective original depth272. Each hook call also incurs one discarded native initialization update, which is not another retained continuation transition.

The run retains40 trajectories,20 paired comparisons and10,280 aligned observed states. Each seed archive contains the entire float32 state tensor with shape[4,2,257,48,10,8], plus logits, masks, predictions, NLL, norms, increments, perturbation arrays, named parameters/buffers and RNG snapshots. Together the five gate NPZs and five extended NPZs contain490 named arrays. The five compressed extended NPZs occupy150,859,169bytes; their raw state slots alone occupy157,900,800bytes before compression.

### Learned results

The ranges below span the four base queries for each fixed checkpoint. Pair gain is final RMS separation divided by the actual initial represented RMS separation. It is measured for the single specified direction.

| Seed | Base RMS at T16 | Base RMS at effective T272 | Base RMS growth factor | Final paired gain |
|---|---:|---:|---:|---:|
| 40 | 0.5708–0.5732 | 2.4593–2.5072 | 4.293–4.392 | 3.049–4.541 |
| 41 | 0.5199–0.5240 | 9.5371–9.5891 | 18.212–18.431 | 5.964–6.059 |
| 42 | 0.5022–0.5077 | 2.5741–2.5810 | 5.070–5.136 | 4.821–5.416 |
| 43 | 0.4267–0.4361 | 2.4730–2.5009 | 5.734–5.796 | 3.463–3.647 |
| 44 | 0.5695–0.5732 | 2.5085–2.5175 | 4.392–4.416 | 1.557–1.582 |

All40 trajectories completed; there were no censored prefixes or nonfinite states. Every state-RMS maximum and all20 pair-gain maxima occurred at the final retained endpoint. This does not demonstrate monotonic growth at every step, nor does it establish what happens afterT272.

Base-state RMS growth factors were4.292553–18.430841. Final paired gain ranged1.556726–6.058958, with absolute final paired RMS distance 1.556741487e-4–6.059019566e-4. The last base-state L2 increments ranged0.572883–3.984878, so the states changed during the final observed transition. This is the adjacent difference between times255 and256, not the final-state fixed-point residual F(s256)−s256; the next transition was not measured. The last observed transition could in principle have landed at a fixed point, and a later settling process also remains possible.

As an explicitly post-run descriptive recount of the prospectively retained endpoint summaries,18 of20 base predictions changed between the first and last endpoint, and19 of20 base-label NLL values increased. The exception is seed42's patient-role query, whose NLL decreased from2.295432895 to1.549068426. At the final endpoint the base and perturbed branches predicted the same token in all20 pairs; this categorical agreement coexists with their measured state separation. These are diagnostics of the known scene, not a new accuracy comparison, independent success rate or confidence interval. They do not identify the mechanism causing earlier poor binding performance.

### Numerical and provenance checks

The unchanged hook/snapshot test passed once, with one pytest call and its setup/teardown events; no tests were skipped. All five native policy/integrity checks passed. The independent checkpoint binder passed429 checks. The separately authored NumPy auditor passed5,761 checks and recomputed every retained state-derived metric, perturbation, readout consistency allowance, gate decision, observation domain and cross-file/event linkage.

The largest float64 scalar-metric discrepancy was3.552713678800501e-15. The largest NumPy64 versus native32 readout discrepancy was6.881229531074951e-6, within the prospectively defined scale-aware bound. The largest error/bound ratio was0.00731425584. PAD logits remained under their exact rule. The audit's numerical comparison is distinct from the earlier gate tolerance against historical logits.

The source-frozen root JavaScript recount passed994 checks, including190 numerical comparisons. It verifies the JSON summaries' norm relationships, temporal arithmetic, conditional counts, complete gain profiles and peak/last values. It does not deserialize raw arrays or independently rerun the recurrence. The separate checkpoint binder verifies weight identity, while NumPy verifies saved-array calculations; these roles are attributed explicitly.

Root independently rehashed all46 text files in the final archive, with matching sizes and SHA-256 values, and verified Git modes, sizes and the exact69-file inventory including the manifest. Binary hashes are corroborated by the recorded recovery, binder, native descriptors, NumPy rereads and final manifest; root did not locally deserialize those binaries. All source and input immutability checks passed.

## Resources and archival completion

| Item | Constructed15A | Learned15B |
|---|---|---|
| Scientific source | ff66dc4bacd19219bf4f754e40944396e7ce8c5b | 771975b4833bfcf20e06f848253912c699b2152e |
| Actions run / job | 37649654847 / 112889349696 | 37654405152 / 112905734552 |
| Final results commit | ff34cc4f44c535e5f4d00276e1e3be194ec70f21 | 205507b71e2f7abb00ce97aa1dbec8321fc0af26 |
| Manifest files, excluding manifest itself | 21 | 68 |
| Archived bytes, excluding manifest itself | 9,839,284 | 164,161,748 |
| Minimum retained available RAM | 14.329380GiB | 14.066124GiB |
| Worker seconds before final archive | 17.604075 | 39.001697 |

Both standard public CPU jobs completed successfully; every final provider step reports success. Numerical threads1, interop1 and Git threads1 remained within the aggregate ceiling4 and RAM floor8GiB. PartB retained15 periodic supervisor samples and25 total selected periodic/boundary RAM observations, all above the floor. Sampling is not a continuous measurement of every transient allocation.

PartB's controller took11.086749s; native processing took8.428628s including process overhead, and its internal measured work8.159340s. The binder and NumPy processes took1.115996s and1.168242s respectively. These are measured diagnostic costs on this worker, not a complete training benchmark or a physical energy measurement.

Final learned manifest SHA-256: a7fc7460987028d3074b0985a232288c71236dbf80310d94429899ac57ebca01. Its source ZIP is9,612,092bytes, SHA-2569bca72a4c271939ef2dda8c0954ed76281ee3e53e92736a1cc2eea437525da8b. Provider completion and zero active runs were checked before releasing the numerical reservation. Local shell execution has remained unavailable since2026-10-07T09:02:53+00:00; the bounded CPU worker and immutable Git archives supplied the execution and retained evidence.

Pre-execution candidate changes, exact preimages and independent static reviews are retained. No scientific tolerance, selected query, checkpoint or stopping rule changed after outcomes. The repeated historical input files and source ZIP preserve the material needed to reproduce the panel.

## Changes and decision for this point

The native learning/update architecture was not altered for item15. New frozen measurement, recovery, binding and audit tools were added. The README's unrestricted “never returns fully to black” wording was replaced with the finite T16 activity interpretation, and its empty-cell description now distinguishes zero PAD identity/initial seeding from later recurrent activation. Raw historical results remain intact.

**Achieved in this point:** a reproducible native stability-measurement census; analytic and numerical witnesses separating output stability, state boundedness, cycles, attraction and transient gain; exact historical checkpoint recovery and parameter binding; a compatible five-checkpoint finite diagnostic with all raw states; independent metric/recount checks; and qualified reporting of successful as well as unfavorable observations.

**Still missing for the stronger claim:** a precisely scoped stability property and metric; a proof or certified bound under explicit input/parameter assumptions where one is claimed; prospective empirical coverage across relevant tasks, independent scenes, perturbation directions/amplitudes, modes and horizons; useful task competence under those same conditions; and independent external reproduction. Local Jacobians or a handful of finite trajectories cannot substitute for those conditions.

These remaining requirements are scientific gaps, not claims that unlimited simulations would settle an all-time theorem. No theorem, energy advantage, external replication or Nobel-level discovery is manufactured by closing the available investigation. The next point is16, causal scanner/null-space analysis; it begins only after this closure is recorded.
