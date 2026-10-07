# Item 15A — finite native dynamical census

## Decision and scope

The constructed measurement census completed and its independent recount found no discrepancies outside the frozen numerical tolerances. The unchanged native NeuroPixel class realizes all seven assigned dynamics on the nonnegative invariant region examined. Constant predictions, constant logits, small residuals, bounded state and attraction are demonstrably different properties.

This completes the constructed-census component of item15. Item15 remains open for a separately frozen retrospective analysis of five available learned checkpoints. No learned checkpoint was loaded and no optimizer step occurred in this census. Items16–30 remain unopened; the historical H1 result stays closed and not supported.

## Immutable evidence

- Scientific source: [ff66dc4bacd19219bf4f754e40944396e7ce8c5b](https://github.com/Agnuxo1/NeuroPixel/tree/ff66dc4bacd19219bf4f754e40944396e7ce8c5b).
- Frozen protocol: [15_probe_protocol.md](https://github.com/Agnuxo1/NeuroPixel/blob/ff66dc4bacd19219bf4f754e40944396e7ce8c5b/docs/research/15_probe_protocol.md).
- Plan SHA-256: `a39782409d87a172ca516a461e4c31a92ebd2ed2828477c02dfb9e38b060460d`, frozen2026-10-07T16:07:56+00:00, before execution.
- [Actions run37649654847](https://github.com/Agnuxo1/NeuroPixel/actions/runs/37649654847), job112889349696; provider reports every step completed successfully.
- Final archive: [ff34cc4f44c535e5f4d00276e1e3be194ec70f21](https://github.com/Agnuxo1/NeuroPixel/tree/ff34cc4f44c535e5f4d00276e1e3be194ec70f21/results/research/15_cloud_runs/37649654847-1-probe).
- [Native report](https://github.com/Agnuxo1/NeuroPixel/blob/ff34cc4f44c535e5f4d00276e1e3be194ec70f21/results/research/15_cloud_runs/37649654847-1-probe/native/report.json),78,996bytes, SHA-256 `752323fa052e37d3fa29800135f90eefffdc6ffcd2554f29d51a7d590c9a9748`.
- [Independent NumPy audit](https://github.com/Agnuxo1/NeuroPixel/blob/ff34cc4f44c535e5f4d00276e1e3be194ec70f21/results/research/15_cloud_runs/37649654847-1-probe/audit/audit.json),60,219bytes, SHA-256 `fac66bf07186ef64caa620e2b0ff58fb538d83ac3725493b11dff61236d1471d`.

The final manifest contains21 files totaling9,839,284bytes, excluding the manifest itself. Manifest:3,393bytes, SHA-256 `c8088103801480ab3d951566426ff8a0c1f0be52d1311e185a5d0b25086bf75e`. Full raw traces are187,153bytes, SHA-256 `97ee04229df6d89232caa3dd3ed8ee47cf3877f6310cfb8e5275186292da3c47`. Source ZIP:9,490,669bytes, SHA-256 `14fe051cecf44b5f5f1fc71ad7e5f8799b76cb78487b460c282a9df75cb36cf4`.

## What was executed

Every parameter of a native four-channel,1x1-grid model was explicitly assigned. In the visited nonnegative orthant the native ReLU update equals the declared affine map As+b. A post-update1 hook injects the specified base or perturbed initial state; the next256 updates use the ordinary native recurrence and common all-PAD input. The actual native forward has257 updates and258 snapshots. The aligned diagnostic series starts after the intervention, at time0.

Seven rules and two initial states produce14 trajectories,3,598 aligned state endpoints and3,612 native snapshots. These are correlated deterministic observations, not3,598 experimental replications. The scalar report contains154 selected rows across11 times and seven whole-horizon summaries. The214 arrays include every state, input, matrix, boundary, logit, local Jacobian, parameter/buffer and RNG witness required by the plan.

The readout observes only channel2. Token2 NLL and predicted token are diagnostic quantities under that deliberately restricted readout; there is no learned task-accuracy assay. The perturbed initial state adds1/16 in channel3. Jacobians are measured at positive ones by native autograd and by native central differences with step2^-20.

## Results

| Assigned rule | Main observed result | What it establishes in this construction |
|---|---|---|
| Identity | Pair gain1 and residual0 at every saved time. Base norm1. | A fixed state can preserve every nearby initial difference. Residual0 does not establish attraction. |
| Decay0.5I | Base norm and pair gain at256 are8.636168555094445e-78. Predicted token remains2; NLL approaches log3. | A categorical answer can persist while its amplitude and confidence vanish. |
| Expansion1.05I | Base norm and pair gain reach265742.2219223645 at256, with token2 still predicted. | Finite numerical execution and constant argmax do not imply a state bound valid for all time. |
| Hidden drift | Base norm grows from1 to256.0019531175495. All logits, token2 NLL0.5514447139320511 and prediction2 remain unchanged. | The readout can entirely miss growth of a hidden coordinate. |
| Two-cycle | Base norm remains1; state alternates between channels2 and3. Predicted token alternates2/1. | A bounded orbit need not converge; sampling only even times would conceal the cycle. |
| Nonnormal transient | Spectral radius0.9, Euclidean operator norm4.193171219946131. Pair gain peaks at15.501661559668227 at time9, then falls to2.1985686523109228e-9 at256. | Eigenvalues inside the unit circle permit substantial transient Euclidean amplification. |
| Forced contraction | Base state reaches the represented equilibrium2e2; final base residual0 and final norm2. Pair gain decays to8.636168555094445e-78. | A stable endpoint can encode a fixed bias supplied by the designer, without storing arbitrary episodic information. |

The algebraic rule for the nonnormal block is A=[[0.9,4],[0,0.9]]. Its off-diagonal power term is4t(0.9)^(t−1). Its eventual decay follows from this explicit matrix formula; the numerical run checks only times through256. Similarly, infinite-time properties of scalar multiplication or drift follow from the assigned formulas, not from extrapolating a finite plot.

All analytical statements here concern the region where the specified affine equation holds. Negative ReLU activation regions can implement different dynamics. This is not a global contraction certificate for the full nonlinear realization, much less for every learned model.

## Verification and errors

One existing contract passed: post-update intervention timing and snapshot ownership. Its setup, call and teardown are three records of one test. All12 native checks passed. There were no test skips, failed native checks or aborted scientific runs.

The separately authored NumPy program ran in a fresh process without Torch or producer imports. It checked214 arrays, reconstructed all affine orbits, verified weights and RNG preservation, recomputed logits, NLL, norms, residuals, perturbation gains and spectra, and checked every reported row and summary. Its3,532 checks reported zero issues.

The largest full-array orbit difference was1.4551915228366852e-11; the same maximum appeared when comparing observed increments with the preceding algebraic residual. The maximum local finite-difference Jacobian discrepancy was9.313227966600834e-11. Autograd matched the literal Jacobians exactly in the recorded float64 values. The largest NLL recount discrepancy was2.220446049250313e-16. All are below their prospective tolerances: orbit and scalar metrics atol1e-10 plus rtol1e-12; finite differences atol1e-9,rtol0; autograd atol1e-12,rtol0.

A separately frozen elementary JavaScript program checked the JSON report with1,667 checks, including826 numerical comparisons, with no discrepancy; its maximum absolute difference was1.1368683772161603e-13. A separate producer-versus-auditor JSON comparison made1,668 checks with1,155 numerical fields and no discrepancy. The JavaScript did not deserialize the NPZ. Root independently rehashed all19 text files and checked all archive file sizes/modes against the Git tree. The NumPy process independently verified the binary trace hash.

Before any execution, static review found that an early producer lacked the promised autograd calculation. That exact preimage was retained, the missing calculation and witness were added, and the corrected source was frozen. No observed scientific outcome was used to change a case, threshold or tolerance. Temporary read timeouts during later archive retrieval did not alter the completed run; subsequent reads and checksums succeeded.

## Resources and scientific limitations

The worker completed in17.604074884s before its final archive. The native subprocess used2.02035136s, including startup; its internal timer recorded1.600717849s. The independent audit subprocess used0.265610006s, with0.179984106s internally. These are diagnostic execution times on this runner, not model-efficiency or physical-energy benchmarks.

Seven supervisor samples and15 total retained RAM observations had minimum available RAM14.32938003540039GiB. This is an observed available-memory minimum, not peak RSS or continuous physical measurement. Numerical, interop and Git threads were1; the aggregate ceiling was4 and the RAM floor8GiB. Deadlines, owned-process supervision and archival reserve were retained. Provider completion and zero active runs were verified before numerical-resource release.

There are no learned weights, external replications, population confidence intervals or representative scene samples in this census. A first-order derivative at one point is not a guarantee over changing activation masks. Directional pair gain is not the worst-case gain. Constant input defines one particular map; removing it would define another. A maximum-absolute-state stop of1e12 is an execution guard, not an architectural theorem.

The historical rest-state JSONs contain finite accuracies at T8–64 but lack raw trajectories, norms, probabilities and perturbation pairs. A visual frame that appears unchanged cannot fill that evidence gap. The additional learned-checkpoint component will remain a descriptive finite diagnostic, separately frozen, using existing exposed validation data; it will not reopen H1 or establish useful memory from weak task acquisition.

## Scientific background

Miller and Hardt, [Stable Recurrent Models](https://arxiv.org/html/1805.10369v4), sections2–3, define a uniform contraction notion stronger than generic Lyapunov stability and derive a finite-context approximation under their assumptions. Revay and Manchester, [Contracting Implicit Recurrent Neural Networks](https://arxiv.org/pdf/1912.10402), section1.1 and Theorem1, show how a suitable metric can certify contraction when a Euclidean bound is restrictive. Those constructions do not automatically certify NeuroPixel's unconstrained update.

Pascanu et al., [On the Difficulty of Training Recurrent Neural Networks](https://proceedings.mlr.press/v28/pascanu13.pdf), equation5 and section3.2, distinguish temporal sensitivity and gradient clipping. Trefethen et al.'s [1992 NASA/ICASE report](https://ntrs.nasa.gov/api/citations/19930018499/downloads/19930018499.pdf), section on transient energy growth, supplies a primary continuous-system example of nonnormal amplification; the discrete formula used here was derived separately.
