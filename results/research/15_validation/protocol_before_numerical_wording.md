# Item 15 — prospective finite dynamical-stability census

## Question, scope and decision

Can finite answer accuracy, a static-looking visualization, or small observed updates establish bounded state, convergence, attraction, or sensitivity guarantees for the native NeuroPixel update? This item distinguishes these propositions by source analysis, already-observed historical evidence, and a complete finite census of explicitly assigned native dynamics. It is a diagnostic of measurement meaning and architecture possibilities. It is not a newly trained benchmark, a replication by an external laboratory, or evidence that historical learned checkpoints realize any constructed case.

The item opens after item14 closure d7c2e07473856ccd38591b2389babb235e8e29b5. Item15 opening is fd57fc2cff0ce471ccca5609571cc27293b8cd05. H1 remains closed and not supported. Items16–30 stay unopened until this feasible investigation is closed.

The frozen machine-readable plan fixes all scripts and this protocol by SHA-256 before its push starts a public CPU worker. No outcome-dependent horizon, tolerance, case removal, retraining or success threshold is allowed. A discrepancy requires retaining its failed archive and a separately justified correction. The independently authored auditor receives raw arrays and the plan; it does not import the native producer or Torch.

## Existing evidence and source derivation

The native fixed-input update is F_u(s)=s+W2 ReLU(Ks+W_i u+b1)+b2, where K combines the direct state and depthwise spatial-perception paths. At a differentiable activation region, J=I+W2 diag(active) K. The implementation imposes no contraction constraint on this matrix and no hard state-norm bound. Zero initialization of the final update layer produces the identity map, which is neutral and preserves initial perturbations. Training-time random firing and gradient clipping do not directly bound the evaluation state or its Jacobian.

For a fixed constant u, this is an autonomous map conditioned on u. Changing or removing u changes the map; comparing those input schedules is not the same experiment as perturbing only the initial state under a common input. A local Jacobian at one state does not certify a nonlinear map over all states or activation masks.

Previously audited rest-state JSONs record one seed per recipe and six intact accuracies at T8–64 plus three damaged accuracies at T16/32/48, with 2,000 examples per endpoint. They do not retain per-example states, norms, logits, NLL, fixed-point residuals or paired perturbations. The recipes also jointly change training depth and damage. The README figure driver retains normalized PNG/GIF visualizations, not raw trajectories. Item11's large finite NLL values remain results of a failed competence pilot, not an asymptotic divergence theorem. Activity-threshold summaries from T16 do not establish that updates tend to zero.

## Constructed cases

Use the unchanged native NeuroPixel class, evaluation mode, CPU float64, vocabulary4, identity channels4, state channels4, hidden width8, grid1x1, output cell(0,0). All parameters are explicitly assigned. The stored dictionary is I4, with the native effective PAD row0. Seed weights are I4, all perception weights and unspecified biases are zero, the first four f1 outputs copy the four state channels, and f2's first four columns are A−I4 with bias b. The only nonzero read weight is row2,column2=1. All input cells contain PAD. There is no optimization and no trained checkpoint is loaded.

On the invariant nonnegative orthant these assignments realize F(s)=As+b. Outside that orthant ReLU changes the equation; no global contraction of the complete ReLU realization is asserted.

| Case | A and b | Diagnostic purpose |
|---|---|---|
| identity | A=I4, b=0 | A fixed state with zero residual need not attract nearby states. |
| decay | A=0.5I4, b=0 | Argmax can persist while signal amplitude shrinks and token probability approaches the tie limit. |
| expansion | A=1.05I4, b=0 | Finite execution and unchanged argmax do not bound state for all time. |
| hidden_drift | A=I4, b=e3 | A read-invisible channel grows while the complete logits remain constant. |
| two_cycle | Swap channels2 and3; other diagonal entries0.5; b=0 | Constant norm does not imply convergence; subsampling can alias a cycle. |
| nonnormal_transient | Channels2,3 block [[0.9,4],[0,0.9]], other diagonal entries0.5; b=0 | Spectral radius below1 coexists with transient Euclidean amplification. |
| forced_contraction | A=0.5I4, b=e2 | Convergence to a cue encoded in fixed bias is not storage of an arbitrary earlier episode. |

Channel numbers and basis vectors are zero-based. Base initial state is e2 except forced_contraction, whose base is0. The paired initial state adds epsilon e3 with epsilon=1/16. This exact binary perturbation avoids unnecessary cancellation and is fixed before execution.

The native forward executes257 updates with trace=True. A hook at the first post-update boundary records the incoming state b and replaces it with the assigned initial state. The retained native frames therefore have shape[7,2,258,4,1,1]. Frames from native index1 onward yield aligned diagnostic times0..256, shape[7,2,257,4]. There are14 trajectories and3,598 aligned state endpoints, not3,598 independent replications. The remaining256 updates follow the unchanged native forward with a no-op hook. This is a conditional state-injection diagnostic; the PAD-only ordinary seed trajectory is not claimed to produce these initial states.

## Measurements and independent reconstruction

Retain every aligned state, native frame, input, A, b, injected state, pre/post-injection boundary, complete logits and final native output. Save all named parameters and buffers before and after each case plus RNG witnesses after literal construction. Native inference must leave them unchanged.

Compute the full4x4 one-step Jacobian at the strictly positive all-ones state using autograd, and independently estimate it by central finite differences with step2^-20 through native forward calls with state injection after update1. Save all eight finite-difference outputs per case. Report sorted eigenvalues, spectral radius, singular values and Euclidean operator norm. These matrices pertain to this deliberately linear activation region.

For every aligned endpoint save state Euclidean norm, analytic affine fixed-point residual ||As+b−s||, prediction and stable token2 negative log likelihood. Save observed adjacent-step differences only for times1..256; time0 has no observed predecessor in this aligned experiment. Pair gain at time t is ||s_perturbed(t)−s_base(t)||/(1/16). The residual at time256 is algebraic evaluation at the observed state, not an extra observed native step. Token2 NLL is a diagnostic scalar; no task accuracy is reported.

Publish154 scalar rows: seven cases x two trajectories x the11 times[0,1,2,3,4,8,16,32,64,128,256]. Include whole-horizon summaries, especially the time and magnitude of peak pair gain. Every numerical comparison uses a prospectively specified tolerance: orbit reconstruction atol1e-10 plus rtol1e-12; finite-difference Jacobian versus A atol1e-9 with rtol0; autograd Jacobian versus A atol1e-12 with rtol0. Recomputed scalar metrics use the same atol1e-10 plus rtol1e-12 as orbit reconstruction. Exact integer, shape, label, input and parameter-preservation checks remain exact. Metric audits derived from raw float64 arrays use explicitly documented bounded numerical tolerances; they cannot be adjusted after seeing results.

The independent NumPy process reconstructs affine orbits from the literal recipe without importing producer functions, checks all raw values and native boundaries, and recalculates all summaries. A separately frozen elementary JavaScript recount checks the JSON scalar report using independent arithmetic; it cannot read the binary NPZ and is not described as doing so. Source SHA bindings, array inventory, dimensions and file checksums make those scopes explicit.

Reuse one relevant existing contract, tests/test_repair_trace.py::RepairTraceContracts::test_post_update_boundary_and_snapshot_ownership. This verifies native hook timing and independent snapshot ownership. It is an implementation gate, not an additional learned result. No new tests merely duplicate the case recipe.

## Analytic expectations and interpretation

The affine identity preserves every initial difference. Scalar decay and forced contraction reduce differences by0.5 per step on this invariant region; the latter tends to2e2. Scalar expansion scales nonzero states by1.05 per step. Hidden drift adds t e3 while the readout observes only channel2. The swap yields a two-cycle. The nonnormal block has powers with off-diagonal term4t(0.9)^(t−1): it can first amplify and ultimately decay. These are deductions from explicit assigned matrices, to be checked against the native implementation. Finite experiments alone do not establish infinite-time claims for unconstrained learned models.

A suitable contraction or Lyapunov certificate over a defined invariant region would give a stronger claim than finite trajectories. A different metric may certify contraction even if Euclidean norm increases temporarily. Strict contraction under a common future input erases dependence on sufficiently distant initial states; preserving useful episodic content therefore requires a clearly specified memory objective and suitable state structure. This observation does not rule out all stable memory systems.

## Resource and completion policy

One standard public Ubuntu24.04 CPU worker, Python3.12.14 and the same pinned dependencies as item14; numerical threads1, interop1, Git1, aggregate active CPU ceiling4. Available RAM must remain at least8GiB. Worker limit900s includes180s reserved for final archive; contract stage120s; probe540s with native and audit components120s each. Workflow limit20min, finite bootstrap deadline1170s, owned process-group supervision every1s, heartbeat60s and partial archives at most300s apart during long stages. Max absolute native state1e12, all values finite, horizon256; violation stops rather than silently clips.

Retain source ZIP, plan, runtime records, raw NPZ, reports, tests, logs, failures and final Git archive manifest. Root verifies provider completion, checksums and numerical audit before release. A successful measurement census closes this feasible item investigation; a learned stability guarantee remains unestablished unless evidence actually proves it. No paid compute, GPU request, main-branch merge, external contact or nomination is part of this item.

## Primary background

- Miller and Hardt, Stable Recurrent Models, ICLR2019, sections2–3: https://arxiv.org/html/1805.10369v4 . Their definition is uniform contraction, stronger than generic Lyapunov stability; their finite-context result is used only under its assumptions.
- Revay and Manchester, Contracting Implicit Recurrent Neural Networks,2020, section1.1 and Theorem1: https://arxiv.org/pdf/1912.10402 . The contraction metric condition explains why a Euclidean singular-value bound is sufficient but can be restrictive. The architecture's weights do not inherit their certificate.
