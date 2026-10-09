# Item 1 — Mechanism specification and contribution boundaries

**Status:** completed specification of the audited implementation; a distinct scientific contribution is **not established** by this specification or the archived experiments.

**Source snapshot:** `Agnuxo1/NeuroPixel`, commit `7da18d1cbd36c40c48228118abb20a7b5ab032f4`.  
**Specification date:** 2026-10-06.  
**Research programme:** item 1, “Delimit what NeuroPixel contributes relative to prior work.”

This document states what the code computes, derives its local information-propagation bound, and separates implementation facts from archived observations and unestablished claims. It specifies the symbolic NeuroPixel core and its directly connected vision and phase-3 extensions. It does not describe every Kaggle-specific variant or a later local snapshot. No training, model change, or implementation of a subsequent programme item was performed to produce this document.

The mathematical description identifies a recurrent, spatially local neural network with a learned input dictionary, a factorized tied decoder, and optional auxiliary supervision. Establishing whether their particular combination contributes something beyond existing alternatives requires the matched references, replication, and ablations in programme items **4–6**. A negative result on novelty is an admissible conclusion of item 1.

## 1. Objects, dimensions, and the meaning of “pixel”

Let the grid be

\[
\Omega=\{0,\ldots,H-1\}\times\{0,\ldots,W-1\},
\]

and let \(X_{b,p}\in\{0,\ldots,V-1\}\) be the token at position \(p\) in batch example \(b\). Token 0 denotes PAD/empty. The default role vocabulary has 35 entries: PAD, four role labels, twelve nouns, ten verbs, and eight places. These are separate token categories in the generator. [TASK]

| Object | Symbol | Default shape or dimension | Code |
|---|---|---|---|
| Learned embedding table | \(E\) | \(V\times d\), \(d=16\) | `embed.weight` |
| Effective input/output dictionary | \(D\) | \(V\times d\) | `dictionary()` |
| Fixed identity input during a rollout | \(z_{b,p}\) | \(d=16\) real channels per cell | `ids` |
| Recurrent state | \(s^t_{b,p}\) | \(C=48\) real channels per cell | `s` |
| Hidden update width | \(M\) | 128 | `hidden` |
| Number of recurrent updates | \(T\) | 16 | `steps` |
| Training update probability | \(\rho\) | 0.5 | `fire_rate` |
| Readout position | \(p_*\) | One grid coordinate | `out_pos` |

The dimensions are configurable; the defaults above describe `NeuroPixel` itself. The implementation uses ordinary PyTorch tensors, convolutional layers, a linear layer, and automatic differentiation. The word “colour” is a description of a latent vector. It does not restrict the state to three RGB channels, unsigned bytes, or physical light. [MODEL]

The separate RGB24 codec maps an integer \(n\in[0,2^{24}-1]\) to

\[
R=(n\mathbin{\gg}16)\mathbin{\&}255,\quad
G=(n\mathbin{\gg}8)\mathbin{\&}255,\quad
B=n\mathbin{\&}255,
\]

with inverse \(n=(R\ll16)\,|\,(G\ll8)\,|\,B\). PNG preserves these bytes. This reversible identifier encoding is not called by the neural forward pass and does not supply a semantic representation or a semantic compression result. [CODEC]

## 2. Effective dictionary and input initialization

### 2.1 Optional perceptual anchors

For token \(v\), let \(a_v\in\{0,1\}\) be its fixed grounding mask and \(g_v\in\mathbb R^3\) its supplied colour. Using channel indices starting at zero, the effective dictionary is

\[
D_v=
\left[
a_v g_v+(1-a_v)E_{v,0:3}\ ;\ E_{v,3:d}
\right].
\]

Without grounding, \(a_v=0\) for every token and \(D=E\). Grounding fixes only three channels of selected tokens; it does not fix their other channels. The supplied RGB values and masks are registered with `persistent=False`, so reproducing a grounded model requires reconstructing these inputs as well as loading its `state_dict`. [MODEL]

### 2.2 Symbolic and camera inputs

Without a camera, \(z_{b,p}=D_{X_{b,p}}\). With camera mask \(c_{b,p}\), the implementation replaces the identity at camera cells:

\[
z_{b,p}=
\begin{cases}
D_{X_{b,p}}, & c_{b,p}=0,\\
[I_{b,p};0_{d-3}], & c_{b,p}=1\text{ and no retina},\\
\mathcal R_\phi(I_b)_p, & c_{b,p}=1\text{ and retina enabled}.
\end{cases}
\]

Here \(I\) is the supplied three-channel image. The task loaders scale camera values to \([-1,1]\); the core forward method does not itself normalize them. The retina \(\mathcal R_\phi\) is four 3×3 convolution/ReLU stages with dilations 1, 2, 4, and 8, followed by a 1×1 projection into \(d\) channels. Its maximum spatial receptive-field radius is \(1+2+4+8=15\), giving a 31×31 bounding window. [MODEL] [CARTILLA]

Define the presence indicator

\[
m_{b,p}=\mathbf 1\{X_{b,p}\ne0\}\lor c_{b,p},
\]

where the camera term is absent when no camera is supplied. The initial state is

\[
s^0_{b,p}=m_{b,p}(A z_{b,p}+a),
\qquad A\in\mathbb R^{C\times d}.
\]

The presence mask is applied only to initialization. An initially empty cell can receive information and become active in later updates. Each ordinary `forward` call creates a new \(s^0\); persistence across separate frames is implemented by the different streaming function described in section 7. [MODEL]

### 2.3 Empty is an intended convention, not a trained invariant

`nn.Embedding(..., padding_idx=0)` initializes the PAD row to zero. The actual lookup subsequently uses `F.embedding(canvas, dictionary())` without forwarding `padding_idx=0`; `dictionary()` also does not project its PAD row back to zero. The usual module lookup protection is therefore absent from this path. The loss can reach the dictionary through input lookup and, for the auxiliary lens, through its output rows. The code does not establish that \(D_0\) stays zero during learning. [MODEL] [TRAIN]

Even if \(D_0=0\) were enforced, a learned nonzero update bias or a neighbouring state can activate an empty cell. The unit test checks that the **initial** state of an empty canvas is zero, not that the state remains zero after learning or throughout a rollout. This issue is recorded here as an implementation boundary; correction belongs to programme item 8. [TESTS]

## 3. Exact local update

Use zero state values outside the finite grid, matching convolution padding. For each state channel \(j\), the perception layer has two learned 3×3 filters and no bias:

\[
(P_\theta s)_{b,p,j,k}
=\sum_{\delta\in\{-1,0,1\}^2}
K_{j,k,\delta}s_{b,p+\delta,j},
\qquad k\in\{1,2\}.
\]

This is a depthwise convolution with output width \(2C\). It sees the centre as well as its eight neighbours. Cross-channel mixing occurs in the following pointwise network:

\[
h^t_{b,p}=[s^{t-1}_{b,p};(P_\theta s^{t-1})_{b,p};z_{b,p}],
\]

\[
u^t_{b,p}=W_2\operatorname{ReLU}(W_1h^t_{b,p}+b_1)+b_2,
\]

where \(W_1\in\mathbb R^{M\times(3C+d)}\) and \(W_2\in\mathbb R^{C\times M}\). The same \(P_\theta,W_1,W_2,b_1,b_2\) are used at every cell and every step. Both \(W_2\) and \(b_2\) start at zero. [MODEL]

The stochastic update mask is one scalar per example, position, and step, broadcast over all \(C\) channels:

\[
f^t_{b,p}\sim\operatorname{Bernoulli}(\rho)
\quad\text{if training and }\rho<1;
\qquad f^t_{b,p}=1\text{ otherwise}.
\]

The code does not divide the retained updates by \(\rho\). Define

\[
\Delta s^t_{b,p}=f^t_{b,p}u^t_{b,p},\qquad
\widetilde s^t_{b,p}=s^{t-1}_{b,p}+\Delta s^t_{b,p},
\]

\[
s^t=\mathcal H_t(\widetilde s^t),\qquad t=1,\ldots,T.
\]

With no hook, \(\mathcal H_t\) is the identity. The phase-3 damage hooks multiply whole cell states by a binary mask at one specified step. The API accepts arbitrary Python hooks, so locality claims require restrictions on the hook, stated in section 6. [MODEL] [PHASE3-TRAIN]

The identity field \(z\) is supplied locally again at **every** update. Deleting \(s^t\) does not delete \(z\). This is a recurrent network conditioned on a persistent input field; state deletion while that field remains available measures input-assisted recovery.

The implementation computes \(P_\theta s\) and the pointwise update densely before multiplying by \(f^t\). It has no active-cell scheduler, event-driven kernel, or early stopping criterion. Small or masked updates do not automatically avoid their preceding arithmetic. The positive integer \(T\) is chosen externally. In the current API, `steps=0` falls back to `self.steps` because the loop uses `steps or self.steps`. [MODEL]

## 4. Readout and scanner

Let \(R\in\mathbb R^{d\times C}\) and \(r\in\mathbb R^d\) be the linear readout parameters. The unmasked token scores at a cell are

\[
\ell_v(s_{b,p})=D_v^\top(Rs_{b,p}+r).
\]

The final answer is read at \(p_*\), with the PAD score overwritten by the finite constant \(-10^4\):

\[
L_{b,v}=\begin{cases}-10^4,&v=0,\\
\ell_v(s^T_{b,p_*}),&v\ne0.
\end{cases}
\qquad
\widehat y_b=\arg\max_v L_{b,v}.
\]

The returned `logits` are scores; cross-entropy and scanner code perform their own softmax operations. `lens_logits` applies \(\ell\) at every cell and does **not** suppress PAD. The visualization helper suppresses PAD itself; auxiliary training and the growth scores use the unsuppressed lens logits. [MODEL] [SCANNER] [GROW]

Two consequences are useful for contribution analysis:

1. The effective decoder matrix is \(DR\), so its rank is at most \(d\). Tying this decoder to the input dictionary and giving it a low-rank factorization are distinct design decisions. An ablation that replaces it with a full, independent \(V\times C\) head changes both decisions.
2. The scanner is not an invertible representation of the entire recurrent state. With the default \(C=48,d=16\), \(R\) has a nullspace of dimension at least 32. A state perturbation in that nullspace leaves all immediate lens logits unchanged, while later update layers may use the changed state. Therefore readable lens outputs alone cannot establish a complete causal explanation.

There is an important time-index detail. When `lens_every=k`, the loop collects the lens **before** the update numbered \(t\), whenever \(t\bmod k=0\). It therefore collects

\[
\ell(s^{k-1}),\ell(s^{2k-1}),\ldots,
\]

not \(\ell(s^k),\ell(s^{2k}),\ldots\). For \(T=16,k=4\), the supervised states are \(s^3,s^7,s^{11},s^{15}\). Trace frames contain detached states \(s^0,\ldots,s^T\), after any hook at each update. The code has no detached recurrent state in the training computation itself. [MODEL]

## 5. Training objective and gradient path

For a batch of \(B\) examples, define answer loss

\[
\mathcal L_{\rm answer}=\frac1B\sum_b\operatorname{CE}(L_b,y_b).
\]

Let \(\mathcal A=\{t:1\le t\le T,\ t\bmod k=0\}\) and
\(\mathcal P=\{(b,p):X_{b,p}\ne0\}\). With the auxiliary lens enabled and \(\mathcal A\) nonempty, the school loss is

\[
\mathcal L_{\rm school}=
\frac{1}{|\mathcal A||\mathcal P|}
\sum_{t\in\mathcal A}\sum_{(b,p)\in\mathcal P}
\operatorname{CE}(\ell(s^{t-1}_{b,p}),X_{b,p}).
\]

The query token is included because it is nonempty. A camera cell with token ID zero is excluded from this symbolic mask even though the camera presence mask activates it. The output cell, initially empty in `RoleTask`, is excluded. [TRAIN] [TASK]

The activity term returned by the model is

\[
\mathcal A_{\rm update}=
\frac{1}{TBCHW}\sum_{t,b,p,j}|\Delta s^t_{b,p,j}|.
\]

This uses the masked update before the hook; a hook's erasure of state is not counted in this quantity. The symbolic training objective is exactly

\[
\mathcal L=\mathcal L_{\rm answer}
+\lambda_s\mathcal L_{\rm school}
+\lambda_a\mathcal A_{\rm update},
\]

with optional terms omitted when disabled. Gradients of this finite, unrolled computation are obtained by backpropagation through time. The random binary masks are sampled values for this differentiation; the code does not differentiate their sampling probabilities. A global gradient-norm clip with maximum norm 1 precedes an AdamW optimizer step with weight decay \(10^{-4}\). A OneCycleLR scheduler changes the learning rate after each optimizer step; its other settings, including whether it also cycles momentum, inherit PyTorch defaults unless specified. [TRAIN] [PHASE3-TRAIN]

The update procedure can be written without assuming unstated optimizer defaults as

\[
g_n=\nabla_\theta\mathcal L_n,\quad
\bar g_n=\operatorname{clip\_grad\_norm}_{2,1}(g_n),\quad
(\theta_{n+1},o_{n+1})=
\operatorname{AdamW}_{\eta_n,\,10^{-4}}(\theta_n,o_n,\bar g_n),
\]

where \(o_n\) denotes optimizer state and the exact optimizer/scheduler defaults belong to the recorded PyTorch version. The code calls `clip_grad_norm_(..., 1.0)`, `opt.step()`, then `sched.step()`. Weight decay is decoupled AdamW decay; it is not an additional L2 term in the written task loss. [TRAIN]

The entry points use different defaults:

| Entry point | Relevant configuration |
|---|---|
| `scripts/train.py` | Batch 128, learning rate 0.002, 2,000 iterations, school weight 0, activity weight 0; OneCycleLR `pct_start=0.1`. |
| `scripts/phase3.py:train` | Batch 512, learning rate 0.002, school weight 0.3 for NeuroPixel by default, activity weight 0; iteration count supplied by caller. |
| Rest-state variant | \(T\) uniformly sampled from integers 12 through 32. With probability 0.5, one post-update cell mask is applied at an update sampled from 2 through \(T-1\); each cell is erased with probability 0.3. |
| `scripts/train_cartilla.py` | Adds a final-image lens loss, optionally adds the symbolic role loss and its school loss, and uses OneCycleLR `pct_start=0.05`; it is a separate multitask objective. |

In cartilla training, the image lens term is

\[
\mathcal L_{\rm image\ lens}=
\frac{1}{B\,32\,32}\sum_b\sum_{p\in\{0,\ldots,31\}^2}
\operatorname{CE}(\ell(s^T_{b,p}),w_b),
\]

where \(w_b\) is the image's global class word, repeated at every image pixel. This is not a per-pixel semantic segmentation target. The full NeuroPixel cartilla loss sums image answer loss, its weighted image lens loss, and, when enabled, a separate batch's symbolic answer and weighted school losses. [CARTILLA-TRAIN]

Local inference does not imply a biologically local learning rule: the optimizer uses gradients from these supervised objectives through the unrolled network. No alternative to backpropagation is implemented here.

## 6. Local propagation theorem and invariance boundaries

### 6.1 Fixed-parameter propagation bound

Define Chebyshev distance

\[
d_\infty(p,q)=\max(|p_1-q_1|,|p_2-q_2|),\qquad
B_t(p)=\{q\in\Omega:d_\infty(p,q)\le t\}.
\]

**Proposition.** Fix the model parameters, grid, positive update count \(T\), readout position, and update-mask realization. Use no hook, or use hooks that are pointwise functions of the current cell state and fixed local masks, with no additional input-dependent nonlocal information. For symbolic input or direct per-pixel RGB input, \(s^t_p\) depends on the input only within \(B_t(p)\). Consequently the answer at \(p_*\) is independent of a change confined to \(q\) if \(d_\infty(p_*,q)>T\).

**Proof.** At \(t=0\), the seed and presence indicator at \(p\) depend only on the local token and, where applicable, local RGB and camera mask. Thus the assertion holds with \(B_0(p)\). Suppose it holds at \(t-1\). The depthwise perception at \(p\) reads only states at \(r\) with \(d_\infty(p,r)\le1\). By induction, each such state depends only on \(B_{t-1}(r)\), whose union is contained in \(B_t(p)\) by the triangle inequality. The other arguments of the update are the old local state and the locally reinjected identity \(z_p\), which add no input outside \(B_t(p)\). Pointwise channel mixing, the fixed update mask, residual addition, and the permitted hook add no spatial dependency. This proves the induction. The decoder reads only \(s^T_{p_*}\) with fixed parameters, so it does not widen the dependency set. ∎

An equivalent statement is that two inputs agreeing throughout \(B_T(p_*)\) yield the same answer logits under the same masks and permitted hooks. With independent input-independent sampling, the result also holds for the conditional distribution of the answer. It does not assert equality between two particular runs using different random masks.

Repeated local reinjection does **not** bypass the bound. A change supplied only to the identity input at update \(t\), with earlier state held fixed, can first change that cell's \(s^t\) and can travel at most \(T-t\) further cells before the final readout. An input token available to the initial seed has at most \(T\) propagation steps. This distinction matters when interpreting recovery with the original input still present.

**Retina extension.** The retina gives each identity vector a raw-image receptive-field radius at most 15. Replacing the base case in the proof by \(B_{15}(p)\) gives a raw-RGB dependency bound of \(B_{T+15}(p)\). The symbolic token path and a change to the local camera mask remain subject to their corresponding local paths. The bound is on potential dependency; a particular set of weights can use a smaller region. [MODEL]

### 6.2 What the proposition does and does not establish

| Property | Consequence and qualification |
|---|---|
| Minimum communication time | For a pure symbolic core to use an independently varying fact at distance \(d\) from the output, \(T\ge d\) is necessary. It is not sufficient for successful binding or for learning it. |
| Conditioning on parameters | Learned weights and the dictionary are held fixed. Across training runs, globally aggregated gradients can change weights everywhere; this theorem is about one forward rollout. |
| Input removal | Erasing state while retaining the input permits a fresh local computation from that input. It is a different condition from reconstructing an unavailable fact from surviving state alone. |
| Arbitrary hooks or control flow | A hook computing a global statistic, or an externally input-dependent choice of steps/readout, can create an additional information path and is outside the proposition. |
| Scanner-based routing | Selecting among experts from scores averaged over all input cells is a global operation outside the single-core forward theorem. |
| Stability | The dependency bound provides no contraction, bounded-state, convergence, oscillation, or infinite-horizon guarantee. Residual updates and biases are unconstrained. |

### 6.3 Equivariance and its limits

The shared convolutional and pointwise rule is translation equivariant on an infinite grid, or when the domain, boundary treatment, inputs, masks, and readout are translated consistently. On a fixed finite rectangle, zero padding can reveal boundary position. The corresponding equality is safe in interior regions whose dependency cones avoid the boundary. A fixed output coordinate is not an invariant task-level readout under arbitrary input translation. Training masks must be translated with the input for pathwise equality; independent identically distributed masks give the corresponding distributional statement. [MODEL]

The learned 3×3 kernels are not constrained to be rotation- or reflection-symmetric. Neither geometric symmetry, arbitrary token permutation for fixed weights, nor scaling invariance is guaranteed. Jointly relabelling token IDs, dictionary rows, grounding data, targets, and PAD consistently is a reparameterization, not evidence that a learned model understands unseen labels. Accepting different grid sizes because convolutional parameters are shared also does not guarantee generalization to those sizes or sufficient propagation time.

## 7. Connected extensions and their exact boundaries

**Streaming state.** For frame \(f\), `np_stream` forms a new local seed \(n^f_p=m^f_p(Az^f_p+a)\). The first frame starts with \(s^{f,0}=n^f\); subsequent frames start with \(s^{f,0}=s^{f-1,T_{f-1}}+n^f\). It then runs the same local update with that frame's \(z^f\). The previous identity field is replaced, so retention of an earlier fact requires the state to carry it. This helper does not include the retina, arbitrary hooks, or lens supervision from `forward`. Its blank frame still follows the implementation's current PAD convention. [STREAM]

**New dictionary entries.** `expand_vocab` copies the network, appends embedding rows, freezes the other parameters, and masks the gradients of old rows. The published adaptation path fits the new row with Adam. It uses an ungrounded model; the expansion helper resets the grounding buffers, so it would not preserve an existing grounded dictionary merely by preserving its learned embedding rows. The probe always asks for the role where that new token was inserted; it does not yet establish correct handling when the token is a distractor. This mechanism is an operational extension of the dictionary, not a demonstrated acquisition of a new semantic concept. [NEWWORD] [NEWWORD-PROBE]

**Growth and selection.** For model \(m\), input-cell probabilities are \(p_m(v\mid b,p)=\operatorname{softmax}(\ell_m(s^T_{b,p}))_v\). The routing score for example \(b\) is

\[
S_m(X_b)=\frac{\sum_{p:X_{b,p}\ne0}p_m(X_{b,p}\mid b,p)}{\#\{p:X_{b,p}\ne0\}}.
\]

The novelty score over a probe batch is

\[
N_m(X)=\frac{\sum_{b,p:X_{b,p}\ne0}\mathbf1\{p_m(X_{b,p}\mid b,p)<0.5\}}{\#\{(b,p):X_{b,p}\ne0\}}.
\]

If no model exists or every model has \(N_m>0.1\), `t_grow2` creates a complete model, copying the least-novel model if one exists, and trains it on the current topic. Otherwise it trains the least-novel existing model briefly. At inference it chooses \(\arg\max_m S_m(X_b)\). All experts are evaluated for their scores; the current script runs them again to obtain predictions. The implementation grows whole networks, not the spatial size of one recurrent state. [GROW]

The source comment calls this “ART-like.” The defined threshold rule alone establishes neither mathematical equivalence to Adaptive Resonance Theory nor a novelty claim relative to it. Such a claim needs an explicit correspondence of state variables, category-learning rules, reset/selection dynamics, and assumptions, or an appropriately controlled empirical comparison. This document assigns neither equivalence nor historical originality from the analogy.

**HRR.** The separate `hrr.py` module implements circular-convolution binding and approximate unbinding and has its own unit test. It is not invoked by the symbolic core or the training paths specified above, so it cannot explain their measured performance. Circular convolution in this implementation is commutative; distinct assignments can nevertheless be represented by summing products with different role vectors. [HRR] [TESTS]

## 8. Claim ledger

The status labels describe evidence, not a progression of marketing claims:

- **Implemented:** a concrete code path computes the stated operation.
- **Observed:** an archived numerical artifact reports the stated result. These are historical records, not new independent replications performed for this document.
- **Not established:** the audited code and available records do not establish the stronger statement.

| ID | Statement | Status | Evidence and boundary |
|---|---|---|---|
| M01 | A trainable recurrent rule updates a grid through local convolutions and pointwise channel mixing. | Implemented | Equations in sections 2–5; conventional backpropagation and AdamW. [MODEL] [TRAIN] |
| M02 | The same effective dictionary writes inputs and participates in the output head. | Implemented | A tied, factorized decoder \(DR\), not a reversible semantic RGB code. [MODEL] |
| M03 | The symbolic core reads distant cells instantly or without enough update steps. | Not established | Excluded by the propagation proposition under its assumptions. |
| M04 | One archived 235,776-parameter run performs well on held-out role combinations. | Observed | `scale_np230k.json` records 0.9965 accuracy. It is one configuration on the same synthetic task family. [SCALE-RESULT] |
| M05 | The result establishes a general architectural limitation of Transformers. | Not established | The scaling script gives NeuroPixel school supervision and the Transformer no school loss; matching iterations does not match all supervision or computation. [SCALE-CODE] |
| M06 | The far task demonstrates successful difficult agent/patient binding. | Not established | The best cited record gives overall 0.743, agent 0.4522, patient 0.4622, verb and place 1.0. The elementary category control has expected accuracy 0.75 by the generator's structure; this is a deduction, not a control run in this item. [TASK] [FAR] [FAR-RESULT] |
| M07 | A model recovers after an independently sampled 50%-erasure cell mask with input retained. | Observed | The rest-state record gives 0.9935 at 16 steps after a step-8 mask. The mask has expected erasure fraction 0.5, not an exactly fixed cell count. [STABLE-RESULT] [PHASE3-TRAIN] |
| M08 | The state is stable at arbitrary horizons or repairs without input support. | Not established | The archived rest-state accuracy is 0.992 at 8 steps and 0.958 at 64; intermediate finite tests do not establish convergence or removal of the input. [STABLE-RESULT] |
| M09 | Earlier frames can be retained temporarily in recurrent state. | Observed | Memory accuracy is 0.992 at delay 0, 0.965 at delay 8, and 0.316 at delay 32. This is finite synthetic retention, not indefinite memory. [STREAM] [MEMORY-RESULT] |
| M10 | A threshold rule creates and selects among full model copies. | Implemented | Novelty and routing are defined in section 7. This establishes no formal ART equivalence. [GROW] |
| M11 | Three growing models can serve three synthetic topics in an archived run. | Observed | The result reports three models and topic accuracies 0.8255, 0.964, and 0.99. Total capacity grows, and there is no equal-capacity result in this record. [GROW-RESULT] |
| M12 | An ungrounded dictionary can be expanded while restricting adaptation to new rows. | Implemented | The stated probe does not test new tokens as distractors or general concept acquisition. [NEWWORD] [NEWWORD-PROBE] |
| M13 | Activity regularization can reduce the fraction of numerically large updates. | Observed | At \(\lambda_a=3\), the archived update fraction is 0.4929, accuracy 0.991, and final active-cell fraction 1.0. These use numerical thresholds. [ENERGY] [ENERGY-RESULT] |
| M14 | Sparse numerical updates establish physical energy savings. | Not established | The dense arithmetic precedes masking; the records contain no joule measurement or demonstrated skipped-work kernel. [MODEL] [ENERGY] |
| M15 | The scanner supplies labels for every cell and selected time. | Implemented | The readout is applied spatially; its dimensional bottleneck and optional auxiliary supervision limit its interpretation. [MODEL] [SCANNER] |
| M16 | Every displayed scanner word faithfully explains the cause of the answer. | Not established | An observational readout and supervision to repeat labels do not supply the required intervention evidence. |
| M17 | Images and symbolic tokens enter a shared latent interface. | Implemented | Camera replacement and optional CNN retina exist; the image lens target is global class repetition. [MODEL] [CARTILLA-TRAIN] |
| M18 | Masked-word generation demonstrates a causal model of the world. | Not established | The archived task reports category accuracy 1.0 and exact-token accuracy 0.0775. It evaluates category-compatible completion. [DREAM] |
| M19 | PAD is guaranteed to remain a zero dictionary vector and empty state after learning. | Not established | The lookup path and tests do not enforce this invariant; section 2.3. [MODEL] [TESTS] |
| M20 | The particular combination has an isolated, previously unestablished scientific advantage. | Not established | Requires comparison with prior mechanisms and the matched references, repeated experiments, and ablations of items 4–6. |

The expected category-only control in M06 follows because the query is uniform over four roles, the verb and place have unique categories, and agent/patient are two distinct nouns:

\[
\mathbb E[\mathrm{accuracy}]=\tfrac14\cdot1+\tfrac14\cdot1+\tfrac14\cdot\tfrac12+\tfrac14\cdot\tfrac12=0.75.
\]

An exact neighbour lookup solves `RoleTask` by construction, and an exact same-row lookup solves `RoleTaskFar`. No numerical execution of those controls is claimed here. The difference between 0.743 and 0.75 alone does not establish statistically significant inferiority. [TASK] [FAR]

## 9. Minimum comparison needed to isolate a contribution

This section specifies the necessary comparisons for the item-1 boundary assessment. It does not implement programme items 2–6 or substitute for their later experimental registration and execution.

### 9.1 Core comparison matrix

Use a shared local recurrent core and the same \(d,C,M,T\), seed/readout positions, data splits, initialization draws for shared tensors, optimizer family, and input examples. Compare the following three binary factors in a complete **2×2×2 matrix**:

| Factor | Condition 0 | Condition 1 | Why the contrast is needed |
|---|---|---|---|
| Decoder tying | Independent \(U\in\mathbb R^{V\times d}\) with head \(UR\); initialize \(U\) equal to \(D\). | Head \(DR\), with the same table used for input. | Separates tied optimization from the decoder's low-rank factorization. |
| School loss | \(\lambda_s=0\). | The same specified nonzero \(\lambda_s\), checkpoint indices, and labels. | Separates the architecture from added supervision and estimates interaction with tying. |
| Input reinjection | Preserve \(s^0\) but replace the update's \(z\) argument by zeros of the same shape. | Supply \(z\) at every update. | Separates persistent access to the input from retention in the evolving state while keeping update tensor dimensions fixed. |

The untied variant adds \(Vd\) parameters. Report this difference explicitly; do not silently replace the factorized head with a full head or call unequal models exactly parameter matched. Report both equal-example comparisons, which isolate a training intervention, and accuracy/cost curves, which show its resource consequence. A companion capacity-matched comparison belongs to item 4. School-loss evaluation itself adds computation, so equal training iterations are insufficient evidence of equal training cost.

A positive effect of one matrix cell is evidence about these components under that protocol. It does not on its own establish priority relative to all prior work. Conversely, if an equally supervised untied local model reproduces the advantage, an attribution specifically to dictionary tying is not supported.

### 9.2 Controls for claims about the recurrent core

An appropriate NCA or recurrent convolutional reference must receive the same input information, output target, and permitted auxiliary targets. Differences in perception filters, update rule, decoder rank, reinjection, and damage training must be enumerated instead of bundled under model names. A spatial Transformer reference needs a suitable positional representation and a recorded tuning budget; failure of one absolute-position configuration is not a mechanism-level impossibility result. Symbolic and category controls establish what the generator itself makes easy. These reference implementations and their execution are dependencies of items 3–4.

A single-update local ablation cannot communicate with an output more than one cell away. Its failure on remote retrieval is predicted by section 6 and cannot isolate a learned benefit of recurrence. To study sharing across time, compare a \(T\)-step shared rule with a \(T\)-layer local network with unshared weights, preserving receptive-field depth and reporting the resulting parameter difference and any companion capacity-matched result. No single comparison can silently equalize width, depth, parameters, and computation when those quantities change together.

### 9.3 Controls for the connected extensions

| Attribution | Minimum controlled contrast |
|---|---|
| Repair from internal memory | Cross retained versus removed input with damage-trained versus undamaged training; use the same cell masks, lesion times, recovery horizon, and access to input in the references. |
| Benefit of growing models | Compare equal total parameter/storage budgets, including a fixed bank of experts, and measure routing plus expert inference. Separate creation decisions from answer accuracy. |
| Benefit of scanner interpretation | Predict the effects of targeted state interventions and test those predictions against interventions matched in size and location. Immediate label readability alone is insufficient. |
| Energy or efficiency | Record actual work, latency, and, for an energy claim, physical energy under a common hardware protocol. Do not use update magnitude as a substitute measurement. |

### 9.4 Dependencies and completion decision

| Programme item | Evidence required before a stronger contribution claim |
|---|---|
| **4 — Better-adjusted references** | Executable local/recurrent, symbolic, and spatial Transformer references with matched information, documented auxiliary supervision, reasonable tuning, and transparent capacity/compute budgets. |
| **5 — Sufficient variation and replication** | Repeated paired runs, distributions and uncertainty, with model-initialization randomness distinguished from dataset/split randomness. Three seeds can be an operational start, not a universal proof of robustness. |
| **6 — Ablations explaining the improvement** | Results for the controlled factors above, including their interactions and relevant damage/growth contrasts; the contribution must be attributed to the measured component rather than the full package by assumption. |

**Item-1 mechanism conclusion:** the audited code is specified and its locality bound is derived. Its components and their combination are concrete enough to test. The present evidence does not isolate a novel advantage of the combination, prove formal equivalence to ART, or demonstrate capabilities beyond the stated task and measurement boundaries. A claim of a distinct scientific contribution remains dependent on items 4–6 and on the accompanying prior-art comparison. This conclusion completes the specification work even if those future comparisons ultimately find no unique advantage.

## 10. Verification record and source map

Verification for this document was static: the implementation was checked against the stated source commit, the referenced definitions and loops were read, equations were checked against their tensor operations and time ordering, and the quoted numbers were read from the committed JSON records. No checkpoint was trained or newly benchmarked. The proposition and rank statement are deductions with stated assumptions, not empirical measurements. Existing unit tests were inspected for coverage; this document does not claim a fresh passing test run.

All source references below pin the audited commit. Relative claims about later code or a different training environment require a separate check.

[MODEL]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/model.py#L20-L103
[TASK]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/task.py#L18-L139
[CODEC]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/codec.py#L12-L47
[CARTILLA]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/cartilla.py#L32-L84
[TRAIN]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/scripts/train.py#L87-L168
[PHASE3-TRAIN]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/scripts/phase3.py#L62-L130
[CARTILLA-TRAIN]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/scripts/train_cartilla.py#L100-L182
[SCANNER]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/scanner.py#L1-L24
[STREAM]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/phase3.py#L15-L80
[FAR]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/phase3.py#L84-L108
[NEWWORD]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/phase3.py#L112-L149
[NEWWORD-PROBE]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/scripts/phase3.py#L378-L409
[GROW]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/scripts/phase3.py#L222-L269
[HRR]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/neuropixel/hrr.py#L13-L34
[TESTS]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/tests/test_core.py#L14-L160
[SCALE-CODE]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/scripts/phase3.py#L412-L450
[ENERGY]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/scripts/phase3.py#L354-L374
[SCALE-RESULT]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/results/phase3/scale_np230k.json
[FAR-RESULT]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/results/phase3/far_np_big_reposo_s0.json
[STABLE-RESULT]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/results/phase3/stable_reposo.json
[MEMORY-RESULT]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/results/phase3/memory.json
[GROW-RESULT]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/results/phase3/grow_novedad.json
[ENERGY-RESULT]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/results/phase3/energy_l3.0.json
[DREAM]: https://github.com/Agnuxo1/NeuroPixel/blob/7da18d1cbd36c40c48228118abb20a7b5ab032f4/results/phase3/dream.json
