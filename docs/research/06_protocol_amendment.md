# Item 6 — Prospective amendment for component attribution

**Status: design for review and a dated item-6 freeze; no item-6 outcome is reported here.**  
**Amendment ID:** NP-SCI-20261006-v1-A06.  
**Date:** 2026-10-07.  
**Programme boundary:** item 6 only. Item 5 was closed at commit `ac0adf1`; item 6 opened at `a0d9f63a8db646d599de7686f76b27f041369929`.

This document preserves the factorial in [the frozen protocol][protocol] and [its machine-readable specification][protocol-json]. It adds explicitly declared experiments for recurrent depth, training damage and optimization budget, and separate fixed, novelty-triggered and resonance-triggered policies for growth. It does not modify either item-2 file. The executable plan, source hashes, environment and amendment must be frozen together before the affected training and evaluations.

The three-factor experiment was specified before items 4–5. The additional arms are informed by those completed observations, including the 28 training probes below the prespecified 95% binding threshold. That flag motivates examining learning, but does not establish why performance was low. These additions are prospective with respect to their own results. The split-0 test distribution and examples have already been used in items 4–5: the entire item-6 investigation remains exploratory, and cannot supply independent confirmation of H1.

## 1. Questions and interpretation of an intervention

The question is which declared changes alter performance within this implementation, task and training policy. There is no assumption that the original package has an improvement waiting to be explained. Operational definitions follow the inspected [ResearchNCA implementation][models] and [training/evaluation utilities][experiment]; their item-6 extensions must preserve the declared controls.

| Component | Training comparison | What a same-checkpoint intervention can establish |
|---|---|---|
| Dictionary tying | Shared input/output dictionary versus an independent output dictionary with the same factorized readout | Cloning the tied output dictionary after training preserves its values and predictions; it does not reproduce untied training. |
| School supervision | Answer loss alone versus answer loss plus the original intermediate token objective | School is already absent from evaluation. Removing a training loss at inference is not an ablation. |
| Reinjection | Preserve input seeding in both arms; supply identities versus same-width zeros at every update | Removing reinjection from trained weights changes input access under those weights; it does not estimate the effect of training without it. |
| Recurrent depth | Train otherwise identical cores for 1, 4 or 16 recurrent steps | Truncating a 16-step checkpoint measures dependence on its inference horizon, separately from learning at that horizon. |
| Training damage | Train with or without the declared state lesion, then evaluate both clean and under the same lesion | Evaluation damage measures checkpoint sensitivity; its interaction with damage training measures a difference in sensitivity. |
| Growth | Compare declared policies for creating/reusing complete experts under a capacity ceiling | Reusing exactly the same expert bank with different routers isolates routing from training that bank. |

These are controlled effects of complete interventions as implemented. They do not automatically isolate a unique latent mechanism. Tying changes free parameters and gradient paths; school changes the objective; fewer recurrent steps change computation and the spatial information available. A positive contrast therefore needs its operational definition alongside its value.

## 2. Preserved factorial: 16 training runs

Let A denote tying (0 independent, 1 tied), B school (0 weight 0, 1 weight 0.3), and C reinjection (0 absent after seeding, 1 retained). Execute all eight (A,B,C) combinations at each initialization seed 20 and 21.

| A | B | C | Decoder | School weight | Repeated identity |
|---:|---:|---:|---|---:|---|
| 0 | 0 | 0 | Independent factorized | 0 | Off |
| 0 | 0 | 1 | Independent factorized | 0 | On |
| 0 | 1 | 0 | Independent factorized | 0.3 | Off |
| 0 | 1 | 1 | Independent factorized | 0.3 | On |
| 1 | 0 | 0 | Tied factorized | 0 | Off |
| 1 | 0 | 1 | Tied factorized | 0 | On |
| 1 | 1 | 0 | Tied factorized | 0.3 | Off |
| 1 | 1 | 1 | Tied factorized | 0.3 | On |

Use split 0, adjacent RoleTask, 8×8 grids, vocabulary 35, identity dimension 16, state width 48 and update hidden width 128. Keep 16 recurrent steps, firing probability 0.5 during training, learning rate 0.003, AdamW with weight decay 0.0001, 1,024 optimizer updates, batch size 64 and gradient clipping at 1. There is no learning-rate search or new checkpoint selection. Use the final checkpoint.

The primary endpoint remains equal-weight macro accuracy over AGENTE and PACIENTE. Retain all four role accuracies, all-role macro accuracy, answer cross-entropy, training curves and the fixed training probe. Validation has 2,048 balanced examples with sampling seed 61001; test has 4,096 balanced examples with seed 61002; the 512-example training probe uses seed 61006. Data splitting, batch sampling seed 9002 and firing seed 9001 retain the item-2 definitions. All applicable tensor contents and artifact hashes must be recorded.

### 2.1 Exact component semantics

**Tying.** In the independent arm, initialize the V×16 output dictionary as an exact detached clone of the effective input dictionary. Preserve the 48-to-16 read projection and its rank limit; do not replace it with a full V×48 classifier. Shared tensors must begin at the same values for a paired initialization. The tied core has 29,824 trainable parameters; untying adds 35×16 = 560, for 30,384. This is a comparison of the tying constraint at fixed core dimensions, not equal free-parameter capacity. Adding unused parameters would not equalize functional capacity. Press and Wolf provide a primary precedent for both input/output tying and its effects on learning dynamics; their language-model result does not establish an effect here.[TYING]

**School.** Keep the original pre-update observations: states s3, s7, s11 and s15. Average token cross-entropy over those observations and occupied input positions, including the query; add 0.3 times that average to answer cross-entropy. Empty positions and the initially empty output position are excluded. With nine occupied positions this supplies 36 intermediate token terms per example before averaging. Tokens are already visible in the input, but the extra optimization objective is a real supervision difference. It neither adds a new model head nor proves that its readable states are a complete causal explanation. Intermediate companion objectives have a clear precedent in deeply supervised nets; this particular objective reconstructs input tokens.[DSN]

**Reinjection.** Both arms seed from the same visible identities. The disabled arm passes zeros of the same width to the update rule afterward. Keep initialization, tensor dimensions and legacy PAD behavior unchanged. Repeated access to a fixed observation already occurs in NCA classification with an immutable image channel, although that channel is not identical to this code's learned token field.[MNIST]

**Evaluation.** The audited core uses synchronous updates in evaluation: its random firing mask is applied only in training. Do not describe the standard final scores as a Monte Carlo average over firing masks. School does not run during standard evaluation, and no damage is implicit in a clean evaluation.

## 3. Additional core arms: 10 training runs

The following additions were agreed before their outcomes. All use tied decoding and reinjection, split 0, initialization seeds 20 and 21, the same optimizer settings and learning rate 0.003. They share the frozen data specification and final-checkpoint policy.

| Group | Steps | School | Optimizer updates | Training damage | New runs | Reused comparison |
|---|---:|---:|---:|---|---:|---|
| Depth | 1 and 4 | 0 | 1,024 | None | 4 | Factorial A1B0C1 at T16 |
| Damage | 16 | 0 | 1,024 | Defined below | 2 | Factorial A1B0C1, clean training |
| Budget diagnostic | 16 | 0 and 0.3 | 8,192 | None | 4 | Corresponding 1,024-update factorial arms |

Thus the core contains **26 training runs**, including the preserved 16-run factorial. The growth module below is separate.

### 3.1 Recurrent depth and truncation of the same weights

Compare models trained for T1, T4 and T16 at the same 1,024 optimizer updates and example exposure. School is zero throughout: T1 has no observation at every fourth update, so adding school there would silently change its definition or yield no school observation.

Also evaluate the clean T16, school-0 factorial checkpoints at inference horizons 1 and 4, with no retraining or parameter changes. Report these four additional evaluations separately from the four models trained at those horizons. The T16 evaluation is reused. Retain training and evaluation horizons as separate metadata. A difference between a T4-trained model and a T16 checkpoint truncated to T4 can reflect adaptation to the training horizon; neither is substituted for the other.

All horizons have the same number of parameters. They do not have equal compute or receptive fields. With the shared local 3×3 rule, a fact outside Chebyshev radius T of the output cannot affect that output after T steps under the audited assumptions.[mechanism] Low T1/T4 accuracy can therefore arise from missing accessible information. This panel does not distinguish recurrence or temporal weight sharing from a feed-forward network with matched depth, communication range and compute. Universal Transformers are a primary precedent for iterative shared computation, but their global attention differs from this local rule.[UT]

Keep the historical firing implementation intact. Resetting the same firing seed before training does not align masks by minibatch across different T: short rollouts consume fewer draws. Consequently these contrasts pair initialization and training examples, while their stochastic update trajectories also differ as part of the horizon policy. Do not claim exact common masks between T1/T4/T16.

### 3.2 Training damage and matched evaluation stress

At each training minibatch, draw one independent Bernoulli indicator with probability 0.5 using a dedicated CPU generator seeded 62001. If damage is applied, generate a [B,1,8,8] mask with independent cell erasure probability 0.3; broadcast it over the state channels. Multiply the recurrent state by this mask **after update 8**, then execute updates 9–16. Leave the seed, input identities, weights and subsequent reinjection intact. Generate the cell mask only on selected minibatches; record this draw convention. Damage randomness must not advance initialization, batch-sampling or firing RNGs.

The original rest-state script used probability 0.5 per batch and erasure fraction 0.3, together with a random horizon of 12–32 and random lesion time from 2 through T−1. The new fixed T16 and lesion time 8 are deliberate prospective choices. This arm does not reproduce that entire original recipe or its OneCycle schedule.[legacy-training] Damage training for NCA regeneration is prior art: the Growing NCA experiments explicitly erase regions during training and compare subsequent robustness.[GROWING]

For evaluation, compare both the clean-trained and damage-trained checkpoints in two conditions: clean, and a lesion applied with probability 1 after update 8, with erasure probability 0.3. Generate the complete 4,096-example mask once using a separate CPU generator seeded 62002; hash it, then use slices keyed by stable example index. Use the same masks across both training conditions and both initialization seeds, independent of evaluation batch size. Clean evaluation is unaffected by the training-damage switch.

The 0.3 erasure rate applies to all cells, including output/query positions and currently empty cells; it is not a targeted semantic lesion. There is one specified lesion distribution and one eight-update post-lesion horizon. Only the final T16 answer is scored; there is no immediate post-lesion readout or recovery curve. The endpoint therefore measures **final robustness to a lesion while the input remains available**. It is compatible with assisted recomputation, but does not show how much performance was lost and then recovered. It does not test recovery from unavailable input, long-term retention, or the broader repair hypothesis H3.

### 3.3 Prespecified optimization-budget diagnostic

Train four additional models from scratch for 8,192 updates: school 0 and 0.3, each at seeds 20 and 21. Keep T16, tying, reinjection, batch 64, all optimizer settings and constant learning rate 0.003. Report the two budgets, not a winner chosen by validation or test.

Compare each long-budget endpoint to its corresponding 1,024-update factorial endpoint. If a 1,024-step checkpoint is recorded during the long run, distinguish this trajectory checkpoint from a new replicate; shared initialization and RNG streams should produce the same prefix under the same environment and implementation. Do not resume from weights alone as though optimizer and RNG states had also been restored.

Retain answer loss separately from school and total loss wherever school is active. Compare the same frozen training probe and validation metrics, and retain the below-95% flag at both endpoints. The flag cannot establish convergence, a sufficient training budget or a cause of failure. A large improvement at 8,192 establishes sensitivity to this eightfold increase in optimizer steps and training exposure. An absent improvement does not prove that further optimization is impossible. A school-by-budget interaction does not establish that school was the sole cause of historical or item-5 differences.

There is no rescue of H1, no new learning-rate selection, no further extension triggered by these test scores, and no claim that 8,192 is an optimal budget.

## 4. Separate growth and routing panel

### 4.1 Definition and shared topic data

“Growth” here concerns a bank of complete 29,824-parameter NeuroPixel experts. It does not enlarge the H×W canvas or demonstrate persistent memory inside a cell state. The audited legacy novelty policy can create a model copy and retain old models; inference selects among full experts using an aggregate input-reconstruction score.[mechanism][legacy-training]

Use two initialization seeds, 20 and 21, and three ordered topics A→B→C defined by noun indices 0–3, 4–7 and 8–11. Filter the already frozen global training/validation/test triple pools to require both agent and patient in the same topic. Each topic contains 4×3×10 = 120 possible triples; the panel covers 360 and excludes cross-topic compositions. Do not reshuffle or repartition after filtering. The global 70/10/20 split does not guarantee those exact fractions within each topic; record the actual topic/split triple lists, counts and hashes.

Use the same three-stage exposure for the fixed snapshot-bank, adaptive-novelty and adaptive-resonance policies: 512 updates per stage, batch 64, T16, tied decoding, reinjection, school 0.3, AdamW learning rate 0.003, weight decay 0.0001 and clipping at 1. Reset the optimizer at each stage in all three policies. For stage j=0,1,2, use training sampling seed 9002+100j and firing seed 9001+100j, so the declared exposure streams are shared across policies and paired initializations. Record the exact batches and all RNG definitions before execution. Training always uses training-pool triples.

Final evaluation is balanced by role within each topic, 1,024 examples per topic, with sampling seeds 62012, 62112 and 62212 for A, B and C respectively. Pooling the three topics gives equal topic weight because their counts are equal; also report every topic and role separately. These are fresh sampled layouts from the same generator family and split policy, not an external task or laboratory replication.

### 4.2 Fixed snapshot bank and controls

For each seed, train one A→B→C trajectory, retaining weights after each stage. Denote those experts M_A, M_B and M_C. They are snapshots of one trajectory, not three independent training replications.

Evaluate the final single expert M_C and the following declared banks:

- The bank [M_A,M_B,M_C] preserves one expert per completed stage.
- The three-slot comparison [M_A,M_B,M_B] withholds the C-adapted expert and duplicates M_B. It equalizes nominal expert slots and the nominal parameter-count budget, **not functional diversity or useful capacity**. Record actual allocated storage; repeated references or deduplicated weights need not occupy three independent copies. Its contrast includes the information acquired during C adaptation.
- A preallocated three-slot bank loaded with exactly [M_A,M_B,M_C] has the same expert functions as the retained bank. Expected equality is an implementation/control check, not a separate training replicate or evidence that allocation timing improves learning.

At most three core experts require 89,472 parameters. Single-to-bank differences include retained capacity and routing and cannot be attributed solely to “growth.” Progressive networks provide a precedent for adding capacity and retaining earlier parameters, but their lateral connections are absent here.[PNN] Net2Net illustrates that changes in network allocation can initially preserve a function; the equivalence check here is simpler and is not a Net2Net implementation.[NET2NET]

### 4.3 Router comparison on exactly the same bank

Cache the same frozen experts' outputs for each declared router. Compare the legacy reconstruction-score selector, a uniform probability mixture, a random expert selector seeded 62013 over one vector for the entire A/B/C-concatenated evaluation set, without restarting at batch boundaries, and the declared learned gate. The legacy selector uses each example's unrounded mean own-token probability over occupied positions and picks the highest-scoring expert, with the first index breaking ties. State the bank used for each comparison; changing bank and router together is a policy contrast.

For the learned gate, use 35 binary visible-token presence features, four query one-hot indicators and a bias: a 40×K matrix W, initially zero, for the bank's K expert slots. Preserve the exact feature construction in the frozen source. Experts remain frozen. Fit on 512 balanced training examples per topic, using seeds 62011, 62111 and 62211 respectively, for 1,536 total examples. Do not provide topic IDs, generator metadata or evaluation targets as inference features.

With w_e(x)=softmax(XW)_e and expert probabilities p_e, minimize

\[
-\frac{1}{1536}\sum_i \log\left(\sum_e w_e(x_i)\,p_e(y_i\mid x_i)\right)
+\frac{10^{-4}}2\|W\|_F^2.
\]

The penalty includes the bias row. Use 128 full-batch gradient-descent steps at learning rate 0.1, evaluated stably through log-softmax/log-sum-exp. There are no hard labels assigning training examples to experts. Freeze the resulting gate before any final scores.

The gate has 40K trainable coordinates, at most 120, and uses 1,536 training examples and 128 optimization steps per fit. When K=1 it cannot choose among experts. Fit a separate gate for each fixed bank [M_A,M_B,M_C], duplicate-slot bank [M_A,M_B,M_B], final adaptive-novelty bank and final adaptive-resonance bank: four fits per seed, eight in total. The preallocated equivalent bank reuses the exact fixed-bank gate; the single expert needs none. Freeze all eight fits before final evaluation. Allocating equal-size frozen arrays to other routers only equalizes storage; it does not equalize useful capacity or supervised exposure. Report the maximum expert-plus-gate count, 89,592, and all gate fitting/inference costs. A learned-versus-fixed router contrast includes learning that router.

Report binding and all-role accuracy per topic, routing frequencies and the “any expert correct” fraction as a target-aware descriptive ceiling for **hard expert selectors only**. A probability mixture can predict the correct answer even when every individual expert's argmax is wrong, so this quantity is not an upper bound for the uniform or learned mixtures. Label the metric hard-selection oracle accuracy/binding. The oracle is not a deployable router or an independently evaluated model. Decisions at inference must use visible inputs and expert outputs, never the true answer or topic label.

### 4.4 Adaptive novelty policy

Execute both inherited creation heuristics as distinct policies, each at seeds 20 and 21. Together with the fixed trajectory, there are six growth trajectories and 18 training stages, totaling 9,216 optimizer updates. They remain two initializations per policy, not six independent initialization replications. The four adaptive trajectories add 6,144 updates to the two-trajectory fixed-bank module.

Before each stage j=0,1,2, construct a balanced 512-example **training-pool** probe for its topic using seed 62010+100j. The probe must not advance the training stream. Creation decisions use visible tokens and model outputs only, not answer correctness, final labels or the nominal topic identity. With no existing expert, create the first core using the trajectory's initialization seed.

For expert m, let p_m(X_bp|X) be the final lens probability of the actual visible token at occupied input position p. Use the same school-compatible lens and all occupied positions, including the query.

**Adaptive novelty.** Compute the fraction of occupied positions across the entire probe batch whose own-token probability is strictly below 0.5. Round each expert's fraction to four decimals before selecting the minimum and comparing it to 0.1. Create a new expert if the bank is empty or the minimum rounded novelty is strictly greater than 0.1; otherwise revisit the selected existing expert. When a parent is available, clone the minimum-novelty expert. Ties choose the first bank index.

**Adaptive resonance.** For each example, average own-token probabilities over its occupied positions, then average those example scores over the probe batch. Round each expert's resulting score to four decimals for selecting the maximum and for the creation comparison. After training the first expert on A, compute its **unrounded** resonance on the same A training probe and freeze

\[
\tau=0.75\,R_A^{\mathrm{post\ training}}.
\]

This is a training-derived threshold, not an absolute threshold of 0.75 and not a test-tuned parameter. At subsequent stages, create a new expert when the maximum rounded resonance is strictly below tau; otherwise revisit the selected expert. Clone the maximum-resonance expert when creating a child. Ties choose the first index. Keep tau fixed thereafter.

In either policy, a new expert is a complete copy of its chosen parent; preserve other experts. Revisit updates the selected existing expert in place. There can be at most three experts because there are three stages and at most one creation per stage. Use exactly 512 training updates with a fresh optimizer for a new or revisited expert. This deliberately removes the legacy difference between long new-expert training and brief revisits, as well as its different learning-rate schedule. It is a budget-controlled adaptation of the inherited heuristic, not a literal reproduction of the complete historical experiment.

Record each probe hash, raw and rounded decision scores, threshold, selected index, clone lineage, created/revisited action, expert count and stage costs. Evaluate final banks under the same declared router comparison and gate-fitting procedure. No trigger threshold, probe or routing rule may be altered after final evaluation.

### 4.5 Scope of growth estimates

Within each seed, report bank/router/policy differences on the same examples, including topic-specific effects. The statistical unit is the complete training trajectory. Fixed-bank snapshots, alternative routers and duplicate slots do not increase n beyond two per policy.

Each adaptive policy has a variable realized expert count. Its comparison with a fixed three-expert bank therefore combines creation/assignment decisions, retained capacity, parameter trajectories and routing. A positive result cannot be attributed to the novelty statistic alone. Report the realized expert count, assignments, threshold decisions and costs even if the policy always creates a new expert or never grows past one. Do not adjust thresholds after observing the final set.

## 5. Estimands and uncertainty

Let Y_s(a,b,c) be final test binding accuracy for seed s in {20,21} and the factorial condition (a,b,c). Compute contrasts within each seed before summarizing across seeds.

The main effect of tying, averaged equally over the other factors, is

\[
D_{A,s}=\frac14\sum_{b=0}^1\sum_{c=0}^1
[Y_s(1,b,c)-Y_s(0,b,c)].
\]

Define the B and C main effects by the corresponding permutation of indices. Also report each conditional contrast, since a positive average can conceal a reversed effect in another background.

The tying-by-school interaction is an average of differences in differences:

\[
I_{AB,s}=\frac12\sum_{c=0}^1
\{[Y_s(1,1,c)-Y_s(0,1,c)]
-[Y_s(1,0,c)-Y_s(0,0,c)]\}.
\]

Define AC and BC similarly, averaging over the remaining factor. The three-way interaction is the difference between those AB differences at C1 and C0:

\[
I_{ABC,s}=
[Y_s(1,1,1)-Y_s(0,1,1)-Y_s(1,0,1)+Y_s(0,0,1)]
-
[Y_s(1,1,0)-Y_s(0,1,0)-Y_s(1,0,0)+Y_s(0,0,0)].
\]

These definitions fix scale and sign. They are not automatically the regression coefficients from a −1/+1-coded model: main effects equal twice the corresponding coefficient, pairwise differences in differences equal four times the interaction coefficient, and the three-way difference equals eight times its coefficient. The complete factorial supports separating these contrasts, rather than attributing an arbitrary portion of the full-package score to one component.[FACTORIAL]

For each depth T in {1,4}, report the trained-depth contrast Y_trainT − Y_train16. Separately report Y_same16weights,evalT − Y_same16weights,eval16. Neither quantity replaces the other.

For damage training z in {0,1} and evaluation lesion l in {0,1}, let Q_s(z,l) be binding accuracy. Report both clean and damaged training contrasts, Q_s(1,l) − Q_s(0,l), and the interaction

\[
I_{\mathrm{damage},s}=
[Q_s(1,1)-Q_s(1,0)]
-[Q_s(0,1)-Q_s(0,0)].
\]

A positive interaction means a less adverse (or more beneficial) lesion effect for damage-trained weights. It is not necessarily high absolute damaged accuracy, so report all four underlying scores.

For school b in {0,1}, define the budget contrast

\[
D_{\mathrm{budget},s}(b)=
Y_s(A1,b,C1;8192)-Y_s(A1,b,C1;1024),
\]

and the school-by-budget interaction as D_budget,s(1) − D_budget,s(0). Restrict its interpretation to these four cells. Do not compare an 8,192-update NeuroPixel endpoint with a 1,024-update reference and call it a matched-budget architecture advantage.

For every score and seed-level contrast, report both raw values, mean, sample standard deviation and range. Following item 2, a paired t interval can be reported as mean ± t_(0.975,1) × SD/√2, explicitly labeled exploratory with only one degree of freedom. Its distributional assumption cannot be meaningfully checked with two seeds. Do not clip a wide t interval to the outcome bounds or conceal its width. If a required arm is missing or non-finite, identify the missing contrast; do not impute it, replace its seed or treat one available pair as two.

The eight factorial backgrounds are not eight independent initialization replications of a component. All test examples, snapshots, masks and router variants derived from a seed also do not increase the number of training trajectories. There are two initialization units, conditional on one split, fixed sampling streams, one environment and a limited optimization policy. This does not estimate split-by-initialization variability, variation across arbitrary hyperparameters or laboratory replication. The variance distinctions follow the primary benchmarking analysis of Bouthillier and colleagues.[VARIANCE]

An optional prespecified example bootstrap must reuse common resampling indices across all compared prediction arrays and remain stratified by role. Any such interval is conditional on those checkpoints and that generator. It is not a substitute for the two-seed uncertainty. Retain the item-2 bootstrap settings when used. No collection of secondary p-values or a favorable interval is used to announce a confirmatory component discovery; H1 is not reconsidered here.

## 6. Capacity, cost and optimization accounting

The design gives each core factorial condition equal optimizer updates and example exposure, except for the explicitly separate budget diagnostic. It does not equalize effective capacity, gradients, compute or ease of optimization.

| Change | Preserved | Changed and therefore reported |
|---|---|---|
| Tying | Core dimensions and factorized decoder rank | 560 free dictionary parameters, parameter sharing and gradient coupling |
| School | Architecture, inputs and answer labels | Intermediate targets/loss terms, backward paths and extra lens computation |
| Reinjection | Seed, input width and core parameter count | Availability of repeated local input |
| T1/T4/T16 | Parameter values at initialization and optimizer-update count | Information radius, unrolled computation, gradient paths and firing RNG consumption |
| Damage training | Parameter count, answer supervision and nominal updates | State corruption distribution and optimization trajectory |
| 1,024/8,192 updates | Architecture, optimizer settings and per-batch size | Eightfold example exposure and number of optimizer steps |
| Growth policies | Declared stage schedule and maximum bank slots | Actual distinct experts, retained states/weights, routing work and any gate fitting |

For the 26 core runs, the planned exposure is 55,296 optimizer updates and 3,538,944 minibatch examples, including repeated prefixes of separately executed long-budget runs. The six growth trajectories add 9,216 updates and 589,824 minibatch examples, for 64,512 updates and 4,128,768 examples across core and growth training. Router fitting is additional and reported separately. These are accounting totals, not independent data or replication counts. At fixed grid size, T×H×W is a useful count of dense cell updates per example; it is not a FLOP estimate or a physical energy measurement. Stochastic firing does not avoid the dense computation performed before masking.

Record actual parameter counts, checkpoint/storage bytes, training wall time, evaluation latency when measured, device, thread count, library versions and peak memory. Include school computation, damage-mask generation, router fitting and all expert evaluations in their relevant costs. If raw expert results are cached to avoid the legacy router's duplicate forward calls, document that implementation change when reporting latency. Fixed expert banks do not have single-core inference cost.

Equal learning rate and optimizer updates do not prove equally successful optimization. Report answer loss separately from the auxiliary objective, probe binding, validation performance and non-finite events. All below-95% probe flags remain visible. Do not infer that a component lacks utility in all training regimes from a low-budget null result, or that a later positive result supplies a unique explanation for item 5.

The planned execution target is a separate GitHub Actions CPU environment, following loss of access to the earlier execution environments. No training is authorized by this document until the root reviewer approves the scientific and execution freeze. Verify available RAM and the thread limit at preflight, and record the actual runner environment. Use the same declared software/device settings within paired comparisons; an environment change must be recorded as an additional difference, not silently assigned to the component. Preserve code and dataset hashes and compare with available manifests. The recovered text of item 5 is not a substitute for missing tensors or checkpoints; no byte identity with unavailable Linux evidence is assumed. Environmental differences must remain visible, rather than being counted as external replication.

## 7. Freeze, execution and reporting contract

Before any affected run, freeze this amendment, the complete arm list and order, resolved model configurations, source files/controllers, additional RNG definitions and the environment record. The original protocol files remain unchanged. A no-damage configuration must preserve the existing initialization path; extra model or mask creation must not consume the data or initialization RNG silently.

Keep training, validation/probe access and final evaluation as separate events. Complete the declared training and any train-only router fitting before the final gate; freeze weights, router settings and selection rules before opening item-6 final predictions. Save configurations, learning curves, dataset/mask hashes, states or checkpoints required for declared comparisons, per-example predictions, losses, timestamps and elapsed times. Persist artifacts atomically and verify readback before accepting them as complete.

Do not choose a factorial corner, training horizon, lesion severity, router or budget because it scores best on the final set. Report all declared outcomes and incomplete runs. Preserve resource stops and failed attempts. Any retry must retain its original attempt and declare whether it corrects only a resource/software fault or changes the scientific configuration. Resource policy remains the item-2 policy: sequential reservation, no eviction of other work, declared RAM/GPU margins and no paid compute; CPU threads remain within its maximum.

A completed item-6 report may conclude that a component effect is unestablished, conditional on another factor, reversed in this setting, or obscured by limited learning. Consistent directions across two seeds are descriptive evidence, not established reproducibility across initialization populations. Interactions preclude a unique percentage allocation of an overall change without an additional attribution convention.

This document specifies experiments; it reports no item-6 measurements and does not initiate later programme items.

## 8. Primary sources and the specific evidence used

Sources were consulted for experimental distinctions, not for a claim of worldwide priority. Their experiments do not establish NeuroPixel outcomes. The statements below are original summaries; quoted titles identify the works.

| Source | Relevant primary evidence | Consequence for this design |
|---|---|---|
| Press & Wolf, 2017, *Using the Output Embedding to Improve Language Models* [TYING] | Analyzes tied input/output embeddings and their update behavior in language models. | Separate sharing from decoder rank and disclose its parameter/gradient consequences. |
| Lee et al., 2015, *Deeply-Supervised Nets* [DSN] | Adds companion objectives at hidden layers and analyzes/trains the combined objective. | School is an explicit training intervention; a benefit does not by itself identify why it helps. |
| Randazzo et al., 2020, *Self-classifying MNIST Digits* [MNIST] | Exposes image intensities through an immutable channel while mutable cellular states communicate and classify. | Persistent access to observations has a concrete NCA precedent; state repair with retained observations needs qualification. |
| Mordvintsev et al., 2020, *Growing Neural Cellular Automata* [GROWING] | Experiment 3 damages pool states during training and evaluates regeneration, distinguishing it from undamaged training. | Separate training damage from evaluation lesions; the present lesion distribution is explicitly different. |
| Dehghani et al., ICLR 2019, *Universal Transformers* [UT] | Uses recurrent iterative computation with self-attention. | Repeated learned computation is established; this study's local communication range must be considered separately. |
| Rusu et al., 2016, *Progressive Neural Networks* [PNN] | Adds network columns while keeping earlier parameters and using lateral connections. | Growth, retained capacity and transfer are distinct choices; a snapshot bank is not a reproduction of progressive networks. |
| Chen, Goodfellow & Shlens, ICLR 2016, *Net2Net* [NET2NET] | Expands networks through transformations that preserve the teacher's function at initialization. | Enlarging or copying a representation can initially preserve predictions; allocation alone need not explain a benefit. |
| NIST/SEMATECH, *Two-level full factorial designs* [FACTORIAL] | Lists the eight cells of a three-factor design and its main, pairwise and three-way terms. | Preserve all corners, interactions and an explicit contrast scaling. |
| Bouthillier et al., MLSys 2021, *Accounting for Variance in Machine Learning Benchmarks* [VARIANCE] | Separates randomness from optimization, data processing and hyperparameter procedures. | State which sources vary; paired examples or intermediate snapshots cannot replace independent training variation. |

[protocol]: 02_protocol.md
[protocol-json]: protocol.json
[mechanism]: 01_mechanism_specification.md
[legacy-training]: ../../scripts/phase3.py
[models]: ../../neuropixel/research/models.py
[experiment]: ../../neuropixel/research/experiment.py
[TYING]: https://aclanthology.org/E17-2025.pdf
[DSN]: https://proceedings.mlr.press/v38/lee15a.pdf
[MNIST]: https://distill.pub/2020/selforg/mnist/
[GROWING]: https://distill.pub/2020/growing-ca/
[UT]: https://arxiv.org/abs/1807.03819
[PNN]: https://arxiv.org/pdf/1606.04671
[NET2NET]: https://arxiv.org/pdf/1511.05641
[FACTORIAL]: https://www.itl.nist.gov/div898/handbook/pri/section3/pri3331.htm
[VARIANCE]: https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf
