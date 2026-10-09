# Item 2 — Prospective hypotheses and decision rules

## Scope and timing

This protocol is frozen before new control scores, new training runs or new checkpoint evaluations in this programme. The historical results in item 1 were already known. Consequently this is a prospective local protocol for **new experiments**, not a retrospective claim that the original research was preregistered. A SHA-256 manifest and Git commit record the frozen files. Any later change must be an explicit, dated amendment made before the affected evaluation, with its reason and consequences for confirmatory status.

The first experiment series is a bounded, resource-accounted pilot on the adjacent synthetic RoleTask (8×8). Far-task controls are included in item 3, while far and other layout tests are secondary distribution-shift probes; they are not additional primary endpoints. Its negative results can reject a claim **within that budget and distribution**; undertraining cannot establish that an architecture is incapable of learning. Its positive results cannot establish broad intelligence, a new biological mechanism, a physical energy advantage or Nobel-level importance. Larger independent studies are separate requirements.

## Central hypothesis

**H1.** On the adjacent RoleTask at an 8×8 grid, under fixed supervision and optimization budgets, the NeuroPixel core with tied decoding and repeated identity input improves difficult role binding over a strong, validation-selected neural reference.

- Primary endpoint: macro accuracy over AGENTE and PACIENTE on previously withheld triples, giving the two roles equal weight.
- Secondary endpoints: macro accuracy over all four roles, each role separately, example count, train/validation performance, parameter count, measured training time and inference latency.
- A practically promising pilot requires NeuroPixel's primary accuracy to be at least 90%, the estimated paired advantage to be at least 5 percentage points, and a 95% interval for the paired advantage to exclude zero. The 5-point criterion is a **point-estimate screening threshold**, not a confidence claim that the true improvement exceeds 5 points. A claim of a true improvement of at least 5 points would require the interval's lower endpoint to exceed 5 points.
- The reference family is selected using validation only. After per-family learning-rate selection, the reference family is frozen using pilot initialization 7, excluding NeuroPixel: highest validation binding accuracy, then lowest validation cross-entropy, then fewest parameters, then alphabetical family name. It is not reselected for each replicate. All specified families are still reported. No family is dropped because its final result is inconvenient.
- Exact symbolic references are accuracy ceilings and task-diagnostic controls. A task solved perfectly by a simple rule cannot establish a general reasoning advantage merely through high neural accuracy.
- If the thresholds fail, H1 is not supported for the declared setting. If learning curves show inadequate optimization, the result is also labeled budget-limited; no post-test tuning rescues the same test as a fresh confirmation.

## Component and extension hypotheses

| ID | Falsifiable prediction | Evidence needed; negative outcome |
|---|---|---|
| H2 — attribution | Tying, intermediate token supervision or reinjection has a reproducible positive effect when the other factors are held fixed. | A 2×2×2 factorial comparison, including interactions. A null or reversed contrast does not support the corresponding benefit in this exploratory pilot. Two training seeds support exploratory estimates only. |
| H3 — repair | Recovery survives removal of the original input and exceeds a comparably treated reference. | Cross retained/removed identity with matched lesion masks and recovery horizons. Good retained-input accuracy alone supports assisted recomputation, not memory recovery. |
| H4 — retention | Delayed facts remain retrievable under interference at a fixed total memory budget. | Delay/capacity/interference curves and recurrent or explicit-memory references. A loss of accuracy with delay or capacity rejects indefinite-memory interpretations. |
| H5 — scanner | Predicted effects of state interventions agree with actual answer changes. | Matched targeted and control interventions, including decoder-nullspace changes. Decodable labels alone are insufficient. |
| H6 — efficiency | A correct-answer cost advantage remains after including routing, preprocessing, execution and relevant training costs. | Measured matched-hardware latency; physical joules separately when telemetry is available. Numerical update magnitude is never substituted for energy. |
| H7 — locality | With fixed parameters, masks and local hooks, an input outside the radius-T light cone cannot affect an output after T steps. | Controlled perturbations and the proof in item 1. This is an architectural consequence, not a newly discovered scientific law. |

## Datasets and independence

For new training, the 1,320 valid agent/verb/patient triples are deterministically shuffled. The first 20% form final evaluation, the next 10% form validation, and the remaining 70% form training. All three sets are pairwise disjoint. Place words and spatial layouts remain generated independently, so this is composition holdout within one generator, not outside-generator generalization. New layout and richer-task protocols must be specified before their affected evaluations.

The primary split seed is 0. Additional split seeds are 101 and 202. Validation has 2,048 examples and final evaluation 4,096 examples, balanced equally across the four query roles. Sampling seeds are specified in `protocol.json`. Fixed validation and final datasets are hashed. Training never reads final metrics; the fixed final checkpoint is evaluated once after hyperparameter selection. Final predictions are retained per example.

Historical checkpoints use their **original** task split seeds (0, 1 and 2 for the scaling runs). Their original sample seed is 5 and their original test count is 2,000. These are retrospective re-evaluations, reported separately from fresh balanced evaluations. Original runs coupled initialization and split randomness; they cannot isolate those sources of variation.

## Neural pilot and selection

Four families are specified: the 29,824-parameter NeuroPixel core; a conventional cellular comparator with an untied factorized decoder and no repeated identity input; a spatial convolutional GRU; and a Transformer with explicit two-dimensional relative-position information. Widths are chosen before outcome inspection to target approximately 30,000 trainable parameters. The actual count and deviation are reported; parameter equality is not claimed where absent.

All four primary comparisons use answer cross-entropy only. Intermediate token supervision is introduced explicitly in the factorial study, not given to one family silently. The two learning-rate trials are 0.001 and 0.003, both using AdamW, weight decay 0.0001, gradient clipping at 1, 1,024 updates and batch size 64. NeuroPixel uses 16 recurrent steps. The final checkpoint is used; there is no best-test or best-validation-checkpoint selection. Optimization is flagged as budget-limited when final binding accuracy is below 95% on a fixed 512-example training probe (sample seed 61006), or when a non-finite loss prevents completion. This diagnostic is defined before results and does not authorize extending training after final evaluation. The learning rate is chosen by validation macro binding accuracy, with lower validation cross-entropy breaking ties, then the lower learning rate. Pilot initialization seed is 7.

Equal updates and examples are the primary exposure budget. They do **not** equalize elapsed time, FLOPs or tuning difficulty, all of which are recorded or explicitly unmeasured. The two-rate search is a limited pilot search, not an exhaustive optimization of each architecture. A separate measured-cost comparison is required before any efficiency claim.

## Replication panels and uncertainty

After validation selection, the initialization panel uses seeds 10, 11, 12, 13 and 14 with split seed 0. The split panel uses initialization 10 with split seeds 0, 101 and 202. Only the five-initialization panel decides H1; all five finite paired runs must complete for a positive decision. Failed or safety-stopped pairs prevent a positive H1 decision. The split panel is exploratory and cannot replace an unfavorable primary panel. These are **two conditional, overlapping panels**, not seven independent joint replications. They do not estimate the initialization-by-split interaction. Report them separately. The three-split panel is exploratory.

Minibatch sampling and update-mask seeds are held fixed within an initialization panel; initialization is generated separately. Deterministic algorithms are requested where supported, and any non-deterministic device behavior is recorded. Different hardware remains an environment factor, not an independent laboratory replication.

Per-role binary rates receive Wilson intervals, labeled as conditional on the checkpoint and generator. Macro rates and paired differences receive a deterministic paired bootstrap stratified by role, but this does not replace training-run uncertainty. Across training seeds, report every value, mean, sample standard deviation and a paired t interval with its small-sample limitation; also report the range. Do not pool thousands of test examples as though they were thousands of independent model replications. Split and initialization panels remain separate. Secondary contrasts and the factorial study are exploratory; no collection of unadjusted secondary p-values is used to announce a discovery.

The factorial pilot uses tying on/off, school weight 0/0.3 and reinjection on/off with seeds 20 and 21, split 0, fixed learning rate 0.003, the same updates/batch/steps as above. The untied decoder keeps the same rank factorization; its additional dictionary parameters are recorded. No-reinjection keeps initialization intact and supplies zeros of the same width during updates. School supervision has the original sampling convention (pre-update states 3, 7, 11 and 15).

## Resources, failure handling and records

Runs are sequential inside the shared GPU queue. A reservation and live preflight must satisfy the user's resource margins; another process is never terminated to make room. CPU fallback uses at most four threads. Out-of-memory, non-finite loss, resource safety stops and interrupted runs retain their logs and are not silently excluded. A retry may correct a software or resource fault without changing the scientific configuration, and must be recorded. No paid compute is authorized by this protocol.

The code revision, environment, configuration, data hashes, seeds, weights, learning curve, per-example predictions and elapsed times are retained. Historical observations, new measurements, simulations and external requirements carry distinct provenance. A completed investigation can conclude that a desired capability is **not established**. Independent replication, independent utility and qualified nomination remain external acts and cannot be marked complete by writing a document.

## Acceptance for item 2

The falsifiable hypotheses, variables, primary decision rule, controls, data separation, limited tuning budget, randomness panels, uncertainty treatment and failure rules are defined before new outcomes. This closes protocol formulation. It does not assert successful experimental outcomes, statistical power for every possible effect, or public third-party preregistration.
