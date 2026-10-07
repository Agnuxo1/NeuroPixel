# Item 11C: learned delayed-copy protocol

## Scientific question

Can the current NeuroPixel stream retain and retrieve one of six presented symbols after the symbol disappears, and how does its performance change after a longer blank interval or a controlled distractor? Compare it with the existing FrameGRU using approximately matched parameter counts. The primary estimand is the NeuroPixel-minus-GRU accuracy difference at eight blank frames, paired across five fresh training realizations.

This is a known, synthetic cue census and finite-horizon experiment. It does not establish novel-content generalization, indefinite persistence, automatic memory between ordinary calls, a full episodic memory system, biological memory, or historical H1. The original production model and phase3 recurrence are unchanged. The strict research helper provides explicit continuation and pre-query interventions and has been tested against the native paths in the preceding diagnostic.

## Information presented to the models

Vocabulary is PAD=0, QUERY=1 and target symbols 2–7; canvas size is 3 by 3. Readout is (2,2), query token position is (2,1). Eight row-major cue positions exclude the readout and include the query position. The complete known census has 6 targets times 8 positions = 48 cases, ordered by target then position. Every sequence presents one cue frame, d gap frames, and one query-only frame. The target is absent from all normal gap and query frames.

NeuroPixel receives 3 local updates on the cue, 3 on each gap and 10 on the query: 13+3d total. FrameGRU receives d+2 recurrent frame transitions. These models receive the same visible sequence; their numbers of operations, encoding mechanisms and dynamic state allocations are not matched.

Training samples one delay uniformly from {0,1,2} for an entire minibatch of 32 independently sampled target-position pairs with replacement. The full census is balanced; an individual training batch need not be. Development evaluates all 48 pairs at delays 3 and 4, and a prerequisite competence probe evaluates delay 2. All content and positions are known by design. Delay 4 has development exposure; final delays 0 and 2 were included in training. Longer delays probe horizon extrapolation, not unseen symbolic content.

## Models, optimization and randomness

NeuroPixel uses c_id=16, c=48, hidden=128, fire_rate=0.5, no retina and no grounding: 29,392 parameters. FrameGRU uses e=8, d=48, hid=70: 29,336 parameters. Both train from fresh initialization with AdamW, weight decay 0.0001, gradient norm clipping 1.0, constant selected learning rate, batch size 32 and answer cross-entropy only. There is no auxiliary school loss, early stopping, checkpoint selection, rescue fit or outcome-dependent update extension.

NeuroPixel allocates 432 float32 state values per example (1,728 bytes); GRU allocates 70 (280 bytes). These are array payload counts, not estimates of representational capacity, resident memory, FLOPs or physical energy. This difference limits architectural attribution even when parameter counts are close.

Each run initializes Torch with its seed s, then resets the training firing stream to 110000+s. A private NumPy PCG64 generator seeded 100000+s draws the entire [updates,32] pair-index matrix first, followed by the entire [updates] minibatch-delay array. Persist both arrays. Sharing s pairs the sampled input stream across architectures and rates; it does not make their random numerical operations identical. Scientific and test-fixture seeds are distinct.

## Development stage and conditional admission

Run exactly four pilots: two architectures times learning rates {0.001,0.003}, seed 69, 1,024 updates each. The earlier prospective 512-update draft was revised to 1,024 before any learned result or execution; both preparation records remain in version control.

For each family, choose the rate by highest mean accuracy over development delays 3 and 4, then lowest mean cross-entropy, then lowest learning rate. Main admission requires that BOTH selected pilots reach at least 95% accuracy on the complete 48-case delay-2 census. Because counts are discrete, this requires at least 46 correct cases. Preserve all four pilots irrespective of selection.

If a complete pilot stage fails that competence requirement, archive it as an operationally completed negative scientific result and do not run the main study. Actual execution/contract failures remain failures and are investigated before any admission. The source/runtime/array and preceding diagnostic audits must verify successfully before main admission.

If competent and operationally admissible, train exactly ten fresh main models: seeds 70–74, paired NeuroPixel and GRU, 2,048 updates each, with the development-selected rate for that family. Do not reuse pilot weights or select a main checkpoint. Completed configs, weights, streams, buffers, logs and all three development outputs must be present for every run before final access.

## Fixed final panel

Every checkpoint has sixteen conditions, totaling 1,488 nominal row evaluations:

| Panel | Conditions | Rows per condition |
|---|---|---:|
| Intact delay curve | Normal delays 0, 2, 4, 8, 16, 32, 64 | 48 |
| Cue/state interventions | Cue erased, state zero, state swap at delays 2 and 8 | 48 |
| Interference | Repeated blank control, same-position distractor, next-position distractor at delay 8 | 288 |

Cue-erased conditions retain the entire frame/update inventory but blank the cue. State zero and swap occur after all cue/gap processing and immediately BEFORE the query. For the swap, recipient row i receives the state of row (i+8) modulo 48: the next cyclic target at the same cue position. Record original-target and donor-target accuracy and save actual before/after intervention tensors. Verify the operation directly, separately from approximate cross-call numerical comparisons.

The distractor replaces the FIRST blank frame at delay 8; it does not add another frame or updates. Fully cross each target-position pair with all six distractor labels, independently of the original target. Same-position places the distractor at the cue location; next-position uses the next of the eight eligible cue positions cyclically. These are 48 same-label re-exposure cases and 240 conflicting-label cases, reported separately as well as together. Also report distractor-target accuracy. A fixed target-to-distractor mapping would reveal the answer and is not used.

The batch-288 blank control repeats each target-position history six times. These repetitions and the multiple delays/conditions are correlated observations. Interference comparisons use that matched batch-288 blank control, with a saved diagnostic comparison against the batch-48 normal case. They are not 288 independent trials of a population generalization hypothesis.

## Analysis and claims

The only inferential interval is the primary paired difference at normal delay 8: the five per-seed NeuroPixel-minus-GRU accuracies, their mean and nominal two-sided Student t 95% interval with four degrees of freedom. Report all five values, sample SD and the strong small-sample assumptions. This concerns training realizations conditional on a fixed task and selected rates, not variation over independent tasks, laboratories or arbitrary content. Do not bootstrap deterministic census rows or treat frames/cells as replications.

For every condition, report counts, accuracy, cross-entropy and per-target/per-position results; include same-label/conflicting-label strata for interference and donor/intruder accuracy where relevant. Secondary paired differences retain all values, means and SDs as descriptive results, without confidence intervals, p-values or selection rules.

Evaluate near-delay competence (delay 2) and finite retention (delay 8) against 95% separately for every main seed and report whether all five pass. A drop after state erasure is interpretable as useful memory dependence only when intact retrieval is competent. Donor tracking offers a content-specific control beyond nonspecific damage. Failure of near-delay learning leaves optimization, representation and generalization unresolved; low intact performance is not evidence of forgetting or repair.

A predictor independent of the target has at most 1/6 accuracy on the balanced cue census if it predicts an answer class, with lower accuracy possible for non-answer outputs. This is an analytical reference, not a measured baseline model. No superiority, Nobel eligibility, capacity or energy claim follows automatically from a positive primary interval.

## Source custody, runtime and independent recount

Eight focused controller contract methods cover literal frame construction, balanced independent distractors, private shared streams, rate selection and completed-negative admission, semantic metric corruption, ten-run artifact/development completeness, consumed-access retry refusal and a two-row single-update factory/serialization fixture. The fixture is not a study model. Any skip, failure or inventory mismatch blocks training.

The parent worker archives all ten training subtrees to a remotely confirmed Git commit before launching the final evaluation. The consumed final-access record precedes model construction and cannot be silently retried. Save raw logits, predictions, labels, float64 NLL, full frames/update counts, exact rows, positions, pre/post-query states and intervention tensors. Save every run's parameter-only checkpoint with its explicit constructor and nonpersistent-buffer metadata.

A separately implemented NumPy/standard-library/SciPy auditor reconstructs the census, labels, frames, sampling streams, selection, scores, intervention arrays and primary statistics from saved artifacts. It does not import Torch, load checkpoint tensors or run inference. The operational wrapper checks full source ZIP/Git identities, complete file inventories, actual training-archive ancestry and identical training subtrees before and after final scoring. These are internal reproducibility controls, not an independent laboratory replication or external blind custody.

Numerical execution uses Python 3.12.14, Torch 2.6.0+cpu, NumPy 2.2.6, SciPy 1.15.1 and the other exact package pins in the executable plan. Training uses two numerical threads and one inter-op thread; Git uses one thread; aggregate active CPU is capped at four and available RAM must remain at least 8 GiB under one-second sampled supervision. Offline analysis uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, psutil 7.2.2 and one numerical thread.

Preflight worker ceiling is 1,800 seconds TOTAL, including 180 seconds for final archival; tests have a 180-second cap and pilots a 1,440-second cap within the shared remainder. Workflow limit is 40 minutes, bounded additionally by 2,370 seconds from the first step.

The main study requires its own frozen resource admission after observing all pilot costs. Its maximum worker allowance is 3,000 seconds including 180 seconds archival, tests at most 180, training at most 2,400, and final at most 180 seconds within the shared remainder. For prospective admission, estimate main training cost as 20 times the sum of the largest observed complete pilot elapsed time in each family, plus 60 seconds; this gives a twofold timing allowance for five runs at twice the updates. Admit only if that projection fits 2,400 seconds. This is an operational allowance, not a statistical cost prediction. Maximum main workflow is 60 minutes with a 3,570-second first-step deadline. Do not increase budgets after observing failed learning.

## Primary background and limits

The distinction between dynamic activation and learned parameter storage is made explicitly in [LSTM, Hochreiter and Schmidhuber 1997](https://www.bioinf.jku.at/publications/older/2604.pdf). Delayed copying after input disappears, episodic reset, and associative retrieval are distinct tasks in [Neural Turing Machines, Graves et al. 2014](https://arxiv.org/pdf/1410.5401). A [Differentiable Neural Computer, Graves et al. 2016](https://www.nature.com/articles/nature20101) adds explicit external read/write memory, a different mechanism from this fixed canvas. These sources motivate controls; the present recipe is not their benchmark replication or a novelty proof.

The historical phase3 curve is documented separately. It has different geometry, four ordered facts, rounded results, incomplete run provenance, different parameter counts/update schedules and no recoverable original weights. New measurements must retain their own identity and cannot retrospectively certify that historical curve.
