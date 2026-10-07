# Item 13 — Vocabulary expansion, new-token labels and distractors

## Status and scope

This protocol is frozen before the first item13 numerical execution. It follows the closure of item12 and opens no item14 work. The source branch remains isolated from main. The preserved item5 hypothesis remains closed and not supported.

Three questions are investigated separately:

1. Does allocating new vocabulary rows preserve the existing native model's configuration, parameters and grounding?
2. Do historical positive-only new-token evaluations identify learned semantic binding, and do transformed labels/splits remain valid?
3. How often did already archived item9 models retain correct answers when irrelevant facts were exchanged?

The first is an implementation contract; the second includes dataset and literal counterexample checks; the third is a descriptive reanalysis of previously seen predictions. None is a fresh confirmatory test of semantic acquisition. No new learning benchmark is admitted: prior item9/item10 binding or acquisition weaknesses remain material, and successful code contracts cannot satisfy that missing scientific condition.

## Source observations made before freezing

The source parent is 3058665ffcaf8a0c4c1d91313cf598d42fabee1a. Historical neuropixel/phase3.py has Git blob b826869ad1816f43a182fbcdbcbc47ea2a123dd5 and SHA256 57c177757b333ee5ba9fdc953cbbd7ef284892edee612e234cca1df83b6814d0. Its expansion function deep-copies the model but replaces both grounding buffers by zeros; newly allocated modules use the default floating dtype. Full preimages are retained, and only the relevant function AST is executed under the current core as a legacy regression witness. This is not an integral replay of historical experiments. The archived new-word callers reconstruct models without explicit grounding; their published scores are not attributed to this grounding defect.

The second expansion implementation in scripts/battery.py, blob cd9670ce42116524fb7b3de6f25c493259f359a3, recreates models using constructor defaults, including a square-grid inference for the Transformer. The same driver's with_new_word(..., ask_new=False) changes a filler while retaining the old target even when that filler is queried. Historical default calls use ask_new=True; this source defect is not attributed to the historical reported scores.

The t_newword and fewshot drivers in scripts/phase3.py insert one new token, force the query to its inserted role, and set every target in their new-example panel to that token. A constant answer solves that positive-only panel; an input-only presence gate combined with the unchanged old predictor also preserves the separate old-panel result. This is a constructive identification problem, not evidence that the trained model actually used that shortcut.

Recorded np_lens30k/norma/k5 scores are 0.986 new, 0.926 old, with old-before 0.9795. These remain historical aggregates, not results of this execution. The README compares that selected new-token maximum with a different plain five-shot scaling procedure for the Transformer. No historical adaptation checkpoints or per-trial predictions are reconstructed from aggregate JSON.

## Minimal implementation corrections

- Native expand_vocab preserves old grounding prefixes and appends ungrounded zero rows, retaining nonpersistent-buffer behavior.
- New vocabulary modules preserve their original devices and floating dtypes; the boolean gradient row mask follows the actual gradient device.
- The native return API is unchanged. Only positive integer additions and the two exact native classes are supported; subclasses with extra vocabulary tables need a separate migration policy.
- Fresh optimizers without weight decay or inherited momentum can use the masked gradients as the historical callers do. A gradient hook is not an optimizer-level freeze. Old optimizer objects still reference the original parameters and are not migrated.
- The battery expansion helper delegates to this native policy. This deliberately changes its former constructor/default/RNG path; historical battery initializer draws are not claimed identical.
- The unforced-query battery branch preserves the query and recalculates the target only when the replaced role is queried. False means “do not force the new query”, not “the new token is guaranteed irrelevant”.

The pure phase3 float32 initialization path is checked against its legacy preimage under a restored RNG state. Configuration/grounding preservation is the intended correction; old scientific scores are not overwritten.

## Stage 1: twelve focused contract methods

Two files contain eight native-expansion methods and four battery methods. The worker requires exactly twelve collected parent methods, no skipped tests and successful completion. Subtests are counted separately in the final receipt; they are not additional independent scientific experiments.

Native contracts reproduce legacy grounding loss and float64 forward failures, preserve NP/Transformer old rows and old-input logits/configuration, verify float32 initialization/RNG compatibility, demonstrate one fresh Adam step with zero decay, inspect half-precision gradient masking after conversion, perform repeated expansion, reject invalid requests before RNG use, demonstrate softmax/global-argmax changes with preserved old logits, and include an AdamW-decay negative control.

Battery contracts enumerate all four queries by both insertion roles in an eight-row manual fixture, compare legacy and corrected labels, retain the forced-query behavior, and exercise rectangular/nondefault Transformer and grounded nondefault NP configurations. The legacy reconstruction failures are expected controls, not hidden run failures.

Exact equality is used for preserved tensor prefixes, masks, old state and integer labels. Old float32 logits/lens comparisons use rtol=1e-5 and atol=1e-6 to allow the changed matrix shape's rounding. Specified float64 readout comparisons use rtol=atol=1e-12. No half-precision convolution or GPU support is inferred from a gradient-only half test.

The new worker uses pytest -s so successful tests' legacy fault witnesses remain in the archived log. A source-review correction to an optional-import test default is preserved separately; it is not an observed cloud failure.

## Stage 2A: retrospective irrelevant-event recount

The pinned original scientific source is 83fe135e301f76bc0c74e30c66bb18e067ca5959. Raw evidence is the final archive b64d0e2ba222df76caf72bc9870c7602873e7236, directory results/research/09_cloud_runs/37593731891-1-study.

Exactly twelve input files are byte-bound in the plan: the outer archive manifest, one final dataset NPZ and ten prediction NPZ files for NeuroPixel and relative Transformer seeds 40–44. No checkpoint is loaded, no model is imported into these fresh NumPy subprocesses, and no new inference, bootstrap or confidence interval is requested.

The original condition order is verified from the historical generator:
base; swap_queried_agent_patient; swap_other_agent_patient; relabel_events; query_switch; layout_permutation.

Only base versus swap_other_agent_patient is recounted. Both contain two events; the transformation exchanges agent/patient fillers in the event that is not queried. It preserves every target, the query, token inventory and layout. It perturbs existing irrelevant content, rather than adding a distractor to an otherwise distractor-free input.

The program validates identities, array shapes, record/group/role pairing, the gold-preserving transformation, and saved predictions against argmax logits. It produces one table for each seed/family, separately for all roles, binding (agent+patient) and each of four roles. N is 2048 overall, 1024 binding and 512 per role; the underlying shared units are 256 bags, not 2048 independent sampled datasets.

For every table record:
- CC: both correct;
- CW: base correct, transformed wrong;
- WC: base wrong, transformed correct;
- WW: both wrong;
- prediction agreement and stable wrong agreement;
- retention conditional on a correct base answer, CC/(CC+CW), or null when that denominator is zero.

All raw counts and denominators remain visible. Means across the five training realizations are descriptive, not fresh inferential estimates. A separate NumPy program reopens the exact raw NPZ inputs and verifies the produced counts without importing the producer. The existing audited JSON projection may be used by the root for an additional independent arithmetic comparison; it does not substitute for raw recounting in the worker.

## Stage 2B: constructed labels and mechanism witnesses

### Label census

Use 32 constructed scenes on an 8x8 grid, all four role queries per scene, and six conditions, totaling 768 examples:
base; one new agent; one new patient; two distinct new noun tokens; an unpaired old noun; an unpaired new token.

The four meaningful facts are unique horizontal role-marker/filler pairs. A query occupies (7,6), the output (7,7) remains empty, and an unpaired distractor is placed in the first blank cell of the first seven rows. It cannot overwrite a fact, query or output or create another role marker. A parser independently scans the visible unique role pairs, without using generator metadata, and verifies every transformed gold.

The two new IDs are 35 and 36. Literal controls include constant 35, “answer 35 if visible, otherwise use a symbolic solver”, and the symbolic solver. The fallback is explicitly an oracle construction used to prove that old-only and new-positive panels cannot identify a learned binding rule; its scores are not NeuroPixel scores. Counts are reported for all questions, agent/patient binding, new targets and old targets. Empty subsets retain a denominator of zero.

### Transformed split census

Construct the current legacy RoleTask 8x8 split with seed 0 and retain all 1056 TRAIN and 264 TEST ordinal triples. Exhaustively project each triple under agent replacement, patient replacement, or both, and count intersecting transformed keys.

The original values are noun/action indices, with new-token sentinels 35/36. This is a census of triple projections, not a claim that particular historical five-shot support canvases and evaluation canvases were identical. Place, spatial layout, the exact historical RNG/software environment and actual adaptation draws are not equated.

### Softmax identity

Two literal float64 fixtures contain two rows each, four old logits (including the overridden PAD logit -10000) and either one or two appended logits. Targets remain old. Check:

p'_old = p_old * Z/(Z+A)
NLL' - NLL = log(Z+A) - log(Z).

Record all logits, probabilities in log form, new-class probability mass and old/global argmax. Reconditioning the expanded probabilities on the old classes recovers the old distribution; this does not mean the expanded readout itself stayed unchanged. These are literal arithmetic fixtures, not learned scores, and no embedding-average bound is asserted.

### Consistent renaming

For each of a tiny NP and tiny Transformer, initialize with seed 130003; fix PAD and role IDs while exchanging noun IDs 5/6 and 7/8. Apply the same old-to-new permutation to input IDs, embedding rows, grounding rows and masks, and the Transformer's output rows/biases.

Use two 3x3 inputs, native float64 models in evaluation mode, dimensions c_id=4, c=8, hidden=12, steps=2 for NP and d=8, layers=1, heads=2, ff=12 for TF. NP's otherwise-zero residual update layer is manually assigned small nonzero weights/biases to avoid a quiescent fixture; it remains untrained. Record both inputs, all output logits, NP states and alignment residuals. Aligned logits must agree at 1e-12 tolerance and NP state exactly.

This is an isomorphism under a simultaneous change of model tables and input coding. It is not generalization by an unchanged model or acquisition of a new meaning. Root JavaScript independently checks saved JSON labels, counts, split intersections, softmax calculations and output/state alignment; it does not independently reimplement Torch inference.

## Execution, failures and custody

Only one standard public CPU workflow is reserved prospectively, with a 20-minute outer deadline. Python 3.12.14 and all package versions are pinned in the JSON plan. Numerical, interop and Git operation thread counts are 1; aggregate active CPU capacity is bounded by 4 and available RAM must remain at least 8 GiB.

Worker budget 900 seconds includes 180 seconds reserved for archival. Contract stage 120 seconds; combined probe 540 seconds, including pinned-input recovery and three serial fresh subprocesses, each at most 120 seconds. The existing owned-process-group supervisor samples resources every second, emits a 60-second heartbeat and admits only finite work. Evidence is archived between stages and at 300-second intervals during longer stages.

The worker verifies committed source/plan hashes, clean source and all bindings before and after execution; writes a source ZIP, logs, resource observations and result manifest; and confirms each results-branch push. The final run archive is immutable. Worker and provider failures remain part of the record and do not justify silently rerunning a changed hypothesis.

The separate auditor and root recount provide independent implementations within one investigation. They are not replication by an outside laboratory. Finite contracts cannot prove universal correctness for arbitrary future callers or all dtype/device/optimizer states.

## Primary sources consulted

- Press & Wolf (2017), Using the Output Embedding to Improve Language Models, section 3: https://arxiv.org/pdf/1608.05859 . Input/output tying couples two roles of one table; its language-model results are not transferred to this task.
- Hewitt (2021), Initializing New Word Embeddings for Pretrained Language Models: https://www.cs.columbia.edu/~johnhew/vocab-expansion.html . Author's technical note, not a peer-reviewed paper; normalizer change after vocabulary expansion. No claim that an initialization grants semantic competence.
- Lake & Baroni (2018), Generalization without Systematicity, experiment 3: https://proceedings.mlr.press/v80/lake18a/lake18a.pdf . Learning a primitive correspondence and using it in unseen compositions are distinct evaluations.
- Shi et al. (2023), Large Language Models Can Be Easily Distracted by Irrelevant Context, sections 3.1–3.2: https://arxiv.org/pdf/2302.00093 . Operationally verify gold-preserving irrelevant additions and grouped variants; the published percentages are not a baseline for NeuroPixel.
- Mirzadeh et al., GSM-Symbolic, ICLR 2025; consulted arXiv v2 dated 2025-08-27, sections 3.1/4.2/4.4: https://arxiv.org/pdf/2410.05229 . Different symbolic transformations require distinct target and validity checks; no universal reasoning claim is inferred.

Read on 2026-10-07. This is a focused primary-source review, not an exhaustive systematic review or proof of priority. Item13 closes when source fixes, bounded tests, recount, independent verification and scoped reporting are completed; unestablished semantic acquisition remains explicitly pending as a scientific capability.
