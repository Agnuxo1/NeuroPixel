# Item 12: continual-learning evidence and bounded diagnostics

## Decision and scope before execution

Only item 12 is open. Items 1–11 are closed as feasible investigations, with their capability failures and limitations retained. Historical H1 remains not supported. This protocol freezes new analysis and diagnostics before their execution. It does **not** preregister the already observed item-6 training outcomes or create an unseen final set.

The investigation distinguishes four propositions:

1. A model acquired useful performance on a task.
2. Its previous task performance survived subsequent updates.
3. A separately stored expert retained its own parameters, buffers and outputs.
4. The growing system still selected or combined suitable experts without a supplied task identity.

These propositions require different evidence. Exact preservation of an ineffective expert is not useful task retention; preservation of all expert outputs does not force a router or mixture to preserve its answer. The current experiment directly checks the third proposition, constructs a counterexample for the fourth, and reconstructs retrospective evidence relevant to the first two.

No six-task or eight-task learned benchmark is admitted here. The already audited item-6 hard-selection oracle bounds the binding accuracy of every saved individual expert: at most 29.1015625%/20.703125%/22.4609375% on A/B/C for seed 20, and 27.734375%/24.609375%/19.7265625% for seed 21. Thus those snapshots cannot supply high binding competence even with label-aware hard selection. This is a reason to avoid interpreting an expanded low-competence assay as successful retention. It is **not** a newly executed six-task preflight failure or a proof that a revised architecture cannot learn continually. A mixture of probability distributions is outside this hard-selection bound.

## Evidence available before this freeze

The historical source paths are [scripts/phase3.py](../../scripts/phase3.py), [neuropixel/phase3.py](../../neuropixel/phase3.py), and the three JSON summaries under results/phase3. Their single recorded A/B/C/A sequence has global acquisition scores 96.9%, 98.45% and 99.3%. After C, A/B are 49.1%/49.35%; after revisiting A, A/B/C are 98.5%/49.55%/49.35%. These rounded historic summaries provide no per-example arrays or matched multi-seed comparison. The expert-growth summaries end with two and three experts, different capacity and, for the recorded resonance revisits, a different update budget. Their mean selected-slot values are not routing accuracy.

The stronger provenance comes from item 6:

- Original experiment source: 08d0d52edd05da6835e71479f3ba4399fcbeabae.
- Pinned raw archive: 15e76456bc2b4cce5faec0b08fb5288fe7844547.
- Directory: results/research/06_cloud_runs/37566497890-1.
- Two seeds, 20 and 21; one split; three noun-disjoint topics.
- Three fixed sequential checkpoint descendants per seed, after A, B and C; 512 updates per stage, with optimizer reset.
- Three final datasets of 1,024 examples each, 256 per queried role, and a saved 3-by-3,072-by-35 array of expert log probabilities for each seed.

The source of those saved outputs precedes item 8's effective PAD correction. Reanalysis preserves its bytes and original interpretation. The new native-copy diagnostic uses the current source. Their results must not be represented as a single retraining experiment.

The frozen input specification contains exactly 15 selected files: the original final manifest and worker status, growth records, partitions, three final datasets, two expert-output archives and six checkpoint files. Checkpoints are hashed, never deserialized. The original full item-6 archive audit is separate evidence; selecting 15 artifacts does not independently repeat that whole audit.

## A. Retrospective task-performance matrices

For each seed, reconstruct all nine checkpoint/task pairs from the saved outputs. The evaluation row order is the concatenation of topic A, B and C. Verify the visible canvas, role marker, adjacent filler, query position, token categories, balanced role counts, frozen TEST triple membership and exact saved label ordering. Verify recorded checkpoint lineage and byte identities. Use the argmax across all 35 output entries, with lowest-index tie breaking.

Let R[i,j] be performance on fixed evaluation task j after training through task i, using zero-based indices. Retain six matrices: global accuracy, macro-average AGENTE/PACIENTE accuracy ("binding"), and each of AGENTE, ACCION, PACIENTE and LUGAR. Retain global correct counts, per-role correct counts and cross entropy for every pair.

For each stage k:

- Average seen-task accuracy: mean of R[k,j] for j = 0,...,k.
- BWT: mean of R[k,j] - R[j,j] for j < k.
- Max-past forgetting: mean of max(R[l,j], 0 <= l < k) - R[k,j] for j < k.
- Post-acquisition forgetting: a separately named variant restricting the maximum to j <= l < k.
- At the first stage, the three old-task summaries are undefined and saved as null.

Both forgetting quantities are signed; improvements can yield negative values. BWT uses each acquisition diagonal, whereas the two maxima can detect intermediate gains that were subsequently lost. The included hand-written nonmonotone example distinguishes all three measures. This is the newest-descendant checkpoint sequence; it is not an invented intermediate trajectory for the final bank's router.

GEM defines backward and forward transfer using a task-performance matrix; its forward-transfer comparison needs an initial baseline [1]. No such initial prediction baseline is saved here, so FWT stays null. RWalk defines forgetting against the maximum over all earlier observed checkpoints and distinguishes failure to learn from forgetting [2]. The post-acquisition variant is explicitly labelled as an additional definition.

No new p-value, confidence interval, bootstrap or confirmatory hypothesis test is added. The two already observed seeds are descriptive. The 18 metric rows, three topics, role queries and repeated views are not extra training replications. The report must show acquisition and final performance together; small forgetting from weak acquisition is not success.

Recount the saved label-aware "any individual argmax correct" oracle per topic as a consistency check. It is an upper bound on a selector of individual hard answers, not on all possible ensembles. A task-ID route is a privileged diagnostic and is not automatically the highest-accuracy individual expert.

## B. Native storage-isolation diagnostic

Use current NeuroPixel, CPU float32, vocabulary 35, 3-by-3 canvas, output (2,2), c_id 16, c 48, hidden 128, four recurrent updates, fire_rate 0.5. Initialize with Torch seed 120001. Use evaluation mode with gradients enabled, so stochastic firing is disabled. No retina or grounding mutation is introduced.

The fixed fixture has 12 examples: the target noun IDs 5 through 16 are visible at (0,0), query token 1 at (2,1), and every other cell blank. This fixture is an integrity witness, not a held-out competence assay.

Create the first expert, then successively deep-copy the newest expert until the bank has eight. At each of the eight stages, perform three SGD updates of the new expert on the same fixture (learning rate 0.01, momentum 0, weight decay 0). There are 24 positive-control optimization steps. Record the actual losses and verify finite losses and gradients. Do not infer task acquisition from a changed loss or output.

Before and after each update group, save every named parameter and every named buffer, including nonpersistent grounding buffers, together with all bank logits on the fixture. state_dict alone is insufficient for the complete buffer census. Hash sorted tensor names, shape/dtype headers and bytes. Check that every old expert's full tensor fingerprint and logits are exactly unchanged, and that the updated expert's tensors and logits actually changed. Record the live process's storage alias classes to check cross-expert disjointness.

Report the inventories at K = 1, 2, 4 and 8, while preserving all eight transitions. Each ungrounded expert is expected to have 29,824 parameter elements, 119,296 parameter bytes and 455 buffer bytes: 119,751 bytes of named tensor payload. Eight experts therefore require 958,008 bytes of that payload. These are analytic tensor counts to be checked, not total process RAM, allocator use, optimizer state, gradients, energy or an information-capacity estimate.

The negative control deliberately puts the same object in two list slots. Three additional SGD updates must change the old slot's tensors and both slots' outputs, while the two slots stay mutually identical. This establishes that the test detects sharing. Total new diagnostic SGD steps are 27. The focused unit contracts separately execute one positive and one alias-control SGD step; they are not extra learned-task runs.

Save initial and final CPU Torch RNG arrays and independently compare them. These snapshots delimit the deterministic diagnostic after model initialization; unchanged RNG says nothing about training-mode stochastic firing. The diagnostic checks native deepcopy/create behavior. A policy that revisits and updates an existing expert can still alter that expert's old-task behavior.

## C. Literal routing counterexample

Freeze eight two-class probability tables on two labelled examples, y = [0,1]. Expert 0 is correct: [[0.9,0.1],[0.1,0.9]]. Each other expert is opposite: [[0.1,0.9],[0.9,0.1]]. Never change any of these outputs.

For K = 1,2,4,8, report correct counts for selecting expert 0, selecting the last expert and averaging all K probability tables. The expected counts are [2,2,2,2], [2,0,0,0] and [2,1,0,0], respectively. The K=2 mixture ties and uses the lowest class index. These are constructed numbers, not measurements from trained NeuroPixel experts. They demonstrate why stable expert outputs alone cannot certify stable system output under a changed selector or mixture.

Progressive Neural Networks similarly isolate old columns, but also use lateral connections and explicitly discuss parameter growth and task identity [3]. Native expert copying here does not establish the progressive-network transfer mechanism. The Task-IL/Domain-IL/Class-IL distinctions depend on available task identity and the inference problem [4]; three noun-disjoint topics with shared other roles should be described explicitly, not assigned an unqualified standard label.

## D. Independent recount and operational controls

Use four fresh interpreters in a fixed serial order: secondary producer, native isolation producer, independent matrix auditor, independent isolation auditor. The matrix auditor imports neither the producer nor the metric helper. The isolation auditor imports no Torch, model or producer. Both independently reconstruct numerical claims from saved arrays and identities. They execute on the same authorized runner and do not constitute external independent replication.

The matrix auditor additionally verifies partition coverage, visible gold and normalization. It preserves all five produced files and compares JSON numerical rows; root review will compare the CSV rows. The isolation auditor independently reconstructs fingerprints, tensor counts, old/new equality, actual mutation, saved RNG equality and the reported storage-alias arithmetic. It does not independently observe live memory pointers, replay SGD or certify an optimizer trajectory.

Eight focused parent test methods are frozen: four metric contracts and four storage-isolation contracts. All collected parents must pass with no skips or failures; preserve subtest events separately from parent counts. Preserve every component log/status, source bindings before/after, input bindings, source archive, full result manifest and workflow outcome.

The standard public CPU workflow has a 20-minute provider limit and an earlier 1,170-second wall deadline. Worker allowance is 900 seconds including a 180-second archive reserve. Contract tests have at most 120 seconds; the shared scientific probe has at most 540 seconds, including input recovery, with each component limited to at most 120 seconds. These are ceilings sharing one deadline, not guaranteed allocations. Use one numerical thread, one inter-op thread and one Git thread; keep aggregate active CPU threads at most four and available RAM at least 8 GiB. The enclosing worker supervises the owned process group every second and emits heartbeats. Failed admission or execution is preserved and is not a scientific null result.

## Work still needed for a substantive continual-learning claim

A subsequent learned study requires independently verified initial acquisition on each proposed task, a prospectively fixed threshold and finite pilot budget before main outcomes. Once feasible, compare a shared sequential model, a shared model with a fixed rehearsal budget and an isolated expert bank. Prespecify supplied-task-ID and inferred-route settings, task orders, overlaps, recurrence and held-out content. Count all experts, router parameters, buffers, optimizer state, rehearsal examples, data exposures and elapsed computation. Evaluate acquisition curves, retention matrices, final end-to-end performance and failures across independent seeds and task constructions.

GEM, EWC and A-GEM provide different memory/constraint/regularization approaches and evaluation considerations [1,5,6]. They motivate appropriate controls; they are not implemented or claimed as beaten by this diagnostic. No biological mechanism, general continual-learning solution, energy advantage, broad scaling law or Nobel-level result follows from storage isolation.

## Primary sources consulted

1. Lopez-Paz and Ranzato, *Gradient Episodic Memory for Continual Learning* (2017), section 2, equations 2–4, and section 3. https://arxiv.org/pdf/1706.08840
2. Chaudhry et al., *Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence* (2018), section 3, equations 3–4. https://arxiv.org/pdf/1801.10112
3. Rusu et al., *Progressive Neural Networks* (2016), architecture and limitations. https://arxiv.org/pdf/1606.04671
4. van de Ven and Tolias, *Three Scenarios for Continual Learning* (2019), sections 2–3. https://arxiv.org/pdf/1904.07734
5. Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (2017), parameter-importance regularization. https://arxiv.org/pdf/1612.00796
6. Chaudhry et al., *Efficient Lifelong Learning with A-GEM* (2019), evaluation protocol and computation/memory discussion. https://arxiv.org/pdf/1812.00420

The source and data evidence is additionally traceable through docs/research/06_results.md and the immutable input specification in 12_probe_plan.json. This protocol makes no literature-exhaustiveness or priority claim.
