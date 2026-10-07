# Item 16 — Prospective scanner and causal-information protocol

## Question and scope

Can the repository's dictionary scanner be read as a complete, causally faithful account of the recurrent computation? This point separates five questions: what the projection displays; when it is sampled; what the token/confidence display discards; whether an invisible current state difference can affect a later answer; and the ambient rank of the actual learned readout matrices.

The investigation includes three regression contracts, deliberately assigned native counterexamples, and retrospective matrix analysis of five previously archived models. It contains no new training, no learned forward pass, no new benchmark score and no search for a semantic circuit. The native model core is unchanged from item 15. An outcome can close the feasible investigation without establishing the broader capability. Historical H1 remains closed and not supported.

The numerical recipe, exact source bindings, test inventory, input hashes, tolerances and resource budgets must be frozen in `16_probe_plan.json` before the first run. This narrative accompanies that machine-readable contract. Source preparation and mathematical deductions are not reported as experimental outcomes.

## 1. Mathematical target

For one cell with recurrent state s in R^C, the raw vocabulary logits are

    z(s) = D (R s + b),

where R has c_id rows and C columns and D is the effective dictionary. For any delta in ker(R), z(s+delta)=z(s). More generally, the full instantaneous decoder has matrix M=DR, so its ambient kernel can be even larger. With the default C=48 and c_id=16, its rank is at most 16 and its ambient kernel has dimension at least 32. Adding readout bias does not change this difference calculation.

These are statements about the ambient instantaneous linear map. They do not establish how many independent directions are reached during learned inference or prove that task-relevant information is encoded in the kernel. Nor do they determine observability from several times with a known transition rule: in a linear system, that question involves the stacked matrices C, CA, CA^2, etc., not C alone. The constructed controls below deliberately distinguish a hidden direction that later couples to the readout from a hidden direction that remains unused.

The scanner returns only the most probable token and its probability. Distinct complete logit vectors can give the same retained pair. We do not claim that softmax necessarily loses information under a fixed PAD anchor: with its finite logit fixed and positive probability retained in exact arithmetic, the full probability vector can identify the remaining logits. The additional compression claimed here concerns the actual top-token/confidence output and the rendered image, which does not retain that full vector.

## 2. Existing measurement and display contracts

Native `forward(trace=True)` returns the seed at frame 0 and the post-update, post-hook state at frame t. Native `lens_every=k` samples before update t, when t is a multiple of k. Thus those training samples correspond to s[k-1], s[2k-1], etc. The scanner applied to trace frames has a different, explicit time index. The test uses changing states and an intervening hook so a mistaken index cannot pass because every state happens to be identical.

Both the standard scanner and final output overwrite the PAD score with -10000. This is a finite numerical convention. It suppresses PAD in the ordinary negative-score witness, but does not absolutely exclude PAD if all other scores are below -10000. The core convention is preserved and its limit is recorded.

The historical animation helper used the unmasked raw lens. The prepared correction delegates to the standard scanner. A regression runs the actual current generator and the preserved historical function with the same small native model and task fixture. Only the image/font/drawing sinks are substituted to observe the tokens passed to rendering without recreating historical image artifacts. No scanner or native forward is mocked. The ordinary negative-score case should give 2112 empty labels for the historical path and 2112 token 1 labels for the corrected path, across 33 displayed frames and 64 cells. This is an integration control, not a reproduction of the trained historical animation.

The scanner's documentation is also corrected: its readout can receive auxiliary supervision at nonempty input cells, and in the image trainer at image cells. Using the learned output head throughout the grid is an implemented feature, but auxiliary readability is not by itself proof of a causal explanation.

## 3. Assigned native causal counterexamples

Use the original `NeuroPixel` class in evaluation mode with V=4, c_id=3, C=4, hidden=8 and a 1x1 grid. Set the effective dictionary to the zero PAD row followed by the three Cartesian basis vectors, and set R=[I3|0], b=0. Every parameter of the constructed model is declared explicitly; no optimizer is used.

Two initial states are A=(1,0,0,0) and B=(1,0,0,2). A post-update 1 hook installs each state, preserving frame 0 as the ordinary initialization and frame 1 as the intervention boundary. One further ordinary native update gives frame 2. The raw current logits and complete current scanner distributions of A and B coincide.

Under the assigned coupled rule, the update adds ReLU(s3) to coordinate 1. B can therefore change the next readout while A does not. The negative control assigns a zero update, so neither state changes its readout on the next step. Both runs preserve the future all-PAD input. This establishes conditional causal influence in the assigned fixture; it does not identify a learned semantic variable or show that B lies on a trained model's reachable manifold.

Retain all native frames, raw and PAD-suppressed logits, complete probabilities, scanner top labels/confidences, sampled pre-update lenses, declared states, and parameter/RNG witnesses. The main contrast uses logit differences as well as categorical output. An exact no-change comparison is retained for the inactive coupling.

Separate PAD witnesses assign ordinary negative non-PAD logits and extreme logits below the finite sentinel. A separate compression witness uses first-three state coordinates log(0.6,0.3,0.1) versus log(0.6,0.1,0.3), with zero hidden coordinate. The most probable token and its confidence coincide within the declared tolerance, while the other scores differ. This demonstrates compression by the reported summary without conflating it with the linear kernel.

## 4. Retrospective learned matrix census

Use only the five gate NPZ files and their JSON descriptors from item15B, plus that archive's manifest and checkpoint-binding receipt. The twelve input files total 1,578,221 bytes. Their source is 771975b4833bfcf20e06f848253912c699b2152e and final archive 205507b71e2f7abb00ce97aa1dbec8321fc0af26, under `results/research/15_cloud_runs/37654405152-1-preflight`.

The item15B binder separately authenticated these arrays against the original item9 checkpoints. Here, first copy the declared regular Git blobs, verify every size and SHA-256, and verify the complete original manifest. Then read the NPZ arrays with pickle disabled. Do not load a checkpoint pickle, call a learned model forward, select new data or compute task accuracy.

For seeds 40–44, retain the readout and effective dictionary, the non-PAD composite matrix, singular values and null-space projectors. Use float64 for matrix diagnostics while recording the original stored dtype. The prospective numerical-rank threshold is max(matrix dimensions) times float64 machine epsilon times the largest singular value, with strict greater-than classification. Report the threshold and singular values so a near-boundary rank is visible. Rank does not measure the number of semantic concepts or the distribution of learned task information.

The independent auditor reconstructs the relevant matrices directly from the authenticated input arrays, recomputes the rank and projector properties, and checks saved evidence. It must not import the producer's numerical functions. Equivalent projectors are compared rather than individual null-space basis vectors, whose signs and rotations need not agree between singular-value decompositions.

## 5. Verification and failure handling

The three focused regression methods run first. Native computation follows read-only input recovery. A fresh NumPy auditor then reconstructs the assigned trajectories, display policy, compression contrasts, input-derived matrices, rank thresholds and invariant checks from raw saved arrays. Root performs an additional elementary JSON recount and checks the archive's declared text hashes and Git file inventory.

Every producer/auditor check has an explicit meaning. Counts of parent tests, subcases, checks, arrays, trajectories and observations must remain separate. We preserve a negative control or negative capability result as data; source, schema, resource or consistency failures remain failures. If a test or audit fails, retain the attempt and its inputs before making the minimum correction under a new frozen source. Do not silently replace a failed attempt.

Previous internal audits of the reused learned weights are provenance. This run's separate native and NumPy calculations are internal cross-checks, not replication by an outside laboratory. The unavailable original trained animation checkpoint prevents regenerating that historical picture; the current integration fixture must not be described as doing so.

## 6. Bounded execution

One standard public CPU workflow is permitted after freeze. Limits: 20 minutes at provider level; worker 900 seconds including 180 seconds for final archival; contract stage 120 seconds; shared diagnostic 540 seconds; recovery, native and audit each at most 120 seconds. Numerical, interop and Git threads are 1; aggregate active CPU ceiling 4; available RAM must remain at least 8 GiB. The supervisor owns the process group, samples resources every second, emits periodic progress and preserves partial/final archives.

No GPU, paid runner, billing/access change, main-branch merge, outside contact or nomination is part of this point. Release the numerical reservation after terminal provider status and the final archive are verified. Point 17 remains unopened until point 16's report is closed.

## Primary methodological references

- Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*, section 4 and limitations: https://arxiv.org/pdf/2303.08112. Lens-to-model causal alignment is tested separately from observational prediction. NeuroPixel's fixed shared head is not this paper's trained translator.
- Zhang and Nanda, *Towards Best Practices of Activation Patching in Language Models*, sections 2–6: https://arxiv.org/html/2309.16042v2. Corruption choices, metrics and out-of-distribution interventions affect interpretation.
- Heimersheim and Nanda, *How to use and interpret activation patching*, sections 2–4: https://arxiv.org/html/2404.15255v1. Restoring behavior and disrupting behavior answer different conditional questions; neither supplies universal minimality or scope.
- Elazar et al., *Amnesic Probing*, section 2.3: https://aclanthology.org/2021.tacl-1.10.pdf. Matched removal controls and correlated information matter when interpreting intervention effects.
- Hewitt and Liang, *Designing and Interpreting Probes with Control Tasks*: https://aclanthology.org/D19-1275.pdf. Probe accuracy requires controls on what the probe itself can learn. No separately trained probe is added here.
- M. Lemmon, *Linear Systems* notes, chapter 4, discrete-time observability on printed page 183: https://academicweb.nd.edu/~lemmon/courses/linear-systems/lecture-book/linsys-book-2024.pdf. The stacked observation map differs from an instantaneous decoder.
