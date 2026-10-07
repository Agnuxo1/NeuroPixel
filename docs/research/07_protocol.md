# Item 7 — Development, selection and final assessment

This item starts after item 6 closed in commit `34ec75bad57825ae71cc7772ed236efb17cfcc94`. It investigates data identity and exposure, strengthens the prospective evaluation interface, and illustrates selection bias under a fully specified null model. It does not train NeuroPixel, reopen H1, or designate any historical dataset as a new independent confirmation. Items 8–30 remain unopened until this investigation closes.

## Questions and acceptance criteria

1. **Partition semantics.** Reconstruct the complete ordered universe of 1,320 directed `(agent, verb, patient)` triples with unequal nouns. For the exact exported Torch permutations, verify legacy 1,056/264 and research 924/132/264 boundaries, disjointness and coverage. Verify the three topic-filtered research partitions against the archived item-6 file. These are exact finite-set checks, not significance tests.
2. **Exposure across revisions.** Retain all declared legacy split seeds 0/1/2, research seeds 0/101/202 and growth topics at split 0. Report every relevant pairwise pool intersection, prior-training union and final-pool overlap. Pool eligibility is not evidence that every member appeared in a historical minibatch. The export runtime is Torch 2.6.0+cpu; compatibility with unarchived complete pools from other historical runtimes remains a stated condition. Do not infer unseen historical lists from the few recovered tensor hashes.
3. **Prospective local contracts.** Reject cross-partition group/content overlap, identity changes, incomplete candidate inventories, inconsistent validation populations, nonfinite metrics and changed source/configuration/checkpoint files. Persist a final-access receipt before resolving or reading final data; failed access remains consumed. Preserve the complete failed initial denominator test and its correction. Verify the development sampler agrees with the existing training/validation sampler and refuses final and inherited unsupported routes.
4. **Selection illustration.** Under equal true accuracy for all fixed candidates, compare the development score selected for being high with the independent final score of the same candidate. Record all four predeclared search sizes, analytical expectations and Monte Carlo precision. This illustrates a known statistical mechanism and does not estimate actual bias in NeuroPixel results.
5. **Closure.** Archive source, complete successful and failed test evidence, the permutation export, independent finite-set audit, simulation arrays and all summaries. Review numerical counts and interpretation against their artifacts. A closed investigation does not certify historical blindness, independent external replication or a Nobel-level contribution.

## Evidence and scientific interpretation

The source audit is [07_gate_source_audit.md](07_gate_source_audit.md), anchored to the closed item-6 source. The primary-method review is [07_methodological_evidence.md](07_methodological_evidence.md). It reads Cawley–Talbot, Varma–Simon, Dwork and colleagues, Russo–Zou, Recht and colleagues, and Kapoor–Narayanan directly and records the versions and relevant sections.

Training fits parameters; validation supports selection; final assessment evaluates the previously fixed procedure. Model selection itself is part of the procedure being assessed. Reusing observed final scores to redesign that procedure adds an adaptive dependency. Reporting all predeclared comparisons can remain informative, whereas relabelling the winning reused score as independent confirmation cannot remove that dependency. Nested evaluation is appropriate when the target includes repeating the selection procedure; it is not automatically required to estimate one already-frozen predictor on new independent draws.

The adjacent task groups examples by directed triple. Places, layouts and queries are generated within each pool; nouns, verbs and the renderer are shared. Different layouts of the same triple are not new compositions. A stronger claim about new templates, unordered role-swap families, vocabulary or domains requires a different grouping/estimand. That distinction does not automatically make a directed compositional split invalid for its stated target.

A public generator can produce fresh random instances of a known process after a predictor is frozen. Such a sample can estimate that predictor's performance on the stated process. It does not establish novel support, a new generator, independence across fitted models, or secret labels. Public seeds and hashes establish reproducibility and recorded identity, not independent custody. Framework reproducibility also depends on runtime: [PyTorch 2.6 documents cross-release/platform limits](https://docs.pytorch.org/docs/2.6/notes/randomness.html), and [NumPy 2.3 states its random-stream compatibility conditions](https://numpy.org/doc/2.3/reference/random/compatibility.html).

## Frozen membership export and software validation

The exact source inventory and cloud recipe are in [07_cloud_export_plan.json](07_cloud_export_plan.json). One standard public Ubuntu 24.04 runner uses Python 3.12.8, Torch 2.6.0+cpu, the existing pinned research dependencies and one numerical/inter-op thread. Two suites contain 17 functional governance checks and four development-sampler checks. The latter use tiny train/validation fixtures; no final task examples or model predictions are requested.

The exporter constructs each private CPU generator afresh for seeds 0, 1, 2, 101 and 202 and records `torch.randperm(1320)` plus independently instantiated legacy/research memberships. It does not substitute a Python or NumPy shuffle. It does not call a sampler, construct a model, load weights or optimize parameters. The separate standard-library auditor reconstructs the ordered universe, slices and topic filters without importing Torch and compares all lists and set relations, including the archived growth partitions with SHA-256 `eff760e91613cd88a75f05231e11fe811674f4ef76d5469a412b9e5ea26f629f`.

Cloud stages require at least 8 GiB available RAM, are supervised each second, and have a 900-second limit each. Only their own process groups may be terminated. The outer workflow limit is 35 minutes including setup and archival. Both ordinary test/export failures are retained; no model performance is contingent on these checks. All resulting files and an exact source ZIP go to the existing isolated results branch using a normal fast-forward push. No paid runner, dependency cache or Actions artifact-storage upload is used.

## Frozen null simulation

[07_execution_plan.json](07_execution_plan.json) binds the complete recipe and source/method-note hashes. The runner additionally requires NumPy 2.3.5. Its fixed design is:

| Quantity | Fixed value |
|---|---|
| Independent Monte Carlo replicates | 10,000 |
| Candidate procedures | 64, each with true accuracy 0.5 |
| Observations per partition/candidate | 256 Bernoulli correctness observations, represented by binomial counts |
| Search sizes | 1, 4, 16, 64 |
| Random streams | PCG64; TRAIN 71001, DEV 71002, FINAL 71003 |
| First selection stage | Rank TRAIN counts descending; lower candidate ID breaks ties |
| Second selection stage | Highest DEV count in each nested TRAIN shortlist; lower ID breaks ties |
| Final generation | Only after selected IDs and their receipt have been durably saved |
| Secondary tail summary | Accuracy at least 0.6, equivalent to count at least 154 of 256 |
| Analytical diagnostic | Flag an absolute discrepancy greater than six analytical Monte Carlo standard errors; never rerun in response |

For shortlist size M and binomial CDF F, the expected winning development score is `sum(k=0..255, 1 - F(k)^M) / 256`. The final score of the selected candidate has expectation 0.5 and variance 0.25/256 because selection occurs without its independent FINAL counts. Second moments of the maximum give an analytical variance and hence an analytical simulation error scale. Paired development-minus-final differences preserve replicate pairing. All four conditions share some selected candidates and final counts; they are not four independent scientific replications.

The design is intentionally a measurement-level model with independent candidate errors and identical population competence. It performs no learning, nested cross-validation, privacy mechanism or real-world replication. A selected development score above 0.5 would be a consequence of selection from noisy estimates, not an improvement in the stipulated population accuracy. The simulation does not quantify how much, if any, adaptive optimism occurred in any historical NeuroPixel result.

The runner records complete count matrices, selected IDs, 40,000 per-replicate/condition CSV rows, all summaries, analytical checks, timestamps, source/plan/runtime identities and resource observations. The summary intervals express Monte Carlo precision of simulated means, not uncertainty about an architecture's real performance. The one-thread local run is admitted only with at least 8 GiB available RAM; source and output remain in the isolated scientific workspace. Existing output directories cannot be reused. A failure is retained for investigation, not silently replaced.

## Prospective implementation and migration boundary

`neuropixel/research/split_governance.py` provides a content-bound manifest and `FinalStudy` contract. The manifest identifies generator/source, vocabulary, shape, RNG/runtime recipe, ordered records, declared group rule, transformations, fitted-preprocessing population and prior exposure scope. Candidate records are tied to a common validation artifact and denominator count. Source/configuration/checkpoint hashes are checked again before the final receipt. The verified final bytes are passed directly to an evaluator callback, avoiding a second file read between checksum and callback. An external immutable plan must preserve the freeze-receipt hash.

`neuropixel/research/development_data.py` is an opt-in train/validation adapter. It rejects final/test names and inherited loop/camera routes, and allows `frozen_dataset` only through its restricted `sample` method. The old public APIs remain available for reproducing historical experiments. They have not been silently replaced. Future drivers must explicitly adopt the new surface or provide an equivalently audited protocol.

These contracts cannot verify that caller-supplied grouping labels match actual semantics, that validation metrics were honestly calculated, that no outside program accessed public files, or that a protocol was committed before supplied selection metrics. The upstream grouping/metric implementation, explicit exposure evidence and actual freeze chronology must be reviewed. Complete-history declarations are conditional on their stated scope. Unknown or previously exposed final groups are refused in confirmatory mode and retained as such in exploratory mode; a new directory does not clear a supplied exposure history.

All seven recovered historical scientific files remain unchanged. H1 remains not supported. Any later change in metric, generator, architecture or evaluation target must be versioned and frozen under its own programme item.
