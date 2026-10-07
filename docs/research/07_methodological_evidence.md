# Item 7 — Methodological evidence for development and final evaluation

**Prepared:** 2026-10-07 UTC. **Scope:** item 7 only. This is a bounded primary-source review and a prospective simulation specification. Root approved its scientific settings before outcomes; the discrepancy diagnostic is strictly greater than six analytic standard errors, and no simulation has yet been run by this author. It does not change the frozen item-2 protocol, reinterpret the closed H1 as supported, or authorize new model training. The simulation below has not been executed by this author; root must freeze its design and implementation before execution.

## 1. Evidence and access record

Six original research sources were selected for complementary coverage: finite-sample selection bias, nested assessment, adaptive reuse, information-dependent selection, new-test-set interpretation, and dependence between group variants. Full-text passages identified below were directly read, rather than relying only on abstracts. This is a targeted methodological review, not a systematic survey or a claim of exhaustive coverage. Access date for every source: **2026-10-07**. Source summaries below are original paraphrases.

| ID | Primary source and directly read location | Supported proposition | Limit of the evidence |
| --- | --- | --- | --- |
| S1 | Gavin C. Cawley and Nicola L. C. Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 11:2079–2107. [Article](https://www.jmlr.org/papers/v11/cawley10a.html); [PDF](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf). Sections 4–4.1 and 5–5.1, especially printed pp. 2084–2088 and 2094–2095. | Optimizing a noisy finite-sample criterion can overfit that criterion. Assessment must include the full fitting-and-selection procedure inside each outer evaluation split. | Their examples establish a mechanism and empirical risk, not the amount of optimism in NeuroPixel. Repeated partitioning is not a method for erasing earlier access. |
| S2 | Sudhir Varma and Richard Simon (2006), *Bias in error estimation when using cross-validation for model selection*, BMC Bioinformatics 7:91. [Publisher full text and DOI](https://link.springer.com/article/10.1186/1471-2105-7-91). Background, Results: “Nested CV with shrunken centroids and SVM,” and Conclusion. | Their null-data simulations show optimistic assessment when the CV score minimized during tuning is reported as performance. Repeating selection inside an outer loop brings assessment close to independent-test error. | Nested CV estimates a specified training procedure at its fold training sizes. It does not retroactively make already-used observations untouched or remove all sampling uncertainty. |
| S3 | Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold and Aaron Roth (2015), *Generalization in Adaptive Data Analysis and Holdout Reuse*, [arXiv:1506.02629v2](https://arxiv.org/abs/1506.02629v2), [full PDF](https://arxiv.org/pdf/1506.02629). Introduction, PDF p. 2; section 4.1 and Figure 1, PDF pp. 15–16. | Adaptive interaction can make a predictor depend on holdout noise. Thresholdout supplies a particular controlled-response mechanism with noise and an explicit reuse budget. | An ordinary exact-score test endpoint, a timestamp, or a query log does not implement Thresholdout and inherits none of its theorem merely by being reproducible. |
| S4 | Daniel Russo and James Zou (2016), *Controlling Bias in Adaptive Data Analysis Using Information Theory*, AISTATS, PMLR 51:1232–1240. [Proceedings](https://proceedings.mlr.press/v51/russo16.html); [PDF](https://proceedings.mlr.press/v51/russo16.pdf). Proposition 1, printed pp. 1233–1234; section 4.1, pp. 1238–1239. | Under the stated sub-Gaussian assumptions, expected selection bias is bounded by statistic variability and mutual information between selection and the statistics. An independently selected index has zero expected bias under this model. | The bound is not a correction computed here. Candidate dependence and information released matter; the number of runs alone does not quantify actual bias. |
| S5 | Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt and Vaishaal Shankar (2019), *Do ImageNet Classifiers Generalize to ImageNet?*, ICML, PMLR 97:5389–5400. [Proceedings](https://proceedings.mlr.press/v97/recht19a.html); [PDF](https://proceedings.mlr.press/v97/recht19a/recht19a.pdf). Sections 2 and 5, PDF pp. 2–3 and 7. | The paper distinguishes adaptivity, distribution differences and finite-sample error. Its new-test-set results show sensitivity to collection/sampling choices and do not identify every accuracy drop as adaptive overfitting. | A fresh sample can assess a frozen classifier under its sampling distribution; it does not establish that the old and new distributions are identical or that all forms of adaptivity are absent. |
| S6 | Sayash Kapoor and Arvind Narayanan (2022), *Leakage and the Reproducibility Crisis in ML-based Science*, [arXiv:2207.07048v1](https://arxiv.org/abs/2207.07048v1), [PDF](https://arxiv.org/pdf/2207.07048). Section 2.4, categories L1.4 and L3.2, PDF pp. 5–6. | Duplicates and observations of the same underlying unit across training and test can invalidate a claim about new units. The relevant independence structure depends on the scientific claim. | This note cites the directly accessed 2022 preprint, not an unread later journal version. Its taxonomy does not make all within-group prediction invalid; the estimand must say whether groups are new or already observed. |

Access limitations: the attempted Cell/Patterns full-text endpoint did not return readable content; the Kapoor–Narayanan source used here is explicitly the accessible arXiv v1. An attempted arXiv HTML v6 endpoint also failed and is not treated as an available version. No proposition depends on either failed endpoint. Dwork's cited text is the full 2015 technical paper; this note does not imply that it has read the separate short Science article.

## 2. Operational distinctions for this programme

The following are applications of S1–S6 to the local study, not claims that those papers audited NeuroPixel. Historical exposure should be established from the programme's own manifests, sample/group identities and access records. Unrecorded exposure remains unknown rather than being silently treated as absent.

**Training, development and final assessment have different roles.** Training fits parameters. Development can choose architectures, learning rates, budgets, stopping points, losses, routers and reporting rules. The final set estimates performance of the procedure frozen before its results become available. A change motivated by a final score makes that score part of development for the revised procedure. Reporting every predeclared arm can remain informative; retrospectively selecting a winner and presenting its same final score as independent confirmation cannot. S1–S4 explain why the selection procedure is part of the dependency chain.

**Nested selection evaluates the procedure, not just its last fitted weights.** In outer resampling, all feature processing, fitting and selection that use data must occur within each outer training partition. Inner validation chooses configurations; the outer held-out groups assess that complete choice rule. Reusing outer scores to redesign the rule creates another adaptive layer. A separately frozen fixed configuration is a different estimand from an algorithm that repeats hyperparameter selection for each new training sample. S1–S2 support the distinction; this note does not require expensive nested training where the actual target is only one already-frozen configuration.

| Evaluation arrangement | What it can support, if access assumptions hold | What it does not establish |
| --- | --- | --- |
| Recount of previously saved predictions | Arithmetic, alignment, provenance and metric reproducibility | New sampled evidence, an independent training replication, or restoration of test independence |
| New layouts/places over previously seen triples | Performance on newly drawn instances of the declared known generator, conditional on the frozen predictor and specified triple distribution | Generalization to unseen compositions or an unknown generator |
| Newly reshuffled split of the old finite triple universe | A documented new partition and, after auditing identities, a clearly scoped resampling analysis | A pristine composition holdout merely because its seed or file hash changed |
| New final instances sampled after model/rule freeze, without using them for selection | Conditional performance of that frozen procedure on the declared sampling distribution, with an appropriate uncertainty analysis | Blinded collection, new latent groups, outside-generator generalization, or independent research-team replication |
| Group-disjoint outer evaluation | Generalization to the held-out group unit defined in advance, subject to its sampling frame and exposure audit | Independence at a broader unit that was not withheld, or automatic protection from later adaptive reuse |

**A split is a relation between identities, not a seed label.** For an unseen-triple claim, adjacent/far layouts, different places, query roles, corruption variants and other renderings of the same agent/verb/patient composition belong to the same declared composition group. Variant bytes can differ while group identity is shared. If the intended claim instead conditions on a known composition and tests a new layout, such sharing can be legitimate and must be named. More demanding claims can require a coarser canonical identity; for example, withholding role-swap families requires a rule beyond an ordered triple. The grouping rule must be specified before assessing results, with all linked variants assigned consistently. This is the local application of S6, not a retroactive discovery that every historical layout overlap was necessarily leakage.

**Reshuffling cannot undo exposure.** A triple used in an earlier training partition does not become a never-trained composition when moved to a new test partition. A former test used to guide decisions does not become untouched development-independent evidence when renamed. Set intersections with historical training/development/test pools must be recorded at the claim's group level, along with which outcomes were exposed. No universal claim that every triple was individually observed is made here: membership in an exposed pool, actual sampled-instance use and access to reported outcomes are different facts. Where the whole old finite universe has participated in training or adaptive assessment across trials, a reshuffle cannot supply historically unexposed groups from that same universe.

**Known generator versus fresh instances.** Knowing a generator's code and distribution does not itself force leakage: new independent draws can estimate a frozen predictor's performance on that known process. For independent Bernoulli correctness conditional on that predictor, ordinary sampling uncertainty concerns those draws. It does not include uncertainty across training seeds, groups or generators unless those are sampled too. With a finite universe, new draws can repeat old identities by design; claims must distinguish fresh sampling from novel support. Performance differences between old and new sets may combine adaptive selection, sample noise and distribution changes; S5 cautions against assigning one cause without evidence.

**Reproducibility is not blindness.** A public seed, manifest or source hash makes a sample reconstructible and a computation auditable. It does not prove that nobody reconstructed the sample, saw its labels or used its outcomes. A local final-access contract can record the freeze, the exact selected procedure and outputs, and a single authorized assessment phase; it is an auditable process control rather than an adversarial guarantee. Blinding requires an actual access barrier or an independent custodian. None of the earlier artifacts should be labelled blind solely because their hashes were fixed. Ordinary final evaluation here is not a reusable-holdout algorithm under S3.

## 3. Proposed bounded null simulation — not executed

**Purpose.** Demonstrate the difference between a development score selected for being high and an independent final score of the same selected candidate. Every candidate has true population accuracy exactly 0.5. This is a deliberately simple measurement-level model of selection; it is not NeuroPixel training, a numerical estimate of historical leakage, or an implementation of nested CV/Thresholdout. The adaptivity is choosing an index from observed scores among a fixed candidate set, not inventing new candidates in response to answers.

### 3.1 Prospective specification approved by root, pending source/plan freeze

| Element | Proposed fixed value or rule |
| --- | --- |
| Independent Monte Carlo replicates | R=10,000 |
| Candidate pool | K=64, IDs 0–63; true accuracy p=0.5 for every candidate |
| Counts per candidate | TRAIN, DEV and FINAL independently follow Binomial(n=256,p=0.5); also independent across candidates and replicates |
| Random streams | NumPy **2.3.5** (runner enforces this version), PCG64 with three explicit independent stream seeds: TRAIN 71001, DEV 71002, FINAL 71003; record Python/platform details |
| TRAIN operation | Sort all 64 TRAIN counts descending, tie by smaller candidate ID. This is a synthetic shortlist operation, not parameter fitting. |
| Search sizes | M in {1,4,16,64}; shortlist is the first M entries of that TRAIN ordering, hence nested within each replicate |
| DEV operation | Among each shortlist, choose the largest DEV count, tie by smaller candidate ID. Freeze every winning ID before generating or opening FINAL. |
| Reported comparisons | Same winner's selected DEV accuracy versus its untouched FINAL accuracy, plus their paired difference; retain every M and every replicate |
| Descriptive tail threshold | q=0.60, interpreted exactly as count >= ceil(256q)=154 |
| Resources | CPU only, no learned models or Torch; approximately 1.92 million binomial counts, bounded arrays; no GPU |
| Execution boundary | Root approved these scientific values before outcomes and freezes source plus plan remotely before any simulation output; this document alone is not execution approval |

A transparent implementation has two phases. First, generate TRAIN and DEV counts and persist all shortlist/winner identities and their digest. The selection routine receives no FINAL counts or RNG. Second, generate FINAL with its own stream and score the already-fixed identities. Save the source/config hashes, versions, winner digest, three count arrays or an equivalently complete reproducible record, and all per-replicate summaries. The NumPy version is pinned because the named bit generator and seeds alone do not pin the distribution-sampling implementation across versions. Public seeds provide reproducibility within the recorded implementation; they are not a claim of blinded custody.

The two reported columns are the same chosen model under different assessment rules: reusing its winning DEV score is the biased practice being illustrated, while FINAL is independent of its selection. Do not choose M, a seed, a replicate subset, or a preferred display from FINAL outcomes. All four M values remain in the report. There is no additional round of selection on FINAL.

### 3.2 Analytic checks to freeze before simulation

Let F_n(k)=P[Binomial(n,0.5)<=k], and let D_M be the selected DEV fraction. The TRAIN shortlist is independent of DEV, so its M counts are still independent Binomial(n,0.5). Thus:

\[
P(nD_M\le k)=F_n(k)^M,\qquad
E[D_M]=\frac1n\sum_{k=0}^{n-1}\left(1-F_n(k)^M\right).
\]

The corresponding exact second moment is

\[
E[D_M^2]=\frac1{n^2}\sum_{k=0}^{n-1}(2k+1)\left(1-F_n(k)^M\right).
\]

For the final fraction Z_M of the already-selected candidate,

\[
E[Z_M]=0.5,\quad \operatorname{Var}(Z_M)=\frac{0.25}{256},\quad
E[D_M-Z_M]=E[D_M]-0.5.
\]

Z_M is independent of D_M in this idealized equal-accuracy, independent-stream construction. Scores for different M within one replicate need not be independent because shortlists overlap and winners can coincide. Monte Carlo replicates, rather than the four M values, supply the replication unit.

For the fixed tail threshold q, let c=ceil(nq). The exact reference probabilities are

\[
P(D_M\ge q)=1-F_n(c-1)^M,\qquad
P(Z_M\ge q)=1-F_n(c-1).
\]

Implementation checks should use a stable binomial CDF or the exact combinatorial definition at p=0.5. Required algebraic identities are: M=1 gives E[D_1]=0.5; expected selected DEV accuracy is nondecreasing in M; expected FINAL accuracy and its tail probability do not depend on M. Under the proposed nested shortlists, each replicate's selected DEV count is nondecreasing in M as well. Checks concern this specified artificial model, not the actual dependence among trained neural models.

### 3.3 Reporting and stopping rules

For each M, report DEV mean, FINAL mean, mean paired optimism, sample SD and Monte Carlo standard error (sample SD/sqrt(R)); show analytic expectations alongside, plus empirical versus analytic tail probabilities at q. A 95% interval for a Monte Carlo mean, if displayed, is mean +/- 1.96 MCSE and is explicitly labelled simulation precision. Empirical 2.5%/97.5% replicate quantiles describe the replicate distribution and must not be labelled a confidence interval for the mean. Report the dependence across M and retain raw per-replicate rows.

Compare simulated means with analytic expectations using prespecified diagnostic bands of six analytic Monte Carlo standard errors; flag discrepancies for arithmetic/implementation review, not as a scientific rejection threshold. The discrepancy gate uses exact population variances: the maximum-binomial variance for DEV, 0.25/256 for FINAL, their sum for paired optimism, and p_tail(1-p_tail)/R for each tail proportion. The observed sample MCSE is reported separately; observing zero tail events must not set the analytic discrepancy tolerance to zero. Random deviations can occur even for correct code. Do not change seeds, enlarge R, repeat until a desired pattern appears, or hide flagged results. Any correction requires a recorded implementation reason and preservation of the failed attempt. No simulation result is filled in by this proposal.

This simulation deliberately makes candidates' errors independent and equally accurate. Real models can have correlated errors, unequal quality, training-induced dependence and adaptive candidate generation; those features change the size of selection bias. Successful analytic agreement validates this illustrative computation only. It supplies no estimate of NeuroPixel's true accuracy, no retrospective debiasing factor, and no new support for H1.

## 4. Boundaries of the proposed remedy

A prospective registry and final-access gate can make later dependency claims concrete: declare the estimand/group key, preserve historical exposure, freeze selection inputs and decisions, and record exactly when final outcomes become available. They cannot erase old test knowledge or manufacture historically unseen compositions from an exhausted finite universe. If future work changes the support or the generator, it needs a separately stated target distribution and a prospectively fixed evaluation; item 7 does not open that future experiment.

**State at writing:** literature review completed; simulation specification approved by root and unexecuted by this author; implementation, freeze and any later run are root-owned. No later programme item is opened.
