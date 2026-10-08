> Recovery copy saved 2026-10-07 from the complete reviewed report text retained in the conversation. The Linux executor was unavailable, so byte-for-byte identity with docs/research/05_results.md could not be verified. Relative links retain the original repository layout; no raw artifacts are reconstructed by this copy.

# Item 5 — Replication results and historical checkpoint reassessment

**Report status: 28 final evaluations and both artifact checks verified. H1: `not_supported`. Updated 2026-10-07.**

The prospective CPU recovery completed all 28 declared runs. On the primary five-initialization panel, mean binding was **12.393% for NeuroPixel and 49.395% for the fixed relative Transformer**. The paired NeuroPixel-minus-reference difference was **−37.002 percentage points**, with a two-sided 95% t interval of **−42.030 to −31.974 pp**. H1 is not supported under the frozen recipe and budget. Both artifact checks found zero discrepancies. [Verified analysis][analysis] [Independent recount][independent]

All 28 fixed training probes also trigger the predefined budget-limited flag. This is an operational optimization diagnostic, not an isolated causal explanation for the outcome. Separately, all six historical checkpoints reproduce their recorded clean global accuracy at four-decimal precision; those results remain retrospective and excluded from H1. [Run tables][tables] [Historical results][hist-results]

| Evidence stream | Verified status | Role in the conclusion |
|---|---|---|
| Six original scaling checkpoints | Retrospective evaluation and artifact audit complete | Consistency of retained weights and historical scores; excluded from H1 |
| Interrupted GPU attempt | No completed training run; no validation or final evaluation | Preserved resource incident and execution cost; excluded from statistical panels |
| Full CPU recovery | 28/28 training runs and 28/28 final evaluations complete; both checks verified | Five complete primary pairs decide H1: not supported |
| External repetition | No independent laboratory repetition supplied by this work | Not established by local hashes, a new device or a second artifact reader |

## 1. Question and frozen methods

### 1.1 Endpoint, data and optimization

Protocol `NP-SCI-20261006-v1` asks whether NeuroPixel improves nominal role binding over the reference selected in the item-4 pilot, under the declared generator, supervision and optimization budget. Binding is the equally weighted mean of AGENT and PATIENT accuracy; global accuracy, four-role macro accuracy and individual role accuracies are also reported. [Protocol][protocol] [Machine-readable protocol][protocol-json]

The adjacent RoleTask uses an 8×8 grid and 35 token IDs. Of 1,320 valid agent/verb/patient triples, 924 form training, 132 validation and 264 final evaluation. These partitions are disjoint within each split. Validation contains 2,048 examples; final evaluation contains 4,096, with 1,024 queries per role. This tests withheld compositions within one generator. It does not test generalization to an independently designed task. [Data specification][data-spec]

Each new run uses 1,024 optimizer updates, batch size 64, AdamW with weight decay 0.0001, gradient clipping at 1.0 and answer cross-entropy only. School loss is absent from all four primary recipes. The final checkpoint is retained. Initialization seeds are separated from the fixed minibatch sampling stream, seed 9002, and NCA update-mask stream, seed 9001. [Execution plan][plan] [Execution methods][execution]

| Family | Parameters | Frozen LR | Computational depth | Defining comparison |
|---|---:|---:|---|---|
| NeuroPixel | 29,824 | 0.003 | 16 recurrent steps | Tied decoding and repeated identity input |
| Adapted NCA | 30,384 | 0.001 | 16 recurrent steps | Untied factorized decoding; initial seed retained without input reinjection |
| Spatial ConvGRU | 27,835 | 0.003 | 16 recurrent steps | Gated spatial recurrence |
| Relative Transformer | 29,547 | 0.001 | Two attention blocks | Learned two-dimensional relative biases; fixed primary reference |

The similar parameter counts do not equalize active capacity, propagation radius, FLOPs, memory, elapsed time or optimization difficulty. The common budget equalizes updates and examples. The original model behavior, including the NeuroPixel PAD dictionary behavior, is retained. In evaluation mode the NCA models perform synchronous full updates: the result is not a Monte Carlo average over the Bernoulli firing masks used during training. [Model specification][model-spec]

Learning rates and the primary reference are fixed from pilot validation. There is no new selection on replication validation or final outcomes. The execution source is `fbf576f0dcfd0149d2969eccb61e442c41c7d47f`; its eight scientific source files match the pilot. Controller hashes are recorded separately. The immutable pilot selection SHA-256 is `a62300cfccc8d01f9d2777042c20a2620ba0ebfbd24d1b2b0fe4a7ea3463673b`. [Execution methods][execution] [Pilot selection][selection]

### 1.2 Two overlapping panels

| Panel | Varied factor | Fixed factor | Unique contribution |
|---|---|---|---:|
| Primary initialization panel | Initializations 10, 11, 12, 13, 14 | Split 0 | 20 runs across four families |
| Exploratory split panel | Splits 0, 101, 202 | Initialization 10 | Eight additional runs; four split-0 runs are shared |

The union contains 28 runs. The shared initialization-10/split-0 checkpoint enters both descriptive panels once, without a second training or evaluation. The panels are not pooled into seven independent joint replications and do not estimate an initialization-by-split interaction. Pilot initialization 7 and the historical checkpoints are excluded from the replication statistics. [Execution plan][plan] [Statistics specification][statistics]

### 1.3 Final evaluation and artifact boundary

The runner persists the complete manifest before training and writes the final-evaluation gate only after all 28 training runs complete with consistent scientific sources and environments. Final datasets and predictions follow that gate. An existing final gate blocks silent re-entry. The independent analyzer verifies the inventory, provenance, event ordering, datasets and checkpoint identities before recounting final metrics from saved predictions; it performs no new training or inference. [Execution methods][execution] [Analyzer source][analyzer]

A fixed 512-example training probe, sampling seed 61006, provides the predeclared optimization diagnostic. Binding below 95% is labeled budget-limited. That label neither permits post-test tuning nor proves that additional training would solve the task. All flags and completed runs must remain in the report. [Protocol][protocol] [Statistics specification][statistics]

## 2. Interrupted GPU attempt and CPU recovery

The GPU attempt was admitted at `2026-10-06T23:09:38.414705+00:00`. The first NeuroPixel run, initialization 10/split 0, reached recorded update 768 before a `ResourceStop`: available RAM was 7.99 GiB, below the unchanged 8 GiB guard. The attempt ended at `23:10:18.798732+00:00`, with zero complete training runs and no validation or final evaluation. Its evidence is preserved. [Runtime decisions][runtime]

At `23:15:13.339416+00:00`, before access to new final outcomes, recovery was fixed as all 28 runs from scratch on CPU with two threads. The scientific source, recipes, selection, seeds and budgets remain unchanged; partial GPU weights are not reused and CPU/GPU runs are not mixed. The CPU worker started at `2026-10-06T23:16:42.119319+00:00` and finished successfully at `2026-10-07T00:06:21.326185+00:00`. All 28 training runs preceded the final gate at `00:05:23.882112+00:00`; the final completion event was recorded at `00:06:20.768621+00:00`. [Runtime decisions][runtime] [Worker status][worker] [Replication record][replication]

The recorded protocol interpretation applies the explicit resource-retry and CPU-fallback provisions. An interrupted attempt remains an interruption; a complete permitted recovery can supply the required pairs. An unresolved missing, failed or non-finite primary pair still prevents positive H1 support. The recovery follows a GPU-selected recipe and conditions the resulting inference on its CPU environment. It does not establish invariance between devices. GPU attempt costs remain separate from the CPU panel's costs. [Protocol][protocol] [Runtime decisions][runtime]

Neither a successful local recovery nor independent checking of its artifacts constitutes an external repetition of training by another laboratory.

## 3. Completed retrospective historical results

### 3.1 Reconstruction and provenance

The six retained scaling checkpoints comprise NeuroPixel with 29,824 parameters and the original TinyTransformer with 44,483 parameters, each at coupled seeds 0, 1 and 2. The original classes and learned PAD weights were restored strictly. For each split, one original RoleTask CPU call generated 2,000 examples with sample seed 5; the paired models shared those examples. Roles were not rebalanced. Evaluation used PyTorch 2.14.1+cpu, two threads and inference batches of 256. [Historical plan][hist-plan] [Historical results][hist-results]

The historical operation is a reassessment of saved weights, not a new training replication. Execution ran from `2026-10-06T23:08:10.466241+00:00` to `23:08:15.646605+00:00`. The weights used as inputs were retained under `runs/research/historical/`; archived copies are also versioned under `results/research/05_historical/checkpoints/`. The input manifest, source hashes, generated datasets and predictions are retained. [Checkpoint manifest][checkpoints] [Archived weight copies][archived-weights] [Historical artifact record][hist-json]

### 3.2 Accuracy by retained checkpoint

Values are percentages; display rounding does not enter the calculations. Binding denotes the AGENT/PATIENT macro mean. Every reconstructed global score matches its corresponding historical `combos_nuevas` value at four-decimal precision. [Historical artifact record][hist-json] [Independent historical audit][hist-audit]

| Family | Coupled seed | Global | Binding | AGENT | PATIENT | Historical global matched |
|---|---:|---:|---:|---:|---:|---|
| NeuroPixel | 0 | 98.400% | 98.985% | 98.592% | 99.378% | Yes |
| TinyTransformer | 0 | 53.800% | 5.624% | 5.231% | 6.017% | Yes |
| NeuroPixel | 1 | 95.400% | 93.647% | 94.970% | 92.324% | Yes |
| TinyTransformer | 1 | 54.550% | 7.142% | 7.646% | 6.639% | Yes |
| NeuroPixel | 2 | 97.250% | 95.763% | 98.994% | 92.531% | Yes |
| TinyTransformer | 2 | 55.550% | 9.192% | 9.256% | 9.129% | Yes |

Each dataset contains 497 AGENT, 531 ACTION, 482 PATIENT and 490 PLACE queries. The historical Transformer achieves 100% on ACTION and PLACE in all three retained evaluations; its global accuracy therefore masks much weaker nominal binding. This describes these trained models, not Transformers as an architectural family. [Historical results][hist-results] [Independent historical audit][hist-audit]

### 3.3 Dispersion and paired differences

SD is the sample standard deviation across three retained historical trainings, with denominator two. Percentages and percentage points (pp) are distinguished explicitly. [Historical artifact record][hist-json]

| Family | Metric | Mean | Sample SD | Observed range |
|---|---|---:|---:|---|
| NeuroPixel | Global | 97.017% | 1.514 pp | 95.400%–98.400% |
| NeuroPixel | Binding | 96.131% | 2.688 pp | 93.647%–98.985% |
| TinyTransformer | Global | 54.633% | 0.878 pp | 53.800%–55.550% |
| TinyTransformer | Binding | 7.320% | 1.791 pp | 5.624%–9.192% |

| NeuroPixel minus TinyTransformer | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD | Range |
|---|---:|---:|---:|---:|---:|---|
| Global difference, pp | 44.600 | 40.850 | 41.700 | 42.383 | 1.966 | 40.850–44.600 |
| Binding difference, pp | 93.361 | 86.504 | 86.570 | 88.812 | 3.940 | 86.504–93.361 |

The seed changes initialization, triple partition and training streams together. These three outcomes cannot isolate the corresponding variances. The six evaluations do not create six paired training replications, and the 12,000 scored predictions do not increase the number of independently trained models. No historical significance decision is added; this panel is excluded from H1. [Historical plan][hist-plan] [Historical results][hist-results]

### 3.4 What was verified, and what remains limited

A separate read-only audit checked 31 file hashes, the six checkpoint files and all 12,000 saved predictions, with zero discrepancies. It performed no inference, data generation or training. This establishes consistency of the retained artifacts and recounted scores, not independent proof of the historical execution. [Independent historical audit][hist-audit]

Historical example hashes and package versions are unavailable. Matching rounded aggregate scores does not prove identical historical input bytes or individual decisions. Seed-0 and seed-1 score snapshots came from the public repository; seed-2 scores are normalized JSON snapshots from the user's existing project. Their provenance is explicit. [Historical source manifest][hist-sources]

The original study used an 80/20 triple split, NeuroPixel school loss weighted 0.3, a different optimizer schedule and batch size, and a different Transformer. Exact historical invocation budgets are not established by the score JSON. Consequently, differences from the prospective experiment cannot be assigned to school loss, parameter count or any other single component. The historical outcome establishes neither equal-supervision/equal-cost superiority nor outside-generator reasoning. [Historical plan][hist-plan] [Historical results][hist-results]

## 4. Verified prospective recovery results

All 28 unique configurations completed, with no missing or duplicate statistical records. Accuracy fractions in the individual-value tables below preserve the reported values without display rounding. Summary tables use percentages and percentage points, rounded for readability; the decisions use the unrounded values in section 5. The complete CSV preserves all 28 runs, role counts, validation scores and training-probe metrics. [Verified analysis][analysis] [Run tables][tables] [Full CSV][csv]

### 4.1 Inventory, provenance and two artifact checks

The first analyzer verified 28 training records and 28 final records, inspected 196 file identities and produced 21 conditional bootstrap intervals. It checked source/controller consistency, the fixed selection, checkpoint hashes, final-data identity and event ordering before recounting scores. Its status is `verified`, with zero issues. Analysis SHA-256: `11fd91ee4ddb8155fc9127e5f6b826535654842190f048fecd23c3dd6f31e436`. [Verified analysis][analysis]

A second implementation independently recounted **114,688 final predictions** across the 28 runs, recorded hashes for 35 inputs, and reproduced the five paired differences, t interval and H1 decision within its declared 1e-12 numerical tolerance. It imports neither Torch nor the project's statistics module. Its status is also `verified`, with zero issues. The probe flags were checked against reported aggregate probe binding; this second reader did not regenerate probe predictions. [Independent recount][independent] [Independent reader source][independent-source]

The retained archive contains **215 files, 6,305,023 bytes**, including all 28 checkpoints; the manifest records byte-identical copies. Source snapshots, the CPU wrapper, worker records and recovery decisions are retained beside the replication artifacts. The archive manifest is separate from those 215 entries. These checks establish internal artifact consistency and a second implementation of the recount, not independent laboratory reproduction of training. [Archive manifest][archive] [Replication archive][replication-archive]

### 4.2 Primary initialization panel: split 0

| Initialization | NeuroPixel | Adapted NCA | ConvGRU | Relative Transformer | NP minus Transformer |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.142578125 | 0.11767578125 | 0.1015625 | 0.4921875 | -0.349609375 |
| 11 | 0.09716796875 | 0.12548828125 | 0.103515625 | 0.501953125 | -0.40478515625 |
| 12 | 0.07373046875 | 0.09228515625 | 0.0849609375 | 0.48779296875 | -0.4140625 |
| 13 | 0.169921875 | 0.10986328125 | 0.08154296875 | 0.4853515625 | -0.3154296875 |
| 14 | 0.13623046875 | 0.111328125 | 0.1142578125 | 0.50244140625 | -0.3662109375 |

| Family | Mean binding | Binding SD | Binding range | Global mean ± SD | Global range | Mean CE |
| --- | --- | --- | --- | --- | --- | --- |
| NeuroPixel | 12.393% | 3.824 pp | 7.373%–16.992% | 21.958% ± 4.382 pp | 16.284%–28.394% | 2.666573 |
| Adapted NCA | 11.133% | 1.230 pp | 9.229%–12.549% | 13.457% ± 0.753 pp | 12.402%–14.185% | 2.516240 |
| ConvGRU | 9.717% | 1.365 pp | 8.154%–11.426% | 14.116% ± 1.268 pp | 12.671%–16.138% | 2.223771 |
| Relative Transformer | 49.395% | 0.792 pp | 48.535%–50.244% | 74.697% ± 0.396 pp | 74.268%–75.122% | 0.427854 |


Each family has five observations. Global accuracy equals four-role macro accuracy here because final role counts are balanced. Every per-run role numerator and denominator is retained in the companion tables and CSV. Mean final accuracy by role is:

| Family | AGENT | ACTION | PATIENT | PLACE |
| --- | --- | --- | --- | --- |
| NeuroPixel | 12.969% | 25.801% | 11.816% | 37.246% |
| Adapted NCA | 11.699% | 15.410% | 10.566% | 16.152% |
| ConvGRU | 11.680% | 12.715% | 7.754% | 24.316% |
| Relative Transformer | 55.332% | 100.000% | 43.457% | 100.000% |

The reference answers every ACTION and PLACE query correctly across these five checkpoints, while its mean AGENT/PATIENT binding is 49.395%. NeuroPixel is below that reference in all five matched initializations. This establishes the observed ordering for this budget and setting, rather than a broad capability limit for either architecture. [Verified analysis][analysis]

### 4.3 Exploratory split panel: initialization 10

Split 0 reuses the same four checkpoints from the initialization panel. These are three conditional observations per family, not additional independent initializations.

| Split | NeuroPixel | Adapted NCA | ConvGRU | Relative Transformer | NP minus Transformer |
| --- | --- | --- | --- | --- | --- |
| 0 (shared) | 0.142578125 | 0.11767578125 | 0.1015625 | 0.4921875 | -0.349609375 |
| 101 | 0.16259765625 | 0.11572265625 | 0.14697265625 | 0.50244140625 | -0.33984375 |
| 202 | 0.14697265625 | 0.1337890625 | 0.11572265625 | 0.48876953125 | -0.341796875 |

| Family | Mean binding | Binding SD | Binding range | Global mean ± SD | Global range | Mean CE |
| --- | --- | --- | --- | --- | --- | --- |
| NeuroPixel | 15.072% | 1.052 pp | 14.258%–16.260% | 22.925% ± 2.631 pp | 20.703%–25.830% | 2.236912 |
| Adapted NCA | 12.240% | 0.992 pp | 11.572%–13.379% | 14.803% ± 0.977 pp | 13.843%–15.796% | 2.410550 |
| ConvGRU | 12.142% | 2.323 pp | 10.156%–14.697% | 13.468% ± 0.485 pp | 13.086%–14.014% | 2.237774 |
| Relative Transformer | 49.447% | 0.712 pp | 48.877%–50.244% | 74.723% ± 0.356 pp | 74.438%–75.122% | 0.416327 |


The paired NeuroPixel-minus-Transformer mean is −34.375 pp, with SD 0.517 pp, observed range −34.961 to −33.984 pp and descriptive 95% t interval −35.659 to −33.091 pp (df=2). This split result is separate from H1 and is not pooled with the initialization panel. Complete secondary-metric means, SDs and ranges are retained in the analysis JSON. [Verified analysis][analysis]

### 4.4 Other neural contrasts and conditional intervals

The following comparisons are descriptive, with no multiplicity adjustment. All four t intervals cross zero; the results do not establish a stable NeuroPixel advantage over the adapted NCA or ConvGRU.

| Panel | Contrast | Mean, pp | Sample SD, pp | 95% t interval, pp |
| --- | --- | --- | --- | --- |
| Five initializations | NP minus Adapted NCA | 1.260 | 3.606 | [-3.217, 5.737] |
| Five initializations | NP minus ConvGRU | 2.676 | 4.050 | [-2.354, 7.705] |
| Three splits | NP minus Adapted NCA | 2.832 | 1.710 | [-1.417, 7.081] |
| Three splits | NP minus ConvGRU | 2.930 | 1.281 | [-0.252, 6.111] |

All 21 per-checkpoint paired percentile bootstrap intervals are retained in `conditional_binding_intervals` in the analysis report. The seven comparisons against the fixed Transformer each have a negative interval, conditional on their respective checkpoints and examples. These intervals are not pooled, do not estimate training-run variance and do not enter H1. [Verified analysis][analysis] [Statistics specification][statistics]

### 4.5 Optimization and descriptive validation behavior

All **28/28** probes fall below the predeclared 95% training-binding threshold. The following ranges describe the seven unique executed configurations of each family; they are an inventory of optimization diagnostics, not a pooled seven-run variance estimate.

| Family | Budget-limited / unique runs | Training-probe binding range |
| --- | --- | --- |
| NeuroPixel | 7/7 | 10.547%–18.750% |
| Adapted NCA | 7/7 | 7.422%–14.844% |
| ConvGRU | 7/7 | 8.594%–14.844% |
| Relative Transformer | 7/7 | 47.656%–58.984% |

The flag means that the recipe did not reach the declared training-probe criterion. It does not identify the cause, establish that more updates would solve the deficit, or isolate the effect of absent school supervision. Per-run probe values and all four role counts remain in the CSV. [Run tables][tables] [Full CSV][csv]

The separate post-hoc validation diagnostic finds that **each of seven relative-Transformer checkpoints chooses a visible filler of the queried category in all 2,048 validation examples**. Its nominal binding nevertheless ranges from 43.750% to 47.559%. All seven ConvGRU checkpoints also produce the correct category in every validation example, but none retrieves a visible filler on every example. These saved-output counts describe different degrees of category and filler retrieval; they neither alter selection/H1 nor establish a causal internal mechanism or positional invariance. [Validation diagnostic][diagnostics]

### 4.6 Figure

![Individual binding accuracies for the five-initialization and three-split panels](../../results/research/05_replication_panels.png)

Each dot is one retained checkpoint evaluation. The four initialization-10/split-0 checkpoints appear in both panels but were trained and evaluated once. The dashed line is the 50% nominal-binding expectation for choosing between visible noun fillers by category alone. No confidence intervals or extra training replications are implied by the dots. [SVG version][figure-svg] [Task controls][controls]

### 4.7 Execution time and memory

| Quantity | Recorded value |
| --- | --- |
| Completed CPU training timers, sum over 28 unique runs | 2876.993445514003 s |
| CPU worker elapsed time, including compatibility/setup/evaluation | 2979.206866 s (49 min 39.207 s) |
| Recorded peak child RSS | 0.517669677734375 GiB |
| Runtime | Python 3.12.14; PyTorch 2.14.1+cpu; NumPy 2.3.5; SciPy 1.17.0; two CPU threads |
| Scientific source | fbf576f0dcfd0149d2969eccb61e442c41c7d47f |
| Interrupted GPU attempt | Preserved separately; last recorded training timer 18.899970799975563 s, not a completed-run cost |
| Inference latency and physical energy | Unmeasured; JSON null |

Training timers do not include all worker overhead. The worker duration excludes the earlier GPU queue/attempt, historical reassessment and subsequent auditing/reporting. The CPU panel's cost therefore is not the total project cost. Similar parameter counts and equal example exposure do not establish equal elapsed cost or physical efficiency. CPU recovery uses a GPU-selected recipe; it is not a controlled comparison of devices. [Replication record][replication] [Worker status][worker] [Runtime decisions][runtime]

## 5. H1 decision and uncertainty

For the primary five pairs, let \(d_i=b_{NP,i}-b_{TF,i}\). The declared two-sided interval is

\[
\bar d\pm t_{0.975,4}\frac{s_d}{\sqrt5}.
\]

NIST's paired analysis operates on the differences, using their sample SD and four degrees of freedom here. The interval is not clipped. Its exact nominal coverage assumes independent, identically distributed normal differences; five observations cannot establish those distributional assumptions. [NIST paired observations][nist-paired] [NIST mean interval][nist-ci] [Inference limits][limits]

| Frozen requirement | Unrounded reported value | Assessment |
| --- | --- | --- |
| Exactly five complete finite primary pairs | 5 | Pass |
| Mean NeuroPixel binding >= 0.90 | 0.12392578125 | Fail |
| Mean paired advantage >= 0.05 | -0.37001953125 | Fail |
| 95% t interval lower endpoint > 0 | -0.42030141465134446 | Fail |
| H1 | not_supported | positive_support = false; eligible = true |

The primary paired summary, in accuracy-fraction units, is:

| Quantity | Value |
| --- | --- |
| Mean NeuroPixel binding | 0.12392578125 |
| Mean reference binding | 0.4939453125 |
| Mean paired difference | -0.37001953125 |
| Sample SD of paired differences | 0.04049556359376132 |
| Paired 95% t interval | [-0.42030141465134446, -0.3197376478486555] |
| Degrees of freedom | 4 |

All conditions were applied without rounding. The five-point screen concerns the estimated mean advantage; it is not a confidence claim that the population advantage exceeds five points. The complete panel is eligible for the rule, but fails all three performance criteria. The observed difference is negative in every primary pair, and the t interval is wholly negative. Thus H1 is not supported in the declared CPU setting and budget. This does not prove equivalence, universal inferiority or inability to learn. The historical panel, exploratory split results and other baseline contrasts cannot replace this frozen comparison. [Verified analysis][analysis] [Independent recount][independent] [Protocol][protocol]

The paired example bootstrap uses common resampled indices within AGENT and PATIENT, equal role weights, seed 61003 and 2,000 percentile resamples. It estimates conditional example uncertainty for fixed checkpoints. It does not turn 4,096 examples into 4,096 training replications or enter the H1 decision. SciPy documents the common-index pairing principle; the role stratification and percentile method are fixed by this protocol. [SciPy bootstrap][scipy-bootstrap] [Statistics specification][statistics]

Bouthillier et al. analyze how multiple sources of randomness affect an ML benchmarking pipeline. Here, the initialization and split panels each hold other factors fixed. They do not estimate repeated hyperparameter-search variability, independent changes of sampling or mask streams, total pipeline variance, or variation across laboratories. Their conditional variances cannot simply be added, and no quantitative variance estimate from that paper is imported into this experiment. [Bouthillier et al.][bouthillier] [Inference limits][limits]

## 6. Interpretation of the completed measurements

This experiment does not support the proposed NeuroPixel binding advantage over the fixed reference under the declared answer-only, 1,024-update recipe. The reference itself remains near the category-only nominal-binding expectation. The result therefore is not evidence that either recipe solves general role reasoning.

All runs meet the procedural completion requirement while all training probes fail the optimization diagnostic. That combination supports a bounded negative finding, with optimization adequacy unresolved. It does not license attributing the outcome solely to insufficient budget, missing school loss, CPU execution or any particular architectural component. Those explanations were not isolated by this experiment.

The successful historical reconstruction and unsuccessful prospective H1 test address different trained weights and recipes. Historical supervision, task partition, comparator, schedule and unverified invocation budgets prevent a causal attribution of their difference to one component. The two conditional panels remain small, share checkpoints and use one generator and one execution environment; they do not establish total pipeline uncertainty or external reproducibility. Artifact agreement strengthens the credibility of the reported numbers without substituting for repetition by an independent laboratory. [Historical results][hist-results] [Inference limits][limits] [Runtime decisions][runtime]

## Sources and artifact links

Local links identify the protocol, implementation and retained evidence. External references below are primary methodological sources checked on 2026-10-06; they justify calculations and scope distinctions. Experimental claims are supported by the linked local artifacts.

- [Verified analysis][analysis], [independent recount][independent], [independent reader source][independent-source], [complete tables][tables] and [CSV][csv].
- [Validation diagnostic][diagnostics], [replication record][replication], [worker status][worker] and [archive manifest][archive].
- [PNG figure][figure-png] and [SVG figure][figure-svg].
- [Frozen protocol][protocol] and [machine-readable settings][protocol-json].
- [Execution plan][plan], [execution methods][execution] and [runtime decisions][runtime].
- [Data specification][data-spec], [model specification][model-spec] and [pilot selection][selection].
- [Statistics specification][statistics], [inference limits][limits] and [artifact analyzer][analyzer].
- [Historical plan][hist-plan], [historical narrative][hist-results], [result JSON][hist-json], [checkpoint manifest][checkpoints], [archived weights][archived-weights], [score provenance][hist-sources] and [independent artifact audit][hist-audit].
- [Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021][bouthillier].
- NIST: [analysis of paired observations][nist-paired] and [confidence limits for the mean][nist-ci].
- SciPy: [bootstrap documentation][scipy-bootstrap].

[protocol]: 02_protocol.md
[protocol-json]: protocol.json
[plan]: 05_execution_plan.json
[execution]: 05_execution.md
[runtime]: 05_runtime_decisions.json
[data-spec]: 04_data_specification.md
[model-spec]: 04_model_specification.md
[selection]: ../../results/research/04_experiment/pilot/selection.json
[statistics]: 05_statistics.md
[limits]: 05_inference_limits.md
[analyzer]: ../../scripts/research_analyze_replications.py
[hist-plan]: 05_historical_reassessment_plan.md
[hist-results]: 05_historical_results.md
[hist-json]: ../../results/research/05_historical/historical_results.json
[checkpoints]: historical_checkpoints.json
[archived-weights]: ../../results/research/05_historical/checkpoints/
[hist-sources]: ../../results/research/05_historical_sources/source_manifest.json
[hist-audit]: ../../results/research/05_historical_verification.json
[bouthillier]: https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf
[nist-paired]: https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm
[nist-ci]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
[scipy-bootstrap]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html

[analysis]: ../../results/research/05_analysis.json
[independent]: ../../results/research/05_independent_check.json
[independent-source]: ../../scripts/research_independent_replication_check.py
[tables]: 05_replication_tables.md
[csv]: ../../results/research/05_replications.csv
[diagnostics]: ../../results/research/05_validation_diagnostics.json
[replication]: ../../results/research/05_experiment/replication/replication_records.json
[replication-archive]: ../../results/research/05_experiment/replication/
[worker]: ../../results/research/05_experiment/metadata/worker_status.json
[archive]: ../../results/research/05_experiment/archive_manifest.json
[figure-png]: ../../results/research/05_replication_panels.png
[figure-svg]: ../../results/research/05_replication_panels.svg
[controls]: 03_controls.md
