# Item 10: prospective neural protocol for external synthetic QA

This protocol evaluates newly trained architectures on the original English 1k bAbI task 4, Two Argument Relations. It is an externally authored synthetic question-answering task, not zero-shot transfer of item-9 checkpoints, real-world generalization, or an architecture-causal comparison. The original paper presents bAbI as prerequisite proxy tasks [1]. Passing this task would not demonstrate capabilities outside its generator. Historical H1 and the closed item-9 conclusions remain unchanged.

This document precedes neural preflight results. The current controller candidate is Git blob `410a7c491f81f704598053e19e5d37890315f0ce`; the final execution plans must bind the reviewed source bytes, dependencies, runtime, inputs and this protocol before their respective jobs. Frozen contract tests run first within the job and must pass before neural work begins.

## 1. Data, representation and exposure

The acquisition archive is `19c8c40ab50141ba128623e7bbbc04cfb2460f31`, from run 37609533224 and source `99ecaeefdd00dffd3dae444a781ad009c19dff5d`. Its data directory is `results/research/10_cloud_runs/37609533224-1-acquisition/data`. Acquisition reports 1,000 official TRAIN episodes, 919 unique normalized inputs, 895 optimization-TRAIN records and 105 DEV records, with no TRAIN/DEV input intersection, OOV inputs, unsupported answers or conflicting raw/encoded labels. Independent acquisition auditing is a prerequisite to neural admission; these observations are not neural results.

The original author URL returned HTTP 404. The documented mirror supplied the archive; no independently authenticated upstream checksum was available. Preserve that provenance limitation, the failed attempt and the attribution recorded in [2]:

- Compressed archive: 11,745,123 bytes; SHA-256 `84f5296ab9a1ad0dc9464e08c491d65cd08830fca3acae9ab86f75e0fb81573c`.
- Selected TRAIN member: 117,470 bytes; SHA-256 `c9370502d85fa382f32d1fea2111e817f81ae9e3e79eb2924a1a5392a90417fa`.

Use the exact original `tasks_1-20_v1-2/en/qa4_two-arg-relations_{train,test}.txt` members. The separately distributed development split and rewritten Lua generator are not substitutes [2,3]. TEST is already present inside the conserved compressed archive, but its payload must remain unopened until the final gate. This is workflow-controlled access under project custody, not independent blinding.

The adapter lowercases text and retains all lexical and punctuation tokens in order. Numeric line IDs are format metadata. All preceding statements in the episode remain visible; only the current question is added. Gold answers, support IDs and earlier questions/answers never construct neural inputs or choose facts.

TRAIN/DEV grouping transitively connects original episodes, identical complete normalized statement sequences and identical normalized visible inputs. Groups are assigned with the previously fixed salt `neuropixel-item10-dev-v1` and DEV fraction 0.1. The vocabulary and geometry are fitted only on assigned optimization TRAIN. This grouping prevents the declared exact-story/input leakage; it does not eliminate shared templates, individual facts, entities or semantic equivalence.

The frozen geometry is **H=4, W=8, V=18**: statements occupy ordered rows, the question starts at row 2, and row 3 is reserved for output at **(3,7)**. PAD=0 and UNK=1. Known-token decoding must recover normalized token order, not original capitalization/whitespace. Maximum input-to-output Chebyshev distance is 7; 16 local updates therefore have no hard geometric exclusion, without establishing learned transport.

## 2. Models and optimization

| Family | Fixed architecture | Parameters |
|---|---|---:|
| NeuroPixel | Direct corrected core; identity 16, state 48, hidden 128, T=16, tied dictionary, reinjection, no retina | 29,552 |
| Relative Transformer | Width 32, two layers, four heads, learned 2D relative bias, feed-forward 144; original non-PAD key mask | 29,562 |

Both consume the same integer canvas and produce logits over the full 18-token vocabulary at the same output position. No answer-category mask, pretrained encoder, support supervision or auxiliary dictionary loss is supplied. NeuroPixel uses firing probability 0.5 during training and dense updates in evaluation; the Transformer has no dropout.

The Transformer feed-forward width was prospectively changed from the unexecuted 128-width draft after TRAIN geometry was known and before any neural result. The source-derived counts are $29,264+16V$ for NeuroPixel and $25,472+65V+8(2H-1)(2W-1)$ for the 128-width Transformer; increasing its feed-forward width by 16 adds $2(65)(16)=2,080$ parameters. The resulting ten-parameter difference is approximately 0.034% of NeuroPixel's count. Factory tests must verify actual counts. Parameter matching does not match computation, memory, optimization difficulty or receptive-field structure.

Training uses batch size 32, AdamW weight decay `1e-4`, gradient-norm clipping at 1 and answer cross-entropy only. Initialization uses the declared run seed. After initialization, the Torch firing RNG is seeded once with `110000 + run_seed`; its initial/final state hashes are retained. No reseeding occurs per minibatch. Sampling uses a separate NumPy `default_rng(100000 + run_seed)`, drawing record indices with replacement. Save the exact planned index array, consumed-prefix digest and unique observed record/raw-input/encoded-input counts. Family pairs share indices, not an asserted identical internal stochastic trajectory.

## 3. Preflight, selection and conditional main budget

Preflight retains all six runs and ordinary failures:

1. Two memorization diagnostics, one per family, seed 58 and LR 0.003. Select the first 32 distinct normalized TRAIN inputs after sorting record IDs. Use at most 4,096 updates; check every 128 updates in evaluation mode. Require two consecutive checks with accuracy 1.0 and CE at most 0.05. Save every check's already-computed 32-row prediction arrays, metadata, loss and streak decision. Stop at the second passing check; no additional inference is needed for retention.
2. Four pilots: both families at LR 0.001 and 0.003, seed 59, exactly 1,024 updates. All candidates use the same DEV population. Save checkpoints, logs, exact streams, full-TRAIN probe predictions and DEV predictions.

Select each family's LR by **DEV accuracy descending, then DEV CE ascending, then LR ascending**. Preserve every candidate. Before main training, authenticate all six expected configurations, artifact hashes, complete pilot budgets, memorization schedules/streaks and saved DEV predictions; independently recount argmax, float64 NLL and selection. Neither memorization performance nor TEST chooses the LR.

The planned main panel is five paired training realizations, seeds **60–64**, both families, **4,096 updates each**, initialized afresh. Its admission requires successful preflight and a prospective cost review based on recorded update/overhead timings. No main run is authorized by this document alone. If the budget is not admitted, record the limitation and prospectively amend the plan before training; do not silently shorten runs. Failure to memorize is a valid negative observation: preserve all six runs and report admission false. An artifact audit may verify that evidence without authorizing main training. The main panel will use fixed endpoints, without DEV early stopping or selecting the best seed. No architecture, learning recipe or candidate search may be revised in response to TEST.

## 4. Durable final gate and complete scoring

Before TEST access, complete and preserve all ten runs, their configurations, checkpoint/summary references, source identities and data/selection hashes in `training_manifest.json`. The worker must archive this inventory and confirm the results commit. It then creates `pre_final_archive_receipt.json`, binding that commit, source, run key and manifest SHA-256.

The final controller checks the complete inventory and receipt, then writes exclusive `final_access.json` before opening the exact TEST member. Failed access remains consumed; no silent retry, replacement checkpoint or regenerated final panel is allowed. Retain raw TEST bytes, parsed records, encoded records, failed encodings and all predictions. An independent archive review must authenticate the intermediate Git objects; a local receipt alone does not establish remote push chronology.

Score every official QA row. Input OOV tokens map to frozen UNK; vocabulary and geometry do not expand. An overflowing input remains in its applicable denominator, receives no neural inference and counts as wrong. An unsupported gold token also counts as wrong; predicting UNK never earns credit. Distinguish:

- `can_encode`: fits the frozen grid;
- `gold_supported`: gold belongs to the full non-reserved output vocabulary;
- `nll_supported = can_encode & gold_supported`: CE can be evaluated.

Logits are float32, NLL is calculated in float64, and arrays are loaded without pickle. Unencodable rows have explicit abstentions and masked placeholders, not claimed model logits. Report each mask count and CE denominator separately; empty populations have null accuracy/CE.

Fix three populations before reading prediction outcomes: all official rows; rows absent from optimization TRAIN by both normalized raw and encoded-input identity; and the strict subset absent from complete official TRAIN, including DEV, plus the complete invented development-fixture ledger. Unencodable rows can remain novel by raw identity and count as wrong; do not drop them. Preserve fixture encoding failures and raw/encoded overlap counts. These subsets measure exact-input exposure, not unseen semantic structures. Full exposure means this declared ledger, not unknown prior human or model exposure.

Report all eight fixed raw-input controls: TRAIN majority, exact ordered-input memory, question-only memory, unordered multiset memory, multinomial naive Bayes, fact-answer-token frequency, that frequency excluding query-mentioned candidates, and the independent direct/inverse symbolic grammar. Learned controls use optimization TRAIN only. Frequency candidates come from TRAIN answer labels. The symbolic control neither supplies neural features nor uses support indices; it may emit a name outside the neural vocabulary. Raw controls need not share neural-grid overflow limits. No control is selected retrospectively.

## 5. Estimands and uncertainty

The primary contrast is the mean paired **NeuroPixel minus Transformer all-official accuracy**, over the five declared training realizations. Report all ten values, paired differences, their mean and a two-sided Student-t 95% interval with df=4. This is conditional on this dataset, split, selection and fixed budget; it is not a causal architecture effect or five independently sampled datasets.

The exploratory competence screen requires NeuroPixel accuracy at least 0.95 in **every** seed on all official rows **and** on the strict subset when nonempty. An empty strict subset makes that part not established/null, rather than a vacuous pass. Report individual failures without replacing historical H1 or presenting this threshold as a general-intelligence criterion.

For fixed-checkpoint sensitivity, independently reconstruct TEST components by transitive episode/full-story/normalized-input connections, with canonical component order. Use one saved **2,000-by-G int64** resampling matrix from **PCG64, seed 104001**, shared across all ten checkpoints and all three subsets. Each draw uses total selected correct divided by total selected rows across sampled components, preserving unequal component sizes. Retain array/hash identities and NumPy linear 2.5/97.5 percentiles. If G<2, bootstrap inference is unavailable; if any draw has zero denominator for an endpoint, its interval is null and the undefined-draw count remains explicit. Never substitute zero accuracy. Numerical undefined draws may be NaN in saved arrays; JSON uses null.

These intervals describe observed-source-component sensitivity conditional on fixed checkpoints, not training variation or multiplicity-adjusted evidence. Raw grouping cannot remove every semantic dependence.

## 6. Execution and limits

Training runtime: Python 3.12.14, Torch 2.6.0+cpu, NumPy 2.2.6, SciPy 1.15.1; two numerical threads, one Torch inter-op thread, at least 8 GiB available RAM. Exact remaining package versions, source bindings and admission thresholds are in the frozen plan. The root worker supervises owned processes and preserves failures. Preflight has a 50-minute workflow ceiling and a 2,970-second deadline from the first step; its worker budget is 2,400 seconds including a 180-second stopped archival reserve, leaving at most 2,220 shared execution seconds. Contract tests are capped at 300 seconds and neural preflight at 1,800 seconds, each further bounded by remaining shared time. No new stage starts without admission. Main timing limits are frozen after pilot cost review. No GPU or paid compute is part of this protocol.

The independent auditor uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and one numerical thread. It reads saved artifacts without model execution. Preserve worker and scientific receipts separately (`study_{phase}_environment.json` and `study_{phase}_status.json` for the controller).

Success would establish bounded performance on this external synthetic QA distribution. It would not resolve unseen languages, broad text comprehension, recursion, real-world transfer, learned mechanism, energy efficiency or independent external replication.

## References

1. Weston et al., [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](https://arxiv.org/abs/1502.05698), especially the task-4 and answer-only evaluation descriptions.
2. [Pinned benchmark provenance review](https://github.com/Agnuxo1/NeuroPixel/blob/99ecaeefdd00dffd3dae444a781ad009c19dff5d/docs/research/10_babi_provenance.md), including publisher loader, licensing, selected members and documented mirror limitations; [publisher dataset card](https://huggingface.co/datasets/facebook/babi_qa/blob/main/README.md).
3. [Pinned author generator README](https://github.com/facebookarchive/bAbI-tasks/blob/ccd8fd6af76d35346109e7bb5f7de9138d055e01/README.rst): the rewrite does not reproduce the published dataset exactly.
