# Item 9 saved-report and figure-receipt review

Review completed at 2026-10-07T10:00:45.060Z by the evidence-review agent within the same project team. This is a read-only review of saved audit outputs, not a new experiment, raw-array recount, external replication, or independent custody attestation. No model, scientific data generator, bootstrap rerun, local executor, or statistical refit was used. Values below are extracted from the saved report; percentages and percentage points multiply saved proportions by 100. Display rounding does not change stored criteria.

## Inputs and provenance

- Offline final archive O: `2b8203f15ec5f6fe190876c80bf29232034603ce`.
- Raw path in O: `results/research/09_offline_runs/37603398040-1-offline/`.
- Reviewed `scientific_audit_summary.json`: Git blob `52cdc2dd0d5c732c7a9eef373347cfb1c091696b`, 715535 ASCII bytes. Root independently verified SHA256 `97d5e1c348343e7a08856d46a5bd67baa5858cdfd27e70e155988cbb869fee89`.
- Reviewed `figures/figure_receipt.json`: Git blob `6e3748650b0b386da57b2115e6fa21a7b7bcf1ec`.
- Summary full-report reference: `scientific_audit_01.json`, 7142199 bytes, SHA256 `2ffcd15900dd583f796f158bac6de9dee1761ba74cf7f4d9ae9ca700d217316e`.
- Scientific source S: `83fe135e301f76bc0c74e30c66bb18e067ca5959`.
- Frozen analysis source SHA256: `b12ac084aed181b1a22aa96f1b28a1c467d55587e205e0de7f18bb7e32b80e05`.
- Scientific recipe SHA256: `7450b1ede40542c5ff940842e42eb2725ac4420946baad6bac920e779a214b97`.
- Inventory archive claim G: `2e58f5deb47f5676e8663d666d9db7ea98e3e080`. Authentication of the Git chain and final archive is supplied by the separate operational audits.
- Scientific analysis runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0; recorded OMP/MKL/OpenBLAS thread variables all equal 1.

The reviewed summary has status `verified` and zero issues. Its projection declares the sole omission `bootstrap.indices`, shape [2000, 256], while retaining the original full report and its identity. This reviewer did not fetch or recompute the omitted index matrix. Root separately reported that the full report, projection equality, and index digest were checked. The gzip is preserved by the worker; root has not decompressed or independently verified that compressed copy. The retained bootstrap algorithm is PCG64 seed 94001, 2000 repetitions, with common bag indices; the saved little-endian int64 digest is `95f989b32c09be3185420ba911b4bcafd0291a9e50aea3fcf88075039f0718c6`.

## Concrete review checks

No discrepancies were found in the checks below:

- The exact ten family/seed runs are present; all 60 final condition cells have 2048 queries and 256 bags.
- Each probe has 512 queries/64 training bags; each validation panel has 1024 queries/128 bags.
- Every stored score-bootstrap denominator is 256 bags. These numerators are sums of per-bag averages, except all-eight, whose numerator counts complete bags, and CE, whose numerator sums per-bag mean NLL.
- Every counterfactual denominator agrees with its condition and role. Empty eligibility returns null estimate and interval, not zero.
- All 60 binding points, 12 family means, five paired differences, paired mean, and t interval in the figure receipt exactly match the saved scientific analysis.
- The figure receipt's full-input bytes and SHA256 match the summary's full-report reference.
- All ten duplicate checks retain 2048/2048 identical argmax predictions and maximum absolute logit difference 0.
- Five matching-family input-stream records each have 4096 updates, equality=true, and saved stream hashes.
- All three competence-screen checks are false, consistently with the displayed values.

The saved maximum final NLL disagreement is at most 8.881784197001252e-16, below the prospectively declared absolute 1e-10 and relative 1e-12 tolerances. This is a reported audit measurement, not a new recount by this review.

## Main results

NeuroPixel mean base binding is **9.62890625%**, versus **49.35546875%** for RelativeTransformer. The saved mean paired NP-minus-TF difference is **-39.7265625 percentage points**, sample SD **1.2302362131463691 pp**, with the untrimmed two-sided t95 interval **[-41.25410251515038, -38.19902248484962] pp**, n=5, df=4.

Base global accuracy is 10.986328125% for NeuroPixel (sample SD 0.6765823467065927 pp; range 10.3515625-11.71875%) and 73.515625% for RelativeTransformer (sample SD 2.584571869627072 pp; range 69.140625-75.341796875%). Base binding ranges are 8.88671875-10.25390625% and 47.8515625-50.68359375%, respectively.

| Model | Seed | Base global % | Base binding % | Base all-eight % | Base CE (nats/query) | Probe binding % | VAL binding % |
|---|---:|---:|---:|---:|---:|---:|---:|
| NeuroPixel | 40 | 10.498047 | 9.765625 | 0.000000 | 2.366164 | 10.546875 | 8.007813 |
| NeuroPixel | 41 | 11.718750 | 10.058594 | 0.000000 | 2.359583 | 7.421875 | 10.937500 |
| NeuroPixel | 42 | 10.351563 | 9.179688 | 0.000000 | 2.361200 | 9.765625 | 8.593750 |
| NeuroPixel | 43 | 11.718750 | 10.253906 | 0.000000 | 2.371649 | 10.937500 | 10.351563 |
| NeuroPixel | 44 | 10.644531 | 8.886719 | 0.000000 | 2.381879 | 10.546875 | 8.203125 |
| RelativeTransformer | 40 | 69.140625 | 47.851563 | 0.000000 | 0.609672 | 47.265625 | 50.000000 |
| RelativeTransformer | 41 | 75.048828 | 50.097656 | 0.000000 | 0.396346 | 49.218750 | 49.804688 |
| RelativeTransformer | 42 | 75.341797 | 50.683594 | 0.000000 | 0.472443 | 48.828125 | 50.000000 |
| RelativeTransformer | 43 | 74.853516 | 49.707031 | 0.390625 | 0.433884 | 48.046875 | 49.804688 |
| RelativeTransformer | 44 | 73.193359 | 48.437500 | 0.000000 | 0.518980 | 48.437500 | 48.828125 |

Per final condition: global n=2048; each role n=512; binding has 1024 queries, with equal agent/patient weighting; all-eight n=256 bags. Probe binding n=256 queries; validation binding n=512. The all-eight result for RelativeTransformer seed43 in base is exactly 1/256; the other four base seeds are 0/256. NeuroPixel is 0/256 for every seed and condition. For the two agent-patient swap conditions, each all-eight bundle comprises eight query-conditioned cases: four queries from each of two distinct fact grids, not eight questions about one common transformed scenario. The base competence-screen endpoint is unchanged.

## All six conditions

The following are saved means and sample SDs over five paired training realizations, not new confidence intervals. A training realization varies initialization and minibatch/firing streams jointly.

| Condition | NP binding mean +/- SD (%) | TF binding mean +/- SD (%) |
|---|---:|---:|
| base | 9.628906 +/- 0.580213 | 49.355469 +/- 1.177153 |
| swap_queried_agent_patient | 9.863281 +/- 0.693978 | 49.375000 +/- 1.620329 |
| swap_other_agent_patient | 9.550781 +/- 0.554151 | 49.355469 +/- 1.166982 |
| relabel_events | 9.902344 +/- 1.097493 | 48.886719 +/- 1.599596 |
| query_switch | 9.628906 +/- 0.580213 | 49.355469 +/- 1.177153 |
| layout_permutation | 9.531250 +/- 0.550698 | 49.199219 +/- 0.982404 |

The query-switch condition reorders the same eight visible queries already represented by base for each bag. Its identical marginal scores are therefore expected, not an additional replication. Its paired response-change diagnostics remain distinct.

The NeuroPixel operational competence screen is not met: all-five base binding >=0.95, all-five base all-eight >=0.90, and every-condition/every-seed binding >=0.90 all evaluate false. The screen is descriptive and predeclared; it is not the historical H1 or a significance test. The observed Transformer advantage in this recipe does not establish general architecture superiority or complete role binding: Transformer binding remains near the 50% analytic role/event-category reference and almost no bags have all eight answers correct. These scores alone do not identify the learned algorithm.

## Stored conditional intervals and counterfactual interpretation

| Seed | NP-minus-TF base binding (pp) | Stored whole-bag bootstrap 95% interval (pp) |
|---|---:|---:|
| 40 | -38.085938 | [-40.039063, -36.035156] |
| 41 | -40.039063 | [-41.894531, -38.281250] |
| 42 | -41.503906 | [-43.261719, -39.843750] |
| 43 | -39.453125 | [-41.308594, -37.597656] |
| 44 | -39.550781 | [-41.406250, -37.500000] |

These intervals condition on each fixed checkpoint pair and sampled bags. They do not replace the five-training-realization t interval above. The report preserves 30 paired checkpoint/condition rows, eight metrics each, without pooling seeds or conditions as independent observations.

For all-role counterfactuals, joint correctness always has denominator 2048 query pairs. Invariant/change eligibility is respectively: base 2048/0; queried agent-patient swap 1024/1024; other-event swap 2048/0; event relabeling 2048/0; query switch 0/2048; layout permutation 2048/0. Role-specific joint denominators are 512.

For query switch, prediction-change rates across seeds40-44 are:
- NeuroPixel: [0.01953125, 0.013671875, 0.01953125, 0.1884765625, 0.0078125].
- RelativeTransformer: [0.8681640625, 1, 1, 0.9990234375, 0.9775390625].

Corresponding all-role joint-correct rates are:
- NeuroPixel: [0, 0, 0, 0.001953125, 0].
- RelativeTransformer: [0.5234375, 0.60546875, 0.6240234375, 0.6298828125, 0.607421875].

Changing a prediction when gold changes is not sufficient for correct role binding. Under the queried agent-patient swap, Transformer changes only [0.09765625, 0.0556640625, 0.0732421875, 0.0517578125, 0.1513671875] of the 1024 changed-gold queries. Role-specific joint-correct rates for this intervention are [0.046875, 0.029296875, 0.052734375, 0.021484375, 0.078125] for agent and [0.037109375, 0.03125, 0.03125, 0.044921875, 0.064453125] for patient. Large all-role joint scores can include the unchanged action/place answers, so the role-specific diagnostics must remain visible.

## Controls, grouping, and duplicates

The saved final panel contains 256 bags, 12288 nominal rows and 10240 unique canvases. Multiplicity counts are 8192 unique canvases appearing once and 2048 appearing twice. Each bag contributes 48 correlated nominal rows. None is an independent training replication.

| Input-only control | Expected global % | Expected binding % |
|---|---:|---:|
| Symbolic parser | 100 | 100 |
| Role-only | 50 | 50 |
| Event-category | 75 | 50 |
| Bag-category | 37.5 | 25 |
| TRAIN role majority | 10.888671875 | 8.7890625 |

Control totals are over 12288 nominal rows (3072 per role, 6144 binding). Uniform controls report exact marginal probabilities, not sampled predictions; there is no declared coupling that would justify reporting random-control joint/all-eight accuracy. The majority control was fitted only to TRAIN. The ledger excludes 100 known fixture groups, with 2048 training, 128 validation, 64 probe and 4 memorization bags; probe/memorization are training subsets. The fixed probe uses 64 TRAIN bags with assignment/layout views distinct from the resampled training minibatches; probe accuracy is not an exact-minibatch memorization measurement.

## Figure receipt and remaining scope

The figure receipt status is `rendered`, helper SHA256 `f6d1a9d9f03858de316036b6af5197550426f35d4a08a922c23861358d74e05a`, and it records Matplotlib 3.10.8, Python 3.12.14, NumPy 2.3.5, all six thread variables equal 1. Minimum sampled available RAM is 14.573951721191406 GiB; this is not a peak-RSS measure.

| Figure | Bytes | SHA256 |
|---|---:|---|
| base_paired_difference.png | 155416 | `2e03d3f022fe211c808f62ff8ed2f5b33724a8724ecb632996bf3dc6f5d602ef` |
| base_paired_difference.svg | 11849 | `5c45b04db9389423c1fbb99e020b4d730c5a656887b4d55d36482ac6923bb5cf` |
| binding_by_condition.png | 206446 | `952e10a1c250838253351e01315de35f3142cbbdd5d37786469b5bee28537ba6` |
| binding_by_condition.svg | 30737 | `beeecf8e4469b852b152ef427b81edb2dddb1cd12365eb50a81cc0f1a16f754a` |

The immutable receipt retains `visual_review_pending=true`. Root subsequently reported that both PNGs were hash-verified and visually inspected without clipping, preserved in commit `109ba71e30ebe1b28aa0d35d1dba413ad560d2d8`. That later visual check is root's observation, not one performed by this reviewer. It does not alter the original receipt.

No numerical or schema discrepancy was found in this review. Historical H1 remains closed and not supported. Training convergence, held-out grammar generalization, broad scientific novelty, and external independent replication are not established by the current result.

## Documentary correction record

Corrected at 2026-10-07T10:03:42.220Z. The original review remains preserved as Git blob `c6769eaffd09f2a8cf427833fd95a7332a0b552f`. This revision only removes the unsupported attribution of gzip verification to root, states the query-conditioned two-grid interpretation of swap all-eight bundles, and clarifies that the TRAIN-bag probe is not exact-minibatch memorization. All numerical values, tables, saved intervals, and scientific criteria are unchanged. No new statistical calculation or execution accompanied these corrections.
