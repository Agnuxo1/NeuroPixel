# Item 3 — Executable elementary controls

## Implementation clarification before outcomes

Recorded on 2026-10-06, after protocol freeze `cab2e08` and before this item's
first evaluation. The programme owner confirmed these choices before execution:

- The frozen control panel uses the original `RoleTask` at 8×8 and `RoleTaskFar`
  at 12×12, following `scripts/phase3.py`. These diagnostic controls use the
  generators' original 80/20 train/test triple partition. The prospective neural
  training partition of 70/10/20 is outside this item's scope.
- For each task, split seed 0/101/202 and train/test partition, generate 1,024
  examples per query role. Initialize one CPU sampling generator with seed 61004
  at the start of each dataset, sample in the declared `ROLES` order, and preserve
  that order. Reset the category draw generator to 61005 for each dataset.
- A separate retrospective Far diagnostic reconstructs the original recipe:
  12×12, task split seed 0, test sample seed 5, 2,000 examples, without forced role
  balance. It uses current software; there is no historical input hash establishing
  byte-for-byte identity with the old run. Do not calculate a paired significance
  comparison against its old 0.743 aggregate without old per-example predictions.
- Counterfactuals swap the AGENTE/PACIENTE fillers. Swapping can move a triple
  across the original split; both memberships are recorded. These are paired task
  diagnostics, not an additional untouched holdout. Counterfactual targets come
  from independently transformed generator fillers, not from the tested solver.
- Wilson intervals apply to each marginal role rate only. Expected probability
  scores are analytic expectations, not realized binary counts. Original and
  swapped examples are dependent pairs; no binomial interval treats them as
  twice as many independent observations.

The frozen protocol files are unchanged. Executed results, commands, hashes and
limitations follow below.

## Executed result

The controls passed on **51,152 original examples**, comprising 49,152 examples in the new balanced panel and 2,000 in the separate retrospective recipe reconstruction. The exact reference made **zero errors**. All **25,523 nominal counterfactual pairs** changed the required answer; the exact reference answered both members correctly in every pair. The category control assigned the same distribution and made the same seeded prediction to both members, correctly answering exactly one. These results concern the original generators and these controls; no neural training or checkpoint evaluation took place.

The final execution completed at `2026-10-06T22:01:04.176128+00:00` in 6.47 seconds on CPU with two PyTorch threads, deterministic algorithms enabled, Python 3.12.14, PyTorch 2.14.1+cpu and NumPy 2.3.5. The implementation starts from Git commit `cab2e086de1fe533079902f12d4318529ff5599e`; the JSON records exact source-file hashes for the executed working-tree code.

### What each control can see

- **Adjacent exact reference:** the visible canvas, public vocabulary and query position. It finds the matching fact role outside the query cell and reads its right neighbor.
- **Far exact reference:** the same public information. It reads the sole filler on the matching fact role's row, independent of horizontal distance.
- **Category probability control:** the query role and the bag of visible tokens. It ignores fact-role locations and token positions. Its candidates are the two visible nouns for AGENTE/PACIENTE, the visible verb for ACCION, and the visible place for LUGAR; candidates receive uniform probability.
- **Category seeded draw:** one draw from that distribution using private RNG seed 61005. Neither targets, task objects, split membership nor generator metadata are accepted by predictor interfaces. The evaluator uses labels only after predictions are returned.

All category knowledge is supplied through the public vocabulary. These controls are deliberately informed about the task schema. They are accuracy ceilings and shortcut diagnostics, not learned baselines or claims about general language understanding.

## New balanced panel

Each of the following datasets contains exactly **1,024 examples per role**, or 4,096 overall. In every dataset, the matching exact reference achieved **1,024/1,024 for each role and 4,096/4,096 overall**. Its per-role 95% Wilson interval is **[99.6263%, 100%]**. ACCION and LUGAR also achieved 1,024/1,024 under the category draw, with the same marginal interval.

The category control's analytic expected score is **512/1,024** on each nominal role, **1,024/1,024** on each other role, and **3,072/4,096 = 75%** overall. The fractional expected score is not an observed binomial success count. The actual seeded draws are shown below. Interval endpoints refer only to the indicated marginal role; no overall binomial interval treats the balanced-role mixture as one homogeneous sample.

| Dataset | AGENTE: correct/n; 95% Wilson | PACIENTE: correct/n; 95% Wilson | Category total correct/n |
|---|---|---|---|
| [adjacent_s0_train](../../results/research/controls/adjacent_s0_train.jsonl.gz) | 507/1,024; [46.46%, 52.57%] | 500/1,024; [45.78%, 51.89%] | 3,055/4,096 (74.585%) |
| [adjacent_s0_test](../../results/research/controls/adjacent_s0_test.jsonl.gz) | 524/1,024; [48.11%, 54.22%] | 506/1,024; [46.36%, 52.47%] | 3,078/4,096 (75.146%) |
| [adjacent_s101_train](../../results/research/controls/adjacent_s101_train.jsonl.gz) | 532/1,024; [48.89%, 55.00%] | 493/1,024; [45.10%, 51.21%] | 3,073/4,096 (75.024%) |
| [adjacent_s101_test](../../results/research/controls/adjacent_s101_test.jsonl.gz) | 532/1,024; [48.89%, 55.00%] | 502/1,024; [45.97%, 52.08%] | 3,082/4,096 (75.244%) |
| [adjacent_s202_train](../../results/research/controls/adjacent_s202_train.jsonl.gz) | 511/1,024; [46.85%, 52.96%] | 536/1,024; [49.28%, 55.39%] | 3,095/4,096 (75.562%) |
| [adjacent_s202_test](../../results/research/controls/adjacent_s202_test.jsonl.gz) | 520/1,024; [47.72%, 53.83%] | 524/1,024; [48.11%, 54.22%] | 3,092/4,096 (75.488%) |
| [far_s0_train](../../results/research/controls/far_s0_train.jsonl.gz) | 507/1,024; [46.46%, 52.57%] | 493/1,024; [45.10%, 51.21%] | 3,048/4,096 (74.414%) |
| [far_s0_test](../../results/research/controls/far_s0_test.jsonl.gz) | 524/1,024; [48.11%, 54.22%] | 519/1,024; [47.62%, 53.74%] | 3,091/4,096 (75.464%) |
| [far_s101_train](../../results/research/controls/far_s101_train.jsonl.gz) | 532/1,024; [48.89%, 55.00%] | 533/1,024; [48.99%, 55.10%] | 3,113/4,096 (76.001%) |
| [far_s101_test](../../results/research/controls/far_s101_test.jsonl.gz) | 532/1,024; [48.89%, 55.00%] | 521/1,024; [47.82%, 53.93%] | 3,101/4,096 (75.708%) |
| [far_s202_train](../../results/research/controls/far_s202_train.jsonl.gz) | 511/1,024; [46.85%, 52.96%] | 507/1,024; [46.46%, 52.57%] | 3,066/4,096 (74.854%) |
| [far_s202_test](../../results/research/controls/far_s202_test.jsonl.gz) | 520/1,024; [47.72%, 53.83%] | 516/1,024; [47.33%, 53.45%] | 3,084/4,096 (75.293%) |

The observed category total ranges from **74.414% to 76.001%** across these datasets, around its exact 75% expectation. The difficult-role expectation remains **50%**. This illustrates why an overall score near 75% does not, by itself, establish correct agent/patient binding. The three split settings reuse the generator, vocabulary and sampling recipe; they are conditional diagnostic panels, not independent laboratory replications.

## Separate retrospective Far recipe reconstruction

The original sampling recipe produced the following actual query counts. No role balancing was imposed in this diagnostic.

| Role | Exact correct/n | Category expected accuracy | Category seeded correct/n | Seeded accuracy; 95% Wilson |
|---|---|---|---|---|
| AGENTE | 471/471 | 50.0% | 244/471 | 51.805%; [47.296%, 56.284%] |
| ACCION | 515/515 | 100.0% | 515/515 | 100.000%; [99.260%, 100.000%] |
| PACIENTE | 476/476 | 50.0% | 243/476 | 51.050%; [46.569%, 55.515%] |
| LUGAR | 538/538 | 100.0% | 538/538 | 100.000%; [99.291%, 100.000%] |

The exact reference achieved **2,000/2,000**. The category draw achieved **1,540/2,000 = 77.0%**. Its role-weighted expected number correct is **515 + 538 + (471 + 476)/2 = 1,526.5**, hence **76.325% expected overall**, rather than 75%. Its nominal macro expectation is still 50%.

The historical model aggregate of 74.3% is below this elementary expectation descriptively. **No paired significance claim follows:** the historical model's individual predictions and input hash are unavailable, and the current software reconstructs its sampling recipe rather than establishing archived input identity. The present control result is separately labeled retrospective.

Dataset SHA-256: `e1c572978ddfd6182499a42c6396e34ca6ac1e64f21be4786e11c2727f763a9d`. Its [per-example predictions](../../results/research/controls/far_historical_recipe_s0_test.jsonl.gz) retain all original canvases, labels and control outputs.

## Paired agent/patient intervention

For every nominal query, interchange the two noun fillers while preserving the role tokens, query position, query token and complete bag of tokens. Targets for the altered inputs are derived from independently transformed generator fillers, not by asking the exact solver for its own ground truth. The swap can cross from an original train triple into a test triple or vice versa; every artifact records both memberships. No filtering changes the declared original sample.

| Diagnostic across 25,523 nominal pairs | Exact reference | Category seeded control |
|---|---|---|
| Pairs with a changed prediction | 25,523 | 0 |
| Pairs with both members correct | 25,523 | 0 |
| Correct members per pair | 2 | 1 |
| Mean accuracy within a pair | 100% | 50% |

The category distribution, input token bag and query are unchanged in all pairs. The repeated seed couples the two draws deliberately; it is not a second independent random trial. No confidence interval treats the two members as independent observations. Per-role pair counts and train/test transition counts are in the JSON.

An elementary bound explains the result. A predictor using only the token bag and query must give both members the same distribution. Their correct noun tokens differ, so the probabilities of the two answers sum to at most one. Its average expected accuracy across the pair is therefore at most one half. The uniform two-noun control attains that bound. This is a deduction from the intervention and information restriction, not a newly discovered learning mechanism.

## Artifacts, tests and reproduction

- [Machine-readable results](../../results/research/03_controls.json): all 13 datasets, complete total and per-role counts, expected scores, marginal Wilson intervals, paired diagnostics, source/environment/configuration metadata and dataset/export hashes.
- [Control implementation](../../neuropixel/research/controls.py), [runner](../../scripts/research_controls.py), and [tests](../../tests/test_research_controls.py).
- Thirteen gzip JSONL exports, linked by dataset above and referenced relative to the result JSON. They contain **51,152 base rows and 25,523 nested counterfactual records**, occupying **1,863,451 bytes**. Each record retains the input and predictions; compressed and uncompressed SHA-256 values are recorded. The integer-array dataset hash includes names, shapes, explicit little-endian int64 dtype and values.
- Full suite: **36 passed, 1 skipped in 1.31 seconds**. The skip is the existing CIFAR-10-dependent test, because those data were not downloaded. The 24 new control cases include hand-authored fixtures, both real generator schemas, input-only interfaces, counterfactual ground truth, reproducibility, Wilson endpoints, deterministic export and interrupted-export preservation.

The successful final artifact check decompressed all 13 files, validated compressed and JSONL hashes, counted every record, and independently recomputed exact and category binary counts from stored predictions.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /workspace/scratch/61d6e3e6e688/neuropixel-env/bin/python \
  scripts/research_controls.py --threads 2

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /workspace/scratch/61d6e3e6e688/neuropixel-env/bin/python \
  -m pytest -o addopts='-p no:nengo' -q
```

Use the equivalent environment interpreter on another machine. An alternative `--output` path preserves the committed reference result during reproduction. The script validates the frozen protocol hashes before execution. A repeated scientific run should reproduce dataset hashes, JSONL hashes and scores in the same environment; timestamps and elapsed times naturally change.

### Recorded persistence incident and recovery

The first execution returned complete metrics, but a subsequent integrity check found three gzip files truncated before their end-of-stream markers: `adjacent_s0_train`, `far_s0_train` and `far_s202_test`. Their observed sizes were 33,836, 98,556 and 32,926 bytes, respectively, rather than the 136,858, 161,485 and 159,491 bytes recorded on write completion. No concurrent benchmark/test writer was found; the cause is **not established**.

The writer was changed to use a unique temporary file, close and `fsync`, then atomic replacement of the final path. The JSON summary uses the same atomic mechanism. A new test checks that an interrupted export preserves the previous complete destination. One complete retry was justified by this concrete artifact failure; no scientific settings or selection rule changed. All dataset hashes, scores, per-example JSONL hashes and intended compressed hashes were identical to the first attempt. The final JSON retains the first manifest's hash, code hashes, timestamps, observed failures and equality checks. The recovery is a persistence check, not an extra independent experimental replicate.

## Conclusion and remaining limits

Item 3 is complete: executable controls and the paired shortcut diagnostic have been implemented, tested, run and retained. The original adjacent and far schemas admit exact elementary solutions. A category shortcut reaches 75% expected accuracy on balanced queries, and 76.325% for the reconstructed historical Far query mixture, while remaining at chance on nominal binding and unable to solve both members of the swap intervention.

These results establish task ceilings and expose an overall-accuracy shortcut. They do not establish which neural architecture learns the intended relation under matched training, whether a learned model uses that shortcut, statistical superiority over unavailable historical predictions, outside-generator generalization, physical energy savings, or a novel mechanism. Those questions remain outside this completed item.
