# Item 8 — Classification input and denominator contracts

## Finding and intended scope

The original `classification_metrics()` checks matching prediction/target/role
vector shapes and requires at least one example from each of the four declared
roles. It does not reject additional unknown roles: those observations enter the
global numerator and denominator but disappear from all four per-role summaries.
It also accepts a scalar, shortened or two-dimensional `nll` input and simply
averages it, although that value is meant to be one loss per evaluated example.
Nonfinite or negative losses can be returned. The later strict JSON writer can
reject nonfinite values, but the metric function itself did not enforce its
stated population contract.

The original source was preserved before editing as
`results/research/08_validation/classification_source_original.py`, 20,174 bytes,
SHA-256 `0a95e445e0654eb3666b3efc12a7bc594a237142dc8a7a2dc9527fac274f41ad`.
It is the exact `neuropixel/research/experiment.py` from the closed item-7 commit
`ef25ed6497cd3554d8eb78a0ef278f7b589e1e1f`. The source receipt records this
preservation before any metric tests or correction.

These are demonstrated misuse paths and a prospective tightening of the declared
RoleTask input contract. They are not evidence that the inputs to items 5 or 6
contained unknown roles, malformed losses, or invalid labels. Their recorded
scores and the closed H1 decision are not replaced by this work.

## Narrow correction

The new NumPy-only `neuropixel/research/classification_contracts.py` validates
inputs before the existing scoring formulas or bootstrap can run:

- Prediction, target and role arrays must be nonempty, aligned one-dimensional
  integer arrays. Booleans, floating labels and nonfinite labels are rejected.
- Roles must contain every declared role and no unknown role IDs. Therefore the
  four per-role populations cover the full evaluated population.
- Predicted PAD/zero remains a possible wrong answer; negative predictions and
  PAD/zero targets are rejected under the RoleTask answer convention.
- If supplied, NLL must have exactly the same shape as the label vector and must
  contain finite, nonnegative real losses. It is not broadcast or silently
  flattened into another population.
- The interval switch must be a real boolean. Requested bootstrap seeds and
  repetition counts are validated before RNG construction.
- Wilson inputs must be integer counts, with `0 <= correct <= n` and positive
  `n`; confidence must be finite and strictly between zero and one.

The helper is called by the actual `classification_metrics()` and `wilson()`
functions, and its hash is added to their source inventory. All aggregation,
Wilson and bootstrap formulas remain unchanged. This is a new version of the
input boundary, not a new estimator of NeuroPixel competence.

The helper cannot infer an experiment's vocabulary from the predictions. Upper
vocabulary bounds, correct semantic labels, grouping dependencies and honest
provenance remain part of dataset/model preparation and independent artifact
validation. The new checks do not make arbitrary supplied predictions trustworthy.

## Tests and preserved negative evidence

`tests/test_classification_contracts.py` contains 11 unittest-compatible methods.
Ten can run in the current NumPy environment. They execute the two actual scoring
function ASTs, extracted unchanged from the selected source file, with their real
NumPy/math/NormalDist dependencies and the role list read from `task.py`. This
avoids importing unrelated Torch training code on a machine without Torch. A
separate integration method imports the complete production module under the
declared cloud environment; it is explicitly skipped locally.

The invalid-bootstrap fixtures replace RNG construction with a failing sentinel.
They verify rejection before RNG access and do not draw any bootstrap samples.
The remaining cases are deterministic, hand-specified arrays. There is no model
training, generated task evaluation, or new final-set access in this local audit.

| Stage | Methods | Passed methods | Failed methods | Failed assertions | Skipped methods |
|---|---:|---:|---:|---:|---:|
| Original source | 11 | 3 | 7 | 21 | 1 |
| Corrected source | 11 | 10 | 0 | 0 | 1 |

`unittest` reports 21 failures in the first log because several of the seven
failing methods contain subtests. This is not 21 failed independent experiments
or 21 failed methods out of 11. The complete original transcript and source
snapshot are retained; the expected failures were not replaced by the passing
run.

The valid unequal-population fixture has role sample counts 3, 1, 2 and 2, with
per-role accuracies 1/3, 1, 1/2 and 1/2. It establishes the intended distinction:
micro accuracy is 4/8 = 0.5, macro accuracy across roles is 7/12, and mean
agent/patient accuracy is 5/12. Mean NLL is 1.75 for the specified loss vector.
Another fixture compares the entire corrected summary with the preserved
original output for valid inputs, including the Wilson intervals, by exact
dictionary equality. The original scoring definition is preserved for those
valid data.

## Execution receipts

The original run finished at 2026-10-07T06:33:17.577995+00:00 with return code 1;
the corrected run finished at 2026-10-07T06:33:40.970817+00:00 with return code 0.
The logs report 0.019 and 0.018 seconds for their respective unittest runs.
Outer command times, including interpreter and setup overhead, were
0.14818654600094305 and 0.14276011000038125 seconds. Both used one numerical
thread; admission available RAM was 9.132316589355469 and 9.133167266845703 GiB.
These are sampled available-memory observations, not process peak memory or
energy measurements.

| Artifact | SHA-256 |
|---|---|
| Initial transcript | `ce5f1f8976270cf72736b97f9fdf9fbff8402d8d021842f1aa0f9ee10037cd77` |
| Corrected transcript | `ff5eee876fdbab07fdddb3ea0bb3b238032489c4c6dbe1b78d14f40f334a5e3a` |
| Corrected experiment source | `5fc9ef4d211b2baca37960b0e79ed50a348a58fee34f2c7d6330cb9b48c9532f` |
| Input-contract helper | `29be6bd5ccac6c95ae3ea61f32f80ec3f1c32a993f93c564cae8b3bbe1c9fdd5` |
| Regression source | `3e7ba9654c80392a1cb22907626030bf70a16ada41daf129ae191b9bf3f96fd8` |

The corresponding `classification_initial.json` and
`classification_corrected.json` bind source hashes, return codes, log hashes and
resource scope. Their recorded local result is 10 passes and one explicit
integration skip after correction. Full-module cloud verification is a separate
pending stage at the time of this note; only the item-8 final report may record
its eventual outcome.
