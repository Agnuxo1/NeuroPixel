# Item 8: routing-summary correction

This is a source-level metric correction and deterministic regression exercise.
It does not train, load or evaluate a model, reconstruct historical routing
assignments, change the closed item-6 study, or open a later programme item.

## Finding and original scope

The original `scripts/phase3.py:t_grow2` calculated `pick = sc.argmax(0)` and
stored `pick.float().mean().item()` as `tema{ti}_elige_lienzo`. Each `pick` is a
zero-based index of one expert in the current canvas list. It is a nominal
assignment, not an ordinal measurement. Its arithmetic mean does not identify
the chosen expert or provide routing frequencies, prediction accuracy, or
agreement with an independently specified ownership mapping.

For example, assignments `[1, 1]` and `[0, 2]` both have mean 1.0, although their
counts over three slots are `[0, 2, 0]` and `[1, 0, 1]`. Renumbering the experts
also changes the numerical mean without changing the underlying assignments.
The preserved fail-before fixture executes the exact original AST expression on
a minimal tensor-like object, using these exactly representable values. It
contains no Torch implementation, model execution, randomness or real examples.

The full callsite inspection found this ambiguous scalar only in `t_grow2`.
`t_grow(..., mode="grow")` uses the same scanner argmax but previously saved
only downstream answer accuracy. Both drivers now report the same descriptive
routing schema. The single-model sequential branch has no expert selection and
is unchanged. `scripts/battery.py` instead explicitly compares assignments to
its declared A-to-slot-0/B-to-slot-1 convention. That is a different statistic
and is outside this patch; it is not evidence of an oracle-optimal choice.

The pre-edit driver is preserved as
[`routing_phase3_original.py`](../../results/research/08_validation/routing_phase3_original.py),
28,698 bytes, SHA-256
`e12ceeeadfcf875ec3b1f0b584f58226fdd7b2f077f05bd09d807fa6eb340ca3`.
The source receipt and expected failing fixture are preserved beside it.

## Corrected contract

The standard-library helper is
`summarize_routing(picks, n_slots, *, topics=None, correct_slot_by_topic=None, argmax_scores=None)`
in [`routing_metrics.py`](../../neuropixel/research/routing_metrics.py).

| Field or input | Meaning and boundary |
|---|---|
| `picks` | One integer slot ID per example, in `[0, n_slots)`. Booleans, floats, negative IDs and IDs outside the declared inventory are rejected. |
| `n` and `counts` | The number of examples and exact assignment counts, including zero-count unused slots. Empty samples are rejected rather than given a fabricated denominator. |
| `fractions` | Each count divided by `n`, without display rounding. Pooling uses all assignments, so unequal batches/topics receive their observed example weights. |
| `modal_slot_ids` | Every slot attaining the largest count. This is a frequency mode, not the per-example decision or an accuracy metric. |
| `unique_modal_slot_id` | The modal index only when unique; `null` when the largest frequencies tie. |
| `topics` / `by_topic` | Optional aligned, genuinely known nonnegative integer labels and their count contingency. Labels are not inferred from the selected experts. Without labels the contingency is `null`. |
| `argmax_scores` / `argmax_ties` | Optional example-by-slot scores. Finite scores must reproduce the supplied picks using the first maximum in slot order. The summary counts examples with multiple exactly equal maximum scores; without scores this statistic is `null`, not zero. |
| `correct_slot_by_topic` / `mapped_slot_agreement` | Optional explicit ownership-label mapping. Every observed topic must have a valid mapped slot; extra valid but unobserved labels are permitted. The metric is agreement with that mapping, not answer accuracy, specialist quality or oracle optimality. Without a mapping the metric is `null`. |

Topic labels in the original driver are known from the `tasks[ti]` source of
each evaluation batch. The driver supplies no correct-slot mapping: adaptive
creation, inherited weights and subsequent updates do not establish a unique
correct expert for each topic. Answer accuracy remains the separately computed
`tema{ti}_escaner` metric, using the original predictions and targets.

The score-tie rule is exactly the original first-index argmax rule. It is not
changed to a random or more favourable expert. Renaming a *fixed assignment*
permutes its counts and preserves mapped-slot agreement when the mapping is
renamed consistently. The routing decision itself can depend on expert order
when scores tie. Frequency ties and score ties are different quantities and
are reported separately.

## Driver output and migration

Both growth drivers now emit:

- `tema{ti}_routing`: a schema-version-1 summary for that known topic;
- `routing`: one pooled summary with an explicit per-topic contingency.

They use the `pick` and `sc` arrays already calculated by the driver. Tensor to
CPU/list conversion and deterministic counting add no forward pass, sampling,
optimization step or random draw. The training calls, task definitions,
thresholds, clone/refresh policy, scanner scores, first-maximum selections and
downstream answer-accuracy expressions remain unchanged.

New `grow2` outputs no longer contain the ambiguous `tema{ti}_elige_lienzo`
field. Consumers must migrate to the counts/fractions, not assume that an old
scalar is a fraction or a new scalar is an accuracy. There is deliberately no
compatibility alias that would preserve the mistaken interpretation.

Existing result JSON files are unchanged. In particular, the historical
[`grow_novedad.json`](../../results/phase3/grow_novedad.json) stores three slots
and scalar means approximately 0.2385, 1.0215 and 1.9955. Those scalars do not
reconstruct the three assignment frequencies. The archived trajectory log also
does not recover per-example picks. Neither the means nor a different later
study's assignments can repair those historical frequencies. No rerun is
claimed by this correction.

## Validation scope

The preserved legacy fixture has one passing source-expression check and one
expected failed distinguishability check: `AssertionError: 1.0 == 1.0`. See
[`routing_legacy_failure.log`](../../results/research/08_validation/routing_legacy_failure.log)
and its JSON receipt. This is a reproduced metric-information loss, not a
failed scientific training run.

[`test_routing_metrics.py`](../../tests/test_routing_metrics.py) covers the
counterexample, unused slots versus empty input, invalid IDs and metadata,
explicit mapping completeness, label permutation, example-weighted pooling,
modal ties, exact score ties, malformed/nonfinite scores, and AST integration
of both original drivers. The AST guard compares original training, model
construction, task-test generation, scanner/prediction stacking and routing
expressions without importing or executing the driver.

Local validation uses one CPU thread and requires at least 8 GiB available RAM.
Its execution log and source-hash receipt are saved under
`results/research/08_validation/routing_*`. The separate pinned cloud validation
is coordinated by root; this note does not claim an unexecuted cloud result.

The corrected local suite ran once on 2026-10-07 under Python 3.12.14:
**16 tests passed, zero failures**, reported test time 0.030 s. Available RAM
was 9.1268310546875 GiB before and 9.113723754882812 GiB after the short process;
these are sampled observations, not a continuous minimum or process peak RSS.
The independent static review found no concrete blocker in the helper,
integration or focused tests. No scientific driver was executed.
