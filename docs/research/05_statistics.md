# Item 5: replication statistics

This implementation specification precedes item-5 result analysis. It implements
the frozen protocol's replication panels and the item-4 selection of
`relative_transformer` as the fixed reference. It does not modify the training
recipe, tune configurations, or pool historical results into the new panels.

## Units and panels

`summarize_replications(records, reference_family, protocol)` consumes evaluation
rows with `status`, `config` and `metrics`. The caller must verify source,
checkpoints, dataset identity and saved predictions before supplying recounted
metrics. This pure statistics function does not read files or execute models.
Extra provenance fields in a configuration do not alter the pairing key.

Runs are keyed by `(family, split_seed, init_seed)`. The initialization panel uses
initializations 10, 11, 12, 13 and 14 on split 0. The split panel uses splits 0,
101 and 202 at initialization 10. Their `(split=0, init=10)` run is shared. These
are seven unique configurations per family, **not seven IID replications**.
The panels are summarized separately and never combined for H1.

Each family retains every planned value in seed order, including `null` for an
unavailable value. Summaries report the observed count, expected count, mean,
sample standard deviation and range for binding, global accuracy, macro accuracy,
cross entropy and each role's accuracy. Missing optional metrics are null. Missing,
unfinished or duplicate runs and invalid/non-finite binding metrics are identified
explicitly. No duplicate is resolved by selecting a preferred result.

## Paired run differences and H1

For each baseline and panel, pair matching seeds and form
\(d_i=b_{NP,i}-b_{reference,i}\), where
\(b=(a_{AGENTE}+a_{PACIENTE})/2\). The summary preserves every signed difference.
For \(n\geq2\), its two-sided 95% Student t interval is

\[
\bar d\;\pm\;t_{0.975,n-1}\frac{s_d}{\sqrt n},\qquad
s_d=\sqrt{\frac{\sum_i(d_i-\bar d)^2}{n-1}}.
\]

The implementation uses SciPy's Student t quantile. The interval is not clipped.
For constant differences, SD is exactly zero and the interval is degenerate.
For one pair, SD and interval are null; for zero pairs, the mean and range are
also null. An interval from an incomplete panel is descriptive only.

H1 has positive support only when **all** of these checks hold, without rounding
the estimates before comparison:

- Exactly five complete, finite initialization pairs against the fixed Transformer.
- Mean NeuroPixel binding accuracy is at least 0.90.
- Mean paired advantage is at least 0.05.
- The paired t interval's lower endpoint is strictly greater than zero.

A missing, failed, duplicate or non-finite primary pair prevents positive support.
The split panel cannot rescue an incomplete or failing initialization panel.
Other baseline comparisons are descriptive and have no multiplicity adjustment.
The t calculation assumes approximately independent, normally distributed run
differences; five or three seeds do not establish those assumptions empirically.
The split interval is conditional on initialization 10, and the initialization
interval is conditional on split 0.

## Conditional paired example bootstrap

`paired_binding_interval(pred_np, pred_ref, roles, seed=61003, repetitions=2000)`
accepts prediction mappings containing integer `prediction` and `target` arrays,
or two boolean correctness vectors. Mappings must have identical ordered targets;
any embedded role vectors must agree with the supplied roles. Raw class IDs are
rejected because accuracy cannot be calculated without targets.

Within each nominal role separately, the method resamples common example indices
with replacement. Equivalently, it resamples signed per-example correctness
differences in \(\{-1,0,1\}\). Agent and patient means each receive weight 0.5,
including when their sample sizes differ. A private NumPy generator controls the
draws. The interval uses the 2.5th and 97.5th percentiles of 2,000 bootstrap
differences. Action and location do not contribute to this endpoint.

This interval is **conditional on the evaluated checkpoints and examples**. It
does not estimate uncertainty across training runs and never enters the H1 rule.

## Output contract

The main result contains `panels.initialization`, `panels.split`, `decision` and
`record_inventory`. Each panel contains `families` and `paired_differences`.
`decision.positive_support` is always Boolean; `decision.status` is `supported`,
`not_supported`, or `incomplete`. Every absent statistic is represented as JSON
`null`, never NaN or infinity. The bootstrap returns `estimate`, `interval_95`,
role counts, method, seed, repetitions and its conditional scope.

## Primary methodological references

- [NIST, Analysis of paired observations](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm): paired differences, sample SD and n−1 degrees of freedom.
- [SciPy, bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html): common resampling indices for paired observations. The implementation here explicitly stratifies nominal roles and uses the percentile method fixed in the protocol.

These references were checked on 2026-10-06. They support the calculation; they do
not establish its assumptions or scientific importance for this small experiment.
