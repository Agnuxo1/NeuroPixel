# Item 9: independent saved-array analysis implementation

Status: ready for the prospective Stage-B source/analysis freeze. This note and
the two implementation files are new additions. The 49 files bound to preflight
source `e63764ccb924ab23d65c2deb94b477387d102c82` were not edited. No Stage-B final
examples, predictions, model runs or outcomes were used to develop this analysis.

The final auditor is `scripts/research_complex_binding_audit.py`; its synthetic
regressions are `tests/test_complex_binding_audit.py`. Stage A has its own
independent `research_item9_preflight_audit.py`; this file intentionally analyzes
only the complete Stage-B study instead of duplicating that preflight workflow.
Neither analyzer is an external independent replication. The final auditor
imports no scientific generator, trainer, model, Torch, or project metric code.

## Exact files and invocation

| File | SHA-256 |
|---|---|
| `scripts/research_complex_binding_audit.py` | `b12ac084aed181b1a22aa96f1b28a1c467d55587e205e0de7f18bb7e32b80e05` |
| `tests/test_complex_binding_audit.py` | `4929c8914653063a7e9a1e9a6c26ee331c12c76a78bf8d657998610ad75a7f99` |

Run only after the complete final archive is explicitly admitted:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/research_complex_binding_audit.py \
  --input /path/to/final-cloud-run/study \
  --recipe /path/to/frozen/docs/research/09_experiment_recipe.json \
  --source-root /path/to/frozen/source \
  --output /path/to/new/item9_analysis.json
```

`--source-root` is optional. When supplied, every source path/hash in the saved
execution plan must match. The complete cloud run's `execution_plan.json` must
be the study directory's immediate sibling in its parent directory. The output
must not exist. A caught artifact/schema/statistical failure writes a failed
receipt with the issue and accumulated input hashes, then returns nonzero; it
does not omit a failed run and analyze a selected subset. Admission and argument
errors before an audit instance exists raise directly.

## Prospective offline runtime binding

The Stage-B execution plan must contain **exactly this value** at the top-level
key **`offline_analysis_runtime`**:

```json
{
  "python": "3.12.14",
  "numpy": "2.3.5",
  "scipy": "1.17.0"
}
```

The auditor requires both the installed runtime and that plan value to equal
its `OFFLINE_RUNTIME` constant before opening development or final arrays. This
is separate from the frozen training runtime: Python 3.12.8, NumPy 2.2.6 and
Torch 2.6.0+cpu. SciPy's version is recorded because it supplies the Student-t
quantile. Thread environment variables are set to one before importing NumPy.
Linux available RAM must be at least 8 GiB at admission and at sampled boundaries
before each run, bootstrap condition and paired-comparison block.

The common bootstrap index matrix is generated explicitly with
`numpy.random.Generator(numpy.random.PCG64(94001)).integers(0, G,
size=(2000, G), dtype=numpy.int64)`. Its complete values are included in the JSON,
with a SHA-256 of their C-order little-endian int64 bytes. No cross-version
bitwise RNG guarantee is claimed. This binds the previously unspecified offline
runtime before final outcomes while retaining the already declared seed, 2000
repetitions, endpoints and grouping rule.

## Structural and numerical checks

The auditor verifies local artifact sizes/hashes, safe relative paths, strict
JSON duplicate/nonfinite handling, NPZ names/shapes/types, ordered target/role/
group/condition/record alignment, and source/config/recipe links. It checks the
complete ten-run inventory and all training/probe/validation records before
opening final prediction arrays. Checkpoints are hashed as bytes without
deserializing or executing them.

The development manifest must match its context anchor, its listed artifacts,
and its fixture-exclusion ledger. Saved train/probe/validation/memorization
scenario facts are inspected directly, with no RNG replay. Probe and memorization
groups must be the declared training prefixes; training/validation/final bags
must obey the canonical hash partition and fixture exclusions. Training-only
majority answers are independently reconstructed from saved training facts.

An independent visible-row parser validates the 10×8 grammar, eight unique
event/role keys, category-correct distinct fillers, query/output cells and bag
hash. It obtains gold directly from the canvas. Each actual final intervention
is then checked against its own base pair, including the exact changed cells
for the two noun swaps, global event relabel and query-only switch. Layout must
preserve the fact set, change positions and remain shared across the eight
queries. `base_target`, `changed_gold`, pair/record hashes and the saved scenario
assignments are checked too. A self-consistent but incorrectly applied
transformation is rejected even when its saved target agrees with the parser.

With eight distinct fillers and the nonidentity layout rule, each bag has 48
nominal final rows but 40 distinct canvases: eight base/query-switch pairs have
multiplicity two and 32 other rows have multiplicity one. The auditor records
the actual multiplicity counts before checking this expectation. For the planned
256 bags the expected counts are 12288 nominal rows and 10240 distinct canvases.
Duplicate canvases must have the same gold, argmax and numerically matching logits;
agreement is not counted as additional independent evidence.

Saved logits must be finite float32 `[N,37]`, with the declared PAD suppression.
First-index argmax is recounted exactly. NLL is recomputed with a stable float64
log-sum-exp after converting the saved float32 logits to exact double inputs.
There are two distinct prospective numerical tolerances:

| Comparison | Absolute tolerance | Relative tolerance | Reason |
|---|---:|---:|---|
| Saved/recomputed NLL and mean CE | `1e-10` | `1e-12` | Both calculations use double arithmetic on the same saved logits |
| Duplicate-canvas float32 logits | `16 × 2^-23` | `16 × 2^-23` | Allow a small float32 library/kernel rounding difference |

Labels, counts, hashes, input alignment, argmax and analytical control probabilities
remain exact. The former shared float32-level tolerance for NLL was deliberately
replaced before freeze; its fail-before witness and exact code preimage are kept.

The four input-only controls are independently reconstructed from visible facts.
Symbolic, role-only, event/category and global bag/category marginal expectations
are compared exactly with saved probabilities; training-majority predictions are
checked against the train-fitted rule. No stochastic draw is invented. In
particular, the auditor does not report randomized-control joint correctness or
all-eight accuracy by multiplying marginals without a declared coupling.

## Estimands, denominators and intervals

For every checkpoint and condition, recount global accuracy, four-role accuracy,
macro agent/patient binding, mean CE and all-eight-scenario accuracy. The complete
point summaries must match those saved by the trainer. With the frozen population,
each condition has 2048 queries, 512 per role, 1024 binding queries and 256 bags.

For each condition relative to base, report both-correct counts, prediction
equality among gold-invariant pairs, and prediction inequality among gold-changing
pairs, for all queries and separately for each role. Eligible denominators are
explicit; zero eligibility produces `null`, not a perfect score. A constant
prediction may be invariant while wrong, so equality is never substituted for
joint correctness.

Bootstrap uses the same 2000 whole-bag draws for every checkpoint, query, condition
and model comparison. Query/condition dependence remains inside each bag. The
implementation first aggregates per-bag means or numerator/denominator counts,
then indexes those small vectors. It never materializes a
`2000 × 256 × 10 × 6 × 8` resampled tensor. The index matrix is approximately
4 MiB; individual numerical resampling temporaries have at most `2000 × 256`
elements. Local resource samples are observations, not a continuous memory proof.

The JSON's conditional score ratios deliberately use sums of **per-bag means**
over a denominator of 256 bags. Such a numerator can be fractional and must not
be read as a query count. Each output labels its units. Counterfactual ratios
instead use actual eligible query counts pooled within each resampled bag; their
denominators can be 2048,1024,512 or zero. All-eight uses counts of wholly correct
bags. CE uses sums of per-bag mean NLL. Paired checkpoint contrasts subtract
signed bag-level metrics before using the common indices. Intervals are percentile
95% intervals conditional on these fixed checkpoints and sampled bags.

The primary across-run interval uses exactly five complete paired NP-minus-
Transformer base-binding differences for seeds40–44. It reports all left/right
values, signed differences, mean, sample SD, range and
`mean ± t(0.975,4) × sample_SD/sqrt(5)`, without clipping. These are five paired
training realizations: initialization and minibatch/firing streams vary together
across seeds. This does not isolate initialization uncertainty. Both families
must have identical logged input hashes and group-index lists at every matched
update. The seed interval and checkpoint-conditional bag intervals are separate;
neither incorporates the whole development/selection/data-generation uncertainty.
Five seeds provide limited evidence for normal-theory coverage.

The predeclared high-competence screen is reported component by component using
unrounded scores. It remains an exploratory operational screen, not historical
H1 or a significance test. Secondary diagnostics do not become independent
replications or a broad architecture-superiority claim.

## Preserved synthetic validation attempts

All tests use the already excluded manual bag `{5,6,7,8}/{17,18}/{27,28}`, literal
semantic truth, or small abstract statistical vectors. They do not call the
scientific generator or create the study's final population. Negative tests
cover self-consistent wrong transformations, geometry/metadata/identity errors,
corrupt artifact hashes and paths, malformed prediction/NLL arrays, first-index
ties, missing/nonfinite seed pairs, signed bootstrap differences, and constant
predictions that are invariant without solving the task.

| Attempt | Outcome | Meaning |
|---|---|---|
| `analysis_test_attempt01` | 18 passed | Initial parser/structure/statistics regressions |
| `analysis_test_attempt02` | 18 passed, 1 failed | New `+1e-8` malformed-NLL witness exposed acceptance under the old float32 tolerance |
| `analysis_test_attempt03` | **19 passed** | Separate float64 NLL/CE tolerances correctly reject that witness |

Attempt03 completed its 19 tests in 0.534 seconds. Each attempt has its own
exclusive JSON receipt and full stdout log under `results/research/09_validation/`;
they contain exact source hashes, runtime, resource admission and thread settings.
Receipt hashes are:

- Attempt01: `04c42d920bd290ed58c1c0ba183b1606937b6624dfa843e68ba39a3f7205eca1`.
- Attempt02: `998d05c0511d65cfb4edb529273f33e16017fc5047f0035d862ba5dda3286d01`.
- Attempt03: `af4a257c3b4ddc2dabbfd2160ea156444215c1f4f5e6c92d9e1fb93e0975b43e`.

`analysis_test_source_manifest.json` maps each attempt to preserved byte-exact
source preimages. The old analyzer/test bytes were recovered by reversing the
recorded local patch and accepted only after their full SHA-256 equaled the
already preserved receipt. No reconstructed source with an unmatched hash was
retained as authentic. Replaying a historical attempt requires placing its two
preimages at their original relative script/test paths in an isolated checkout;
the evidence copies are not extra automatically collected tests.

The independent architecture reviewer reported no schema/gate/calculation blocker.
The methodological reviewer reported no numerical blocker in the five-pair
t interval, common bag bootstrap, signed contrasts or eligibility rules. Their
unit-label and SciPy-provenance clarifications are included here and in output.
Those reviews were static; the real Stage-B saved-artifact audit has not yet run.
The script's remote-provenance, checkpoint-content and sampled-resource limits
remain explicit in its resulting report. Item10 remains outside scope.
