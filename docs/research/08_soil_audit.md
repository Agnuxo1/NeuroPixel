# Item 8 — Soil evaluation audit and narrow corrections

Date: 2026-10-07 UTC. Scope: isolated source audit and deterministic synthetic fixtures. No competition data, historical jobs, model training, competition submission or new Soil performance estimate was used. Item 9 remains unopened. The historical claim of 24 labelled samples comes from coordination notes; this audit did not recover those samples or their predictions.

## Preserved source and findings

Original `kaggle/soil/soil.py` was preserved before modification as [soil_source_original.py](../../results/research/08_validation/soil_source_original.py), with [provenance](../../results/research/08_validation/soil_source_original.json): **9,621 bytes**, SHA256 **`501ed7c643c8564c5b28efd1f6a027fa3c7f40f6d13d231f2e59cfafd83bb70f`**, parent item-7 closure `ef25ed6497cd3554d8eb78a0ef278f7b589e1e1f`.

| Finding in inspected source | Correction | Interpretation limit |
|---|---|---|
| `mean_cdf` was computed from all eligible labelled samples before CV | Fit a separate mean from each fold's training IDs; supply only those labels to model training | The old baseline used held-out labels. This finding alone does not show the model's gradients used those labels, or quantify the correction to any historical score |
| A log labelled by fold printed accumulated means from all folds so far | Emit separately named `fold_metrics` and `pooled_to_date_metrics`; pooled means weight each sample once | The old final pooled mean was not itself an unweighted mean of fold means; the defect was the intermediate log's ambiguous scope |
| CDF conversion clipped negative increments and renormalized malformed input | Validate shape, finite values, bounds, monotonicity and endpoint before metric/conversion; never repair invalid input inside the validator | The existing model inference path still clamps its output to [0,100]; “no clipping” describes the new validator, not that unchanged path |
| Normalized label keys could overwrite one another | Reject duplicate/colliding normalized label and phone names; retain original label IDs and photo-to-group/device metadata | Filename parsing is still heuristic. Collisions or aliases in unavailable physical samples cannot be audited from source alone |
| Fold counts/membership were not explicitly checked | Reject empty folds, invalid counts, duplicates, unknown IDs, overlap and incomplete validation coverage | These checks use the declared normalized sample identity, not independent physical-sample authentication |
| Scores and predictions were not retained per sample | Save unrounded CDFs, scores, group IDs, fold membership, reference curves, source hashes and versions in a unique CV evidence file | Preserved prediction CDFs are the output of the existing inference function, after its historical clamp |

The NumPy-only [evaluation helper](../../kaggle/soil/evaluation.py) is integrated into [soil.py](../../kaggle/soil/soil.py). No tiling, augmentation, optimizer, dynamics or architecture was changed in this Soil patch. The seven historical scientific sources were not edited by this task; the driver now records the actual `neuropixel/model.py` hash as a dependency because other item-8 work may change that dependency.

## Metric definition and CDF contract

The historical calculation is retained, under the explicit name `trapezoidal_absolute_cdf_log10_mm_percent`:

\[
D(C,\widehat C)=\sum_{i=0}^{9}\frac{|C_i-\widehat C_i|+|C_{i+1}-\widehat C_{i+1}|}{2}
\left[\log_{10}(s_{i+1})-\log_{10}(s_i)\right].
\]

The eleven diameters are `[0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200]` mm. CDF values are percentages, so score units are percentage points times log10 diameter ratio; lower is better. The operation is the composite trapezoidal rule on absolute differences at those nodes, consistent with NumPy's definition [S1]. The existing training loss uses CDF fractions, giving the corresponding factor-of-100 scale difference before its additional auxiliary term.

This is not generally the exact Wasserstein-1 distance between discrete point masses at the support values. SciPy's discrete Wasserstein API integrates the absolute difference of the actual step CDFs [S2]; for common support this gives interval rectangles rather than the endpoint-average rule above. The deterministic first-support versus last-support fixture has discrete log-W1 × 100 equal to **500**, whereas this repository's trapezoidal score is **`500 − 50 log10(200/63)`**. Nor does applying trapezoids after taking absolute differences necessarily integrate the absolute difference of linearly interpolated CDFs exactly when their difference crosses zero. These distinctions are definitions, not evidence that the historical competition scorer is wrong.

The Kaggle competition search extract describes a logarithmically weighted EMD/Wasserstein-1 score at eleven support points [S3]. Direct opening of its main and evaluation pages returned no readable body. The exact official scorer, interpolation convention and numerical tolerance were therefore **not independently recovered**. This patch preserves the repository calculation; it does not assert byte-for-byte equivalence with Kaggle's hidden scorer.

The new CDF contract requires exactly 11 numeric, finite values, strictly within [0,100] and nondecreasing. The final value must be within **10⁻⁴ percentage points of 100**, with zero relative tolerance, solely to admit float32 accumulation round-off. Bounds and monotonicity have no slack. Validation returns an unmodified copy; it does not clip, sort or project. `cdf_to_pdf` uses successive differences and normalizes the already validated terminal mass. This tolerance is an explicit local engineering rule, not a verified competition tolerance. The analytically defined uniform baseline/fallback has its endpoint set to exactly 100 when constructed. Model outputs are checked before submission's existing endpoint formatting; malformed decreasing predictions are no longer silently repaired by `maximum.accumulate`.

## CV grouping, aggregation and evidence

Each fold holds out complete **normalized sample IDs**. Its reference curve uses only the other sample IDs, and the model training call receives only their images and labels. A fixture that raises on any held-out label lookup confirms the helper does not read one. Altering a held-out label leaves the fitted reference unchanged. All folds together must score each eligible sample exactly once.

Scores are first computed per sample and then pooled with equal sample weight. Fold means are reported separately; unequal fold sizes are not silently given equal weight. Raw records include target, model, training-mean and uniform CDFs; all three scores; normalized and original sample IDs; fold index; and associated photo/device metadata. JSON uses unrounded floating values and rejects nonfinite numbers. The legacy summary field names remain available, now with unrounded sample-weighted values and an explicit metric name.

Each CV invocation reserves a unique `cv_evaluation_<time_ns>.json`. Full planned folds, IDs, source hashes and versions are persisted before training; each successful prediction is persisted before the next one. Writes replace a temporary file atomically within that owned evidence path. An exception records `failed`, active fold/sample, exception type and message while retaining prior successes, then re-raises. This is evidence preservation, not partial-training resume. The legacy `result.json` remains a convenience summary and can be replaced by a later invocation; the unique evidence file preserves the CV details.

The current shared `predict_sample` evaluates each tile and its horizontal mirror for both CV and submission. September notes that a particular historical CV ran without mirrors are not a description of the current source. The historical training/evaluation mismatch cannot be recalculated without its exact code, data and predictions.

Random sample folds do **not** constitute a genuine held-out-device study. The source records device identities but does not partition by device. Absence of an iPhone/device holdout limits cross-device generalization; it does not by itself prove that the same physical sample leaked across train and validation. Remaining unresolved risks include aliases that normalize differently, distinct physical samples that normalize alike, filename-derived phone identification and the retained 15-px/mm fallback for unknown phones. Metadata now exposes that fallback. Unlabelled photos and labels without photos are reported; their identities have not been independently validated.

Training still uses random crops (default 128) with rotations/reflections, while prediction uses its existing 256-pixel tiling and mirror average. The existing tiling can omit residual border strips, and aggregation weights tiles rather than making each photo contribute equally. These behaviours and the lack of an image-content duplicate audit remain documented limitations; they were not changed or tested on competition images. The submission no-photo fallback remains a declared uniform curve, not a model prediction.

## Failure witnesses and validation

All numerical fixtures ran with **one numerical CPU thread**, admission RAM above **8 GiB**, and no Torch import. The integration test compiles only the actual `main` AST into a namespace with inert training/prediction stubs; no model is instantiated. Full module import under the pinned cloud environment remains part of root's wider item-8 validation.

| Preserved stage | Result | Evidence |
|---|---|---|
| Three expected witnesses extracted from original source | 3 failures: reference changed with held-out label; “fold” score was 30 instead of current-fold 90; malformed CDF was silently repaired | [Log](../../results/research/08_validation/soil_legacy_witness.log), [receipt](../../results/research/08_validation/soil_legacy_witness.json), [witness source](../../results/research/08_validation/soil_legacy_witness_source.py) |
| First corrected suite | 19 passed, 0 failed, 0.043 s | [Log](../../results/research/08_validation/soil_corrected_tests_01.log), [receipt](../../results/research/08_validation/soil_corrected_tests_01.json) |
| Review-added witnesses, before their fixes | 18 passed, 1 failure, 2 errors: uniform endpoint `100.00000000000001`; initial evidence lacked metadata; prediction failure lost partial-fold evidence | [Log](../../results/research/08_validation/soil_review_witness_02.log), [receipt](../../results/research/08_validation/soil_review_witness_02.json), [source](../../results/research/08_validation/soil_review_source_02.py), [tests](../../results/research/08_validation/soil_review_tests_02.py) |
| Corrected suite after review | **21 passed, 0 failed, 0.062 s** | [Log](../../results/research/08_validation/soil_corrected_tests_03.log), [receipt](../../results/research/08_validation/soil_corrected_tests_03.json) |

The final suite covers invalid CDFs, metric identity/symmetry and an independent NumPy trapezoid calculation, the discrete-Wasserstein counterexample, normalization collisions, fold validity, training-only references, raw precision, unequal-fold sample weighting, full CV wiring and persistence after an injected failure on the second prediction. The exact no-photo fallback constructor is also checked. Test names and source hashes are listed in [soil_final_inventory.json](../../results/research/08_validation/soil_final_inventory.json).

These are software invariants and synthetic witnesses. They do not estimate a corrected historical EMD, a device-generalization gain or the effect of any NeuroPixel component. Neither the historical performance claims nor a new submission were validated here.

## Primary documentation consulted

All accessed 2026-10-07.

- **[S1] NumPy 2.3 manual, `numpy.trapezoid`.** [Official versioned documentation](https://numpy.org/doc/2.3/reference/generated/numpy.trapezoid.html). Defines composite trapezoidal integration and use of supplied x spacing; retrieved full relevant text. This supports the arithmetic description, not a competition-specific rule.
- **[S2] SciPy 1.16.2 manual, `scipy.stats.wasserstein_distance`.** [Official versioned documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.stats.wasserstein_distance.html). Defines one-dimensional discrete Wasserstein-1, probability-mass weighting and the equivalent integral of absolute CDF differences. Retrieved definition, Notes and examples; no SciPy dependency is needed by the helper.
- **[S3] Competition organizer, “Predicting Soil Grain Size Distributions from Images.”** [Official Kaggle page](https://www.kaggle.com/competitions/soil-grain-size-from-photos), [evaluation URL](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview/evaluation). Only the indexed evaluation extract was available; both direct page opens returned zero readable lines. It is retained as limited context, not verification of the exact official scorer.
