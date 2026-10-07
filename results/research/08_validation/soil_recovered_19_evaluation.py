"""NumPy-only, sample-level Soil evaluation contracts.

The metric preserves the repository's trapezoidal absolute-CDF difference on
log10 diameter. It is not generally discrete Wasserstein-1 on point masses.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np

SIZES = (0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0)
LOGW = np.diff(np.log10(SIZES))
CDF_ENDPOINT_ATOL = 1e-4  # percentage points; float32 accumulation round-off only
METRIC_NAME = 'trapezoidal_absolute_cdf_log10_mm_percent'
SCORE_NAMES = ('model', 'train_mean', 'uniform')


def normalize_sample_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('sample IDs must be strings')
    value = value.lower().replace('ü', 'ue').replace('ö', 'oe').replace('ä', 'ae').replace('ß', 'ss')
    key = ''.join(ch for ch in value if ch.isalnum())
    if not key:
        raise ValueError('empty normalized sample ID')
    return key


def unique_normalized_ids(raw_ids: Sequence[str]) -> dict[str, str]:
    """Reject duplicate rows and distinct names collapsing to the same key."""
    result = {}
    for raw in raw_ids:
        key = normalize_sample_id(raw)
        if key in result:
            raise ValueError(f'duplicate or colliding normalized ID: {result[key]!r}, {raw!r}')
        result[key] = raw
    if not result:
        raise ValueError('empty sample inventory')
    return result


def validate_cdf(cdf, *, name='CDF') -> np.ndarray:
    """Check, never project/clip: shape11, finite, bounds, monotonic, end100.

    The terminal tolerance is explicit and absolute (no relative tolerance).
    Bounds and monotonicity are strict. The returned values are unmodified.
    """
    try:
        values = np.asarray(cdf, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: nonnumeric CDF') from exc
    if values.shape != (len(SIZES),):
        raise ValueError(f'{name}: expected {len(SIZES)} support values')
    if not np.isfinite(values).all():
        raise ValueError(f'{name}: nonfinite values')
    if np.any(values < 0) or np.any(values > 100):
        raise ValueError(f'{name}: values outside [0,100]')
    if np.any(np.diff(values) < 0):
        raise ValueError(f'{name}: decreasing CDF')
    if abs(float(values[-1]) - 100.0) > CDF_ENDPOINT_ATOL:
        raise ValueError(f'{name}: terminal value must be 100 within {CDF_ENDPOINT_ATOL} pp')
    return values.copy()


def cdf_to_pdf(cdf) -> np.ndarray:
    values = validate_cdf(cdf)
    # Unit-mass representation; only already validated terminal round-off is normalized.
    return np.diff(np.r_[0., values]) / values[-1]


def emd(cdf_true, cdf_pred) -> float:
    """Repository metric in percentage-point × log10(mm) units, lower is better."""
    difference = np.abs(validate_cdf(cdf_true, name='target') - validate_cdf(cdf_pred, name='prediction'))
    return float(np.sum(0.5 * (difference[:-1] + difference[1:]) * LOGW))


def _canonical_ids(ids, name):
    values = list(ids)
    unique_normalized_ids(values)
    if any(normalize_sample_id(value) != value for value in values):
        raise ValueError(f'{name}: IDs must already be normalized')
    return values


def validate_folds(sample_ids, folds) -> list[list[str]]:
    samples = _canonical_ids(sample_ids, 'samples')
    groups = [list(fold) for fold in folds]
    if not 2 <= len(groups) <= len(samples):
        raise ValueError('need 2..n_samples nonempty folds')
    seen = set()
    for fold in groups:
        ids = _canonical_ids(fold, 'fold')
        if not set(ids) <= set(samples):
            raise ValueError('unknown validation sample')
        if seen.intersection(ids):
            raise ValueError('overlapping validation folds')
        seen.update(ids)
        if len(ids) == len(samples):
            raise ValueError('empty training fold')
    if seen != set(samples):
        raise ValueError('folds must cover each sample exactly once')
    return groups


def make_folds(sample_ids, n_folds: int, seed=0) -> list[list[str]]:
    samples = _canonical_ids(sample_ids, 'samples')
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or not 2 <= n_folds <= len(samples):
        raise ValueError('n_folds must be an integer from 2 to n_samples')
    groups = [part.tolist() for part in np.array_split(np.random.default_rng(seed).permutation(samples), n_folds)]
    return validate_folds(samples, groups)


def train_mean_cdf(labels: Mapping, train_ids, validation_ids) -> np.ndarray:
    """Fit only from train IDs; no validation-label lookup is performed."""
    train_ids = _canonical_ids(train_ids, 'train')
    validation_ids = _canonical_ids(validation_ids, 'validation')
    if set(train_ids).intersection(validation_ids):
        raise ValueError('train/validation sample overlap')
    curves = [validate_cdf(labels[s], name=f'train label {s}') for s in train_ids]
    return validate_cdf(np.mean(curves, axis=0), name='train mean')


def score_sample(sample_id, fold_index, target, prediction, reference) -> dict:
    _canonical_ids([sample_id], 'sample')
    if isinstance(fold_index, bool) or not isinstance(fold_index, int) or fold_index < 0:
        raise ValueError('invalid fold index')
    target = validate_cdf(target, name='target')
    prediction = validate_cdf(prediction, name='prediction')
    reference = validate_cdf(reference, name='train mean')
    uniform = np.arange(1, len(SIZES) + 1, dtype=float) * (100. / len(SIZES))
    uniform[-1] = 100.
    return {'sample_id': sample_id, 'group_id': sample_id, 'fold': fold_index,
            'target_cdf': target.tolist(), 'prediction_cdf': prediction.tolist(),
            'train_mean_cdf': reference.tolist(), 'uniform_cdf': uniform.tolist(),
            'scores': {'model': emd(target, prediction), 'train_mean': emd(target, reference),
                       'uniform': emd(target, uniform)}}


def summarize_scores(records: Sequence[dict]) -> dict:
    """Each held-out sample contributes once, irrespective of photo/fold count."""
    if not records:
        raise ValueError('empty score collection')
    _canonical_ids([r['sample_id'] for r in records], 'scored samples')
    values = np.asarray([[r['scores'][key] for key in SCORE_NAMES] for r in records], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('scores must be finite and nonnegative')
    return {'n_samples': len(records), 'sample_weighting': 'equal_per_sample',
            'metric': METRIC_NAME,
            'means': {key: float(values[:, i].mean()) for i, key in enumerate(SCORE_NAMES)}}
