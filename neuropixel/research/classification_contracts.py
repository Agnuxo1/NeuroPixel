"""Input contracts for RoleTask classification summaries (no Torch dependency).

The scoring formulas are unchanged. Token vocabulary membership beyond PAD and
nonnegative IDs belongs to the dataset/model contract; this module cannot infer
a vocabulary from predictions. Prediction zero is a possible wrong answer;
target zero is not a valid RoleTask answer.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def validate_binary_counts(correct, n, confidence):
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           for value in (correct, n)):
        raise ValueError("binary counts must be integers, not booleans")
    if n <= 0 or not 0 <= correct <= n:
        raise ValueError("invalid binary counts")
    if (isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, Real)
            or not math.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError("confidence must be finite and strictly between zero and one")


def validate_classification_inputs(prediction, target, roles, nll, *, role_count,
                                   intervals, bootstrap_seed, repetitions):
    arrays = tuple(np.asarray(value) for value in (prediction, target, roles))
    prediction, target, roles = arrays
    if any(value.ndim != 1 for value in arrays) or not len(target):
        raise ValueError("prediction, target and roles must be nonempty vectors")
    if any(value.shape != target.shape for value in arrays):
        raise ValueError("prediction, target and roles must be equal-length vectors")
    if any(value.dtype.kind not in "iu" for value in arrays):
        raise ValueError("prediction, target and role IDs must have integer dtypes")
    if np.any(prediction < 0) or np.any(target < 1):
        raise ValueError("predictions must be nonnegative and targets must exclude PAD")
    if type(role_count) is not int or role_count < 1:
        raise ValueError("role_count must be a positive integer")
    if not np.array_equal(np.unique(roles), np.arange(role_count)):
        raise ValueError("roles must contain every declared role and no unknown IDs")
    if type(intervals) is not bool:
        raise ValueError("intervals must be a boolean")
    if intervals:
        if (isinstance(repetitions, (bool, np.bool_)) or not isinstance(repetitions, Integral)
                or repetitions < 1):
            raise ValueError("bootstrap repetitions must be a positive integer")
        if (isinstance(bootstrap_seed, (bool, np.bool_)) or not isinstance(bootstrap_seed, Integral)
                or bootstrap_seed < 0):
            raise ValueError("bootstrap seed must be a nonnegative integer")
    if nll is not None:
        raw = np.asarray(nll)
        if raw.shape != target.shape or raw.dtype.kind not in "iuf":
            raise ValueError("NLL must be one real-valued entry per prediction")
        nll = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(nll).all() or np.any(nll < 0):
            raise ValueError("NLL must be finite and nonnegative")
    return prediction, target, roles, nll
