"""Point estimands for balanced eight-query scenario blocks; no resampling."""
from __future__ import annotations

import numpy as np

from neuropixel.research.classification_contracts import validate_classification_inputs


def point_metrics(pred, target, roles, nll, group_index):
    if nll is None:
        raise ValueError("one negative log likelihood per query is required")
    pred, target, roles, nll = validate_classification_inputs(
        pred, target, roles, nll, role_count=4, intervals=False,
        bootstrap_seed=0, repetitions=1)
    group_index = np.asarray(group_index)
    if (group_index.shape != pred.shape or group_index.dtype.kind not in "iu"
            or (group_index < 0).any()):
        raise ValueError("scenario indices must be nonnegative integers matching query rows")
    correct = pred == target
    by_role = [float(correct[roles == role].mean()) for role in range(4)]
    groups = np.unique(group_index)
    blocks = []
    for group in groups:
        pick = group_index == group
        if pick.sum() != 8 or not np.array_equal(np.bincount(roles[pick], minlength=4), [2, 2, 2, 2]):
            raise ValueError("each scenario must contain exactly two queries for each of four roles")
        blocks.append(bool(correct[pick].all()))
    return {"n": len(pred), "groups": len(groups), "correct": int(correct.sum()),
            "global": float(correct.mean()), "per_role": by_role,
            "binding": (by_role[0] + by_role[2]) / 2,
            "macro": sum(by_role) / 4, "cross_entropy": float(nll.mean(dtype=np.float64)),
            "all_eight": float(np.mean(blocks))}
